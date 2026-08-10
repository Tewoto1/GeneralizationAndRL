"""
Run directories. Every stage reads one and writes one; this is the only writer.

    runs/<run>/
        manifest.json     config, git sha, timestamp, chat provenance
        <stream>.jsonl    append-only records
        <stage>.complete  written ONLY on clean exit

The `.complete` marker is not decoration. A previous version of this project
resumed on "does the directory exist", which was true for a stage that died one
second in (the logger creates the directory up front), and then on "is the jsonl
non-empty", which was true for a stage that died at 297 of 360 records. Both
produced silently truncated runs. Existence proves nothing; only the marker does.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import ROOT

RUNS = ROOT / "runs"


# ------------------------------------------------------------------ terminal --
# Colour is decided ONCE, here, and only when stdout is a real terminal.
# `./run.sh night` redirects stdout to a log file; escape codes in that file turn
# every `grep` and every later read of the run into a fight with `\x1b[32m`. The
# isatty check is what lets the live view be readable without making the durable
# artefact worse. NO_COLOR is honoured because it costs one clause.
_TTY = sys.stdout.isatty() and not __import__("os").environ.get("NO_COLOR")

_CODES = {"reset": 0, "bold": 1, "dim": 2, "red": 31, "green": 32, "yellow": 33,
          "blue": 34, "magenta": 35, "cyan": 36, "grey": 90}


def c(text: Any, *styles: str) -> str:
    """Colourise, or return the text untouched when not on a terminal."""
    if not _TTY or not styles:
        return str(text)
    on = "".join(f"\033[{_CODES[s]}m" for s in styles if s in _CODES)
    return f"{on}{text}\033[0m"


def _hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def box(title: str, rows: list, style: str = "cyan", width: int = 66) -> None:
    """A titled box. `rows` are strings, or (label, value) pairs to be aligned.

    Used for the things worth not scrolling past: the pilot's timing table, the
    validation gates, the end-of-run summary.
    """
    print(c("┌─ " + title + " " + "─" * max(0, width - len(title) - 4) + "┐", style))
    for r in rows:
        if isinstance(r, (tuple, list)):
            label, value = r[0], r[1]
            pad = width - 4 - len(str(label)) - len(_strip(str(value)))
            line = f"{label}{' ' * max(1, pad)}{value}"
        else:
            line = str(r)
        print(c("│ ", style) + line + " " * max(0, width - 3 - len(_strip(line)))
              + c("│", style))
    print(c("└" + "─" * (width - 1) + "┘", style))


def _strip(s: str) -> str:
    """Visible length helper: drop ANSI codes so padding maths stays right."""
    out, i = [], 0
    while i < len(s):
        if s[i] == "\033":
            while i < len(s) and s[i] != "m":
                i += 1
            i += 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


@dataclass
class Progress:
    """One line per item, with rate and ETA.

    Rate and ETA are measured, not guessed, and they are the reason this exists:
    a stage that prints nothing for forty minutes is indistinguishable from a
    stage that has hung, and the previous project lost time to exactly that.
    """
    label: str
    total: int
    style: str = "cyan"
    _t0: float = field(default_factory=time.time)
    _n: int = 0

    def step(self, detail: str = "") -> None:
        self._n += 1
        elapsed = time.time() - self._t0
        rate = elapsed / self._n
        eta = rate * (self.total - self._n)
        bar_w = 18
        filled = int(bar_w * self._n / max(self.total, 1))
        bar = "█" * filled + "░" * (bar_w - filled)
        print(f"{c(self.label, self.style, 'bold')} "
              f"{c(bar, self.style)} "
              f"{self._n:>3}/{self.total} "
              f"{c(f'{rate:5.1f}s/it', 'grey')} "
              f"{c('eta ' + _hms(eta), 'grey')}  {detail}", flush=True)

    def done(self) -> float:
        total = time.time() - self._t0
        print(c(f"{self.label} finished {self.total} in {_hms(total)}", "grey"))
        return total


def git_sha(default: str = "unknown") -> str:
    """Current commit, `-dirty` suffixed if the tree has uncommitted changes."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=5, cwd=ROOT)
        if r.returncode != 0:
            return default
        sha = r.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, timeout=5, cwd=ROOT)
        return sha + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return default


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


@dataclass
class Run:
    """A run directory.

        run = Run.open("r0", config={...})
        run.write("judgments", record)
        run.mark_complete("survey")
    """
    dir: Path

    @classmethod
    def open(cls, name: str, config: dict | None = None,
             root: Path | None = None) -> "Run":
        d = Path(root or RUNS) / name
        d.mkdir(parents=True, exist_ok=True)
        manifest = d / "manifest.json"
        if not manifest.exists():
            manifest.write_text(json.dumps({
                "run": name,
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_sha": git_sha(),
                "config": _jsonable(config or {}),
            }, indent=2))
        return cls(dir=d)

    # -- streams --------------------------------------------------------------
    def path(self, stream: str) -> Path:
        return self.dir / f"{stream}.jsonl"

    def write(self, stream: str, record: dict) -> None:
        with self.path(stream).open("a") as f:
            f.write(json.dumps(_jsonable(record)) + "\n")

    def read(self, stream: str) -> Iterator[dict]:
        p = self.path(stream)
        if not p.exists():
            return
        with p.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def count(self, stream: str) -> int:
        return sum(1 for _ in self.read(stream))

    # -- stage completion -----------------------------------------------------
    def is_complete(self, stage: str) -> bool:
        return (self.dir / f"{stage}.complete").exists()

    def mark_complete(self, stage: str, **facts) -> None:
        """Call only on the success path. `facts` are for the human reading it later."""
        (self.dir / f"{stage}.complete").write_text(json.dumps({
            "stage": stage,
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **_jsonable(facts),
        }, indent=2))

    def clear(self, stage: str, *streams: str) -> None:
        """Delete a stage's wreckage so a re-run starts clean."""
        (self.dir / f"{stage}.complete").unlink(missing_ok=True)
        for s in streams:
            self.path(s).unlink(missing_ok=True)

    def note(self, **facts) -> None:
        """Merge facts into manifest.json (chat provenance, model name, versions)."""
        p = self.dir / "manifest.json"
        m = json.loads(p.read_text()) if p.exists() else {}
        m.update(_jsonable(facts))
        p.write_text(json.dumps(m, indent=2))


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
