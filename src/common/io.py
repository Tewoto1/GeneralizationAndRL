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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import ROOT

RUNS = ROOT / "runs"


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
