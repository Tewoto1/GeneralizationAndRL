"""
Human-facing and operational stages: `label`, `view`, `peek`, `sync`.

Nothing here touches a GPU. `label` is the only place a human's judgment enters
the system; `view` and `peek` are the only things safe to run against a live
run, because neither opens a file for writing.
"""
from __future__ import annotations

import json
import shutil
import sys
import textwrap

from ..common.io import Run, _strip, box, c
from ..judge import validate as V
from .base import load_all


def cmd_label(a) -> None:
    """Label pairs by hand, fast. Blind to the judge's verdict by default.

    Blind matters: seeing the judge's answer first anchors you onto it, and the
    whole value of these labels is that they are an INDEPENDENT check on
    whether the judge is right about the cases it called easy.

    Keys: a / b / t(ie) / s(kip) / q(uit). Writes after every label, so
    stopping and resuming costs nothing, and it never asks twice about a pair
    already in the file.
    """
    cfg = load_all()
    run = Run.open(a.run)
    path = cfg["judge"]["validate"]["labels"]
    done = set(V.load_labels(path))

    if not run.is_complete("pairs"):
        sys.exit(f"[label] run '{a.run}' has no completed `pairs` stage. There is "
                 f"nothing to label until the model has written some answers.")

    pairs = [p for p in run.read("pairs") if p["pair_id"] not in done]
    results = {r["pair_id"]: r for r in run.read("results")}

    # Labelling BEFORE `judge` is the preferred order: with no verdicts in
    # existence there is nothing to anchor on, so the labels are blind by
    # construction rather than by discipline. --boundary-only is the exception,
    # since "boundary" is a judge output.
    if a.boundary_only:
        if not run.is_complete("judge"):
            sys.exit("[label] --boundary-only needs the `judge` stage; without it "
                     "no pair has been classified yet. Drop the flag to label "
                     "everything blind, which is the better order anyway.")
        pairs = [p for p in pairs if not results[p["pair_id"]]["clear"]]
    if not pairs:
        print("[label] nothing left to label.")
        return

    crit_ids = [x["id"] for x in cfg["rubric"]["criteria"]]
    print(f"[label] {len(pairs)} unlabelled. criteria: {', '.join(crit_ids)}")
    print("[label] a/b = better answer, t = tie, s = skip, q = save and quit\n")

    added, total = 0, len(done)
    for i, p in enumerate(pairs, 1):
        print("=" * 70)
        print(f"[{i}/{len(pairs)}]  {c(p['pair_id'], 'bold')}")
        print("=" * 70)
        print(f"\nREQUEST:\n{p['prompt']}\n")
        print(f"--- ANSWER A ({p['len_a']} chars) ---\n{p['answer_a']}\n")
        print(f"--- ANSWER B ({p['len_b']} chars) ---\n{p['answer_b']}\n")
        if a.show_judge and p["pair_id"] in results:
            r = results[p["pair_id"]]
            print(c(f"[judge said: {r['winner']} margin={r['margin']:.2f} "
                    f"{'clear' if r['clear'] else 'boundary'}]\n", "grey"))

        v = input("better? [a/b/t/s/q] ").strip().lower()
        if v == "q":
            break
        if v not in ("a", "b", "t"):
            continue
        reason = input("why? (one line, optional) ").strip()
        crit = input(f"deciding criterion? ({'/'.join(crit_ids)}, blank=none) ").strip()

        # Several criteria are normal ("tact, actionable") and the old code
        # tested the whole string for membership, so anything but a single
        # exact id became None. That silently threw away the field the
        # constitution-writer leans on hardest.
        named = [x.strip().lower() for x in crit.replace(" and ", ",").split(",")]
        known = [x for x in named if x in crit_ids]
        unknown = [x for x in named if x and x not in crit_ids]
        total = V.add_label(path, {
            "pair_id": p["pair_id"],
            "verdict": "TIE" if v == "t" else v,
            "deciding_criterion": known[0] if known else None,
            "deciding_criteria": known,
            "criterion_raw": crit if unknown else None,
            "reasoning": reason,
            "confidence": None,
            "notes": "",
        })
        if unknown:
            print(c(f"  (kept {crit!r} verbatim; {unknown} are not rubric ids)",
                    "yellow"))
        added += 1
        print()

    print(c(f"[label] +{added} labels, {total} total -> {path}", "green"))
    print("[label] re-run `validate` to score the judge against them.")


# ------------------------------------------------------------------- viewing --
# A run directory is JSON that a person has to read, and `python -m json.tool`
# on a 2.5 MB judgments.jsonl is not reading. Three things make it readable:
# `_`-prefixed design keys are dimmed rather than hidden (they are the design
# document, per CLAUDE.md, but they are not what you came for); multi-KB text
# fields are folded to a few lines unless asked for; and a .jsonl is a table
# first and records second. Read-only by construction — nothing here opens a
# file for writing, so it is safe against a run a GPU is still appending to.

_LONG = 100          # a string this long, or containing a newline, is "text"
_FOLD = 6            # wrapped lines of a text field shown without --full


def _term() -> int:
    return max(60, min(shutil.get_terminal_size((100, 24)).columns, 120))


def _scalar(v) -> str:
    if v is None:
        return c("null", "grey")
    if isinstance(v, bool):
        return c(str(v), "green" if v else "red")
    if isinstance(v, (int, float)):
        return c(f"{v:.4g}" if isinstance(v, float) else str(v), "yellow")
    return str(v)


def _is_text(v) -> bool:
    return isinstance(v, str) and (len(v) > _LONG or "\n" in v)


def _text_block(v: str, pad: str, width: int, full: bool) -> list[str]:
    """Fold a long string. Real newlines are kept; each line is wrapped."""
    lines: list[str] = []
    for para in v.split("\n"):
        lines += textwrap.wrap(para, width - len(pad)) or [""]
    hidden = 0
    if not full and len(lines) > _FOLD:
        hidden, lines = len(lines) - _FOLD, lines[:_FOLD]
    out = [pad + c(ln, "grey") for ln in lines]
    if hidden:
        out.append(pad + c(f"… +{hidden} lines  (--full)", "blue"))
    return out


def _render(obj, full: bool, width: int, depth: int = 0) -> list[str]:
    """Pretty-print one JSON value. Keys aligned per level; `_` keys dimmed."""
    pad = "  " * depth
    if isinstance(obj, dict):
        if not obj:
            return [pad + c("{}", "grey")]
        klen = max(len(k) for k in obj)
        out = []
        for k, v in obj.items():
            key = c(f"{k}:", "grey") if k.startswith("_") else c(f"{k}:", "cyan")
            gap = " " * (klen - len(k) + 1)
            if _is_text(v):
                out.append(pad + key)
                out += _text_block(v, pad + "  ", width, full)
            elif isinstance(v, (dict, list)) and v:
                out.append(pad + key)
                out += _render(v, full, width, depth + 1)
            else:
                out.append(pad + key + gap + _scalar(v)
                           if not isinstance(v, (dict, list))
                           else pad + key + gap + c("[]" if isinstance(v, list) else "{}", "grey"))
        return out
    if isinstance(obj, list):
        # A list of scalars is one line; a list of objects gets one block each.
        if all(not isinstance(x, (dict, list)) and not _is_text(x) for x in obj):
            joined = ", ".join(_scalar(x) for x in obj)
            if len(joined) < width - len(pad):
                return [pad + joined]
        out = []
        for i, x in enumerate(obj):
            out.append(pad + c(f"- [{i}]", "grey"))
            out += (_text_block(x, pad + "  ", width, full) if _is_text(x)
                    else _render(x, full, width, depth + 1))
        return out
    return [pad + _scalar(obj)]


def _table(records: list[dict], width: int, keys: list[str] | None) -> None:
    """One line per record. Columns are the short scalar fields they share."""
    if keys is None:
        counts: dict[str, int] = {}
        for r in records:
            for k, v in r.items():
                if not _is_text(v) and not isinstance(v, (dict, list)):
                    counts[k] = counts.get(k, 0) + 1
        # A field only some records carry is not a column; it hides in a table.
        keys = [k for k, n in counts.items() if n >= len(records) * 0.8]
    if not keys:
        return
    def plain(v) -> str:
        return "" if v is None and False else _strip(_scalar(v))

    w = {k: min(28, max(len(k), max((len(plain(r.get(k))) for r in records),
                                    default=0))) for k in keys}
    # Drop columns from the right until the row fits, rather than letting the
    # terminal wrap every line and destroy the alignment the table exists for.
    while len(keys) > 1 and sum(w[k] + 2 for k in keys) > width:
        keys.pop()
    print(c("  ".join(f"{k[:w[k]]:<{w[k]}}" for k in keys), "bold"))
    print(c("─" * min(sum(w[k] + 2 for k in keys), width), "grey"))
    for r in records:
        cells = []
        for k in keys:
            s = plain(r.get(k))
            cut = (s[: w[k] - 1] + "…") if len(s) > w[k] else s
            body = cut if len(cut) != len(s) else _scalar(r.get(k))
            cells.append(body + " " * (w[k] - len(_strip(body))))
        print("  ".join(cells).rstrip())
    print(c(f"\n{len(records)} records", "grey"))


def _match(rec: dict, wheres: list[str]) -> bool:
    """`key=value` / `key!=value`, compared as lowercase strings."""
    for expr in wheres:
        neg = "!=" in expr
        k, _, v = expr.partition("!=" if neg else "=")
        got = str(rec.get(k.strip())).lower()
        hit = got == v.strip().lower()
        if hit == neg:
            return False
    return True


def cmd_view(a) -> None:
    """Read any file in a run directory, formatted for a human. Read-only."""
    run = Run.open(a.run)
    if not run.dir.exists():
        sys.exit(f"[view] no run directory {run.dir}")
    width = _term()

    if not a.file:                                   # `view r1` — what is here
        rows = []
        for p in sorted(run.dir.iterdir()):
            if p.is_dir():
                rows.append((c(p.name + "/", "blue"), f"{len(list(p.iterdir()))} files"))
            else:
                n = sum(1 for _ in run.read(p.stem)) if p.suffix == ".jsonl" else None
                rows.append((p.name, f"{n} records" if n is not None
                             else f"{p.stat().st_size / 1024:.0f} KB"))
        box(f"run {run.dir.name}", rows)
        print(c(f"view {a.run} <file>   [--id X] [-w key=val] [-n N] [--full]", "grey"))
        return

    path = run.dir / a.file
    if not path.exists():                            # forgive the extension
        cand = [p for p in run.dir.iterdir() if p.stem == a.file or p.name.startswith(a.file)]
        if len(cand) != 1:
            sys.exit(f"[view] no {a.file} in {run.dir}"
                     + (f"; did you mean {[p.name for p in cand]}?" if cand else ""))
        path = cand[0]

    if a.raw:
        print(path.read_text())
        return
    if path.suffix not in (".json", ".jsonl", ".complete"):
        # console.log and friends: tail, because the end is the interesting part.
        lines = path.read_text().splitlines()
        keep = lines if a.n <= 0 else lines[-a.n:]
        print(c(f"── {path.name}  (last {len(keep)} of {len(lines)} lines)", "grey"))
        print("\n".join(keep))
        return

    if path.suffix != ".jsonl":                      # a single JSON object
        print(c(f"── {path.name}", "cyan", "bold"))
        print("\n".join(_render(json.loads(path.read_text()), a.full, width)))
        return

    records = [r for r in run.read(path.stem)]
    key = "pair_id" if records and "pair_id" in records[0] else "prompt_id"
    if a.id:
        records = [r for r in records if a.id in str(r.get(key, ""))]
    if a.where:
        records = [r for r in records if _match(r, a.where)]
    if not records:
        sys.exit(f"[view] no records in {path.name} matched")

    keys = [k.strip() for k in a.keys.split(",")] if a.keys else None
    # A table is the right default for many records and the wrong one for a few:
    # if you have already narrowed to a handful, you want to read them.
    if not a.expand and (len(records) > a.n or a.table):
        print(c(f"── {path.name}", "cyan", "bold"))
        _table(records, width, keys)
        print(c(f"(showing as table; add --expand, or --id / -w to narrow)", "grey"))
        return
    for r in records[: a.n if a.n > 0 else None]:
        title = str(r.get(key, path.stem))
        if "order_first" in r:                       # judgments: 4 per pair_id
            title += f"   order_first={r['order_first']} sample={r.get('sample')}"
        shown = {k: r[k] for k in keys if k in r} if keys else r
        print(c("─" * width, "grey"))
        print(c(title, "bold"))
        print("\n".join(_render(shown, a.full, width)))


def cmd_peek(a) -> None:
    """Read a run without disturbing it. Safe mid-experiment."""
    run = Run.open(a.run)
    stages = [p.stem for p in sorted(run.dir.glob("*.complete"))]
    rows = [("run", c(run.dir.name, "bold")),
            ("complete stages", c(", ".join(stages) or "(none)",
                                  "green" if stages else "yellow"))]
    for stream in ("answers", "pairs", "judgments", "results"):
        if run.path(stream).exists():
            rows.append((f"  {stream}", f"{run.count(stream)} records"))

    results = list(run.read("results"))
    if results:
        clear = [r for r in results if r["clear"]]
        flips = [r for r in results if not r["swap_consistent"]]
        rows += ["",
                 ("clear", c(f"{len(clear)}/{len(results)}", "green")),
                 ("boundary", c(len(results) - len(clear), "yellow")),
                 ("swap-flips", c(len(flips), "red" if flips else "green"))]
    box("peek", rows)
    if not results:
        return

    ctrl_bad = [r for r in results if r.get("is_control") and not r["clear"]]
    if ctrl_bad:
        print(c(f"!! {len(ctrl_bad)} CONTROL pair(s) landed in boundary: "
                f"{', '.join(r['pair_id'] for r in ctrl_bad)} — a control that is "
                f"not clear means the judge is noisy, not that the case is hard.",
                "red", "bold"))
    if a.verbose:
        for r in results:
            print(f"  {c(r['pair_id'], 'bold'):16s} {r['winner']:>9} "
                  f"m={r['margin']:.2f} "
                  + (c("clear", "green") if r["clear"] else c("bound", "yellow"))
                  + " " + c(r.get("variant_b") or r.get("tension") or "", "grey"))


def cmd_sync(a) -> None:
    """Push or pull a run directory / adapter to the Hub.

    Repo ids come from configs/hub.json; the token is HF_TOKEN, read from a
    .env found by walking up from the repo root (see src/common/hub.py).
    """
    from ..common import hub

    if a.what == "whoami":
        print(json.dumps({k: hub.whoami().get(k)
                          for k in ("name", "fullname", "type")}, indent=2))
    elif a.what == "push-run":
        hub.push_run(a.run, message=a.message)
    elif a.what == "pull-run":
        hub.pull_run(a.run)
    elif a.what == "push-adapter":
        if not a.path:
            sys.exit("push-adapter needs --path <local adapter dir>")
        hub.push_adapter(a.run, a.path, message=a.message)
    elif a.what == "pull-adapter":
        hub.pull_adapter(a.run)


def register(add) -> None:
    s = add("label", cmd_label)
    s.add_argument("--boundary-only", action="store_true",
                   help="only pairs the judge called boundary")
    s.add_argument("--show-judge", action="store_true",
                   help="reveal the judge's verdict (anchors you — off by default)")

    s = add("view", cmd_view, stage=False)
    s.add_argument("run", help="run name, e.g. r1")
    s.add_argument("file", nargs="?",
                   help="file in the run dir; omit to list what is there. "
                        "The extension is optional (`results` finds results.jsonl).")
    s.add_argument("--id", help="jsonl: substring match on pair_id / prompt_id")
    s.add_argument("-w", "--where", action="append", default=[], metavar="K=V",
                   help="jsonl: filter, e.g. -w clear=false -w winner!=TIE. Repeatable.")
    s.add_argument("-k", "--keys", metavar="A,B,C", help="show only these fields")
    s.add_argument("-n", type=int, default=8, metavar="N",
                   help="max records to expand / log lines to tail (0 = all)")
    s.add_argument("--full", action="store_true",
                   help="print long text fields entire instead of folding them")
    s.add_argument("--expand", action="store_true",
                   help="expand records even when there are many")
    s.add_argument("--table", action="store_true", help="force the table")
    s.add_argument("--raw", action="store_true", help="the bytes, unformatted")

    add("peek", cmd_peek).add_argument("-v", "--verbose", action="store_true")

    s = add("sync", cmd_sync)
    s.add_argument("what", choices=["push-run", "pull-run", "push-adapter",
                                    "pull-adapter", "whoami"])
    s.add_argument("--path", help="local adapter dir, for push-adapter")
    s.add_argument("--message", help="commit message")
