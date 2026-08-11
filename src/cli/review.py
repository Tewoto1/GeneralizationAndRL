"""
Human-facing and operational stages: `label`, `peek`, `sync`.

Nothing here touches a GPU. `label` is the only place a human's judgment enters
the system, and `peek` is the only thing safe to run against a live run.
"""
from __future__ import annotations

import json
import sys

from ..common.io import Run, box, c
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

    add("peek", cmd_peek).add_argument("-v", "--verbose", action="store_true")

    s = add("sync", cmd_sync)
    s.add_argument("what", choices=["push-run", "pull-run", "push-adapter",
                                    "pull-adapter", "whoami"])
    s.add_argument("--path", help="local adapter dir, for push-adapter")
    s.add_argument("--message", help="commit message")
