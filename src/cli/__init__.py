"""
Single entrypoint. Every stage is a subcommand; `run.sh` is a thin wrapper.

    python -m src.cli sample --run r1 --domain honesty_tact
    python -m src.cli judge  --run r1
    python -m src.cli spread --run r1

The pipeline, in order:

    constitute   model writes judging criteria from your labelled examples
    sample       answer pool: drafts (the control) + prefills + revisions
    pair         choose which pairs are worth judging          no GPU
    judge        k judgments per pair, in BOTH presentation orders
    validate     audit the judge against your labels  -- THE GATE
    spread       per-variant signal report                     no GPU

Stages read a run directory and write a run directory, and each writes
`<stage>.complete` only on the success path. `--fresh` clears a stage's
wreckage and redoes it; without it, a stage that was interrupted RESUMES —
`sample` skips (prompt, variant) pairs already written and `judge` skips
pair_ids already judged, so an interrupted multi-hour run continues instead
of appending a second copy of everything it already did.

`--stub` swaps the model for a canned generator, so every stage runs on a
laptop with no torch. That is what the smoke test uses, and it is the fastest
way to check a config or prompt change did not break the plumbing.

This module only assembles the parser. The stages live in:

    cli/base.py     config loading, generator construction, shared defaults
    cli/pool.py     constitute, sample, pair      -- what gets judged
    cli/score.py    judge, validate, spread, pilot -- measuring it
    cli/review.py   label, peek, sync              -- humans and ops
"""
from __future__ import annotations

import argparse

from . import pool, review, score

MODULES = (pool, score, review)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="src.cli", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, stage: bool = True):
        """`stage=False` for read-only commands that take their run positionally.

        The pipeline flags (--run/--fresh/--stub/--push) describe a stage that
        writes. `view` writes nothing and reads two positional arguments, so
        inheriting them would put a required `--run` in front of a command whose
        whole point is `view r1 spread.json`.
        """
        s = sub.add_parser(name, help=(fn.__doc__ or "").strip().split("\n")[0])
        if stage:
            s.add_argument("--run", required=True)
            s.add_argument("--fresh", action="store_true",
                           help="discard this stage's output and redo it")
            s.add_argument("--stub", action="store_true",
                           help="canned generator, no model")
            # Push on the success path only. A run that died mid-stage must not
            # appear on the Hub looking complete.
            s.add_argument("--push", action="store_true",
                           help="mirror the run directory to the Hub on success")
        s.set_defaults(fn=fn)
        return s

    for m in MODULES:
        m.register(add)
    return ap


def main(argv: list[str] | None = None) -> None:
    # Load .env at STARTUP, not lazily inside the first hub call. huggingface_hub
    # reads HF_TOKEN from the real process environment when `from_pretrained`
    # downloads weights, and that happens long before any of our sync code runs
    # — so a lazily-loaded token produced "sending unauthenticated requests"
    # during model download while `whoami` worked fine, which looks like a
    # contradiction and is only an ordering bug.
    from ..common.hub import load_env
    load_env()

    a = build_parser().parse_args(argv)
    a.fn(a)
