"""
Single entrypoint. Every stage is a subcommand; `run.sh` is a thin wrapper.

    python -m src.cli pairs    --run r0 --domain honesty_tact
    python -m src.cli judge    --run r0
    python -m src.cli validate --run r0
    python -m src.cli peek     --run r0

Stages read a run directory and write a run directory, and each writes
`<stage>.complete` only on the success path. `--fresh` clears a stage's
wreckage first; without it, a finished stage is skipped rather than redone.

`--stub` swaps the model for a canned generator so every stage runs on a laptop
with no torch. That is what the smoke test uses, and it is also the fastest way
to check a prompt or config change did not break the plumbing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import config as cfg_mod
from .common.io import Run
from .judge import judge as J
from .judge import validate as V

DEFAULTS = {"model": "configs/model.json", "judge": "configs/judge.json"}


def _load_all(domain: str | None = None) -> dict:
    out = {k: cfg_mod.load(v) for k, v in DEFAULTS.items()}
    cfg_mod.env_override(out["model"], {"MODEL": "name", "ADAPTER": "adapter_path"})
    out["rubric"] = cfg_mod.load(out["judge"]["rubric"])
    out["protocol"] = cfg_mod.load(out["judge"]["protocol"])
    if domain:
        out["domain"] = cfg_mod.load(f"configs/domains/{domain}.json")
    return out


def _generator(cfg: dict, stub: bool):
    if stub:
        sys.path.insert(0, str(cfg_mod.ROOT))
        from tests.fixtures.stub import stub_generator
        return stub_generator(), {"model": "STUB", "chat": {"system_policy": "none"}}
    return J.hf_generator(cfg["model"])


# ------------------------------------------------------------------- stages ---
def cmd_pairs(a) -> None:
    """Generate the answer pairs to be judged.

    Two independent samples of the same model on the same prompt. Deliberately
    not "one blunt, one tactful": prompting for the contrast would plant the
    tension the experiment is supposed to discover, and these are also exactly
    the pairs a preference-training stage would later consume.
    """
    cfg = _load_all(a.domain)
    run = Run.open(a.run, config={"domain": a.domain, "model": cfg["model"]})
    if a.fresh:
        run.clear("pairs", "pairs")
    if run.is_complete("pairs"):
        print(f"[pairs] already complete ({run.count('pairs')} pairs) — use --fresh to redo")
        return

    generate, prov = _generator(cfg, a.stub)
    run.note(answer_model=prov)

    n = 0
    for p in cfg["domain"]["prompts"]:
        outs = generate(p["text"], 2)
        if len(outs) < 2:
            print(f"[pairs] {p['id']}: model returned {len(outs)} < 2 samples, skipped")
            continue
        run.write("pairs", {
            "pair_id": p["id"], "prompt": p["text"],
            "tension": p.get("tension"), "is_control": bool(p.get("_control")),
            "answer_a": outs[0].strip(), "answer_b": outs[1].strip(),
            "len_a": len(outs[0].strip()), "len_b": len(outs[1].strip()),
        })
        n += 1
        print(f"[pairs] {p['id']} ({n}/{len(cfg['domain']['prompts'])})", flush=True)

    run.mark_complete("pairs", n_pairs=n)
    print(f"[pairs] {n} pairs -> {run.path('pairs')}")


def cmd_judge(a) -> None:
    """Judge every pair, k samples in each presentation order."""
    cfg = _load_all()
    run = Run.open(a.run)
    if not run.is_complete("pairs"):
        sys.exit(f"[judge] run '{a.run}' has no completed `pairs` stage. Run pairs first.")
    if a.fresh:
        run.clear("judge", "judgments", "results")
    if run.is_complete("judge"):
        print(f"[judge] already complete ({run.count('results')} pairs) — use --fresh to redo")
        return

    jc = cfg["judge"]
    generate, prov = _generator(cfg, a.stub)
    run.note(judge_model=prov, rubric_version=cfg["rubric"].get("version"),
             protocol_version=cfg["protocol"].get("version"))

    pairs = list(run.read("pairs"))
    n_unparseable = 0
    for i, p in enumerate(pairs, 1):
        result, records = J.judge_pair(
            p["pair_id"], p["prompt"], p["answer_a"], p["answer_b"],
            protocol=cfg["protocol"], rubric=cfg["rubric"], generate=generate,
            k=jc.get("k_samples", 3), clear_min=jc["margin"]["clear_min"],
            swap=jc.get("swap_orders", True),
        )
        for r in records:
            run.write("judgments", r)
        run.write("results", {**result.as_record(),
                              "len_a": p.get("len_a"), "len_b": p.get("len_b"),
                              "tension": p.get("tension"),
                              "is_control": p.get("is_control", False)})
        n_unparseable += result.n_unparseable
        print(f"[judge] {p['pair_id']} ({i}/{len(pairs)}) "
              f"winner={result.winner} margin={result.margin:.2f} "
              f"{'clear' if result.clear else 'BOUNDARY'}"
              f"{'' if result.swap_consistent else ' SWAP-FLIP'}", flush=True)

    run.mark_complete("judge", n_pairs=len(pairs), n_unparseable=n_unparseable)
    if n_unparseable:
        print(f"[judge] WARNING {n_unparseable} unparseable completions "
              f"(kept in judgments.jsonl with raw text)")


def cmd_validate(a) -> None:
    """Audit the judge against docs/human_label. This is the gate."""
    cfg = _load_all()
    run = Run.open(a.run)
    if not run.is_complete("judge"):
        sys.exit(f"[validate] run '{a.run}' has no completed `judge` stage.")

    vc = cfg["judge"]["validate"]
    rep = V.report(list(run.read("results")), list(run.read("judgments")),
                   V.load_labels(vc["labels"]), vc["gates"])
    rep["rubric_version"] = cfg["rubric"].get("version")
    rep["protocol_version"] = cfg["protocol"].get("version")
    (run.dir / "validation.json").write_text(json.dumps(rep, indent=2))

    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("margin_calibration", "gate_results")}, indent=2))
    print("\ngates:")
    for name, g in rep["gate_results"].items():
        print(f"  {'PASS' if g['passed'] else 'FAIL'}  {name:18s} "
              f"{g['value']:<6} ({g['kind']} {g['threshold']})")
    if rep.get("warning"):
        print(f"\nWARNING: {rep['warning']}")

    if rep["passed"]:
        run.mark_complete("validate", **{k: rep[k] for k in
                                         ("swap_invariance", "self_consistency",
                                          "clear_agreement") if k in rep})
        print("\n[validate] PASSED — survey may run against this judge.")
    else:
        (run.dir / "validate.complete").unlink(missing_ok=True)
        print(f"\n[validate] FAILED: {', '.join(rep['failed_gates'])}")
        print("Downstream stages are blocked. Fix the protocol or rubric, "
              "re-judge, and validate again.")
        sys.exit(1)


def cmd_peek(a) -> None:
    """Read a run without disturbing it."""
    run = Run.open(a.run)
    stages = [p.stem for p in sorted(run.dir.glob("*.complete"))]
    print(f"run: {run.dir}\ncomplete stages: {', '.join(stages) or '(none)'}")
    for stream in ("pairs", "judgments", "results"):
        if run.path(stream).exists():
            print(f"  {stream:12s} {run.count(stream)} records")

    results = list(run.read("results"))
    if not results:
        return
    clear = [r for r in results if r["clear"]]
    flips = [r for r in results if not r["swap_consistent"]]
    ctrl_bad = [r for r in results if r.get("is_control") and not r["clear"]]
    print(f"\nclear: {len(clear)}/{len(results)}   "
          f"boundary: {len(results) - len(clear)}   swap-flips: {len(flips)}")
    if ctrl_bad:
        print(f"  !! {len(ctrl_bad)} CONTROL pair(s) landed in boundary: "
              f"{', '.join(r['pair_id'] for r in ctrl_bad)} — "
              f"a control that is not clear means the judge is noisy, not that the case is hard.")
    if a.verbose:
        for r in results:
            print(f"  {r['pair_id']:6s} {r['winner']:5s} m={r['margin']:.2f} "
                  f"{'clear' if r['clear'] else 'bound'} {r.get('tension') or ''}")


# --------------------------------------------------------------------- main ---
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="src.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        s = sub.add_parser(name, help=(fn.__doc__ or "").strip().split("\n")[0])
        s.add_argument("--run", required=True)
        s.add_argument("--fresh", action="store_true", help="redo this stage")
        s.add_argument("--stub", action="store_true", help="canned generator, no model")
        s.set_defaults(fn=fn)
        return s

    add("pairs", cmd_pairs).add_argument("--domain", default="honesty_tact")
    add("judge", cmd_judge)
    add("validate", cmd_validate)
    add("peek", cmd_peek).add_argument("-v", "--verbose", action="store_true")

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
