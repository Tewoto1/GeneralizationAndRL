"""
Measuring: `judge`, `validate`, `spread`, `pilot`.

The judge is the instrument. Everything here either runs it or audits it, and
`validate` is the gate that stops a broken instrument producing data.
"""
from __future__ import annotations

import json
import sys
import time

from ..common.io import Progress, Run, box, c
from ..judge import judge as J
from ..judge import validate as V
from .base import DEFAULT_DOMAIN, generator, judgments_per_pair, load_all


def cmd_judge(a) -> None:
    """Judge every pair, k samples in each presentation order.

    Resumable by pair_id: an interrupted run continues rather than re-judging
    (and duplicating) the pairs it already finished.
    """
    cfg = load_all()
    run = Run.open(a.run)
    if not run.is_complete("pairs"):
        sys.exit(f"[judge] run '{a.run}' has no completed `pairs` stage.")
    if a.fresh:
        run.clear("judge", "judgments", "results")
    if run.is_complete("judge"):
        print(f"[judge] already complete ({run.count('results')} pairs)")
        return

    jc = cfg["judge"]
    pairs = [p for p in run.read("pairs")
             if p["pair_id"] not in run.done("results", "pair_id")]
    if not pairs:
        run.mark_complete("judge", n_pairs=run.count("results"), resumed=True)
        print(c("[judge] nothing left to judge", "green"))
        return

    generate, prov = generator(cfg, a.stub, role="judge")
    run.note(judge_model=prov, rubric_version=cfg["rubric"].get("version"),
             protocol_version=cfg["protocol"].get("version"))

    tally = {"unparseable": 0, "clear": 0, "flip": 0, "trunc": 0}
    bar = Progress("judge", len(pairs), style="magenta")
    for p in pairs:
        result, records = J.judge_pair(
            p["pair_id"], p["prompt"], p["answer_a"], p["answer_b"],
            protocol=cfg["protocol"], rubric=cfg["rubric"], generate=generate,
            k=jc.get("k_samples", 3), clear_min=jc["margin"]["clear_min"],
            swap=jc.get("swap_orders", True))
        for r in records:
            run.write("judgments", r)
        # Carry the pair's provenance onto the result. `spread` groups by
        # `variant_b`; dropping these here silently collapsed every variant
        # into one bucket labelled "?".
        run.write("results", {
            **result.as_record(),
            **{k: p[k] for k in ("len_a", "len_b", "tension", "prompt_id",
                                 "variant_a", "variant_b", "conditioning_b",
                                 "constitution_b") if k in p},
            "is_control": p.get("is_control", False)})

        tally["unparseable"] += result.n_unparseable
        tally["trunc"] += result.n_truncated
        tally["clear"] += bool(result.clear)
        tally["flip"] += not result.swap_consistent
        bar.step(f"{c(p['pair_id'], 'bold')} {result.winner:>9} "
                 f"m={result.margin:.2f} "
                 + (c("clear", "green") if result.clear else c("boundary", "yellow"))
                 + ("" if result.swap_consistent else " " + c("SWAP-FLIP", "red"))
                 + ("" if not result.n_unparseable else
                    " " + c(f"{result.n_unparseable} unparseable", "red")))
    bar.done()

    n = run.count("results")
    run.mark_complete("judge", n_pairs=n, n_unparseable=tally["unparseable"])
    box("judge", [
        ("pairs judged", c(n, "bold")),
        ("clear / boundary",
         f"{c(tally['clear'], 'green')} / {c(len(pairs) - tally['clear'], 'yellow')}"),
        ("swap-flips", c(tally["flip"], "red" if tally["flip"] else "green")),
        ("unparseable",
         c(tally["unparseable"], "red" if tally["unparseable"] else "green")),
        ("truncated", c(tally["trunc"], "red" if tally["trunc"] else "green")),
    ], style="magenta")
    if a.push:
        from ..common import hub
        hub.push_run(a.run, message=f"{a.run}: judged {n} pairs")


def cmd_validate(a) -> None:
    """Audit the judge against docs/human_label. This is the gate."""
    cfg = load_all()
    run = Run.open(a.run)
    if not run.is_complete("judge"):
        sys.exit(f"[validate] run '{a.run}' has no completed `judge` stage.")

    vc = cfg["judge"]["validate"]
    rep = V.report(list(run.read("results")), list(run.read("judgments")),
                   V.load_labels(vc["labels"]), vc["gates"])
    rep["rubric_version"] = cfg["rubric"].get("version")
    rep["protocol_version"] = cfg["protocol"].get("version")
    (run.dir / "validation.json").write_text(json.dumps(rep, indent=2))

    rows = [
        ("pairs / judgments", f"{rep['n_pairs']} / {rep['n_judgments']}"),
        ("human labels matched",
         c(rep["n_labelled"], "green" if rep["n_labelled"] else "red")),
        ("parse rate", f"{rep['parse_rate']:.0%}"),
        ("step compliance", f"{rep['step_compliance']:.0%}"),
        "",
    ]
    for name, g in rep["gate_results"].items():
        mark = c("PASS", "green", "bold") if g["passed"] else c("FAIL", "red", "bold")
        rows.append((f"  {mark}  {name}",
                     c(f"{g['value']:<6} ({g['kind']} {g['threshold']})", "grey")))
    box(f"validate  rubric={rep['rubric_version']} protocol={rep['protocol_version']}",
        rows, style="green" if rep["passed"] else "red")

    if rep.get("warning"):
        print(c("WARNING: " + rep["warning"], "yellow"))

    if rep["passed"]:
        run.mark_complete("validate", **{k: rep[k] for k in
                                         ("swap_invariance", "self_consistency",
                                          "clear_agreement") if k in rep})
        print(c("PASSED — survey may run against this judge.", "green", "bold"))
        if a.push:
            from ..common import hub
            hub.push_run(a.run, message=f"{a.run}: judge validated")
    else:
        (run.dir / "validate.complete").unlink(missing_ok=True)
        print(c(f"FAILED: {', '.join(rep['failed_gates'])}", "red", "bold"))
        print("Downstream stages are blocked. Fix the protocol or rubric, "
              "re-judge, and validate again.")
        sys.exit(1)


def cmd_spread(a) -> None:
    """Per-variant summary: did any conditioning produce judgeable difference?"""
    run = Run.open(a.run)
    results = list(run.read("results"))
    if not results:
        sys.exit(f"[spread] no results.jsonl in '{a.run}'. Run `judge` first.")
    if not run.is_complete("judge"):
        print(c(f"note: judge did not finish — {len(results)} pairs so far", "yellow"))

    rep = V.spread(results)
    (run.dir / "spread.json").write_text(json.dumps(rep, indent=2))

    rows = [("variant", c("n  decided  margin  clear  swap  win", "grey")), ""]
    for name, v in rep["variants"].items():
        win = f"{v['win_rate']:.0%}" if v["win_rate"] is not None else "  -"
        rows.append((c(name, "grey" if name == "draft_1" else "bold"),
                     f"{v['n']:>2}  {v['decided']:>6.0%}  {v['mean_margin']:>6.2f}  "
                     f"{v['clear']:>5.0%}  {v['swap_stable']:>4.0%}  {win:>4}"))
    if "control_mean_margin" in rep:
        rows += ["", ("control (draft_1) mean margin",
                      c(f"{rep['control_mean_margin']:.2f}", "bold"))]
    box("spread", rows, style="cyan")

    if rep.get("warning"):
        print(c("WARNING: " + rep["warning"], "yellow"))
    print(c("Read `decided` first: it is the fraction the judge could rank at all. "
            "A variant whose `decided` is no better than the control produced no "
            "usable training signal, however it scored elsewhere.", "grey"))
    run.mark_complete("spread", n_variants=len(rep["variants"]))


def cmd_pilot(a) -> None:
    """Real model, a random subset of prompts, end to end. Run before the sweep.

    `./run.sh test` proves the plumbing with a canned generator. It cannot tell
    you whether the real model fits in VRAM, emits a parseable verdict block, or
    how long a pair takes — the three things that actually waste a paid box.

    Calls the ordinary stages with `--limit`, so there is no second copy of the
    pipeline to drift out of step with the real one. All it adds is a clock and
    `judge.validate.preflight`.
    """
    from .pool import cmd_pair, cmd_sample

    cfg = load_all(a.domain)
    total = len(cfg["domain"]["prompts"])
    a.fresh, a.limit, a.push = True, a.n, False

    t0 = time.time()
    cmd_sample(a)
    t_sample = (time.time() - t0) / max(a.n, 1)
    cmd_pair(a)

    run = Run.open(a.run)
    n_pairs = run.count("pairs")
    t1 = time.time()
    a.fresh = True
    cmd_judge(a)
    t_judge = (time.time() - t1) / max(n_pairs, 1)

    prov = json.loads((run.dir / "manifest.json").read_text()).get("judge_model", {})
    gen = cfg["model"]["gen"]
    pre = V.preflight(
        list(run.read("results")), list(run.read("judgments")), prov,
        max_new_tokens=gen.get("judge", {}).get("max_new_tokens",
                                                gen["max_new_tokens"]))

    pairs_per_prompt = n_pairs / max(a.n, 1)
    est_min = (t_sample * total + t_judge * pairs_per_prompt * total) / 60
    ok = lambda v, good: c(v, "green" if good else "red")  # noqa: E731

    box("pilot", [
        ("answer generation", f"{t_sample:6.1f} s/prompt"),
        (f"judging ({judgments_per_pair(cfg)} judgments/pair)",
         f"{t_judge:6.1f} s/pair"),
        ("pairs per prompt", f"{pairs_per_prompt:6.1f}"),
        "",
        (f"full domain ({total} prompts)",
         c(f"{est_min:.0f} min = ${est_min / 60 * a.rate:.2f} at ${a.rate}/hr", "bold")),
        "",
        ("truncated (hit token cap)",
         ok(f"{pre['truncated']}/{pre['n_judgments']}", not pre["truncated"])),
        ("unparseable", ok(f"{pre['unparseable']}/{pre['n_judgments']}",
                           not pre["unparseable"])),
        ("missing self-check", ok(f"{pre['no_selfcheck']}/{pre['n_judgments']}",
                                  not pre["no_selfcheck"])),
        ("system_policy", ok(pre["system_policy"], pre["system_policy"] == "none")),
    ], style="green" if pre["ok"] else "red")

    if not pre["ok"]:
        print(c("DO NOT START THE FULL RUN:", "red", "bold"))
        for p in pre["problems"]:
            print(c("  - " + p, "red"))
        sys.exit(1)
    print(c("pilot clean. Safe to start the full run.", "green", "bold"))


def register(add) -> None:
    add("judge", cmd_judge)
    add("validate", cmd_validate)
    add("spread", cmd_spread)

    s = add("pilot", cmd_pilot)
    s.add_argument("--domain", default=DEFAULT_DOMAIN)
    s.add_argument("-n", type=int, default=2, help="prompts to pilot")
    s.add_argument("--rate", type=float, default=0.75, help="$/hr, for the estimate")
