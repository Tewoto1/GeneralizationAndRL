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
from .common.io import Progress, Run, box, c
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


def _generator(cfg: dict, stub: bool, role: str = "answer"):
    """`role` picks the generation profile — see judge.hf_generator."""
    if stub:
        return J.stub_generator(), {"model": "STUB", "role": role,
                                    "chat": {"system_policy": "none"}}
    return J.hf_generator(cfg["model"], role=role)


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

    generate, prov = _generator(cfg, a.stub, role="answer")
    run.note(answer_model=prov)

    prompts = cfg["domain"]["prompts"]
    if getattr(a, "limit", None):
        prompts = prompts[:a.limit]
    n = 0
    bar = Progress("pairs", len(prompts), style="blue")
    for p in prompts:
        outs = generate(p["text"], 2)
        if len(outs) < 2:
            bar.step(c(f"{p['id']} model returned {len(outs)} < 2 samples, skipped", "red"))
            continue
        run.write("pairs", {
            "pair_id": p["id"], "prompt": p["text"],
            "tension": p.get("tension"), "is_control": bool(p.get("_control")),
            "answer_a": outs[0].strip(), "answer_b": outs[1].strip(),
            "len_a": len(outs[0].strip()), "len_b": len(outs[1].strip()),
        })
        n += 1
        bar.step(f"{c(p['id'], 'bold')} {c(f'{len(outs[0])}/{len(outs[1])} chars', 'grey')}")
    bar.done()

    run.mark_complete("pairs", n_pairs=n)
    print(c(f"{n} pairs -> {run.path('pairs')}", "green"))
    if a.push:
        from .common import hub
        hub.push_run(a.run, message=f"{a.run}: {n} pairs")


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
    generate, prov = _generator(cfg, a.stub, role="judge")
    run.note(judge_model=prov, rubric_version=cfg["rubric"].get("version"),
             protocol_version=cfg["protocol"].get("version"))

    pairs = list(run.read("pairs"))
    n_unparseable = 0
    n_clear = n_flip = n_trunc = 0
    bar = Progress("judge", len(pairs), style="magenta")
    for p in pairs:
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
        n_trunc += result.n_truncated
        n_clear += bool(result.clear)
        n_flip += not result.swap_consistent
        bar.step(
            f"{c(p['pair_id'], 'bold')} "
            f"{result.winner:>9} m={result.margin:.2f} "
            + (c("clear", "green") if result.clear else c("boundary", "yellow"))
            + ("" if result.swap_consistent else " " + c("SWAP-FLIP", "red"))
            + ("" if not result.n_unparseable else
               " " + c(f"{result.n_unparseable} unparseable", "red")))
    bar.done()

    run.mark_complete("judge", n_pairs=len(pairs), n_unparseable=n_unparseable)
    box("judge", [
        ("pairs judged", c(len(pairs), "bold")),
        ("clear / boundary", f"{c(n_clear, 'green')} / {c(len(pairs) - n_clear, 'yellow')}"),
        ("swap-flips", c(n_flip, "red" if n_flip else "green")),
        ("unparseable", c(n_unparseable, "red" if n_unparseable else "green")),
        ("truncated", c(n_trunc, "red" if n_trunc else "green")),
    ], style="magenta")
    if n_unparseable:
        print(c(f"WARNING {n_unparseable} unparseable completions "
                f"(kept in judgments.jsonl with raw text)", "red"))
    if a.push:
        from .common import hub
        hub.push_run(a.run, message=f"{a.run}: judged {len(pairs)} pairs")


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

    rows = [
        ("pairs / judgments", f"{rep['n_pairs']} / {rep['n_judgments']}"),
        ("human labels matched", c(rep["n_labelled"],
                                   "green" if rep["n_labelled"] else "red")),
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
            from .common import hub
            hub.push_run(a.run, message=f"{a.run}: judge validated")
    else:
        (run.dir / "validate.complete").unlink(missing_ok=True)
        print(c(f"FAILED: {', '.join(rep['failed_gates'])}", "red", "bold"))
        print("Downstream stages are blocked. Fix the protocol or rubric, "
              "re-judge, and validate again.")
        sys.exit(1)


def cmd_pilot(a) -> None:
    """Real model, first N prompts, end to end. Run BEFORE renting the night.

    `./run.sh test` proves the plumbing with a canned generator. It cannot tell
    you whether the real model fits in VRAM, emits a parseable verdict block, or
    how long a pair takes — the three things that actually waste a paid night.

    Runs the ordinary `pairs` and `judge` stages with `--limit`, so there is no
    second copy of the pipeline to drift out of step with the real one. All it
    adds is a clock and `judge.validate.preflight`.
    """
    import time
    total = len(_load_all(a.domain)["domain"]["prompts"])

    a.fresh, a.limit, a.push = True, a.n, False
    t0 = time.time()
    cmd_pairs(a)
    t_pairs = (time.time() - t0) / max(a.n, 1)
    t1 = time.time()
    cmd_judge(a)
    t_judge = (time.time() - t1) / max(a.n, 1)

    cfg = _load_all()
    run = Run.open(a.run)
    prov = json.loads((run.dir / "manifest.json").read_text()).get("judge_model", {})
    pre = V.preflight(list(run.read("results")), list(run.read("judgments")), prov,
                      max_new_tokens=cfg["model"]["gen"].get("judge", {}).get(
                          "max_new_tokens", cfg["model"]["gen"]["max_new_tokens"]))

    per_pair = t_pairs + t_judge
    eta_min = per_pair * total / 60
    ok = lambda v, good: c(v, "green" if good else "red")  # noqa: E731

    box("pilot", [
        ("answer generation", f"{t_pairs:6.1f} s/pair"),
        (f"judging ({cfg['judge']['k_samples']} samples x 2 orders)",
         f"{t_judge:6.1f} s/pair"),
        ("TOTAL", c(f"{per_pair:6.1f} s/pair", "bold")),
        "",
        (f"full domain ({total} prompts)",
         c(f"{eta_min:.0f} min = ${eta_min / 60 * a.rate:.2f} at ${a.rate}/hr", "bold")),
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


def cmd_label(a) -> None:
    """Label pairs by hand, fast. Blind to the judge's verdict by default.

    Blind matters: seeing the judge's answer first anchors you onto it, and the
    whole value of these labels is that they are an INDEPENDENT check on whether
    the judge is right about the cases it called easy.

    Keys: a / b / t(ie) / s(kip) / q(uit). One optional line of reasoning.
    Appends to docs/human_label/labels.json, and never asks twice about a pair
    you have already labelled, so it is safe to stop and resume.
    """
    cfg = _load_all()
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

    crit_ids = [c["id"] for c in cfg["rubric"]["criteria"]]
    print(f"[label] {len(pairs)} unlabelled. criteria: {', '.join(crit_ids)}")
    print("[label] a/b = better answer, t = tie, s = skip, q = save and quit\n")

    added, total = 0, len(done)
    for i, p in enumerate(pairs, 1):
        print("=" * 70)
        print(f"[{i}/{len(pairs)}]  {p['pair_id']}")
        print("=" * 70)
        print(f"\nREQUEST:\n{p['prompt']}\n")
        print(f"--- ANSWER A ({p['len_a']} chars) ---\n{p['answer_a']}\n")
        print(f"--- ANSWER B ({p['len_b']} chars) ---\n{p['answer_b']}\n")
        if a.show_judge and p["pair_id"] in results:
            r = results[p["pair_id"]]
            print(f"[judge said: {r['winner']} margin={r['margin']:.2f} "
                  f"{'clear' if r['clear'] else 'boundary'}]\n")

        v = input("better? [a/b/t/s/q] ").strip().lower()
        if v == "q":
            break
        if v not in ("a", "b", "t"):
            continue
        reason = input("why? (one line, optional) ").strip()
        crit = input(f"deciding criterion? ({'/'.join(crit_ids)}, blank=none) ").strip()

        total = V.add_label(path, {
            "pair_id": p["pair_id"],
            "verdict": "TIE" if v == "t" else v,
            "deciding_criterion": crit if crit in crit_ids else None,
            "reasoning": reason,
            "confidence": None,
            "notes": "",
        })
        added += 1
        print()

    print(f"[label] +{added} labels, {total} total -> {path}")
    print("[label] re-run `validate` to score the judge against them.")


def cmd_sync(a) -> None:
    """Push or pull a run directory / adapter to the Hub.

    Repo ids come from configs/hub.json; the token is HF_TOKEN, read from a
    .env found by walking up from the repo root (see src/common/hub.py).
    """
    from .common import hub

    if a.what == "whoami":
        print(json.dumps({k: hub.whoami().get(k) for k in ("name", "fullname", "type")},
                         indent=2))
        return
    if a.what == "push-run":
        hub.push_run(a.run, message=a.message)
    elif a.what == "pull-run":
        hub.pull_run(a.run)
    elif a.what == "push-adapter":
        if not a.path:
            sys.exit("push-adapter needs --path <local adapter dir>")
        hub.push_adapter(a.run, a.path, message=a.message)
    elif a.what == "pull-adapter":
        hub.pull_adapter(a.run)


def cmd_peek(a) -> None:
    """Read a run without disturbing it."""
    run = Run.open(a.run)
    stages = [p.stem for p in sorted(run.dir.glob("*.complete"))]
    rows = [("run", c(run.dir.name, "bold")),
            ("complete stages", c(", ".join(stages) or "(none)",
                                  "green" if stages else "yellow"))]
    for stream in ("pairs", "judgments", "results"):
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
                  + " " + c(r.get("tension") or "", "grey"))


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
        # Push on the success path only. A run that died mid-stage should not
        # appear on the Hub looking complete.
        s.add_argument("--push", action="store_true",
                       help="mirror the run directory to the Hub when the stage succeeds")
        s.set_defaults(fn=fn)
        return s

    add("pairs", cmd_pairs).add_argument("--domain", default="honesty_tact")
    add("judge", cmd_judge)

    s = add("pilot", cmd_pilot)
    s.add_argument("--domain", default="honesty_tact")
    s.add_argument("-n", type=int, default=2, help="prompts to pilot")
    s.add_argument("--rate", type=float, default=0.75, help="$/hr, for the estimate")

    s = add("label", cmd_label)
    s.add_argument("--boundary-only", action="store_true",
                   help="only pairs the judge called boundary")
    s.add_argument("--show-judge", action="store_true",
                   help="reveal the judge's verdict (anchors you — off by default)")
    add("validate", cmd_validate)
    add("peek", cmd_peek).add_argument("-v", "--verbose", action="store_true")

    s = add("sync", cmd_sync)
    s.add_argument("what", choices=["push-run", "pull-run", "push-adapter",
                                    "pull-adapter", "whoami"])
    s.add_argument("--path", help="local adapter dir, for push-adapter")
    s.add_argument("--message", help="commit message")

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
