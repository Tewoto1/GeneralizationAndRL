"""
Building the answer pool: `constitute`, `sample`, `pair`.

These are the stages that decide WHAT gets judged. r0 established that the
choice matters more than anything downstream: two plain samples of the same
model produced 41/42/67-tie across 150 judgments, which is a coin flip, and no
judge or optimiser recovers signal from a pool that has none.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

from .. import sample as S
from ..common import config as cfg_mod
from ..common.io import Progress, Run, box, c
from .base import DEFAULT_DOMAIN, generator, judgments_per_pair, load_all, pick_prompts


def cmd_constitute(a) -> None:
    """Model writes its own judging criteria from your labelled examples.

    Every constitution in the run is generated from the IDENTICAL examples and
    differs only by sampling seed. The question is how far the model diverges
    from a shared starting point, not how it converges from different evidence
    — so the examples are held fixed and the seed is the variable.
    """
    cfg = load_all()
    writer = cfg_mod.load("prompts/sample/constitution_writer.json")
    run = Run.open(a.run)
    if a.fresh:
        run.clear("constitute", "constitutions")
    if run.is_complete("constitute"):
        print("[constitute] already complete — use --fresh to redo")
        return

    labels = S.load_labels_list(cfg["judge"]["validate"]["labels"])
    if len(labels) < a.min_labels:
        sys.exit(f"[constitute] only {len(labels)} usable labels; need at least "
                 f"{a.min_labels}. Run `./run.sh label <run>` first — the model "
                 f"cannot generalise preferences it has not been shown.")

    # Where do the labelled pairs live? Defaulting to the current run was wrong
    # by construction: labels reference pair_ids from an EARLIER run, and the
    # run being created has no pairs yet, so the default could never work. The
    # labels name the pair_ids they need, so look for them.
    want = {l["pair_id"] for l in labels}
    src = _run_with_pairs(a.labels_from, want)
    pairs = {p["pair_id"]: p for p in src.read("pairs")}
    examples = S.format_examples(labels, pairs, writer["example_block"], a.n_examples)
    if not examples:
        sys.exit(f"[constitute] no pairs in '{src.dir.name}' match any label. "
                 f"Labels reference {sorted(want)[:5]}...")
    print(c(f"[constitute] {len(want & set(pairs))} labelled pairs from "
            f"run '{src.dir.name}'", "grey"))

    prompt = cfg_mod.render(writer["template"], {"examples": examples})
    generate, prov = generator(cfg, a.stub, role="answer")
    run.note(constitution_model=prov, writer_version=writer.get("version"))

    (run.dir / "constitutions").mkdir(exist_ok=True)
    done = run.done("constitutions", "version")
    seeds = S.seeds_for(a.n)
    bar = Progress("constitute", a.n, style="yellow")
    n_ok = sum(1 for v in done)

    for i, seed in enumerate(seeds):
        cid = f"c{i}"
        if cid in done:
            bar.step(c(f"{cid} already written, skipped", "grey"))
            continue
        if not a.stub:
            import torch
            torch.manual_seed(seed)
        text = generate(prompt, 1)[0]
        rec = S.constitution_record(cid, S.parse_constitution(text), text,
                                    seed, len(labels))
        (run.dir / "constitutions" / f"{cid}.json").write_text(json.dumps(rec, indent=2))
        run.write("constitutions", {k: v for k, v in rec.items() if k != "raw"})
        n_ok += bool(rec["ok"])
        ids = ",".join(x["id"] for x in rec["criteria"]) or c("UNPARSEABLE", "red")
        bar.step(f"{c(cid, 'bold')} {ids}")
    bar.done()

    if not n_ok:
        sys.exit("[constitute] no constitution parsed. Raw text is in "
                 "constitutions/*.json — read it before re-running.")
    run.mark_complete("constitute", n=a.n, n_ok=n_ok, n_examples=len(labels))
    _push(a, run, f"{n_ok} constitutions")


def _run_with_pairs(explicit: str | None, want: set) -> Run:
    """The run holding the pairs the labels refer to.

    Explicit `--labels-from` wins. Otherwise every run directory is searched and
    the one covering the most labelled pair_ids is used, because that is a fact
    on disk rather than a guess.
    """
    from ..common.io import RUNS
    if explicit:
        return Run.open(explicit)

    best, best_n = None, 0
    for f in sorted(RUNS.glob("*/pairs.jsonl")):
        r = Run.open(f.parent.name)
        n = len(want & {p["pair_id"] for p in r.read("pairs")})
        if n > best_n:
            best, best_n = r, n
    if best is None:
        have = sorted(d.name for d in RUNS.glob("*") if d.is_dir())
        sys.exit(f"[constitute] no run contains the pairs these labels refer to. "
                 f"Runs on disk: {have or '(none)'}. If the labelled run is only "
                 f"on the Hub, `./run.sh pull <run>` first, or pass --labels-from.")
    return best


def cmd_sample(a) -> None:
    """Generate the answer pool: drafts, prefills, and self-review revisions.

    The draft is both the control AND the input to every self-review variant,
    so the control costs nothing extra.

    Resumable at (prompt, variant) granularity: an interrupted run picks up
    where it stopped instead of appending a second copy of everything it
    already did.
    """
    cfg = load_all(a.domain)
    vcfg = cfg_mod.load("configs/variants.json")
    review = cfg_mod.load("prompts/sample/self_review.json")
    run = Run.open(a.run, config={"domain": a.domain, "model": cfg["model"]})
    if a.fresh:
        run.clear("sample", "answers", "reviews")
    if run.is_complete("sample"):
        print(f"[sample] already complete ({run.count('answers')} answers)")
        return

    cdir = run.dir / "constitutions"
    consts = [json.loads(p.read_text()) for p in sorted(cdir.glob("*.json"))] \
        if cdir.is_dir() else []
    variants = S.expand_variants(vcfg, consts)
    prompts = pick_prompts(cfg["domain"], a.limit)

    n_draft = vcfg.get("n_drafts", 2)
    todo = [(p, v) for p in prompts for v in
            [{"id": f"draft_{i}", "kind": "draft", "conditioning": "none"}
             for i in range(n_draft)]
            + [v for v in variants if v["kind"] != "draft"]]
    done = run.done("answers", "prompt_id", "variant")
    todo = [(p, v) for p, v in todo if (p["id"], v["id"]) not in done]
    if not todo:
        run.mark_complete("sample", n_answers=run.count("answers"), resumed=True)
        print(c("[sample] nothing left to generate", "green"))
        return

    generate, prov = generator(cfg, a.stub, role="answer")
    run.note(answer_model=prov, variants_version=vcfg.get("version"),
             n_constitutions=len(consts))

    # draft_0 is the input to every self-review, so it must exist before the
    # reviews for that prompt run. Sorting by prompt then variant guarantees it
    # within a fresh run; on a resume it may already be on disk, so it is read
    # back from the stream rather than assumed to be in memory.
    todo.sort(key=lambda pv: (pv[0]["id"], pv[1]["id"]))
    drafts: dict[str, str] = {r["prompt_id"]: r["text"] for r in run.read("answers")
                              if r["variant"] == "draft_0"}

    bar = Progress("sample", len(todo), style="blue")
    n_failed = 0
    for p, v in todo:
        base = {"prompt_id": p["id"], "prompt": p["text"],
                "tension": p.get("tension"), "is_control": bool(p.get("_control"))}
        ok = True
        if v["kind"] == "draft":
            text = generate(p["text"], 1)[0].strip()
            if v["id"] == "draft_0":
                drafts[p["id"]] = text
        elif v["kind"] == "prefill":
            out = generate(p["text"], 1, prefill=v["prefill"])[0]
            # The prefill is part of the PROMPT, so the model never emits it;
            # prepend it or the recorded answer starts mid-sentence.
            text = (v["prefill"] + out).strip()
        else:
            draft = drafts.get(p["id"])
            if draft is None:
                bar.step(c(f"{p['id']} {v['id']}: no draft_0, skipped", "red"))
                continue
            con = next(x for x in consts if x["version"] == v["constitution"])
            raw = generate(cfg_mod.render(review["template"], {
                "prompt": p["text"], "draft": draft,
                "criteria": S.criteria_block(con)}), 1)[0]
            text, ok = S.extract_revision(raw, draft)
            n_failed += not ok
            run.write("reviews", {**base, "variant": v["id"],
                                  "constitution": v["constitution"], "raw": raw})

        run.write("answers", {**base, "variant": v["id"],
                              "conditioning": v["conditioning"],
                              "constitution": v.get("constitution"),
                              "revision_ok": ok, "text": text})
        bar.step(f"{c(p['id'], 'bold')} {v['id']} {c(f'{len(text)}c', 'grey')}"
                 + ("" if ok else " " + c("no <<<REVISED>>> block", "red")))
    bar.done()

    run.mark_complete("sample", n_answers=run.count("answers"),
                      n_variants=len(variants), n_review_failed=n_failed)
    box("sample", [
        ("prompts", len(prompts)),
        ("variants", len(variants)),
        ("answers", c(run.count("answers"), "bold")),
        ("revisions with no marker", c(n_failed, "red" if n_failed else "green")),
    ], style="blue")
    _push(a, run, f"{run.count('answers')} answers")


def cmd_pair(a) -> None:
    """Select which pairs to judge, from the answer pool. No GPU.

    Rebuilt from scratch every time: it is a deterministic function of
    answers.jsonl, so there is nothing to resume and a stale pairs.jsonl would
    silently mismatch the pool it claims to describe.
    """
    cfg = load_all()
    vcfg = cfg_mod.load("configs/variants.json")
    run = Run.open(a.run)
    if not run.is_complete("sample"):
        sys.exit(f"[pair] run '{a.run}' has no completed `sample` stage.")
    run.clear("pairs", "pairs")

    answers = list(run.read("answers"))
    pairs = S.select_pairs(answers, vcfg)
    for p in pairs:
        run.write("pairs", p)
    run.mark_complete("pairs", n_pairs=len(pairs))

    per_pair = judgments_per_pair(cfg)
    dropped = len([a_ for a_ in answers if a_["variant"] != "draft_0"]) - len(pairs)
    rows = [("pairs", c(len(pairs), "bold")),
            ("judgments to come", f"{len(pairs) * per_pair} "
                                  f"({per_pair} per pair)")]
    if dropped > 0:
        rows.append(("dropped as identical to draft_0",
                     c(dropped, "yellow")))
    rows.append("")
    rows += [(f"  vs {k}", v) for k, v in
             sorted(Counter(p["variant_b"] for p in pairs).items())]
    box("pair", rows, style="blue")


def _push(a, run, msg: str) -> None:
    if getattr(a, "push", False):
        from ..common import hub
        hub.push_run(a.run, message=f"{a.run}: {msg}")


def register(add) -> None:
    s = add("constitute", cmd_constitute)
    s.add_argument("-n", type=int, default=3, help="how many constitutions")
    s.add_argument("--n-examples", type=int, default=10, help="few-shot examples")
    s.add_argument("--min-labels", type=int, default=5)
    s.add_argument("--labels-from", help="run holding the pairs the labels refer to")

    s = add("sample", cmd_sample)
    s.add_argument("--domain", default=DEFAULT_DOMAIN)
    s.add_argument("--limit", type=int, help="random subset of N prompts")

    add("pair", cmd_pair)
