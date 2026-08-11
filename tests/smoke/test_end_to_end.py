"""
End-to-end smoke test: the whole pipeline, wired together, on the stub model.

    constitute -> sample -> pair -> judge -> validate -> spread -> peek

What it proves, in order:

  1. `constitute` turns labelled examples into parseable constitutions, and
     different seeds give different criteria.
  2. `sample` produces one answer per (prompt, variant), drafts included.
  3. `pair` anchors every pair on draft_0 and drops un-judgeable ones.
  4. `judge` writes per-judgment and per-pair records, and carries the pair's
     variant onto the result so `spread` can group by it.
  5. The stub's planted pathologies are caught rather than absorbed.
  6. `validate` BLOCKS on a bad judge -- the gate is real.
  7. RESUME: an interrupted stage continues instead of duplicating. This is
     the property that matters most on a multi-hour run and it is the easiest
     to break silently.

Runs in seconds. No torch, no GPU, no network.
"""
import json

import pytest

from src.cli import main
from src.common.io import Run


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One fixture builds the whole pipeline, so there is a single finalizer.

    The label path is redirected with the LABELS env var rather than by
    monkeypatching config loading: if a test has to reach inside the loader to
    change where ground truth comes from, that path is not configurable enough,
    and a held-out label split will need exactly this knob later.
    """
    from _pytest.monkeypatch import MonkeyPatch
    m = MonkeyPatch()
    root = tmp_path_factory.mktemp("runs")
    m.setattr("src.common.io.RUNS", root)

    # A seed run holding the pairs the labels refer to.
    seed = Run.open("seedrun", root=root)
    for i in range(6):
        seed.write("pairs", {"pair_id": f"s{i}", "prompt": f"question {i}?",
                             "answer_a": f"A{i}", "answer_b": f"B{i}",
                             "len_a": 2, "len_b": 2})
    seed.mark_complete("pairs", n_pairs=6)

    labels = tmp_path_factory.mktemp("labels") / "labels.json"
    labels.write_text(json.dumps({"labels": [
        {"pair_id": f"s{i}", "verdict": "a" if i % 2 else "b",
         "deciding_criterion": "truth", "reasoning": f"reason {i}"}
        for i in range(6)]}))
    m.setenv("LABELS", str(labels))

    main(["constitute", "--run", "smoke", "--stub", "-n", "2",
          "--labels-from", "seedrun"])
    main(["sample", "--run", "smoke", "--stub", "--limit", "4"])
    main(["pair", "--run", "smoke"])
    main(["judge", "--run", "smoke", "--stub"])
    yield Run.open("smoke", root=root)
    m.undo()


# ------------------------------------------------------------------ stages ---
def test_1_constitutions_parse_and_differ(run):
    """Same examples, different seeds. If every constitution came out identical
    the divergence question would have no data, so difference is the point."""
    cons = list(run.read("constitutions"))
    assert run.is_complete("constitute") and len(cons) == 2
    assert all(x["ok"] and x["criteria"] for x in cons)
    assert len({x["seed"] for x in cons}) == 2
    assert {x["criteria"][0]["id"] for x in cons} != {cons[0]["criteria"][0]["id"]} \
        or len(cons) == 1


def test_2_sample_writes_one_answer_per_prompt_variant(run):
    """2 drafts + 2 prefills + 1 review per constitution = 6 per prompt."""
    answers = list(run.read("answers"))
    assert run.is_complete("sample")
    keys = {(a["prompt_id"], a["variant"]) for a in answers}
    assert len(keys) == len(answers), "duplicate (prompt, variant)"
    assert len({a["prompt_id"] for a in answers}) == 4
    assert {"draft_0", "draft_1"} <= {a["variant"] for a in answers}
    assert any(a["conditioning"] == "prefill" for a in answers)
    assert any(a["conditioning"] == "self_review" for a in answers)


def test_3_prefill_text_is_prepended_not_lost(run):
    """The prefill is part of the PROMPT, so the model never emits it. If it
    were not prepended the recorded answer would start mid-sentence."""
    from src.common import config as C
    pres = {v["id"]: v["prefill"] for v in C.load("configs/variants.json")["variants"]
            if v["kind"] == "prefill"}
    for a in run.read("answers"):
        if a["variant"] in pres:
            assert a["text"].startswith(pres[a["variant"]])


def test_4_every_pair_is_anchored_on_draft_0(run):
    pairs = list(run.read("pairs"))
    assert run.is_complete("pairs") and pairs
    assert all(p["variant_a"] == "draft_0" for p in pairs)
    assert all(p["answer_a"] != p["answer_b"] for p in pairs), \
        "a pair of identical answers is four judgments bought for nothing"


def test_5_results_carry_the_variant(run):
    """`spread` groups by variant_b. Dropping it here silently collapsed every
    variant into one bucket labelled '?' — the bug this test now pins."""
    results = list(run.read("results"))
    assert results and all(r.get("variant_b") for r in results)


def test_6_judgment_count_matches_the_config(run):
    from src.common import config as C
    jc = C.load("configs/judge.json")
    per_pair = jc["k_samples"] * (2 if jc.get("swap_orders", True) else 1)
    assert run.count("judgments") == run.count("results") * per_pair


# ------------------------------------------------------------------ resume ---
def test_7_sample_resumes_instead_of_duplicating(run, capsys):
    """The `.complete` marker says whether a stage FINISHED; it does not say
    which items it got through. A stage that died at item 100 of 150 and was
    re-run must not append 100 duplicates on top of the originals."""
    (run.dir / "sample.complete").unlink()
    before = list(run.read("answers"))
    dropped = before[-3:]
    run.path("answers").write_text(
        "".join(json.dumps(r) + "\n" for r in before[:-3]))

    main(["sample", "--run", run.dir.name, "--stub", "--limit", "4"])

    after = list(run.read("answers"))
    assert len(after) == len(before), "resume duplicated or lost work"
    keys = {(a["prompt_id"], a["variant"]) for a in after}
    assert len(keys) == len(after)
    assert {(d["prompt_id"], d["variant"]) for d in dropped} <= keys


def test_8_judge_resumes_by_pair_id(run):
    (run.dir / "judge.complete").unlink()
    before = list(run.read("results"))
    run.path("results").write_text(
        "".join(json.dumps(r) + "\n" for r in before[:-2]))
    n_judgments_before = run.count("judgments")

    main(["judge", "--run", run.dir.name, "--stub"])

    after = list(run.read("results"))
    assert len(after) == len(before)
    assert len({r["pair_id"] for r in after}) == len(after)
    assert run.count("judgments") > n_judgments_before, \
        "the two missing pairs should have been re-judged"


def test_9_completed_stage_is_skipped_without_fresh(run, capsys):
    before = run.count("answers")
    main(["sample", "--run", run.dir.name, "--stub", "--limit", "4"])
    assert run.count("answers") == before
    assert "already complete" in capsys.readouterr().out


# -------------------------------------------------------------- gate + read ---
def test_10_validate_blocks_a_bad_judge(run):
    """The stub judge has planted position bias, so validate must exit non-zero
    and must NOT write validate.complete."""
    with pytest.raises(SystemExit) as e:
        main(["validate", "--run", run.dir.name])
    assert e.value.code == 1
    rep = json.loads((run.dir / "validation.json").read_text())
    assert rep["passed"] is False and not run.is_complete("validate")


def test_11_spread_groups_by_variant_and_anchors_on_the_control(run, capsys):
    main(["spread", "--run", run.dir.name])
    rep = json.loads((run.dir / "spread.json").read_text())
    assert "?" not in rep["variants"], "results lost their variant tag"
    assert "draft_1" in rep["variants"], "the control pair is missing"
    assert "control_mean_margin" in rep
    assert all("margin_over_control" in v for v in rep["variants"].values())


def test_12_peek_reports_without_mutating(run, capsys):
    before = run.count("results")
    main(["peek", "--run", run.dir.name, "-v"])
    out = capsys.readouterr().out
    assert "clear" in out and "boundary" in out
    assert run.count("results") == before
