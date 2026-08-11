"""
Answer-pool construction: constitution parsing, variant expansion, revision
extraction, pair selection, and the spread summary.

All pure functions over fabricated inputs. The failure these guard against is
the one r0 actually hit: producing a pool whose members are indistinguishable,
and not noticing until 150 judgments had been paid for.
"""
import pytest

from src import sample as S
from src.judge.validate import spread


# ------------------------------------------------------- constitution parse --
_GOOD = ('reasoning...\n```json\n{"criteria": [{"id": "Truth", "question": "true?"},'
         '{"id": "wastes time", "question": "padded?"}], '
         '"tensions": ["truth x wastes_time: when honesty needs length"]}\n```')


def test_constitution_ids_are_normalised():
    """Model output casing is inconsistent — r0's judgments came back with
    'Completeness', 'completeness' and the literal string 'null' all as
    deciding criteria. Ids are lowercased and spaces become underscores so a
    criterion cannot be counted twice under two spellings."""
    c = S.parse_constitution(_GOOD)
    assert [x["id"] for x in c["criteria"]] == ["truth", "wastes_time"]


def test_constitution_keeps_tensions():
    """A constitution whose criteria never conflict produces no boundary cases,
    so the tensions it declares are part of the artefact, not commentary."""
    assert S.parse_constitution(_GOOD)["tensions"]


def test_constitution_any_fence_label():
    for lab in ("", "json", "python"):
        assert S.parse_constitution(
            "```%s\n{\"criteria\": [{\"id\": \"a\", \"question\": \"q\"}]}\n```" % lab)


def test_constitution_last_block_wins():
    """The writer prompt shows a format example; that example must not be read
    as the answer."""
    text = ('format: {"criteria": [{"id": "example", "question": "q"}]}\n'
            '```json\n{"criteria": [{"id": "real", "question": "q"}]}\n```')
    assert S.parse_constitution(text)["criteria"][0]["id"] == "real"


def test_constitution_unparseable_returns_none_not_raises():
    """Failure is recorded with the raw text, never thrown."""
    assert S.parse_constitution("I think good answers are honest.") is None
    rec = S.constitution_record("c0", None, "raw text here", seed=7, n_examples=10)
    assert rec["ok"] is False and rec["raw"] == "raw text here"
    assert rec["criteria"] == []


def test_constitution_record_pins_the_seed():
    """Every constitution in a run is written from the IDENTICAL examples and
    differs only by sampling. The seed is therefore the independent variable
    and has to be in the record, or divergence is unattributable."""
    rec = S.constitution_record("c1", S.parse_constitution(_GOOD), "", 42, 10)
    assert rec["seed"] == 42 and rec["provenance"] == "model_generated"


def test_seeds_are_deterministic():
    """A re-run must reproduce the same constitutions."""
    assert S.seeds_for(3) == S.seeds_for(3)
    assert len(set(S.seeds_for(5))) == 5


# ------------------------------------------------------------ variant expand --
VCFG = {
    "variants": [
        {"id": "draft", "kind": "draft", "conditioning": "none"},
        {"id": "pre_x", "kind": "prefill", "conditioning": "prefill", "prefill": "So"},
    ],
    "self_review": {"id_prefix": "review", "conditioning": "self_review",
                    "revise_from": "draft_0"},
    "pairing": {"strategy": "anchor_on_draft", "include_control_pair": True},
}


def test_one_review_variant_per_constitution():
    consts = [{"version": "c0", "criteria": [{"id": "a", "question": "q"}]},
              {"version": "c1", "criteria": [{"id": "b", "question": "q"}]}]
    vs = S.expand_variants(VCFG, consts)
    assert [v["id"] for v in vs] == ["draft", "pre_x", "review_c0", "review_c1"]
    assert all(v["revise_from"] == "draft_0" for v in vs if v["kind"] == "self_review")


def test_unparseable_constitution_produces_no_variant():
    """A constitution with no criteria cannot drive a review, and generating
    against it would burn GPU time producing an unlabelled arm."""
    vs = S.expand_variants(VCFG, [{"version": "c0", "criteria": []}])
    assert not any(v["kind"] == "self_review" for v in vs)


# ------------------------------------------------------------------ revision --
def test_revision_extracted_between_markers():
    text = "STEP 1 review...\n<<<REVISED>>>\nThe real answer.\n<<<END>>>"
    body, ok = S.extract_revision(text, "DRAFT")
    assert ok and body == "The real answer."


def test_missing_end_marker_still_extracts():
    """Truncation after <<<REVISED>>> should still yield the revision rather
    than silently falling back to the draft."""
    body, ok = S.extract_revision("<<<REVISED>>>\npartial answer", "DRAFT")
    assert ok and body == "partial answer"


def test_no_marker_falls_back_to_the_draft():
    """Critical: the fallback is the DRAFT, never the review text. If review
    notes entered the answer pool the judge would score the model's own case
    for its answer as though it were an answer."""
    body, ok = S.extract_revision("I reviewed it and it's fine.", "DRAFT")
    assert body == "DRAFT" and ok is False


def test_empty_revision_falls_back():
    body, ok = S.extract_revision("<<<REVISED>>>\n   \n<<<END>>>", "DRAFT")
    assert body == "DRAFT" and ok is False


# ------------------------------------------------------------------- pairing --
def _ans(pid, variant, text=None, **kw):
    """Distinct text per variant by default — select_pairs drops pairs whose
    two answers are identical, so a fixture using one string everywhere would
    silently test nothing."""
    return {"prompt_id": pid, "prompt": "Q?", "variant": variant,
            "text": text if text is not None else f"answer from {variant}",
            "tension": "t", **kw}


def test_every_pair_is_anchored_on_the_same_draft():
    """N variants cost N pairs, not N-choose-2 — and every variant is measured
    against the SAME control, so the numbers are comparable across variants."""
    answers = [_ans("d01", "draft_0"), _ans("d01", "draft_1"),
               _ans("d01", "pre_x"), _ans("d01", "review_c0")]
    pairs = S.select_pairs(answers, VCFG)
    assert len(pairs) == 3
    assert all(p["variant_a"] == "draft_0" for p in pairs)
    assert {p["variant_b"] for p in pairs} == {"draft_1", "pre_x", "review_c0"}


def test_control_pair_can_be_dropped():
    cfg = {**VCFG, "pairing": {"strategy": "anchor_on_draft",
                               "include_control_pair": False}}
    pairs = S.select_pairs([_ans("d01", "draft_0"), _ans("d01", "draft_1"),
                            _ans("d01", "pre_x")], cfg)
    assert {p["variant_b"] for p in pairs} == {"pre_x"}


def test_pairs_keep_the_existing_schema():
    """judge / label / validate are untouched by this feature, which is only
    true if the pair records still look exactly like the ones they read."""
    p = S.select_pairs([_ans("d01", "draft_0"), _ans("d01", "pre_x")], VCFG)[0]
    assert {"pair_id", "prompt", "answer_a", "answer_b", "len_a", "len_b"} <= set(p)


def test_prompt_without_a_draft_is_skipped_not_crashed():
    assert S.select_pairs([_ans("d01", "pre_x")], VCFG) == []


def test_identical_answers_are_not_paired():
    """A self-review that emitted no <<<REVISED>>> block falls back to the
    draft, so the "pair" would be one answer against itself: a guaranteed TIE,
    four judgments of GPU bought for nothing, and a fake tie dragging that
    variant's `decided` rate down as though the judge had failed to rank two
    real alternatives."""
    answers = [_ans("d01", "draft_0", "SAME"),
               _ans("d01", "review_c0", "SAME"),      # revision fell back
               _ans("d01", "review_c1", "actually different")]
    pairs = S.select_pairs(answers, VCFG)
    assert {p["variant_b"] for p in pairs} == {"review_c1"}


def test_unknown_strategy_raises():
    cfg = {**VCFG, "pairing": {"strategy": "everything_vs_everything"}}
    with pytest.raises(ValueError, match="unknown pairing strategy"):
        S.select_pairs([_ans("d01", "draft_0")], cfg)


# -------------------------------------------------------------------- spread --
def _res(variant, winner, margin, clear=True, swap=True):
    return {"variant_b": variant, "winner": winner, "margin": margin,
            "clear": clear, "swap_consistent": swap, "conditioning_b": "x"}


def test_spread_anchors_every_variant_on_the_control():
    """The control is the null hypothesis: it is how much margin plain
    resampling produces on its own. A variant is only interesting relative to
    it, so the delta is computed rather than left to the reader."""
    rs = [_res("draft_1", "TIE", 0.0, clear=False)] * 4 + \
         [_res("pre_x", "b", 1.0)] * 4
    rep = spread(rs)
    assert rep["control_mean_margin"] == 0.0
    assert rep["variants"]["pre_x"]["margin_over_control"] == 1.0


def test_spread_reports_decided_separately_from_win_rate():
    """A variant can be reliably distinguishable AND reliably worse — still a
    usable training signal. Collapsing the two would hide that."""
    rs = [_res("v", "a", 1.0)] * 3      # variant loses every time, decisively
    rep = spread(rs)["variants"]["v"]
    assert rep["decided"] == 1.0 and rep["win_rate"] == 0.0


def test_spread_win_rate_is_none_when_nothing_was_decided():
    rep = spread([_res("v", "TIE", 0.0, clear=False)] * 3)["variants"]["v"]
    assert rep["win_rate"] is None and rep["decided"] == 0.0


def test_spread_warns_without_a_control():
    rep = spread([_res("pre_x", "b", 1.0)])
    assert "warning" in rep and "control_mean_margin" not in rep
