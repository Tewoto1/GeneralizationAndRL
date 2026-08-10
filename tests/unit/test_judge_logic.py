"""
Aggregation and validation arithmetic — the numbers the whole project reads.

These are pure functions over fabricated inputs, so a wrong definition shows up
here in milliseconds instead of after a training run. The predecessor project
shipped a metric (`hack_rate`) whose column was an exact copy of another column
for a whole run; a test at this level would have caught it the same afternoon.
"""
import pytest

from src.judge.judge import PairResult, judge_pair
from src.judge.validate import preflight, report

PROTO = {"version": "t", "template": ["{{prompt}}|{{answer_a}}|{{answer_b}}|{{criteria}}|{{label_a}}{{label_b}}"],
         "criterion_line": "- {{id}}: {{question}}"}
RUBRIC = {"version": "t", "criteria": [{"id": "truth", "question": "true?"}]}


def _gen(script):
    """Generator returning canned completions; `script` is a list per call."""
    calls = {"i": 0}

    def generate(prompt, n):
        out = script[calls["i"] % len(script)]
        calls["i"] += 1
        return [out] * n
    return generate


def _v(w):
    return 'STEP 4 - SELF-CHECK ok\n```json\n{"winner": "%s", "confidence": 0.8}\n```' % w


# ------------------------------------------------------------ aggregation ----
def test_consistent_winner_is_clear_and_order_independent():
    """A judge that picks the same ANSWER in both orders (so label A in one and
    label B in the other) yields margin 1.0, swap-consistent, clear."""
    res, recs = judge_pair("p1", "q", "ANS_A", "ANS_B", PROTO, RUBRIC,
                           _gen([_v("A"), _v("B")]), k=2, clear_min=0.7)
    assert res.winner == "a" and res.margin == 1.0
    assert res.swap_consistent and res.clear
    assert len(recs) == 4  # 2 orders x k=2


def test_position_bias_is_caught_by_swap_not_by_margin():
    """A judge that always says 'A' regardless of content splits its votes 50/50
    in answer space. Margin collapses AND swap_consistent goes False — the pair
    lands in boundary, which is the intended behaviour: presentation-order
    dependence makes a case boundary by construction."""
    res, _ = judge_pair("p2", "q", "ANS_A", "ANS_B", PROTO, RUBRIC,
                        _gen([_v("A")]), k=3, clear_min=0.7)
    assert res.margin == 0.0
    assert res.swap_consistent is False
    assert res.clear is False


def test_ties_count_in_the_denominator():
    """Otherwise a pair the judge mostly calls a tie would report a confident
    margin on the strength of one stray vote."""
    res, _ = judge_pair("p3", "q", "A1", "B1", PROTO, RUBRIC,
                        _gen([_v("TIE"), _v("TIE")]), k=2, clear_min=0.7)
    assert res.votes["TIE"] == 4 and res.n_valid == 4
    assert res.margin == 0.0 and res.winner == "TIE" and not res.clear


def test_tie_winner_is_never_clear():
    """A tie is a real result but it is not a training signal."""
    res, _ = judge_pair("p4", "q", "A1", "B1", PROTO, RUBRIC,
                        _gen([_v("TIE")]), k=1, clear_min=0.0)
    assert res.winner == "TIE" and res.clear is False


def test_unparseable_are_counted_and_do_not_crash():
    """All-unparseable yields UNDECIDED with the count preserved, rather than a
    divide-by-zero or a fabricated verdict."""
    res, recs = judge_pair("p5", "q", "A1", "B1", PROTO, RUBRIC,
                           _gen(["no json at all"]), k=2, clear_min=0.7)
    assert res.winner == "UNDECIDED" and res.n_valid == 0
    assert res.n_unparseable == 4 and len(recs) == 4


def test_missing_selfcheck_is_counted():
    """steps_missing tracks judgments that skipped step 4."""
    res, _ = judge_pair("p6", "q", "A1", "B1", PROTO, RUBRIC,
                        _gen(['```json\n{"winner": "A"}\n```']), k=1)
    assert res.steps_missing == 2


# -------------------------------------------------------------- validation ---
def _pair(pid, winner, margin, clear, swap=True, la=100, lb=100, orders=None):
    return PairResult(pid, winner, margin, swap, clear, 4, 0,
                      {"a": 4, "b": 0, "TIE": 0},
                      orders or {"a": {"a": 2, "b": 0, "TIE": 0},
                                 "b": {"a": 2, "b": 0, "TIE": 0}}
                      ).as_record() | {"len_a": la, "len_b": lb}


GATES = {"swap_invariance_min": 0.9, "self_consistency_min": 0.7,
         "clear_agreement_min": 0.85, "length_bias_max": 0.3}


def test_clean_judge_passes_all_gates():
    pairs = [_pair(f"p{i}", "a", 1.0, True) for i in range(5)]
    labels = {f"p{i}": {"pair_id": f"p{i}", "verdict": "a"} for i in range(5)}
    rep = report(pairs, [{"ok": True, "steps_present": [1, 2, 3, 4, 5]}], labels, GATES)
    assert rep["passed"] and rep["clear_agreement"] == 1.0
    assert rep["swap_invariance"] == 1.0


def test_swap_failure_fails_the_gate():
    """The cheapest and most fatal check: if verdicts flip with presentation
    order, nothing else in the report is worth reading."""
    pairs = [_pair(f"p{i}", "a", 1.0, False, swap=False) for i in range(5)]
    rep = report(pairs, [{"ok": True, "steps_present": [4]}], {}, GATES)
    assert not rep["passed"] and "swap_invariance" in rep["failed_gates"]


def test_clear_but_wrong_fails_the_load_bearing_gate():
    """The loop's core assumption is clear => safe to train on. A judge that is
    confidently wrong must be blocked, however tidy its margins are."""
    pairs = [_pair(f"p{i}", "a", 1.0, True) for i in range(5)]
    labels = {f"p{i}": {"pair_id": f"p{i}", "verdict": "b"} for i in range(5)}
    rep = report(pairs, [{"ok": True, "steps_present": [4]}], labels, GATES)
    assert rep["clear_agreement"] == 0.0
    assert not rep["passed"] and "clear_agreement" in rep["failed_gates"]


def test_length_bias_detected():
    """Longer answer always wins => |r| = 1.0, over the max."""
    pairs = [_pair(f"p{i}", "a", 1.0, True, la=500, lb=100) for i in range(4)]
    pairs += [_pair(f"q{i}", "b", 1.0, True, la=100, lb=500) for i in range(4)]
    for p in pairs[4:]:
        p["votes"] = {"a": 0, "b": 4, "TIE": 0}
    rep = report(pairs, [{"ok": True, "steps_present": [4]}], {}, GATES)
    assert rep["length_bias"] == pytest.approx(1.0)
    assert "length_bias" in rep["failed_gates"]


def test_no_labels_produces_a_warning_not_a_silent_pass():
    """Passing every gate while never having checked against a human is the
    most dangerous possible green light."""
    pairs = [_pair(f"p{i}", "a", 1.0, True) for i in range(3)]
    rep = report(pairs, [{"ok": True, "steps_present": [4]}], {}, GATES)
    assert rep["clear_agreement"] is None and rep["warning"]


def test_margin_calibration_is_reported():
    """Agreement bucketed by margin. A flat curve means the margin carries no
    information about correctness — the finding that would send triage to the
    activation probe instead."""
    pairs = [_pair("p1", "a", 0.2, False), _pair("p2", "a", 1.0, True)]
    labels = {"p1": {"pair_id": "p1", "verdict": "b"},
              "p2": {"pair_id": "p2", "verdict": "a"}}
    rep = report(pairs, [{"ok": True, "steps_present": [4]}], labels, GATES)
    assert rep["margin_calibration"]
    assert sum(b["n"] for b in rep["margin_calibration"].values()) == 2


# --------------------------------------------------------------- preflight ---
# The cheap half of validation: needs no human labels, answers "is the
# instrument functioning at all". This is what gates a rented overnight run.

_CLEAN = {"chat": {"system_policy": "none"}}


def _js(n, ok=True, steps=(1, 2, 3, 4, 5)):
    return [{"ok": ok, "steps_present": list(steps)} for _ in range(n)]


def test_preflight_passes_a_healthy_pilot():
    assert preflight([{"pair_id": "d01"}], _js(6), _CLEAN)["ok"]


def test_preflight_blocks_on_unparseable_output():
    """>5% unparseable means the parser and the model disagree about format —
    the exact failure that cost a previous project a 150-rollout sweep."""
    pre = preflight([{"pair_id": "d01"}], _js(3) + _js(3, ok=False), _CLEAN)
    assert not pre["ok"] and "unparseable" in pre["problems"][0]


def test_preflight_blocks_on_skipped_selfcheck_and_blames_the_token_budget():
    """Step 4 missing is nearly always truncation, so the message must say so —
    a pilot that reports a symptom without the likely cause wastes the night
    it was supposed to save."""
    pre = preflight([{"pair_id": "d01"}], _js(6, steps=(1, 2, 3)), _CLEAN,
                    max_new_tokens=900)
    assert not pre["ok"] and "max_new_tokens (900)" in pre["problems"][0]


def test_preflight_blocks_on_an_injected_system_prompt():
    """A persona in context contaminates every judgment, silently."""
    pre = preflight([{"pair_id": "d01"}], _js(6),
                    {"chat": {"system_policy": "template_default"}})
    assert not pre["ok"] and "system_policy" in pre["problems"][0]


def test_preflight_blocks_when_nothing_was_judged():
    assert not preflight([], [], _CLEAN)["ok"]
