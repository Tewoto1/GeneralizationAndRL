"""
Parser tests. Highest-value tests in the repo.

Context: the predecessor project lost a 150-rollout sweep to a parser that
required one exact fence label, and lost the evidence too because unparseable
output was discarded. Every test here encodes one thing that must not regress.
"""
from src.judge.parse import parse

_TAIL = '{"winner": "B", "confidence": 0.7, "deciding_criterion": "tact", "tension": null}'


def test_plain_json_fence():
    """The happy path: ```json ... ``` at the end of the completion."""
    j = parse(f"STEP 5 - VERDICT\n```json\n{_TAIL}\n```")
    assert j.ok and j.winner == "B" and j.confidence == 0.7
    assert j.deciding_criterion == "tact"


def test_any_fence_label_accepted():
    """A model fine-tuned on code emits ```python. That must still parse —
    this exact mismatch is what produced 100% unparseable last time."""
    for label in ("", "json", "python", "JSON", "js"):
        j = parse(f"```{label}\n{_TAIL}\n```")
        assert j.ok, f"fence label {label!r} failed to parse"


def test_bare_json_without_fence():
    """No fence at all is still parseable."""
    assert parse(f"my verdict: {_TAIL}").ok


def test_last_json_block_wins():
    """A judge that quotes the template's example block mid-reasoning must not
    have that example read as its verdict. The final block is the verdict."""
    text = ('Here is the format I will use: {"winner": "A", "confidence": 0.1}\n'
            f'...reasoning...\n```json\n{_TAIL}\n```')
    j = parse(text)
    assert j.winner == "B" and j.confidence == 0.7


def test_tie_synonyms_normalise():
    """TIE / NEITHER / EQUAL / DRAW all mean the same verdict."""
    for word in ("TIE", "tie", "NEITHER", "Equal", "draw"):
        j = parse('{"winner": "%s", "confidence": 0.5}' % word)
        assert j.ok and j.winner == "TIE", word


def test_verbose_winner_forms():
    """'Answer A' and 'ANSWER_A' resolve to label A."""
    for form in ("A", "Answer A", "ANSWER_A", "answer a"):
        j = parse('{"winner": "%s"}' % form)
        assert j.ok and j.winner == "A", form


def test_unparseable_keeps_raw_and_never_raises():
    """Failure is recorded, not thrown and not discarded. The raw text survives
    so a human can see what the model actually did."""
    text = "I think the second answer is better, honestly."
    j = parse(text)
    assert j.ok is False and j.winner is None
    assert j.raw == text and j.error


def test_steps_detected_even_when_verdict_unparseable():
    """Process compliance is measured independently of verdict extraction —
    otherwise a parse failure would silently zero the step statistics."""
    j = parse("STEP 1 - COMMITMENTS ...\nSTEP 4 - SELF-CHECK ...\nno json here")
    assert j.ok is False
    assert 1 in j.steps_present and 4 in j.steps_present


def test_missing_selfcheck_is_visible():
    """A verdict reached without step 4 must be detectable — the protocol's
    whole claim is stepwise self-verified reasoning."""
    j = parse(f"STEP 1 - COMMITMENTS\nSTEP 5 - VERDICT\n```json\n{_TAIL}\n```")
    assert j.ok and 4 not in j.steps_present


def test_confidence_clamped_and_junk_tolerated():
    """Out-of-range or non-numeric confidence must not kill an otherwise good
    verdict."""
    assert parse('{"winner": "A", "confidence": 5}').confidence == 1.0
    assert parse('{"winner": "A", "confidence": "high"}').confidence is None
