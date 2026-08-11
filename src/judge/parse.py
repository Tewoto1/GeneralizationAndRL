"""
Parsing judge output. Deliberately permissive, and it never throws work away.

Why this file is defensive out of proportion to its size: in the prior repo a
tool-call parser required a literal ```tool fence, the model had just been
fine-tuned to emit ```python, and the result was 150 of 150 rollouts
unparseable — with the unparseable completions *discarded*, so a whole sweep
produced zero evidence of what went wrong. The rules that follow from that:

  1. Accept any fence label, or no fence at all.
  2. Prefer the LAST JSON object in the text (the protocol says "finish with"),
     so a judge that writes an example block mid-reasoning doesn't fool us.
  3. If nothing parses, return a record that says so and carries the raw text.
     Never raise, never return None-and-forget.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_FENCE = re.compile(r"```[\w]*\s*(\{.*?\})\s*```", re.DOTALL)
_BARE = re.compile(r"\{[^{}]*\"winner\"[^{}]*\}", re.DOTALL)

# Step headings the protocol asks for. Presence is a cheap process check: a
# verdict reached without step 4 is not the thing being measured.
# Keyed by NAME, not by position. The protocol gained a step and everything
# that asked "did it do step 4" would have started reading the criteria step
# instead. Names survive reordering; numbers do not.
_STEPS = {
    "locate": re.compile(r"LOCATE THE ANSWER", re.IGNORECASE),
    "commitments": re.compile(r"COMMITMENTS", re.IGNORECASE),
    "consequences": re.compile(r"CONSEQUENCES", re.IGNORECASE),
    "criteria": re.compile(r"\bCRITERIA\b", re.IGNORECASE),
    "selfcheck": re.compile(r"SELF.?CHECK", re.IGNORECASE),
    "verdict": re.compile(r"\bVERDICT\b", re.IGNORECASE),
}


def did_selfcheck(steps) -> bool:
    """Whether a judgment performed the self-check step.

    Accepts both the current name list and the integer list used before the
    protocol was renumbered, so old runs stay readable.
    """
    return "selfcheck" in steps or 4 in steps


@dataclass
class Judgment:
    """One parsed judge output.

    `winner` is the LABEL the judge named ("A"/"B"/"TIE"), not the answer id —
    resolving label to answer is judge.py's job, because only it knows which way
    round the pair was shown.
    """
    ok: bool
    winner: str | None = None
    confidence: float | None = None
    deciding_criterion: str | None = None
    tension: str | None = None
    steps_present: list[int] = field(default_factory=list)
    error: str | None = None
    raw: str = ""

    def as_record(self) -> dict:
        return {
            "ok": self.ok, "winner": self.winner, "confidence": self.confidence,
            "deciding_criterion": self.deciding_criterion, "tension": self.tension,
            "steps_present": self.steps_present, "error": self.error,
            "raw": self.raw,
        }


def _candidates(text: str) -> list[str]:
    """Every plausible JSON blob, last first."""
    out = [m.group(1) for m in _FENCE.finditer(text)]
    out += [m.group(0) for m in _BARE.finditer(text)]
    return list(reversed(out))


def _norm_winner(v, label_a: str, label_b: str) -> str | None:
    if not isinstance(v, str):
        return None
    v = v.strip().strip('"').upper()
    if v in ("TIE", "NEITHER", "EQUAL", "DRAW"):
        return "TIE"
    for lab in (label_a, label_b):
        # Accept "A", "ANSWER A", "Answer_A" for label "A".
        if v == lab.upper() or v.endswith(" " + lab.upper()) or v.endswith("_" + lab.upper()):
            return lab
    return None


def parse(text: str, label_a: str = "A", label_b: str = "B") -> Judgment:
    """Parse one judge completion. Always returns a Judgment; never raises."""
    steps = [name for name, rx in _STEPS.items() if rx.search(text)]

    for blob in _candidates(text):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "winner" not in obj:
            continue
        winner = _norm_winner(obj.get("winner"), label_a, label_b)
        if winner is None:
            continue

        conf = obj.get("confidence")
        try:
            conf = min(1.0, max(0.0, float(conf))) if conf is not None else None
        except (TypeError, ValueError):
            conf = None

        crit = obj.get("deciding_criterion")
        tension = obj.get("tension")
        return Judgment(
            ok=True, winner=winner, confidence=conf,
            deciding_criterion=crit if isinstance(crit, str) else None,
            tension=tension if isinstance(tension, str) else None,
            steps_present=steps, raw=text,
        )

    return Judgment(
        ok=False,
        error="no JSON object with a resolvable `winner` field",
        steps_present=steps, raw=text,
    )
