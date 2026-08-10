"""
A canned generator standing in for the model, so every stage runs in
milliseconds with no torch, no GPU and no network.

The point is not to fake success. The stub deliberately emits the failure modes
the harness is supposed to survive, keyed off what it is asked to judge:

  normal pair    -> a clean, well-formed, order-stable verdict
  pair "d05"     -> POSITION BIAS: always picks whichever answer was shown
                    first, so swap-detection must catch it
  pair "d07"     -> UNPARSEABLE: prose with no JSON block at all
  pair "d12"     -> SKIPS STEP 4: verdict reached without the self-check, so
                    step_compliance must notice
  pair "d03"     -> genuinely split votes, so margin lands in the boundary band

A stub that only ever returns clean output would let a broken harness pass.
"""
from __future__ import annotations

import re

_STEPS = (
    "STEP 1 - COMMITMENTS\nAnswer A commits to X. Answer B commits to Y.\n\n"
    "STEP 2 - CONSEQUENCES\nIf they act on A they do P. If they act on B they do Q.\n\n"
    "STEP 3 - CRITERIA\n- truth: A cites specifics, B hedges. A better.\n"
    "- time: A is shorter. A better.\n\n"
    "STEP 4 - SELF-CHECK\nThe weakest point is the time criterion; brevity is "
    "not the same as respecting time. It does not change the verdict.\n\n"
    "STEP 5 - VERDICT\n"
)

_NO_SELFCHECK = (
    "STEP 1 - COMMITMENTS\nA asserts X.\n\n"
    "STEP 2 - CONSEQUENCES\nThey act on X.\n\n"
    "STEP 3 - CRITERIA\n- truth: A better.\n\n"
    "STEP 5 - VERDICT\n"
)


def _verdict(winner: str, conf: float = 0.8, crit: str = "truth") -> str:
    return ('```json\n{"winner": "%s", "confidence": %s, '
            '"deciding_criterion": "%s", "tension": null}\n```' % (winner, conf, crit))


def _which_pair(prompt: str, pair_texts: dict[str, str]) -> str | None:
    for pid, text in pair_texts.items():
        if text[:60] in prompt:
            return pid
    return None


def stub_generator(pair_texts: dict[str, str] | None = None):
    """Build a `generate(prompt, n) -> list[str]` callable.

    `pair_texts` maps pair_id -> the request text, so the stub can recognise
    which pair it is being asked about and act out that pair's failure mode.
    Defaults to the honesty_tact domain, loaded lazily.
    """
    if pair_texts is None:
        from src.common import config as cfg_mod
        dom = cfg_mod.load("configs/domains/honesty_tact.json")
        pair_texts = {p["id"]: p["text"] for p in dom["prompts"]}

    counter = {"n": 0}

    def generate(prompt: str, n: int) -> list[str]:
        counter["n"] += 1

        # Answer generation (the `pairs` stage) — prompt is the raw request.
        if "STEP 1" not in prompt:
            return [f"Stub answer {i + 1}. " + "detail " * (4 + 6 * i)
                    for i in range(n)]

        pid = _which_pair(prompt, pair_texts)

        # Which answer text is currently under label A? The stub reads it back
        # out of its own rendered prompt so it can behave positionally.
        m = re.search(r"ANSWER A:\n(.*?)\n\nANSWER B:", prompt, re.DOTALL)
        first = (m.group(1) if m else "").strip()
        first_is_1 = first.startswith("Stub answer 1")

        if pid == "d07":
            return ["I considered both answers carefully and I think the second "
                    "one reads better overall, though it is a close call."] * n
        if pid == "d12":
            return [_NO_SELFCHECK + _verdict("A", 0.9)] * n
        if pid == "d05":
            # Position bias: label A always wins, whatever is under it.
            return [_STEPS + _verdict("A", 0.85)] * n
        if pid == "d03":
            # Split: alternate, so votes end up near even.
            return [_STEPS + _verdict("A" if i % 2 == 0 else "B", 0.5)
                    for i in range(n)]

        # Stable content preference: answer 1 always wins, whichever label it holds.
        win = "A" if first_is_1 else "B"
        return [_STEPS + _verdict(win, 0.8)] * n

    return generate
