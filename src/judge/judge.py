"""
The judge harness. Given a (prompt, answer_a, answer_b), produce a verdict and
a margin, plus everything needed to tell whether that verdict meant anything.

Design, and why:

ORDER IS ALWAYS SWAPPED. Every pair is judged in both presentation orders. This
is not only a validity check bolted on at the end — it is part of the triage
signal. A pair whose verdict depends on which answer was shown first is a
boundary case by construction, whatever its margin says. Cheap to compute, and
it catches the single most common judge pathology before it contaminates
training pairs.

NO SYSTEM TURN. `render_chat(..., system=None)` and the reason is recorded in
the run manifest. Qwen's template injects "You are Qwen, a helpful assistant"
whenever the caller supplies none; a judge scoring helpfulness with that
sentence in context is not impartial, and in the prior repo it silently sat in
every prompt of every run for weeks.

THE MODEL IS AN ARGUMENT, NOT AN IMPORT. `generate` is any callable
`(prompt, n, prefill=None) -> list[str]`, constructed by the caller. The
backends live in `generators.py`; they are re-exported here so existing callers
keep working, but nothing in this file knows which one it was handed. That is
what lets the whole harness be tested on a laptop with no torch installed.

CAPTURE SEAM. `capture` is an optional callable invoked with the judging prompt
and each completion. It is the one hook the internals work needs: probing the
judge's own forward pass is how "is this a real value-conflict representation or
just position bias" gets answered. Left unimplemented on purpose — the seam is
three lines, building it before there is data to probe is not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from ..common import config as cfg_mod
from .generators import hf_generator, stub_generator  # noqa: F401
from .parse import Judgment, did_selfcheck, parse

# (prompt, n, prefill=None) -> n completions. `prefill` seeds the assistant
# turn; backends that do not support it ignore the argument, and judge_pair
# never passes it, so the judging path is unaffected.
Generate = Callable[..., list[str]]


class Capture(Protocol):
    """Optional activation-capture hook. See module docstring."""
    def on_judgment(self, key: str, prompt: str, completion: str) -> None: ...


# --------------------------------------------------------------------- rubric --
def criteria_block(rubric: dict, protocol: dict) -> str:
    """Render the rubric's criteria into the protocol's criterion_line format."""
    line = protocol.get("criterion_line", "- {{id}}: {{question}}")
    return "\n".join(
        cfg_mod.substitute(line, {"id": c["id"], "question": c["question"]})
        for c in rubric["criteria"]
    )


def build_prompt(protocol: dict, rubric: dict, prompt: str,
                 first: str, second: str,
                 label_a: str = "A", label_b: str = "B") -> str:
    """Render the full judging prompt. `first` is shown under `label_a`."""
    return cfg_mod.render(protocol["template"], {
        "prompt": prompt,
        "label_a": label_a, "label_b": label_b,
        "answer_a": first, "answer_b": second,
        "criteria": criteria_block(rubric, protocol),
    })


# --------------------------------------------------------------------- result --
@dataclass
class PairResult:
    """Aggregate of all judgments on one pair, in both orders.

    `winner` is in answer space ("a"/"b"/"TIE"), never label space.

    `margin` = |n_a - n_b| / n_valid, over every valid judgment in both orders.
    Ties count in the denominator: a pair the judge repeatedly calls a tie is
    genuinely undecided, and a margin definition that ignored ties would call it
    a confident 1.0 on the strength of one stray vote.

    `swap_consistent` = the two orders agree on the winner. `clear` requires
    both a wide margin and swap consistency — either alone is not enough.
    """
    pair_id: str
    winner: str
    margin: float
    swap_consistent: bool
    clear: bool
    n_valid: int
    n_unparseable: int
    n_truncated: int = 0
    votes: dict = field(default_factory=dict)
    per_order: dict = field(default_factory=dict)
    confidence_mean: float | None = None
    deciding_criteria: list = field(default_factory=list)
    tensions: list = field(default_factory=list)
    steps_missing: int = 0

    def as_record(self) -> dict:
        return {"kind": "pair", **self.__dict__}


def _tally(judgments: list[Judgment], label_a: str, label_b: str,
           first_is: str) -> dict:
    """Map label-space votes to answer space for one presentation order."""
    out = {"a": 0, "b": 0, "TIE": 0}
    second_is = "b" if first_is == "a" else "a"
    for j in judgments:
        if not j.ok:
            continue
        if j.winner == label_a:
            out[first_is] += 1
        elif j.winner == label_b:
            out[second_is] += 1
        else:
            out["TIE"] += 1
    return out


def judge_pair(pair_id: str, prompt: str, answer_a: str, answer_b: str,
               protocol: dict, rubric: dict, generate: Generate,
               k: int = 3, clear_min: float = 0.7,
               swap: bool = True, capture: Capture | None = None,
               ) -> tuple[PairResult, list[dict]]:
    """Judge one pair k times per order. Returns (aggregate, raw judgment records).

    Raw records are returned rather than written so the caller owns logging; the
    unparseable ones are in there too, on purpose.
    """
    orders = [("a", answer_a, answer_b)] if not swap else [
        ("a", answer_a, answer_b),   # answer_a shown first, as label A
        ("b", answer_b, answer_a),   # answer_b shown first, as label A
    ]

    records: list[dict] = []
    per_order: dict[str, dict] = {}
    all_conf: list[float] = []
    crits: list[str] = []
    tensions: list[str] = []
    n_unparseable = 0
    n_truncated = 0
    steps_missing = 0

    for first_is, first, second in orders:
        jp = build_prompt(protocol, rubric, prompt, first, second)
        completions = generate(jp, k)
        # Optional, generator-supplied: which completions hit the token cap.
        # `getattr` rather than a required field so the stub and any future
        # backend stay valid Generate callables without implementing it.
        trunc = list(getattr(generate, "truncated", []) or [])
        js = []
        for i, text in enumerate(completions):
            j = parse(text)
            js.append(j)
            if capture is not None:
                capture.on_judgment(f"{pair_id}:{first_is}:{i}", jp, text)
            was_cut = trunc[i] if i < len(trunc) else False
            if was_cut:
                n_truncated += 1
            if not j.ok:
                n_unparseable += 1
            if not did_selfcheck(j.steps_present):
                steps_missing += 1
            if j.confidence is not None:
                all_conf.append(j.confidence)
            if j.deciding_criterion:
                crits.append(j.deciding_criterion)
            if j.tension:
                tensions.append(j.tension)
            records.append({
                "kind": "judgment", "pair_id": pair_id, "order_first": first_is,
                "sample": i, "rubric_version": rubric.get("version"),
                "protocol_version": protocol.get("version"),
                "truncated": was_cut, **j.as_record(),
            })
        per_order[first_is] = _tally(js, "A", "B", first_is)

    votes = {k_: sum(o[k_] for o in per_order.values()) for k_ in ("a", "b", "TIE")}
    n_valid = sum(votes.values())

    if n_valid == 0:
        return PairResult(pair_id, "UNDECIDED", 0.0, False, False, 0,
                          n_unparseable, n_truncated, votes, per_order,
                          steps_missing=steps_missing), records

    margin = abs(votes["a"] - votes["b"]) / n_valid
    winner = "TIE" if votes["a"] == votes["b"] else ("a" if votes["a"] > votes["b"] else "b")

    def order_winner(t: dict) -> str:
        if t["a"] == t["b"]:
            return "TIE"
        return "a" if t["a"] > t["b"] else "b"

    swap_consistent = (len(per_order) < 2 or
                       order_winner(per_order["a"]) == order_winner(per_order["b"]))

    return PairResult(
        pair_id=pair_id, winner=winner, margin=margin,
        swap_consistent=swap_consistent,
        clear=bool(margin >= clear_min and swap_consistent and winner != "TIE"),
        n_valid=n_valid, n_unparseable=n_unparseable, n_truncated=n_truncated,
        votes=votes, per_order=per_order,
        confidence_mean=(sum(all_conf) / len(all_conf)) if all_conf else None,
        deciding_criteria=sorted(set(crits)), tensions=tensions,
        steps_missing=steps_missing,
    ), records
