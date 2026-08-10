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
`(prompt: str, n: int) -> list[str]`. The HF implementation lives in
`hf_generator` and is constructed by the caller. That is what lets the whole
harness be tested on a laptop with no torch installed.

CAPTURE SEAM. `capture` is an optional callable invoked with the judging prompt
and each completion. It is the one hook the internals work needs: probing the
judge's own forward pass is how "is this a real value-conflict representation or
just position bias" gets answered. Left unimplemented on purpose — the seam is
three lines, building it before there is data to probe is not.
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Callable, Protocol

from ..common import config as cfg_mod
from ..common.chat import render_chat
from .parse import Judgment, parse

Generate = Callable[[str, int], list[str]]


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
    steps_missing = 0

    for first_is, first, second in orders:
        jp = build_prompt(protocol, rubric, prompt, first, second)
        completions = generate(jp, k)
        js = []
        for i, text in enumerate(completions):
            j = parse(text)
            js.append(j)
            if capture is not None:
                capture.on_judgment(f"{pair_id}:{first_is}:{i}", jp, text)
            if not j.ok:
                n_unparseable += 1
            if 4 not in j.steps_present:
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
                "protocol_version": protocol.get("version"), **j.as_record(),
            })
        per_order[first_is] = _tally(js, "A", "B", first_is)

    votes = {k_: sum(o[k_] for o in per_order.values()) for k_ in ("a", "b", "TIE")}
    n_valid = sum(votes.values())

    if n_valid == 0:
        return PairResult(pair_id, "UNDECIDED", 0.0, False, False, 0,
                          n_unparseable, votes, per_order,
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
        n_valid=n_valid, n_unparseable=n_unparseable,
        votes=votes, per_order=per_order,
        confidence_mean=(sum(all_conf) / len(all_conf)) if all_conf else None,
        deciding_criteria=sorted(set(crits)), tensions=tensions,
        steps_missing=steps_missing,
    ), records


# ------------------------------------------------------------------ generators --
def hf_generator(model_cfg: dict) -> tuple[Generate, dict]:
    """Build a `generate` backed by a real HF model. Imports torch lazily.

    Returns (generate, provenance) — provenance goes straight into the run
    manifest and includes what the chat template *would* have injected, so a
    reader six weeks later can tell which regime produced the judgments.
    """
    import torch  # noqa: F401  (imported for side effect of failing loudly here)

    from ..common.chat import chat_provenance
    from ..common.model import load_base_model, load_tokenizer, load_with_adapter

    name = model_cfg["name"]
    tok = load_tokenizer(name)
    if model_cfg.get("adapter_path"):
        model = load_with_adapter(name, model_cfg["adapter_path"],
                                  dtype=model_cfg.get("dtype", "bfloat16"),
                                  load_in_4bit=model_cfg.get("load_in_4bit", True))
    else:
        model = load_base_model(name, dtype=model_cfg.get("dtype", "bfloat16"),
                                load_in_4bit=model_cfg.get("load_in_4bit", True),
                                device_map=model_cfg.get("device_map", "auto"))

    system = model_cfg.get("system")  # None => no system turn. See chat.py.
    gen = model_cfg.get("gen", {})

    def generate(prompt: str, n: int) -> list[str]:
        text = render_chat(tok, [{"role": "user", "content": prompt}], system=system)
        enc = tok([text] * n, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=gen.get("max_new_tokens", 900),
            temperature=gen.get("temperature", 0.7),
            top_p=gen.get("top_p", 0.95),
            do_sample=True,
            pad_token_id=tok.pad_token_id,
        )
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                skip_special_tokens=True)

    return generate, {"model": name, "chat": chat_provenance(tok, system)}


# ------------------------------------------------------------- stub backend --
# A canned `Generate` that needs no torch, no GPU and no network. It lives here
# rather than in tests/ because `--stub` is a product feature -- the way to
# exercise a config or prompt change on a laptop -- and because src importing
# from the test package would be backwards: tests may depend on src, never the
# reverse.
#
# It does not fake success. It acts out, per pair id, the failure modes the
# harness must survive:
#
#   "d05"  POSITION BIAS   always picks whichever answer is shown first
#   "d07"  UNPARSEABLE     prose with no JSON block at all
#   "d12"  NO SELF-CHECK   reaches a verdict without STEP 4
#   "d03"  SPLIT VOTES     disagrees with itself, margin lands in boundary
#   others                 stable content preference, order-independent
#
# A stub that only returned clean output would let a broken harness pass -- and
# "verified against a fixture written from the author's own assumptions" is
# exactly how a previous harness shipped a metric that was identically zero.


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
        from ..common import config as cfg_mod
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
