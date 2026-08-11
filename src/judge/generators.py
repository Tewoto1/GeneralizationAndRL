"""
Model backends for the judge harness.

Split from `judge.py` because they are a different concern: `judge.py` decides
what a verdict MEANS, this file decides where completions come from. Both
implement the same `Generate` contract — `(prompt, n, prefill=None) -> list[str]`
— so the judging logic never learns which one it has.

`hf_generator` imports torch lazily, and the stub imports nothing, which is why
the whole harness and its tests run on a laptop with no torch installed.
"""
from __future__ import annotations

import json
import re

from ..common.chat import render_chat

def hf_generator(model_cfg: dict, role: str = "answer") -> tuple[Generate, dict]:
    """Build a `generate` backed by a real HF model. Imports torch lazily.

    `role` selects a generation profile: `gen` merged with `gen[role]`. The two
    roles want opposite things and sharing one budget breaks both of them.

      answer  short. The domain rewards brevity, and every token here is also a
              token the judge must later read twice (once per presentation
              order), so a long answer costs three times over.
      judge   long, and cooler. The five-step protocol quotes both answers back
              before it reaches the verdict block, so a budget sized for an
              answer truncates mid-STEP-3 and the JSON block is never emitted —
              which surfaces as "unparseable", not as "truncated". Lower
              temperature because a judge that samples wildly cannot be
              self-consistent or order-invariant.

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
                                device_map=model_cfg.get("device_map"))

    system = model_cfg.get("system")  # None => no system turn. See chat.py.
    base = {k: v for k, v in model_cfg.get("gen", {}).items() if not isinstance(v, dict)}
    gen = {**base, **model_cfg.get("gen", {}).get(role, {})}

    cap = gen.get("max_new_tokens", 4000)

    def generate(prompt: str, n: int, prefill: str | None = None) -> list[str]:
        text = render_chat(tok, [{"role": "user", "content": prompt}],
                           system=system, prefill=prefill)
        enc = tok([text] * n, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=cap,
            temperature=gen.get("temperature", 0.7),
            top_p=gen.get("top_p", 0.95),
            do_sample=True,
            pad_token_id=tok.pad_token_id,
        )
        new = out[:, enc["input_ids"].shape[1]:]

        # Did generation stop because the model finished, or because it ran out
        # of budget? Without this the two are indistinguishable downstream: a
        # truncated completion has no verdict block, so it is recorded as
        # `unparseable`, which points at the parser instead of at the cap.
        generate.truncated = [                        # type: ignore[attr-defined]
            bool(row.shape[0] >= cap and tok.eos_token_id not in row.tolist())
            for row in new
        ]
        return tok.batch_decode(new, skip_special_tokens=True)

    generate.truncated = []                           # type: ignore[attr-defined]

    return generate, {"model": name, "role": role, "gen": gen,
                      "chat": chat_provenance(tok, system)}


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

    def generate(prompt: str, n: int, prefill: str | None = None) -> list[str]:
        counter["n"] += 1

        # Constitution writer: emit a parseable constitution that differs per
        # call, so `constitute` exercises its real parse + divergence path.
        if "STEP 4 - CRITERIA" in prompt:
            i = counter["n"]
            return ['STEP 1 ...\nSTEP 4 - CRITERIA\n```json\n' + json.dumps({
                "criteria": [{"id": f"crit{i}", "question": "does it?"},
                             {"id": "shared", "question": "is it true?"}],
                "tensions": [f"crit{i} x shared: when they disagree"]}) + '\n```'] * n

        # Self-review: must emit the markers or extract_revision falls back.
        if "<<<REVISED>>>" in prompt:
            return ["STEP 1 - REVIEW ...\nSTEP 2 - KEEP ...\nSTEP 3 - REVISE\n"
                    "<<<REVISED>>>\nStub revision " + str(counter["n"]) +
                    ". detail detail detail\n<<<END>>>"] * n

        # Answer generation (the `sample` stage) — raw request.
        #
        # Varies by CALL as well as by position in the batch. `sample` asks for
        # n=1 per variant, so keying only on the batch index made every draft
        # byte-identical -- and select_pairs then (correctly) dropped the
        # control pair as having nothing to judge. A stub whose outputs cannot
        # differ cannot exercise the path that compares them.
        if "STEP 1" not in prompt:
            k = counter["n"]
            return [f"Stub answer {k}.{i} " + "detail " * (4 + 3 * (k + i) % 11)
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
