"""
Building the answer pool: constitutions, variants, and pair selection.

Why this exists at all
----------------------
r0 generated both answers of every pair by sampling the same model twice at
temperature 0.7. Result: 47 of 50 answers were markdown listicles, the median
within-pair length difference was 13%, and the judge split 41 / 42 / 67-tie
across 150 judgments. That is a coin flip. Under GRPO the advantage is
`(r_i - mean(r)) / std(r)`, so a group with no reward variance has no gradient
— the same dead-gradient failure this project already paid for once.

Temperature does not fix it: it raises entropy over TOKENS, not over
STRATEGIES. RLHF collapsed the output distribution onto one mode, so hotter
sampling gives the same listicle with worse word choices. The other modes are
still in the weights; reaching them takes CONDITIONING.

This module implements three kinds of conditioning plus a true control, and
the selection logic for which pairs are worth spending judging budget on.
Everything about which variants exist lives in `configs/variants.json`.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .common import config as cfg_mod

# The revision is emitted between markers so the model's self-review never
# leaks into the answer pool. If the judge saw the review it would be reading
# the model's own case for its answer -- exactly the fluency-over-substance
# failure the rubric warns about.
_REVISED = re.compile(r"<<<REVISED>>>(.*?)(?:<<<END>>>|$)", re.DOTALL)


# ------------------------------------------------------------- constitutions --
def format_examples(labels: list[dict], pairs: dict[str, dict],
                    block: list[str], limit: int | None = None) -> str:
    """Render human-labelled pairs as few-shot examples for the writer prompt.

    `labels` come from docs/human_label/labels.json, `pairs` is pair_id -> the
    pair record holding the actual answer texts. A label whose pair is missing
    is skipped rather than rendered with a hole in it.
    """
    out, n = [], 0
    for lab in labels:
        p = pairs.get(lab["pair_id"])
        if not p or not lab.get("verdict"):
            continue
        n += 1
        out.append(cfg_mod.render(block, {
            "n": n,
            "prompt": p["prompt"],
            "answer_a": p["answer_a"],
            "answer_b": p["answer_b"],
            "verdict": {"a": "ANSWER A", "b": "ANSWER B"}.get(lab["verdict"], "NEITHER"),
            "reasoning": lab.get("reasoning") or "(no reason given)",
        }))
        if limit and n >= limit:
            break
    return "\n\n".join(out)


def parse_constitution(text: str) -> dict | None:
    """Extract `{criteria, tensions}` from a constitution-writer completion.

    Same permissiveness rules as the judgment parser: any fence label or none,
    last block first, and a `None` return rather than an exception so the caller
    can record the raw text and move on.
    """
    blocks = re.findall(r"```[\w]*\s*(\{.*?\})\s*```", text, re.DOTALL)
    blocks += re.findall(r"\{[^{}]*\"criteria\"[^{}]*\[.*?\][^{}]*\}", text, re.DOTALL)
    for blob in reversed(blocks):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        crits = obj.get("criteria")
        if not isinstance(crits, list) or not crits:
            continue
        clean = [
            {"id": str(c["id"]).strip().lower().replace(" ", "_"),
             "question": str(c["question"]).strip()}
            for c in crits
            if isinstance(c, dict) and c.get("id") and c.get("question")
        ]
        if clean:
            return {"criteria": clean,
                    "tensions": [str(t) for t in obj.get("tensions", [])
                                 if isinstance(t, str)]}
    return None


def constitution_record(cid: str, parsed: dict | None, raw: str,
                        seed: int, n_examples: int) -> dict:
    """One constitution, ready to write to disk.

    `provenance` is `model_generated` and the seed is recorded, because every
    constitution in a run is written from the IDENTICAL examples and differs
    only by sampling. The question being asked is how far the model diverges
    from a shared starting point — so the seed is the independent variable and
    has to be in the record.
    """
    return {
        "version": cid,
        "provenance": "model_generated",
        "seed": seed,
        "n_examples": n_examples,
        "criteria": (parsed or {}).get("criteria", []),
        "tensions": (parsed or {}).get("tensions", []),
        "ok": parsed is not None,
        "raw": raw,
    }


# ------------------------------------------------------------------ variants --
def expand_variants(cfg: dict, constitutions: list[dict]) -> list[dict]:
    """Static variants from config, plus one self-review variant per constitution.

    Self-review variants cannot be listed in the config file because the
    constitutions do not exist until a run has produced them.
    """
    out = [dict(v) for v in cfg["variants"]]
    sr = cfg.get("self_review")
    if sr:
        for c in constitutions:
            if not c.get("criteria"):
                continue
            out.append({
                "id": f"{sr['id_prefix']}_{c['version']}",
                "kind": "self_review",
                "conditioning": sr.get("conditioning", "self_review"),
                "constitution": c["version"],
                "revise_from": sr.get("revise_from", "draft_0"),
            })
    return out


def criteria_block(constitution: dict) -> str:
    return "\n".join(f"- {c['id']}: {c['question']}"
                     for c in constitution["criteria"])


def extract_revision(text: str, fallback: str) -> tuple[str, bool]:
    """Pull the revised answer out of a self-review completion.

    Returns (answer, ok). On failure the DRAFT is returned unchanged with
    ok=False, so a broken revision degrades to "no revision happened" rather
    than putting the model's review notes into the answer pool where the judge
    would score them as an answer.
    """
    m = _REVISED.search(text)
    if not m:
        return fallback, False
    body = m.group(1).strip()
    return (body, True) if body else (fallback, False)


# ------------------------------------------------------------------- pairing --
def select_pairs(answers: list[dict], cfg: dict) -> list[dict]:
    """Choose which (answer, answer) pairs to spend judging budget on.

    `anchor_on_draft`: every pair is draft_0 vs one other variant, plus one
    draft_0 vs draft_1 control. Anchoring on a shared reference makes N variants
    cost N pairs rather than N-choose-2, and — more importantly — measures every
    variant against the SAME control instead of against each other, so the
    numbers are comparable across variants.

    The returned records use the existing pairs.jsonl schema, so `judge`,
    `label` and `validate` need no changes at all.
    """
    by_prompt: dict[str, dict[str, dict]] = {}
    for a in answers:
        by_prompt.setdefault(a["prompt_id"], {})[a["variant"]] = a

    strategy = cfg["pairing"].get("strategy", "anchor_on_draft")
    if strategy != "anchor_on_draft":
        raise ValueError(f"unknown pairing strategy {strategy!r}")

    out = []
    for pid, variants in sorted(by_prompt.items()):
        anchor = variants.get("draft_0")
        if anchor is None:
            continue
        others = [v for k, v in sorted(variants.items()) if k != "draft_0"]
        if not cfg["pairing"].get("include_control_pair", True):
            others = [v for v in others if v["variant"] != "draft_1"]

        # Drop pairs whose two answers are the SAME TEXT. A self-review whose
        # output had no <<<REVISED>>> block falls back to the draft, so the
        # "pair" would be one answer against itself: guaranteed TIE, four
        # judgments of GPU time bought for nothing, and — worse — a fake tie
        # dragging that variant's `decided` rate down as though the judge had
        # failed to rank two real alternatives.
        anchor_text = anchor["text"].strip()
        others = [v for v in others if v["text"].strip() != anchor_text]

        for other in others:
            out.append({
                "pair_id": f"{pid}::{other['variant']}",
                "prompt_id": pid,
                "prompt": anchor["prompt"],
                "tension": anchor.get("tension"),
                "is_control": bool(anchor.get("is_control")),
                "variant_a": "draft_0",
                "variant_b": other["variant"],
                "conditioning_b": other.get("conditioning"),
                "constitution_b": other.get("constitution"),
                "answer_a": anchor["text"],
                "answer_b": other["text"],
                "len_a": len(anchor["text"]),
                "len_b": len(other["text"]),
            })
    return out


# ------------------------------------------------------------------- reading --
def load_labels_list(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_absolute():
        p = cfg_mod.ROOT / p
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    items = data.get("labels", data if isinstance(data, list) else [])
    return [l for l in items if l.get("verdict")
            and not l["pair_id"].startswith("EXAMPLE")]


def seeds_for(n: int, base: int = 0) -> list[int]:
    """Deterministic seeds so a re-run produces the same constitutions."""
    rng = random.Random(base)
    return [rng.randrange(1, 10**6) for _ in range(n)]
