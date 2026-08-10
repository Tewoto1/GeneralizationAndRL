"""
Judge validation. The gate.

The judge is the measurement instrument: triage margins, training pairs and
every internals result downstream are a function of it. So it gets audited
before it is trusted, and `survey` refuses to run against a rubric version with
no passing report (see src/cli.py).

Six checks, cheapest-and-most-fatal first:

  swap_invariance   Does the verdict survive flipping which answer is shown
                    first? Fails => the judge is scoring position, not content.
                    Nothing else is worth reading until this passes.
  self_consistency  Do repeated judgments of the same pair, same order, agree?
                    Fails => margins are sampling noise wearing a number.
  parse_rate        Fraction of completions that yielded a verdict.
  step_compliance   Fraction that actually performed the self-check step. The
                    protocol's whole claim is stepwise reasoning; if the judge
                    skips step 4, it is a different instrument than the one
                    specified.
  length_bias       Correlation between "which answer is longer" and "which
                    answer won". Silently shapes every training pair.
  clear_agreement   On pairs the judge called CLEAR, how often does it agree
                    with the human label? This is the load-bearing one: the
                    entire loop assumes clear => safe to train on. If clarity
                    does not predict correctness, the triage signal is invalid
                    no matter how pretty the margins look.

Also reported, not gated: margin calibration — agreement rate bucketed by
margin. A flat curve means the margin carries no information about correctness
and should be replaced (this is what motivates probing for boundary-ness in
activation space instead).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..common.config import ROOT


def _corr(xs: list[float], ys: list[float]) -> float:
    """Pearson r without numpy. Returns 0.0 when undefined."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def load_labels(path: str | Path) -> dict[str, dict]:
    """Human labels keyed by pair_id. Entries with verdict null are skipped."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    items = data.get("labels", data if isinstance(data, list) else [])
    return {l["pair_id"]: l for l in items if l.get("verdict")}


def report(pairs: list[dict], judgments: list[dict],
           labels: dict[str, dict], gates: dict) -> dict:
    """Compute the six checks plus calibration. Pure function — easy to test."""
    n_pairs = len(pairs)
    out: dict = {"n_pairs": n_pairs, "n_judgments": len(judgments)}

    if not n_pairs:
        return {**out, "passed": False, "reason": "no pairs"}

    # -- parse + step compliance ---------------------------------------------
    n_ok = sum(1 for j in judgments if j.get("ok"))
    out["parse_rate"] = n_ok / len(judgments) if judgments else 0.0
    out["step_compliance"] = (
        sum(1 for j in judgments if 4 in (j.get("steps_present") or [])) / len(judgments)
        if judgments else 0.0
    )

    # -- swap invariance ------------------------------------------------------
    swappable = [p for p in pairs if len(p.get("per_order", {})) == 2]
    out["swap_invariance"] = (
        sum(1 for p in swappable if p["swap_consistent"]) / len(swappable)
        if swappable else 0.0
    )

    # -- self consistency (within one order, do the k samples agree?) ---------
    fracs = []
    for p in pairs:
        for tally in p.get("per_order", {}).values():
            tot = sum(tally.values())
            if tot:
                fracs.append(max(tally.values()) / tot)
    out["self_consistency"] = sum(fracs) / len(fracs) if fracs else 0.0

    # -- length bias ----------------------------------------------------------
    xs, ys = [], []
    for p in pairs:
        la, lb = p.get("len_a"), p.get("len_b")
        if la is None or lb is None or p["winner"] not in ("a", "b"):
            continue
        xs.append(1.0 if la > lb else -1.0)
        ys.append(1.0 if p["winner"] == "a" else -1.0)
    out["length_bias"] = abs(_corr(xs, ys))
    out["n_length_pairs"] = len(xs)

    # -- agreement with humans ------------------------------------------------
    labelled = [p for p in pairs if p["pair_id"] in labels]
    out["n_labelled"] = len(labelled)

    def agrees(p: dict) -> bool:
        return p["winner"] == labels[p["pair_id"]]["verdict"]

    clear = [p for p in labelled if p.get("clear")]
    boundary = [p for p in labelled if not p.get("clear")]
    out["n_labelled_clear"] = len(clear)
    out["clear_agreement"] = (sum(1 for p in clear if agrees(p)) / len(clear)
                              if clear else None)
    out["boundary_agreement"] = (sum(1 for p in boundary if agrees(p)) / len(boundary)
                                 if boundary else None)
    out["overall_agreement"] = (sum(1 for p in labelled if agrees(p)) / len(labelled)
                                if labelled else None)

    # -- margin calibration (reported, not gated) -----------------------------
    buckets: dict[str, list[bool]] = {}
    for p in labelled:
        b = f"{min(int(p['margin'] * 5) / 5, 0.8):.1f}-{min(int(p['margin'] * 5) / 5 + 0.2, 1.0):.1f}"
        buckets.setdefault(b, []).append(agrees(p))
    out["margin_calibration"] = {
        b: {"n": len(v), "agreement": sum(v) / len(v)} for b, v in sorted(buckets.items())
    }

    # -- gates ----------------------------------------------------------------
    checks = {
        "swap_invariance": (out["swap_invariance"], gates.get("swap_invariance_min", 0.9), "min"),
        "self_consistency": (out["self_consistency"], gates.get("self_consistency_min", 0.7), "min"),
        "length_bias": (out["length_bias"], gates.get("length_bias_max", 0.3), "max"),
    }
    if out["clear_agreement"] is not None:
        checks["clear_agreement"] = (out["clear_agreement"],
                                     gates.get("clear_agreement_min", 0.85), "min")

    failed = [
        name for name, (val, thresh, kind) in checks.items()
        if (val < thresh if kind == "min" else val > thresh)
    ]
    out["gate_results"] = {
        name: {"value": round(val, 3), "threshold": thresh, "kind": kind,
               "passed": (val >= thresh if kind == "min" else val <= thresh)}
        for name, (val, thresh, kind) in checks.items()
    }
    out["passed"] = not failed
    out["failed_gates"] = failed
    if out["clear_agreement"] is None:
        out["warning"] = ("no human labels matched these pairs — clear_agreement, "
                          "the load-bearing check, was not evaluated. Fill "
                          "docs/human_label/labels.json before trusting this pass.")
    return out
