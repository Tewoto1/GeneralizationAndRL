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
from .parse import did_selfcheck


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


def preflight(results: list[dict], judgments: list[dict], provenance: dict,
              max_new_tokens: int | None = None) -> dict:
    """The cheap half of validation: is the instrument even functioning?

    Separate from `report` because it needs no human labels and no statistics —
    just "did the model emit parseable verdicts, did it do the steps, and is
    the prompt clean". These are the three ways a rented overnight run comes
    back worthless, and all three are visible after two pairs.

    Returns `{"ok": bool, "problems": [str], ...}`. Pure function; `cmd_pilot`
    prints it and exits non-zero on `not ok`.
    """
    n = len(judgments) or 1
    unparseable = sum(1 for j in judgments if not j.get("ok"))
    truncated = sum(1 for j in judgments if j.get("truncated"))
    no_selfcheck = sum(1 for j in judgments
                       if not did_selfcheck(j.get("steps_present") or []))
    policy = (provenance.get("chat") or {}).get("system_policy")

    problems = []
    # Truncation is checked FIRST and reported as itself. It is the upstream
    # cause of both other symptoms — a completion cut off before the verdict
    # block has no JSON (reads as unparseable) and no STEP 4 (reads as skipped
    # self-check) — so naming it first stops the other two sending you to the
    # parser and the protocol for a problem that is neither.
    if truncated > n * 0.05:
        problems.append(
            f"{truncated}/{n} completions hit the token cap"
            + (f" (max_new_tokens={max_new_tokens})" if max_new_tokens else "")
            + " — raise it; the cap is a runaway guard, not a budget, and it "
              "only costs when it binds")
    if unparseable > n * 0.05:
        problems.append(
            f"{unparseable}/{n} completions unparseable"
            + (" — likely downstream of the truncation above" if truncated else
               " — read the raw text in judgments.jsonl before spending a night on this"))
    if no_selfcheck > n * 0.2:
        problems.append(
            f"{no_selfcheck}/{n} judgments skipped the self-check step"
            + (" — likely downstream of the truncation above" if truncated else
               " — the protocol asks for STEP 4 explicitly, so check the raw text"))
    if policy != "none":
        problems.append(
            f"system_policy is {policy!r}, not 'none' — a persona in context "
            f"contaminates every judgment")
    if not results:
        problems.append("no pairs were judged at all")

    return {"ok": not problems, "problems": problems,
            "n_judgments": len(judgments), "unparseable": unparseable,
            "truncated": truncated, "no_selfcheck": no_selfcheck,
            "system_policy": policy}


def spread(results: list[dict]) -> dict:
    """Per-variant summary: did conditioning produce judgeable difference?

    The question this answers is not "which variant is best" but "is there a
    signal at all". Under GRPO the advantage is `(r_i - mean(r)) / std(r)`, so
    a set of samples the judge cannot rank has no gradient regardless of the
    optimiser. r0's `draft_0 vs draft_1` control is the null: if it comes back
    near zero again and a conditioned variant does not, the conditioning worked
    and the difference is attributable to it.

    Reported per variant:
      n              pairs judged
      decided        fraction where the judge picked a side (not TIE/UNDECIDED)
      mean_margin    mean |votes_a - votes_b| / n_valid -- the continuous
                     quantity, logged next to the thresholded `clear` so a
                     broken threshold cannot masquerade as a strong effect
      clear          fraction meeting the full clear bar
      swap_stable    fraction whose verdict survived the order flip
      win_rate       fraction where the VARIANT beat the draft. Distinct from
                     `decided`: a variant can be reliably distinguishable and
                     reliably worse, which is still a usable training signal
    """
    by: dict[str, list[dict]] = {}
    for r in results:
        by.setdefault(r.get("variant_b") or "?", []).append(r)

    out: dict = {"variants": {}}
    for name, rs in sorted(by.items()):
        n = len(rs)
        decided = [r for r in rs if r["winner"] in ("a", "b")]
        out["variants"][name] = {
            "n": n,
            "decided": len(decided) / n if n else 0.0,
            "mean_margin": sum(r["margin"] for r in rs) / n if n else 0.0,
            "clear": sum(1 for r in rs if r.get("clear")) / n if n else 0.0,
            "swap_stable": sum(1 for r in rs if r.get("swap_consistent")) / n if n else 0.0,
            "win_rate": (sum(1 for r in decided if r["winner"] == "b") / len(decided)
                         if decided else None),
            "conditioning": next((r.get("conditioning_b") for r in rs
                                  if r.get("conditioning_b")), None),
        }

    ctrl = out["variants"].get("draft_1")
    if ctrl:
        base = ctrl["mean_margin"]
        for name, v in out["variants"].items():
            v["margin_over_control"] = round(v["mean_margin"] - base, 3)
        out["control_mean_margin"] = base
    else:
        out["warning"] = ("no draft_1 control pair — every variant number here is "
                          "unanchored, because there is nothing to say how much "
                          "margin plain resampling produces on its own")
    return out


def add_label(path: str | Path, entry: dict) -> int:
    """Append one human label. Returns the new total.

    APPEND-ONLY, to a sibling .jsonl. It used to read the whole JSON file,
    append in memory, and write it back — which is a lost-update race the
    moment anything else touches the file. It was not hypothetical: a second
    process read the file, and by the time it wrote back, nine labels made in
    between were gone. Hand-written labels are the only ground truth this
    project has and the most expensive thing in it to reproduce.

    An `open(..., "a")` of one short line cannot lose a previously written
    line, whatever else is writing. `load_labels` reads both files and the
    .jsonl wins on conflict, so an existing labels.json keeps working.
    """
    p = _resolve(path)
    stream = p.with_suffix(".jsonl")
    with stream.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return len(load_labels(path))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_labels(path: str | Path) -> dict[str, dict]:
    """Human labels keyed by pair_id, from labels.json AND labels.jsonl.

    The .json is the reviewable, hand-editable form; the .jsonl is where new
    labels are appended. Later entries win, so re-labelling a pair overrides
    the old verdict rather than silently keeping both.
    """
    p, out = _resolve(path), {}
    if p.exists():
        data = json.loads(p.read_text())
        items = data.get("labels", data if isinstance(data, list) else [])
        out.update({l["pair_id"]: l for l in items if l.get("verdict")})
    stream = p.with_suffix(".jsonl")
    if stream.exists():
        for line in stream.read_text().splitlines():
            if line.strip():
                l = json.loads(line)
                if l.get("verdict"):
                    out[l["pair_id"]] = l
    return out


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
        sum(1 for j in judgments if did_selfcheck(j.get("steps_present") or []))
        / len(judgments)
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
