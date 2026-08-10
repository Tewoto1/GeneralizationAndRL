"""
End-to-end smoke test: the whole judge slice, wired together, on the stub model.

What it proves, in order:

  1. `pairs` produces one answer pair per domain prompt and marks itself done.
  2. `judge` reads those pairs, judges both orders, and writes both the
     per-judgment records and the per-pair aggregate.
  3. The stub's planted pathologies are all caught by the harness rather than
     silently absorbed:
        d05 position bias  -> swap_consistent False, not clear
        d07 unparseable    -> counted, raw text preserved, no crash
        d12 skipped step 4 -> steps_missing > 0
        d03 split votes    -> lands in the boundary bucket
  4. `validate` runs, writes a report, and BLOCKS (exit 1) on a judge this bad —
     the gate is real, not decorative.
  5. Re-running a completed stage is a no-op without --fresh.

Runs in seconds. No torch, no GPU, no network.
"""
import json
import sys

import pytest

from src.cli import main
from src.common.io import Run


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, monkeypatch_module):
    """Run pairs + judge once into a temp runs/ root; share across tests."""
    root = tmp_path_factory.mktemp("runs")
    monkeypatch_module.setattr("src.common.io.RUNS", root)
    main(["pairs", "--run", "smoke", "--domain", "honesty_tact", "--stub"])
    main(["judge", "--run", "smoke", "--stub"])
    return Run.open("smoke", root=root)


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_1_pairs_stage_produced_one_pair_per_prompt(run_dir):
    """Every domain prompt becomes exactly one (answer_a, answer_b) pair."""
    from src.common import config as C
    n_prompts = len(C.load("configs/domains/honesty_tact.json")["prompts"])
    pairs = list(run_dir.read("pairs"))
    assert len(pairs) == n_prompts
    assert run_dir.is_complete("pairs")
    assert all(p["answer_a"] and p["answer_b"] for p in pairs)
    assert all(p["len_a"] and p["len_b"] for p in pairs)


def test_2_judge_stage_wrote_judgments_and_aggregates(run_dir):
    """k samples x 2 presentation orders per pair, plus one aggregate each."""
    from src.common import config as C
    k = C.load("configs/judge.json")["k_samples"]
    results = list(run_dir.read("results"))
    judgments = list(run_dir.read("judgments"))
    assert run_dir.is_complete("judge")
    assert len(judgments) == len(results) * k * 2
    assert {"pair_id", "winner", "margin", "clear", "swap_consistent"} <= set(results[0])


def test_3a_position_bias_is_caught(run_dir):
    """d05's stub judge always picks whichever answer is shown first. The
    harness must mark it swap-inconsistent and refuse to call it clear."""
    r = next(r for r in run_dir.read("results") if r["pair_id"] == "d05")
    assert r["swap_consistent"] is False
    assert r["clear"] is False


def test_3b_unparseable_is_counted_and_raw_text_kept(run_dir):
    """d07 emits prose with no JSON. It must be counted, never discarded —
    losing the raw text is how a previous sweep produced zero evidence."""
    r = next(r for r in run_dir.read("results") if r["pair_id"] == "d07")
    assert r["n_unparseable"] > 0 and r["n_valid"] == 0
    raws = [j for j in run_dir.read("judgments")
            if j["pair_id"] == "d07" and not j["ok"]]
    assert raws and all(j["raw"].strip() for j in raws)


def test_3c_skipped_selfcheck_is_visible(run_dir):
    """d12 reaches a verdict without step 4. Process compliance must notice."""
    r = next(r for r in run_dir.read("results") if r["pair_id"] == "d12")
    assert r["steps_missing"] > 0


def test_3d_split_votes_land_in_boundary(run_dir):
    """d03's judge genuinely disagrees with itself, so the margin must fall
    below clear_min rather than rounding to a confident verdict."""
    from src.common import config as C
    clear_min = C.load("configs/judge.json")["margin"]["clear_min"]
    r = next(r for r in run_dir.read("results") if r["pair_id"] == "d03")
    assert r["margin"] < clear_min and r["clear"] is False


def test_3e_well_behaved_pairs_come_out_clear(run_dir):
    """The harness must not mark everything boundary — a control that never
    passes is as useless as one that always does."""
    results = list(run_dir.read("results"))
    clear = [r for r in results if r["clear"]]
    assert len(clear) >= len(results) // 2


def test_4_validate_blocks_a_bad_judge(run_dir, capsys):
    """The gate is load-bearing: this stub judge has planted position bias, so
    validate must exit non-zero and must NOT write validate.complete."""
    with pytest.raises(SystemExit) as e:
        main(["validate", "--run", "smoke"])
    assert e.value.code == 1
    rep = json.loads((run_dir.dir / "validation.json").read_text())
    assert rep["passed"] is False
    assert "swap_invariance" in rep["gate_results"]
    assert not run_dir.is_complete("validate")


def test_5_completed_stage_is_not_silently_redone(run_dir, capsys):
    """Re-running `judge` without --fresh must skip, not append a second copy
    of every record. Resuming on directory existence rather than a completion
    marker is a mistake this project has already paid for twice."""
    before = run_dir.count("judgments")
    main(["judge", "--run", "smoke", "--stub"])
    assert run_dir.count("judgments") == before
    assert "already complete" in capsys.readouterr().out


def test_6_peek_reports_without_mutating(run_dir, capsys):
    """`peek` is safe to run mid-experiment."""
    before = run_dir.count("results")
    main(["peek", "--run", "smoke", "-v"])
    out = capsys.readouterr().out
    assert "clear:" in out and "boundary:" in out
    assert run_dir.count("results") == before
