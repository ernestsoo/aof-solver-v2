"""Tests for src/nodelock.py — compare_vs_nash and helper functions."""

import numpy as np
import pytest

from src.hands import COMBO_WEIGHTS
from src.solver import SolverResult, STRATEGY_NAMES
from src.nodelock import compare_vs_nash, lock_from_range_pct, lock_from_hands


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(exploitability: float = 0.0, ev_shift: float = 0.0) -> SolverResult:
    """Build a minimal SolverResult with synthetic ev_table data."""
    strategies = {name: np.zeros(169, dtype=np.float64) for name in STRATEGY_NAMES}
    ev_table = {
        name: np.full(169, ev_shift, dtype=np.float64) for name in STRATEGY_NAMES
    }
    return SolverResult(
        strategies=strategies,
        ev_table=ev_table,
        iterations=1,
        converged=True,
        exploitability=exploitability,
    )


# ---------------------------------------------------------------------------
# compare_vs_nash — unit tests (no equity matrix needed)
# ---------------------------------------------------------------------------


class TestCompareVsNash:
    def test_returns_dict_with_all_strategy_keys(self):
        nash = _make_result()
        nl = _make_result()
        out = compare_vs_nash(nash, nl)
        for name in STRATEGY_NAMES:
            assert name in out, f"Missing key: {name}"

    def test_returns_exploitability_summary_keys(self):
        nash = _make_result(exploitability=0.05)
        nl = _make_result(exploitability=0.30)
        out = compare_vs_nash(nash, nl)
        assert "exploitability_nash" in out
        assert "exploitability_nodelock" in out
        assert "exploitability_delta" in out

    def test_exploitability_values_correct(self):
        nash = _make_result(exploitability=0.05)
        nl = _make_result(exploitability=0.30)
        out = compare_vs_nash(nash, nl)
        assert pytest.approx(out["exploitability_nash"], abs=1e-9) == 0.05
        assert pytest.approx(out["exploitability_nodelock"], abs=1e-9) == 0.30
        assert pytest.approx(out["exploitability_delta"], abs=1e-9) == 0.25

    def test_identical_results_give_zero_ev_diffs(self):
        nash = _make_result(ev_shift=0.5)
        nl = _make_result(ev_shift=0.5)
        out = compare_vs_nash(nash, nl)
        for name in STRATEGY_NAMES:
            assert pytest.approx(out[name], abs=1e-9) == 0.0, f"Expected 0 diff for {name}"

    def test_ev_diff_is_nodelock_minus_nash(self):
        nash = _make_result(ev_shift=1.0)
        nl = _make_result(ev_shift=3.0)
        out = compare_vs_nash(nash, nl)
        for name in STRATEGY_NAMES:
            assert pytest.approx(out[name], abs=1e-9) == 2.0, (
                f"Expected +2.0 diff for {name}, got {out[name]}"
            )

    def test_negative_ev_diff_allowed(self):
        nash = _make_result(ev_shift=3.0)
        nl = _make_result(ev_shift=1.0)
        out = compare_vs_nash(nash, nl)
        for name in STRATEGY_NAMES:
            assert pytest.approx(out[name], abs=1e-9) == -2.0

    def test_missing_ev_table_entry_returns_zero(self):
        """Gracefully handles SolverResult with incomplete ev_table."""
        nash = _make_result()
        nl = _make_result()
        # Remove one entry from each ev_table.
        del nash.ev_table["push_co"]
        del nl.ev_table["push_co"]
        out = compare_vs_nash(nash, nl)
        assert out["push_co"] == 0.0

    def test_missing_only_in_nodelock_returns_zero(self):
        nash = _make_result(ev_shift=1.0)
        nl = _make_result(ev_shift=2.0)
        del nl.ev_table["call_bb_vs_co"]
        out = compare_vs_nash(nash, nl)
        assert out["call_bb_vs_co"] == 0.0

    def test_ev_diff_uses_mean_over_169_hands(self):
        """Mean of a non-uniform array should equal numpy mean of per-hand diffs."""
        nash = _make_result()
        nl = _make_result()
        rng = np.random.default_rng(42)
        arr_nash = rng.standard_normal(169)
        arr_nl = rng.standard_normal(169)
        nash.ev_table["push_co"] = arr_nash
        nl.ev_table["push_co"] = arr_nl
        out = compare_vs_nash(nash, nl)
        expected = float(np.mean(arr_nl - arr_nash))
        assert pytest.approx(out["push_co"], abs=1e-9) == expected

    def test_total_key_count(self):
        """Output dict should have exactly 14 strategy keys + 3 exploitability keys."""
        out = compare_vs_nash(_make_result(), _make_result())
        assert len(out) == 14 + 3


# ---------------------------------------------------------------------------
# lock_from_range_pct (smoke tests — no equity matrix)
# ---------------------------------------------------------------------------


class TestLockFromRangePct:
    def test_returns_169_array(self):
        arr = lock_from_range_pct(30.0, COMBO_WEIGHTS)
        assert arr.shape == (169,)

    def test_values_are_binary(self):
        arr = lock_from_range_pct(30.0, COMBO_WEIGHTS)
        assert set(arr.tolist()).issubset({0.0, 1.0})

    def test_zero_pct_all_zeros(self):
        arr = lock_from_range_pct(0.0, COMBO_WEIGHTS)
        assert arr.sum() == 0.0

    def test_100_pct_all_ones(self):
        arr = lock_from_range_pct(100.0, COMBO_WEIGHTS)
        assert arr.sum() == 169.0


# ---------------------------------------------------------------------------
# lock_from_hands (smoke tests — no equity matrix)
# ---------------------------------------------------------------------------


class TestLockFromHands:
    def test_returns_169_array(self):
        arr = lock_from_hands(["AA", "KK", "QQ"])
        assert arr.shape == (169,)

    def test_correct_hands_set(self):
        from src.hands import HAND_MAP
        arr = lock_from_hands(["AA", "KK"])
        assert arr[HAND_MAP["AA"].index] == 1.0
        assert arr[HAND_MAP["KK"].index] == 1.0
        assert arr[HAND_MAP["QQ"].index] == 0.0

    def test_empty_list_all_zeros(self):
        arr = lock_from_hands([])
        assert arr.sum() == 0.0
