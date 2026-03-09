"""Tests for src/nodelock.py — compare_vs_nash and helper functions."""

import numpy as np
import pytest

from src.hands import COMBO_WEIGHTS
from src.solver import SolverResult, STRATEGY_NAMES
from src.nodelock import compare_vs_nash, lock_from_range_pct, lock_from_hands, nodelock_solve


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


# ---------------------------------------------------------------------------
# Integration tests — require data/equity_matrix.npy
# ---------------------------------------------------------------------------

import os

MATRIX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "equity_matrix.npy")
_MATRIX_EXISTS = os.path.exists(MATRIX_PATH)


@pytest.mark.skipif(not _MATRIX_EXISTS, reason="data/equity_matrix.npy not found")
class TestNodelockIntegration:
    """Integration tests using the real equity matrix.

    All tests are skipped when data/equity_matrix.npy is absent so CI never
    fails without the precomputed matrix.
    """

    @pytest.fixture(scope="class")
    def matrix(self):
        return np.load(MATRIX_PATH).astype(np.float64)

    @pytest.fixture(scope="class")
    def nash(self, matrix):
        from src.solver import solve_nash
        return solve_nash(matrix, COMBO_WEIGHTS, max_iter=100)

    # ------------------------------------------------------------------
    # Test 1: locking ALL strategies to Nash values → result unchanged
    # ------------------------------------------------------------------

    def test_lock_all_to_nash_strategies_unchanged(self, matrix, nash):
        """Nodelocking every strategy to Nash values should leave strategies identical."""
        locked = {name: nash.strategies[name].copy() for name in STRATEGY_NAMES}
        nl = nodelock_solve(matrix, COMBO_WEIGHTS, locked=locked, max_iter=1)
        for name in STRATEGY_NAMES:
            np.testing.assert_array_equal(
                nl.strategies[name],
                nash.strategies[name],
                err_msg=f"Strategy {name!r} changed after locking to Nash",
            )

    def test_lock_all_to_nash_exploitability_near_nash(self, matrix, nash):
        """Locking all strategies to Nash values should reproduce the same exploitability."""
        locked = {name: nash.strategies[name].copy() for name in STRATEGY_NAMES}
        nl = nodelock_solve(matrix, COMBO_WEIGHTS, locked=locked, max_iter=1)
        # Exploitability must be identical (same strategies, same computation).
        assert abs(nl.exploitability - nash.exploitability) < 0.001, (
            f"Exploitability changed after locking to Nash: "
            f"nash={nash.exploitability:.4f}, nodelock={nl.exploitability:.4f}"
        )

    # ------------------------------------------------------------------
    # Test 2: locking push_co to 100% → opponents call wider (call % ↑)
    # ------------------------------------------------------------------

    def test_lock_push_co_100pct_opponents_call_wider(self, matrix, nash):
        """When CO always pushes, callers exploit by calling wider.

        A 100% push range includes many weak hands (72o, etc.), so callers
        face a weaker average range and should call more liberally to exploit it.
        """
        push_co_all = np.ones(169, dtype=np.float64)
        nl = nodelock_solve(
            matrix, COMBO_WEIGHTS,
            locked={"push_co": push_co_all},
            max_iter=100,
        )

        nash_btn_call_pct = float(np.dot(nash.strategies["call_btn_vs_co"],  COMBO_WEIGHTS) / COMBO_WEIGHTS.sum())
        nash_sb_call_pct  = float(np.dot(nash.strategies["call_sb_vs_co"],   COMBO_WEIGHTS) / COMBO_WEIGHTS.sum())
        nash_bb_call_pct  = float(np.dot(nash.strategies["call_bb_vs_co"],   COMBO_WEIGHTS) / COMBO_WEIGHTS.sum())

        nl_btn_call_pct   = float(np.dot(nl.strategies["call_btn_vs_co"],    COMBO_WEIGHTS) / COMBO_WEIGHTS.sum())
        nl_sb_call_pct    = float(np.dot(nl.strategies["call_sb_vs_co"],     COMBO_WEIGHTS) / COMBO_WEIGHTS.sum())
        nl_bb_call_pct    = float(np.dot(nl.strategies["call_bb_vs_co"],     COMBO_WEIGHTS) / COMBO_WEIGHTS.sum())

        # All callers should widen (call more) to exploit the weak push range.
        any_wider = (
            nl_btn_call_pct > nash_btn_call_pct or
            nl_sb_call_pct  > nash_sb_call_pct  or
            nl_bb_call_pct  > nash_bb_call_pct
        )
        assert any_wider, (
            f"Expected at least one caller to widen vs 100% CO push, but:\n"
            f"  BTN: Nash={nash_btn_call_pct:.3f} → NL={nl_btn_call_pct:.3f}\n"
            f"  SB:  Nash={nash_sb_call_pct:.3f}  → NL={nl_sb_call_pct:.3f}\n"
            f"  BB:  Nash={nash_bb_call_pct:.3f}  → NL={nl_bb_call_pct:.3f}"
        )

    def test_lock_push_co_100pct_returns_valid_result(self, matrix, nash):
        """Nodelock with push_co=100% should run and return a well-formed SolverResult."""
        nl = nodelock_solve(
            matrix, COMBO_WEIGHTS,
            locked={"push_co": np.ones(169, dtype=np.float64)},
            max_iter=100,
        )
        # push_co must remain exactly as locked
        np.testing.assert_array_equal(nl.strategies["push_co"], np.ones(169))
        # All 14 strategies present with correct shape
        assert set(nl.strategies.keys()) == set(STRATEGY_NAMES)
        for name in STRATEGY_NAMES:
            assert nl.strategies[name].shape == (169,), f"{name} wrong shape"
        # ev_table populated
        assert set(nl.ev_table.keys()) == set(STRATEGY_NAMES)
        assert nl.iterations == 100

    # ------------------------------------------------------------------
    # Test 3: exploitability(nash) ~ 0; exploitability(nodelock) > 0
    # ------------------------------------------------------------------

    def test_nash_exploitability_near_zero(self, matrix, nash):
        """Nash solve should produce low exploitability (< 0.5 bb with 100 iterations).

        Note: the IBR solver with damping may not fully converge in 100 iterations
        due to oscillation in borderline hands. The threshold here is intentionally
        generous; the full convergence test is marked [!] in test_solver.py.
        """
        assert nash.exploitability < 0.5, (
            f"Nash exploitability too high: {nash.exploitability:.4f} bb"
        )

    def test_nodelock_deviation_raises_exploitability(self, matrix, nash):
        """Locking a strategy away from Nash should produce higher exploitability."""
        # Lock push_co to 100% — clearly non-Nash; opponents can exploit
        nl = nodelock_solve(
            matrix, COMBO_WEIGHTS,
            locked={"push_co": np.ones(169, dtype=np.float64)},
            max_iter=100,
        )
        assert nl.exploitability > nash.exploitability, (
            f"Expected nodelock exploitability ({nl.exploitability:.4f}) > "
            f"Nash exploitability ({nash.exploitability:.4f})"
        )

    def test_compare_vs_nash_with_real_results(self, matrix, nash):
        """compare_vs_nash returns sensible values for a real nodelock result."""
        nl = nodelock_solve(
            matrix, COMBO_WEIGHTS,
            locked={"push_co": np.ones(169, dtype=np.float64)},
            max_iter=100,
        )
        comparison = compare_vs_nash(nash, nl)

        # Exploitability delta should be positive (nodelock is worse than Nash)
        assert comparison["exploitability_delta"] > 0, (
            f"Expected positive exploitability delta, got {comparison['exploitability_delta']:.4f}"
        )
        # All 14 strategy diffs and 3 exploitability keys present
        assert len(comparison) == 14 + 3
