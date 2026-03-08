"""Tests for src/solver.py — Phase 3 (tasks 3.1, 3.2, 3.3).

Uses tests/fixtures/tiny_equity.npy (169×169, mostly 0.5 except AA/KK/QQ/72o).
Tests requiring the real equity matrix are skipped automatically.
"""

import numpy as np
import pytest

from src.hands import COMBO_WEIGHTS, HAND_MAP
from src.solver import (
    STRATEGY_NAMES,
    SolverResult,
    call_prob,
    ev_push_co,
    fold_prob,
    initial_strategies,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_PATH = "tests/fixtures/tiny_equity.npy"


@pytest.fixture
def tiny_matrix() -> np.ndarray:
    """Load the tiny 169×169 test equity matrix."""
    return np.load(FIXTURE_PATH).astype(np.float64)


@pytest.fixture
def all_call_strategies() -> dict[str, np.ndarray]:
    """All 14 strategy arrays set to 1.0 (everyone always calls/pushes)."""
    return {name: np.ones(169) for name in STRATEGY_NAMES}


@pytest.fixture
def all_fold_strategies() -> dict[str, np.ndarray]:
    """All 14 strategy arrays set to 0.0 (everyone always folds)."""
    return {name: np.zeros(169) for name in STRATEGY_NAMES}


@pytest.fixture
def nash_init_strategies() -> dict[str, np.ndarray]:
    """Strategies from initial_strategies() (task 3.1 baseline)."""
    return initial_strategies(COMBO_WEIGHTS)


# ---------------------------------------------------------------------------
# Task 3.1 — SolverResult and initial_strategies
# ---------------------------------------------------------------------------

class TestSolverResult:
    def test_default_fields(self):
        strats = {name: np.zeros(169) for name in STRATEGY_NAMES}
        result = SolverResult(strategies=strats)
        assert result.iterations == 0
        assert result.converged is False
        assert result.exploitability == 0.0
        assert result.ev_table == {}

    def test_strategies_dict(self):
        strats = {name: np.zeros(169) for name in STRATEGY_NAMES}
        result = SolverResult(strategies=strats)
        assert set(result.strategies.keys()) == set(STRATEGY_NAMES)


class TestInitialStrategies:
    def test_returns_14_arrays(self, nash_init_strategies):
        assert len(nash_init_strategies) == 14

    def test_all_correct_shape(self, nash_init_strategies):
        for name, arr in nash_init_strategies.items():
            assert arr.shape == (169,), f"{name}: expected (169,)"

    def test_all_float64(self, nash_init_strategies):
        for name, arr in nash_init_strategies.items():
            assert arr.dtype == np.float64, f"{name}: expected float64"

    def test_binary_values(self, nash_init_strategies):
        for name, arr in nash_init_strategies.items():
            assert np.all((arr == 0.0) | (arr == 1.0)), f"{name}: non-binary values"

    def test_push_ordering(self, nash_init_strategies):
        """SB open >= BTN open >= CO push (wider by position)."""
        sb = nash_init_strategies["push_sb_open"].sum()
        btn = nash_init_strategies["push_btn_open"].sum()
        co = nash_init_strategies["push_co"].sum()
        assert sb >= btn >= co

    def test_call_narrower_than_co_push(self, nash_init_strategies):
        """All call ranges start narrower than CO push."""
        co_count = nash_init_strategies["push_co"].sum()
        for name in STRATEGY_NAMES:
            if name.startswith("call_"):
                assert nash_init_strategies[name].sum() <= co_count


# ---------------------------------------------------------------------------
# Task 3.2 — fold_prob / call_prob
# ---------------------------------------------------------------------------

class TestFoldCallProb:
    def test_fold_prob_all_zeros(self):
        assert fold_prob(np.zeros(169), COMBO_WEIGHTS) == pytest.approx(1.0)

    def test_fold_prob_all_ones(self):
        assert fold_prob(np.ones(169), COMBO_WEIGHTS) == pytest.approx(0.0)

    def test_call_prob_all_zeros(self):
        assert call_prob(np.zeros(169), COMBO_WEIGHTS) == pytest.approx(0.0)

    def test_call_prob_all_ones(self):
        assert call_prob(np.ones(169), COMBO_WEIGHTS) == pytest.approx(1.0)

    def test_complement_identity(self, nash_init_strategies):
        for name, strat in nash_init_strategies.items():
            fp = fold_prob(strat, COMBO_WEIGHTS)
            cp = call_prob(strat, COMBO_WEIGHTS)
            assert fp + cp == pytest.approx(1.0, abs=1e-12), f"Failed for {name}"


# ---------------------------------------------------------------------------
# Task 3.3 — ev_push_co
# ---------------------------------------------------------------------------

class TestEvPushCo:
    def test_returns_shape(self, tiny_matrix, all_call_strategies):
        ev = ev_push_co(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.shape == (169,)

    def test_returns_float64(self, tiny_matrix, all_call_strategies):
        ev = ev_push_co(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.dtype == np.float64

    def test_all_fold_opponents_gives_steal(self, tiny_matrix, all_fold_strategies):
        """When all opponents fold, every hand's EV = +1.5bb (steal blind + antes)."""
        ev = ev_push_co(tiny_matrix, COMBO_WEIGHTS, all_fold_strategies)
        np.testing.assert_allclose(ev, 1.5, atol=1e-10)

    def test_aa_beats_72o(self, tiny_matrix, nash_init_strategies):
        """AA should have higher EV pushing than 72o."""
        ev = ev_push_co(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        aa_idx  = HAND_MAP["AA"].index    # 0
        o72_idx = HAND_MAP["72o"].index   # 168
        assert ev[aa_idx] > ev[o72_idx], (
            f"AA EV={ev[aa_idx]:.4f} should exceed 72o EV={ev[o72_idx]:.4f}"
        )

    def test_72o_may_be_negative(self, tiny_matrix, all_call_strategies):
        """72o EV against tight all-call opponents should be negative (fish hand)."""
        ev = ev_push_co(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        o72_idx = HAND_MAP["72o"].index
        # With uniform 0.5 equity and all callers, EV = 0.5 * pot - 10 < 0
        assert ev[o72_idx] < 0.0, f"72o EV={ev[o72_idx]:.4f} should be negative"

    def test_aa_positive_all_call(self, tiny_matrix, all_call_strategies):
        """AA (index 0) has equity ~0.82+ vs KK in fixture; should be positive EV."""
        ev = ev_push_co(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        aa_idx = HAND_MAP["AA"].index
        assert ev[aa_idx] > 0.0, f"AA EV={ev[aa_idx]:.4f} should be positive"

    def test_no_nan_or_inf(self, tiny_matrix, nash_init_strategies):
        ev = ev_push_co(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert np.all(np.isfinite(ev)), "EV array contains NaN or Inf"

    def test_ev_fold_zero(self, tiny_matrix, nash_init_strategies):
        """EV(fold) for CO = 0. Pushing is better when ev > 0."""
        ev = ev_push_co(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        # Strongest hands should have positive EV (worth pushing)
        aa_idx = HAND_MAP["AA"].index
        assert ev[aa_idx] > 0.0  # AA strictly better than folding (EV=0)

    def test_equity_matrix_dtype_tolerance(self, tiny_matrix, all_call_strategies):
        """Function must handle float32 matrix (as loaded from disk) without error."""
        matrix32 = tiny_matrix.astype(np.float32)
        ev = ev_push_co(matrix32, COMBO_WEIGHTS, all_call_strategies)
        assert ev.shape == (169,)
        assert np.all(np.isfinite(ev))

    def test_steal_component_correct(self, tiny_matrix):
        """Sanity-check the all-fold terminal (Terminal 8) in isolation.

        With all opponents folding (fold_prob = 1.0), EV = 1.5 for every hand.
        """
        strats = {name: np.zeros(169) for name in STRATEGY_NAMES}
        ev = ev_push_co(tiny_matrix, COMBO_WEIGHTS, strats)
        np.testing.assert_allclose(ev, 1.5, atol=1e-10)
