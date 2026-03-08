"""Tests for src/solver.py — Phase 3 (tasks 3.1–3.8).

Uses tests/fixtures/tiny_equity.npy (169×169, mostly 0.5 except AA/KK/QQ/72o).
Tests requiring the real equity matrix are skipped automatically.
"""

import numpy as np
import pytest

from src.hands import COMBO_WEIGHTS, HAND_MAP
from src.solver import (
    STRATEGY_NAMES,
    SolverResult,
    best_response,
    call_prob,
    ev_call_bb_vs_btn,
    ev_call_bb_vs_btn_sb,
    ev_call_bb_vs_co,
    ev_call_bb_vs_co_btn,
    ev_call_bb_vs_co_btn_sb,
    ev_call_bb_vs_co_sb,
    ev_call_bb_vs_sb,
    ev_call_btn_vs_co,
    ev_call_sb_vs_btn,
    ev_call_sb_vs_co,
    ev_call_sb_vs_co_btn,
    ev_push_btn_open,
    ev_push_co,
    ev_push_sb_open,
    fold_prob,
    initial_strategies,
    solve_nash,
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


# ---------------------------------------------------------------------------
# Task 3.4 — ev_push_btn_open and ev_call_btn_vs_co
# ---------------------------------------------------------------------------


class TestEvPushBtnOpen:
    def test_returns_shape(self, tiny_matrix, all_call_strategies):
        ev = ev_push_btn_open(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.shape == (169,)

    def test_returns_float64(self, tiny_matrix, all_call_strategies):
        ev = ev_push_btn_open(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.dtype == np.float64

    def test_no_nan_or_inf(self, tiny_matrix, nash_init_strategies):
        ev = ev_push_btn_open(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert np.all(np.isfinite(ev))

    def test_all_fold_gives_steal(self, tiny_matrix, all_fold_strategies):
        """All opponents fold → BTN steals 1.5bb (SB 0.5 + BB 1.0 posted)."""
        ev = ev_push_btn_open(tiny_matrix, COMBO_WEIGHTS, all_fold_strategies)
        np.testing.assert_allclose(ev, 1.5, atol=1e-10)

    def test_aa_beats_72o(self, tiny_matrix, nash_init_strategies):
        ev = ev_push_btn_open(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        aa_idx  = HAND_MAP["AA"].index
        o72_idx = HAND_MAP["72o"].index
        assert ev[aa_idx] > ev[o72_idx]

    def test_72o_negative_all_callers(self, tiny_matrix, all_call_strategies):
        """72o has near-0.5 equity; EV = 0.5 * pot - 10 < 0 against callers."""
        ev = ev_push_btn_open(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev[HAND_MAP["72o"].index] < 0.0

    def test_aa_positive_all_callers(self, tiny_matrix, all_call_strategies):
        ev = ev_push_btn_open(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev[HAND_MAP["AA"].index] > 0.0


class TestEvCallBtnVsCo:
    def test_returns_shape(self, tiny_matrix, all_call_strategies):
        ev = ev_call_btn_vs_co(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.shape == (169,)

    def test_returns_float64(self, tiny_matrix, all_call_strategies):
        ev = ev_call_btn_vs_co(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.dtype == np.float64

    def test_no_nan_or_inf(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_btn_vs_co(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert np.all(np.isfinite(ev))

    def test_aa_beats_72o(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_btn_vs_co(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        aa_idx  = HAND_MAP["AA"].index
        o72_idx = HAND_MAP["72o"].index
        assert ev[aa_idx] > ev[o72_idx]

    def test_aa_positive_vs_nash_co(self, tiny_matrix, nash_init_strategies):
        """AA calling CO push should have positive EV in fixture (AA equity ~0.82+ vs KK)."""
        ev = ev_call_btn_vs_co(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert ev[HAND_MAP["AA"].index] > 0.0

    def test_72o_negative_all_callers(self, tiny_matrix, all_call_strategies):
        """72o should have negative EV calling when everyone else also calls."""
        ev = ev_call_btn_vs_co(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev[HAND_MAP["72o"].index] < 0.0


# ---------------------------------------------------------------------------
# Task 3.5 — ev_push_sb_open, ev_call_sb_vs_co, ev_call_sb_vs_btn,
#             ev_call_sb_vs_co_btn
# ---------------------------------------------------------------------------


class TestEvPushSbOpen:
    def test_returns_shape(self, tiny_matrix, all_call_strategies):
        ev = ev_push_sb_open(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.shape == (169,)

    def test_returns_float64(self, tiny_matrix, all_call_strategies):
        ev = ev_push_sb_open(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.dtype == np.float64

    def test_no_nan_or_inf(self, tiny_matrix, nash_init_strategies):
        ev = ev_push_sb_open(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert np.all(np.isfinite(ev))

    def test_all_fold_gives_steal_net_1bb(self, tiny_matrix, all_fold_strategies):
        """BB always folds → SB steals, net profit = +1.0bb (wins BB's 1.0)."""
        ev = ev_push_sb_open(tiny_matrix, COMBO_WEIGHTS, all_fold_strategies)
        np.testing.assert_allclose(ev, 1.0, atol=1e-10)

    def test_aa_beats_72o(self, tiny_matrix, nash_init_strategies):
        ev = ev_push_sb_open(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        aa_idx  = HAND_MAP["AA"].index
        o72_idx = HAND_MAP["72o"].index
        assert ev[aa_idx] > ev[o72_idx]

    def test_steal_dominant_term(self, tiny_matrix):
        """With BB folding 90% of the time, steal component should dominate EV."""
        strats = {name: np.zeros(169) for name in STRATEGY_NAMES}
        # BB calls 10% of the time (top 10% of hands)
        from src.hands import top_n_percent
        strats["call_bb_vs_sb"] = top_n_percent(10.0)
        ev = ev_push_sb_open(tiny_matrix, COMBO_WEIGHTS, strats)
        # Fold prob ~ 0.9 → steal component ~ 0.9 * 1.0 = 0.9bb
        # All EVs should be roughly near +1.0 for weak hands (steal dominant)
        o72_idx = HAND_MAP["72o"].index
        assert ev[o72_idx] > 0.5, f"72o EV={ev[o72_idx]:.4f}: steal should dominate with 90% BB fold"


class TestEvCallSbVsCo:
    def test_returns_shape(self, tiny_matrix, all_call_strategies):
        ev = ev_call_sb_vs_co(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.shape == (169,)

    def test_returns_float64(self, tiny_matrix, all_call_strategies):
        ev = ev_call_sb_vs_co(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.dtype == np.float64

    def test_no_nan_or_inf(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_sb_vs_co(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert np.all(np.isfinite(ev))

    def test_aa_beats_72o(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_sb_vs_co(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert ev[HAND_MAP["AA"].index] > ev[HAND_MAP["72o"].index]

    def test_aa_positive_vs_nash_co(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_sb_vs_co(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert ev[HAND_MAP["AA"].index] > 0.0

    def test_72o_negative_all_callers(self, tiny_matrix, all_call_strategies):
        ev = ev_call_sb_vs_co(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev[HAND_MAP["72o"].index] < 0.0


class TestEvCallSbVsBtn:
    def test_returns_shape(self, tiny_matrix, all_call_strategies):
        ev = ev_call_sb_vs_btn(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.shape == (169,)

    def test_returns_float64(self, tiny_matrix, all_call_strategies):
        ev = ev_call_sb_vs_btn(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.dtype == np.float64

    def test_no_nan_or_inf(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_sb_vs_btn(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert np.all(np.isfinite(ev))

    def test_aa_beats_72o(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_sb_vs_btn(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert ev[HAND_MAP["AA"].index] > ev[HAND_MAP["72o"].index]

    def test_72o_negative_all_callers(self, tiny_matrix, all_call_strategies):
        ev = ev_call_sb_vs_btn(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev[HAND_MAP["72o"].index] < 0.0


class TestEvCallSbVsCoBtn:
    def test_returns_shape(self, tiny_matrix, all_call_strategies):
        ev = ev_call_sb_vs_co_btn(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.shape == (169,)

    def test_returns_float64(self, tiny_matrix, all_call_strategies):
        ev = ev_call_sb_vs_co_btn(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.dtype == np.float64

    def test_no_nan_or_inf(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_sb_vs_co_btn(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert np.all(np.isfinite(ev))

    def test_aa_beats_72o(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_sb_vs_co_btn(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert ev[HAND_MAP["AA"].index] > ev[HAND_MAP["72o"].index]

    def test_72o_negative_all_callers(self, tiny_matrix, all_call_strategies):
        """Calling into two callers with 72o should be deeply negative."""
        ev = ev_call_sb_vs_co_btn(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev[HAND_MAP["72o"].index] < 0.0


# ---------------------------------------------------------------------------
# Task 3.6 — BB EV functions
# ---------------------------------------------------------------------------


class TestEvCallBbVsSb:
    def test_returns_shape(self, tiny_matrix, all_call_strategies):
        ev = ev_call_bb_vs_sb(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.shape == (169,)

    def test_returns_float64(self, tiny_matrix, all_call_strategies):
        ev = ev_call_bb_vs_sb(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.dtype == np.float64

    def test_no_nan_or_inf(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_bb_vs_sb(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert np.all(np.isfinite(ev))

    def test_aa_beats_72o(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_bb_vs_sb(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert ev[HAND_MAP["AA"].index] > ev[HAND_MAP["72o"].index]

    def test_aa_positive_all_callers(self, tiny_matrix, all_call_strategies):
        """AA vs uniform 0.5 range: 0.5 * 20.0 - 10.0 = 0.0 (boundary); fixture pushes AA above."""
        ev = ev_call_bb_vs_sb(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev[HAND_MAP["AA"].index] >= 0.0

    def test_72o_negative_all_callers(self, tiny_matrix, all_call_strategies):
        ev = ev_call_bb_vs_sb(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev[HAND_MAP["72o"].index] <= 0.0


class TestEvCallBbVsBtn:
    def test_returns_shape(self, tiny_matrix, all_call_strategies):
        ev = ev_call_bb_vs_btn(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.shape == (169,)

    def test_returns_float64(self, tiny_matrix, all_call_strategies):
        ev = ev_call_bb_vs_btn(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.dtype == np.float64

    def test_no_nan_or_inf(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_bb_vs_btn(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert np.all(np.isfinite(ev))

    def test_aa_beats_72o(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_bb_vs_btn(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert ev[HAND_MAP["AA"].index] > ev[HAND_MAP["72o"].index]

    def test_uniform_half_equity_gives_correct_ev(self, tiny_matrix):
        """All hands have 0.5 equity in tiny fixture default; EV = 0.5*20.5-10 = 0.25."""
        strats = {name: np.ones(169) for name in STRATEGY_NAMES}
        ev = ev_call_bb_vs_btn(tiny_matrix, COMBO_WEIGHTS, strats)
        # tiny_matrix is mostly 0.5; weighted average equity ~ 0.5 → EV ~ 0.25
        assert np.all(np.isfinite(ev))


class TestEvCallBbVsCo:
    def test_returns_shape(self, tiny_matrix, all_call_strategies):
        ev = ev_call_bb_vs_co(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.shape == (169,)

    def test_returns_float64(self, tiny_matrix, all_call_strategies):
        ev = ev_call_bb_vs_co(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.dtype == np.float64

    def test_no_nan_or_inf(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_bb_vs_co(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert np.all(np.isfinite(ev))

    def test_aa_beats_72o(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_bb_vs_co(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert ev[HAND_MAP["AA"].index] > ev[HAND_MAP["72o"].index]


class TestEvCallBbVsBtnSb:
    def test_returns_shape(self, tiny_matrix, all_call_strategies):
        ev = ev_call_bb_vs_btn_sb(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.shape == (169,)

    def test_returns_float64(self, tiny_matrix, all_call_strategies):
        ev = ev_call_bb_vs_btn_sb(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.dtype == np.float64

    def test_no_nan_or_inf(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_bb_vs_btn_sb(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert np.all(np.isfinite(ev))

    def test_aa_beats_72o(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_bb_vs_btn_sb(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert ev[HAND_MAP["AA"].index] > ev[HAND_MAP["72o"].index]

    def test_72o_negative_all_callers(self, tiny_matrix, all_call_strategies):
        """72o calling into two opponents: deeply negative."""
        ev = ev_call_bb_vs_btn_sb(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev[HAND_MAP["72o"].index] < 0.0


class TestEvCallBbVsCoSb:
    def test_returns_shape(self, tiny_matrix, all_call_strategies):
        ev = ev_call_bb_vs_co_sb(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.shape == (169,)

    def test_returns_float64(self, tiny_matrix, all_call_strategies):
        ev = ev_call_bb_vs_co_sb(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.dtype == np.float64

    def test_no_nan_or_inf(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_bb_vs_co_sb(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert np.all(np.isfinite(ev))

    def test_aa_beats_72o(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_bb_vs_co_sb(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert ev[HAND_MAP["AA"].index] > ev[HAND_MAP["72o"].index]


class TestEvCallBbVsCoBtn:
    def test_returns_shape(self, tiny_matrix, all_call_strategies):
        ev = ev_call_bb_vs_co_btn(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.shape == (169,)

    def test_returns_float64(self, tiny_matrix, all_call_strategies):
        ev = ev_call_bb_vs_co_btn(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.dtype == np.float64

    def test_no_nan_or_inf(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_bb_vs_co_btn(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert np.all(np.isfinite(ev))

    def test_aa_beats_72o(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_bb_vs_co_btn(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert ev[HAND_MAP["AA"].index] > ev[HAND_MAP["72o"].index]


class TestEvCallBbVsCoOrBtnSb:
    def test_returns_shape(self, tiny_matrix, all_call_strategies):
        ev = ev_call_bb_vs_co_btn_sb(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.shape == (169,)

    def test_returns_float64(self, tiny_matrix, all_call_strategies):
        ev = ev_call_bb_vs_co_btn_sb(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev.dtype == np.float64

    def test_no_nan_or_inf(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_bb_vs_co_btn_sb(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert np.all(np.isfinite(ev))

    def test_aa_beats_72o(self, tiny_matrix, nash_init_strategies):
        ev = ev_call_bb_vs_co_btn_sb(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        assert ev[HAND_MAP["AA"].index] > ev[HAND_MAP["72o"].index]

    def test_72o_negative_all_callers(self, tiny_matrix, all_call_strategies):
        """72o calling into three opponents in a 4-way pot is deeply negative."""
        ev = ev_call_bb_vs_co_btn_sb(tiny_matrix, COMBO_WEIGHTS, all_call_strategies)
        assert ev[HAND_MAP["72o"].index] < 0.0


# ---------------------------------------------------------------------------
# Task 3.7 — best_response
# ---------------------------------------------------------------------------


class TestBestResponse:
    def test_returns_shape(self):
        ev = np.zeros(169)
        old = np.zeros(169)
        result = best_response(ev, 0.0, old)
        assert result.shape == (169,)

    def test_returns_float64(self):
        ev = np.zeros(169)
        old = np.zeros(169)
        result = best_response(ev, 0.0, old)
        assert result.dtype == np.float64

    def test_pure_best_no_damping(self):
        """With alpha=1.0 (no damping) result should be exactly 0/1 binary."""
        ev = np.array([1.0, -1.0, 0.5, -0.5] + [0.0] * 165)
        old = np.zeros(169)
        result = best_response(ev, 0.0, old, alpha=1.0)
        assert result[0] == 1.0   # 1.0 > 0.0
        assert result[1] == 0.0   # -1.0 not > 0.0
        assert result[2] == 1.0   # 0.5 > 0.0
        assert result[3] == 0.0   # -0.5 not > 0.0

    def test_damping_blends_with_old(self):
        """Damping should blend: alpha * pure_best + (1-alpha) * old."""
        ev = np.ones(169) * 1.0   # all positive → pure_best = all 1.0
        old = np.zeros(169)
        result = best_response(ev, 0.0, old, alpha=0.9)
        # pure_best = 1.0, old = 0.0 → result = 0.9 * 1.0 + 0.1 * 0.0 = 0.9
        np.testing.assert_allclose(result, 0.9, atol=1e-12)

    def test_fold_always_when_ev_below_fold(self):
        """When all EVs are below ev_fold, all hands should fold."""
        ev = np.full(169, -2.0)
        old = np.ones(169)
        result = best_response(ev, -1.0, old, alpha=1.0)
        assert np.all(result == 0.0)

    def test_aa_gets_one_with_positive_ev(self, tiny_matrix, nash_init_strategies):
        """AA should get best response = 1.0 (push) when its EV exceeds fold."""
        ev = ev_push_co(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        old = nash_init_strategies["push_co"]
        result = best_response(ev, 0.0, old, alpha=1.0)
        assert result[HAND_MAP["AA"].index] == 1.0

    def test_no_nan_or_inf(self, tiny_matrix, nash_init_strategies):
        ev = ev_push_co(tiny_matrix, COMBO_WEIGHTS, nash_init_strategies)
        old = nash_init_strategies["push_co"]
        result = best_response(ev, 0.0, old)
        assert np.all(np.isfinite(result))

    def test_boundary_ev_equal_to_fold_folds(self):
        """Hands with EV exactly equal to ev_fold should fold (not strictly greater)."""
        ev = np.zeros(169)  # EV = 0.0 = ev_fold
        old = np.zeros(169)
        result = best_response(ev, 0.0, old, alpha=1.0)
        assert np.all(result == 0.0)


# ---------------------------------------------------------------------------
# Task 3.8 — solve_nash
# ---------------------------------------------------------------------------


class TestSolveNash:
    def test_returns_solver_result(self, tiny_matrix):
        result = solve_nash(tiny_matrix, COMBO_WEIGHTS, max_iter=10)
        assert isinstance(result, SolverResult)

    def test_strategies_have_correct_keys(self, tiny_matrix):
        result = solve_nash(tiny_matrix, COMBO_WEIGHTS, max_iter=10)
        assert set(result.strategies.keys()) == set(STRATEGY_NAMES)

    def test_all_strategy_shapes(self, tiny_matrix):
        result = solve_nash(tiny_matrix, COMBO_WEIGHTS, max_iter=10)
        for name, arr in result.strategies.items():
            assert arr.shape == (169,), f"{name}: expected (169,)"

    def test_all_strategy_float64(self, tiny_matrix):
        result = solve_nash(tiny_matrix, COMBO_WEIGHTS, max_iter=10)
        for name, arr in result.strategies.items():
            assert arr.dtype == np.float64, f"{name}: expected float64"

    def test_iterations_bounded(self, tiny_matrix):
        result = solve_nash(tiny_matrix, COMBO_WEIGHTS, max_iter=50)
        assert 1 <= result.iterations <= 50

    def test_converged_or_max_iter(self, tiny_matrix):
        """Either converged before limit or ran exactly max_iter iterations."""
        result = solve_nash(tiny_matrix, COMBO_WEIGHTS, max_iter=50)
        assert result.converged or result.iterations == 50

    def test_ev_table_has_all_keys(self, tiny_matrix):
        result = solve_nash(tiny_matrix, COMBO_WEIGHTS, max_iter=10)
        assert set(result.ev_table.keys()) == set(STRATEGY_NAMES)

    def test_ev_table_shapes(self, tiny_matrix):
        result = solve_nash(tiny_matrix, COMBO_WEIGHTS, max_iter=10)
        for name, arr in result.ev_table.items():
            assert arr.shape == (169,), f"ev_table[{name}]: expected (169,)"

    def test_aa_pushed_from_co(self, tiny_matrix):
        """AA should always be pushed from CO (highest EV hand)."""
        result = solve_nash(tiny_matrix, COMBO_WEIGHTS, max_iter=50)
        assert result.strategies["push_co"][HAND_MAP["AA"].index] == pytest.approx(1.0, abs=0.15)

    def test_no_nan_in_ev_table(self, tiny_matrix):
        result = solve_nash(tiny_matrix, COMBO_WEIGHTS, max_iter=10)
        for name, arr in result.ev_table.items():
            assert np.all(np.isfinite(arr)), f"ev_table[{name}] contains NaN/Inf"

    def test_exploitability_placeholder(self, tiny_matrix):
        """exploitability is 0.0 placeholder until task 3.9."""
        result = solve_nash(tiny_matrix, COMBO_WEIGHTS, max_iter=10)
        assert result.exploitability == 0.0
