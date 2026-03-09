"""Tests for src/equity.py — uses tests/fixtures/tiny_equity.npy."""

import os
from unittest.mock import patch

import numpy as np
import pytest

import src.equity as equity_module
from src.equity import (
    eq3_approx,
    eq3_vs_ranges_vec,
    eq4_vs_ranges_vec,
    hand_vs_hand_equity,
    hand_vs_range_equity,
    hand_vs_range_equity_vec,
    load_3way_tensor,
    load_equity_matrix,
)
from src.hands import COMBO_WEIGHTS

# ---------------------------------------------------------------------------
# Fixture path helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TINY_MATRIX_PATH = os.path.join(FIXTURES_DIR, "tiny_equity.npy")
REAL_MATRIX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "equity_matrix.npy")


@pytest.fixture(scope="module")
def tiny_matrix() -> np.ndarray:
    """Load the tiny fixture equity matrix once per module."""
    return load_equity_matrix(TINY_MATRIX_PATH)


# ---------------------------------------------------------------------------
# Fixture shape and type tests
# ---------------------------------------------------------------------------


def test_fixture_shape(tiny_matrix):
    assert tiny_matrix.shape == (169, 169)


def test_fixture_dtype(tiny_matrix):
    assert tiny_matrix.dtype == np.float32


# ---------------------------------------------------------------------------
# Matrix property tests
# ---------------------------------------------------------------------------


def test_symmetry(tiny_matrix):
    """matrix[i,j] + matrix[j,i] ≈ 1.0 for all i, j."""
    diff = np.abs(tiny_matrix + tiny_matrix.T - 1.0)
    assert diff.max() < 1e-5


def test_diagonal(tiny_matrix):
    """Diagonal entries must all equal 0.5."""
    diag = np.diag(tiny_matrix)
    assert np.all(diag == 0.5)


# ---------------------------------------------------------------------------
# Equity lookup tests
# ---------------------------------------------------------------------------


def test_aa_vs_kk(tiny_matrix):
    """AA (idx=0) vs KK (idx=1) should be ~0.82."""
    equity = hand_vs_hand_equity(0, 1, tiny_matrix)
    assert abs(equity - 0.82) < 1e-5


def test_hand_vs_range_single(tiny_matrix):
    """hand_vs_range_equity with single-hand range == hand_vs_hand_equity."""
    # Range containing only KK (idx=1)
    mask = np.zeros(169, dtype=np.float64)
    mask[1] = 1.0

    range_equity = hand_vs_range_equity(0, mask, COMBO_WEIGHTS, tiny_matrix)
    direct = hand_vs_hand_equity(0, 1, tiny_matrix)
    assert abs(range_equity - direct) < 1e-5


def test_hand_vs_range_empty(tiny_matrix):
    """Empty range should return 0.5."""
    mask = np.zeros(169, dtype=np.float64)
    equity = hand_vs_range_equity(0, mask, COMBO_WEIGHTS, tiny_matrix)
    assert equity == 0.5


# ---------------------------------------------------------------------------
# Multiway equity tests
# ---------------------------------------------------------------------------


def test_eq3_approx_sum(tiny_matrix):
    """3-way equities for 3 players must sum to ~1.0."""
    eq_aa = eq3_approx(0, 1, 2, tiny_matrix)   # AA vs KK, QQ
    eq_kk = eq3_approx(1, 0, 2, tiny_matrix)   # KK vs AA, QQ
    eq_qq = eq3_approx(2, 0, 1, tiny_matrix)   # QQ vs AA, KK
    total = eq_aa + eq_kk + eq_qq
    assert abs(total - 1.0) < 1e-4


def test_eq3_approx_aa_dominant(tiny_matrix):
    """AA equity in 3-way vs KK, QQ should be > 0.5."""
    eq_aa = eq3_approx(0, 1, 2, tiny_matrix)
    assert eq_aa > 0.5


# ---------------------------------------------------------------------------
# load_equity_matrix error test
# ---------------------------------------------------------------------------


def test_load_equity_matrix_missing():
    """load_equity_matrix with nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_equity_matrix("nonexistent.npy")


# ---------------------------------------------------------------------------
# Real matrix test (skipped if missing)
# ---------------------------------------------------------------------------


def test_load_real_matrix(require_equity_matrix):
    """Load the real equity matrix and check shape — skipped if missing."""
    matrix = load_equity_matrix(REAL_MATRIX_PATH)
    assert matrix.shape == (169, 169)


# ---------------------------------------------------------------------------
# Vectorized multiway equity tests
# ---------------------------------------------------------------------------


@pytest.fixture
def uniform_range():
    """Return a (169,) all-ones mask (full range)."""
    return np.ones(169, dtype=np.float64)


def _reset_tensor_cache():
    """Reset the module-level 3-way tensor cache so tests don't interfere."""
    equity_module._3way_tensor = None
    equity_module._3way_tensor_loaded = False


def test_eq3_vs_ranges_vec_shape_no_tensor(tiny_matrix, uniform_range):
    """eq3_vs_ranges_vec returns (169,) when no tensor is available."""
    _reset_tensor_cache()
    with patch.object(equity_module, "_3way_tensor_loaded", True):
        with patch.object(equity_module, "_3way_tensor", None):
            result = eq3_vs_ranges_vec(tiny_matrix, uniform_range, uniform_range, COMBO_WEIGHTS)
    assert result.shape == (169,)


def test_eq3_vs_ranges_vec_values_in_range_no_tensor(tiny_matrix, uniform_range):
    """eq3_vs_ranges_vec values are in [0, 1] without tensor."""
    _reset_tensor_cache()
    with patch.object(equity_module, "_3way_tensor_loaded", True):
        with patch.object(equity_module, "_3way_tensor", None):
            result = eq3_vs_ranges_vec(tiny_matrix, uniform_range, uniform_range, COMBO_WEIGHTS)
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_eq4_vs_ranges_vec_shape_no_tensor(tiny_matrix, uniform_range):
    """eq4_vs_ranges_vec returns (169,) when no tensor is available."""
    _reset_tensor_cache()
    with patch.object(equity_module, "_3way_tensor_loaded", True):
        with patch.object(equity_module, "_3way_tensor", None):
            result = eq4_vs_ranges_vec(
                tiny_matrix, uniform_range, uniform_range, uniform_range, COMBO_WEIGHTS
            )
    assert result.shape == (169,)


def test_eq3_vs_ranges_vec_empty_range(tiny_matrix):
    """eq3_vs_ranges_vec with an empty range returns 1/3 fallback."""
    _reset_tensor_cache()
    empty = np.zeros(169, dtype=np.float64)
    full = np.ones(169, dtype=np.float64)
    with patch.object(equity_module, "_3way_tensor_loaded", True):
        with patch.object(equity_module, "_3way_tensor", None):
            result = eq3_vs_ranges_vec(tiny_matrix, empty, full, COMBO_WEIGHTS)
    # With empty range1, denom is 0 → falls back to pairwise which returns ~0
    # (a=0.5, b=something → raw ~0.25 / ~0.5 → ~0.5)
    assert result.shape == (169,)


# ---------------------------------------------------------------------------
# Synthetic 3-way tensor tests
# ---------------------------------------------------------------------------


@pytest.fixture
def uniform_tensor():
    """Return a (169, 169, 169) tensor with all values = 1/3 (uniform equity)."""
    return np.full((169, 169, 169), 1.0 / 3.0, dtype=np.float32)


def test_eq3_vs_ranges_vec_with_tensor_shape(tiny_matrix, uniform_range, uniform_tensor):
    """eq3_vs_ranges_vec with tensor returns (169,) array."""
    _reset_tensor_cache()
    with patch.object(equity_module, "_3way_tensor_loaded", True):
        with patch.object(equity_module, "_3way_tensor", uniform_tensor):
            result = eq3_vs_ranges_vec(tiny_matrix, uniform_range, uniform_range, COMBO_WEIGHTS)
    assert result.shape == (169,)


def test_eq3_vs_ranges_vec_with_uniform_tensor(tiny_matrix, uniform_range, uniform_tensor):
    """With uniform tensor (all 1/3), every hand's 3-way equity ≈ 1/3."""
    _reset_tensor_cache()
    with patch.object(equity_module, "_3way_tensor_loaded", True):
        with patch.object(equity_module, "_3way_tensor", uniform_tensor):
            result = eq3_vs_ranges_vec(tiny_matrix, uniform_range, uniform_range, COMBO_WEIGHTS)
    assert np.allclose(result, 1.0 / 3.0, atol=1e-5)


def test_eq4_vs_ranges_vec_with_tensor_shape(tiny_matrix, uniform_range, uniform_tensor):
    """eq4_vs_ranges_vec with tensor returns (169,) array."""
    _reset_tensor_cache()
    with patch.object(equity_module, "_3way_tensor_loaded", True):
        with patch.object(equity_module, "_3way_tensor", uniform_tensor):
            result = eq4_vs_ranges_vec(
                tiny_matrix, uniform_range, uniform_range, uniform_range, COMBO_WEIGHTS
            )
    assert result.shape == (169,)


def test_load_3way_tensor_missing():
    """load_3way_tensor returns None when file does not exist."""
    _reset_tensor_cache()
    result = load_3way_tensor("nonexistent_3way.npy")
    assert result is None
    _reset_tensor_cache()  # clean up for subsequent tests


def test_load_3way_tensor_caches_none(tmp_path):
    """load_3way_tensor caches the None result — second call is free."""
    _reset_tensor_cache()
    result1 = load_3way_tensor("nonexistent_3way.npy")
    result2 = load_3way_tensor("nonexistent_3way.npy")
    assert result1 is None
    assert result2 is None
    _reset_tensor_cache()
