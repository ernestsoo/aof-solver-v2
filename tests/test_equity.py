"""Tests for src/equity.py — uses tests/fixtures/tiny_equity.npy."""

import os

import numpy as np
import pytest

from src.equity import (
    eq3_approx,
    hand_vs_hand_equity,
    hand_vs_range_equity,
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
