"""Shared pytest fixtures for AoF solver test suite."""

import os
import pytest


@pytest.fixture
def require_equity_matrix():
    """Skip test if data/equity_matrix.npy is not present.

    Usage:
        def test_something(require_equity_matrix):
            ...  # only runs if matrix exists
    """
    matrix_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "equity_matrix.npy"
    )
    if not os.path.exists(matrix_path):
        pytest.skip("data/equity_matrix.npy not found — skipping equity-dependent test")
