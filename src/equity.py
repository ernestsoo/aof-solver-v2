"""Equity matrix loading and lookup functions for AoF solver.

All equity functions operate on the precomputed 169x169 equity matrix.
No Monte Carlo at solve time — all lookups are O(1) or vectorized numpy.
"""

import os

import numpy as np

# ---------------------------------------------------------------------------
# 3-way equity tensor (lazy-loaded cache)
# ---------------------------------------------------------------------------
# Shape: (169, 169, 169), dtype float32.
# tensor[h, i, j] = equity of hand h in a 3-way pot vs hands i and j.
# Loaded from data/equity_3way.npy on first use; None if file not present.

_3way_tensor: np.ndarray | None = None
_3way_tensor_loaded: bool = False  # True once we've attempted the load


def load_3way_tensor(path: str = "data/equity_3way.npy") -> np.ndarray | None:
    """Load the 3-way equity tensor from disk (lazy, cached).

    Returns the (169, 169, 169) float32 tensor if the file exists, or None
    if the file is not present. The result is cached in the module-level
    ``_3way_tensor`` variable so subsequent calls are free.

    Args:
        path: Path to the .npy file. Default: "data/equity_3way.npy".

    Returns:
        np.ndarray of shape (169, 169, 169), dtype float32, or None.
    """
    global _3way_tensor, _3way_tensor_loaded
    if _3way_tensor_loaded:
        return _3way_tensor
    _3way_tensor_loaded = True
    if os.path.exists(path):
        _3way_tensor = np.load(path).astype(np.float32)
    return _3way_tensor


def load_equity_matrix(path: str = "data/equity_matrix.npy") -> np.ndarray:
    """Load the precomputed 169x169 equity matrix from disk.

    Args:
        path: Path to the .npy file. Default: "data/equity_matrix.npy".

    Returns:
        np.ndarray of shape (169, 169), dtype float32.
        matrix[i][j] = equity of hand i vs hand j (heads-up).

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    try:
        matrix = np.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Equity matrix not found at '{path}'. "
            "Run scripts/generate_equities.py offline to generate it."
        )
    return matrix.astype(np.float32)


def hand_vs_hand_equity(idx1: int, idx2: int, matrix: np.ndarray) -> float:
    """Return the heads-up equity of hand idx1 against hand idx2.

    O(1) matrix lookup.

    Args:
        idx1: Index of the first hand (0-168).
        idx2: Index of the second hand (0-168).
        matrix: (169, 169) equity matrix.

    Returns:
        Float equity of idx1 vs idx2 (0.0 to 1.0).
    """
    return float(matrix[idx1, idx2])


def hand_vs_range_equity(
    hand_idx: int,
    range_mask: np.ndarray,
    combo_weights: np.ndarray,
    matrix: np.ndarray,
) -> float:
    """Return combo-weighted equity of one hand against a range.

    Fully vectorized — no Python loops over hands. Called thousands of
    times per solve iteration; keep this tight.

    Formula:
        numerator   = sum over j: matrix[hand_idx][j] * range_mask[j] * combo_weights[j]
        denominator = sum over j: range_mask[j] * combo_weights[j]
        equity      = numerator / denominator

    Args:
        hand_idx: Index of the hero hand (0-168).
        range_mask: (169,) array, 1.0 for hands in the opponent range, 0.0 elsewhere.
                    Values may be fractional (mixed strategies).
        combo_weights: (169,) array of combo counts (6/4/12) for each hand.
        matrix: (169, 169) equity matrix.

    Returns:
        Float equity (0.0 to 1.0). Returns 0.5 if the range is empty
        (denominator == 0).
    """
    weighted = range_mask * combo_weights
    denom = weighted.sum()
    if denom == 0.0:
        return 0.5
    return float(np.dot(matrix[hand_idx], weighted) / denom)


# ---------------------------------------------------------------------------
# Multiway equity approximation (pairwise independence approximation)
# ---------------------------------------------------------------------------
# Reference: CLAUDE.md "Multiway Equity — Pairwise Independence Approximation"
#
# For 3-way/4-way pots we approximate multiway equity from the 169x169 pairwise
# matrix. The approximation treats pairwise win probabilities as independent.
# Error is typically < 2% equity — acceptable for Nash convergence at 10bb.


def eq3_approx(h: int, h1: int, h2: int, matrix: np.ndarray) -> float:
    """Approximate 3-way equity of hand h vs specific hands h1 and h2.

    Uses the pairwise independence approximation:
        p_h  = matrix[h][h1]  * matrix[h][h2]
        p_h1 = matrix[h1][h]  * matrix[h1][h2]
        p_h2 = matrix[h2][h]  * matrix[h2][h1]
        equity_h = p_h / (p_h + p_h1 + p_h2)

    Args:
        h:  Index of the hero hand (0-168).
        h1: Index of opponent 1's hand (0-168).
        h2: Index of opponent 2's hand (0-168).
        matrix: (169, 169) equity matrix.

    Returns:
        Float equity of h in the 3-way pot (0.0 to 1.0).
        Returns 1/3 if all products are zero (degenerate case).
    """
    p_h  = float(matrix[h,  h1]) * float(matrix[h,  h2])
    p_h1 = float(matrix[h1, h])  * float(matrix[h1, h2])
    p_h2 = float(matrix[h2, h])  * float(matrix[h2, h1])
    total = p_h + p_h1 + p_h2
    if total == 0.0:
        return 1.0 / 3.0
    return p_h / total


def eq3_vs_ranges(
    h_idx: int,
    range1: np.ndarray,
    range2: np.ndarray,
    combo_weights: np.ndarray,
    matrix: np.ndarray,
) -> float:
    """Approximate 3-way equity of hand h_idx against two ranges.

    Vectorized approximation: compute pairwise equity vs each range, then
    apply the multiway normalization formula.

    Let a = hand_vs_range_equity(h_idx, range1, ...)
        b = hand_vs_range_equity(h_idx, range2, ...)

    Then approximate 3-way equity:
        raw = a * b
        equity = raw / (raw + (1-a)*b + a*(1-b))

    This corresponds to the pairwise independence approximation where:
        p_hero      = a * b        (hero beats both opponents)
        p_range1    = (1-a) * b   (range1 hand beats hero, range2 folds behind)
        p_range2    = a * (1-b)   (range2 hand beats hero, range1 folds behind)

    Args:
        h_idx: Index of the hero hand (0-168).
        range1: (169,) mask for opponent 1's range (may be fractional).
        range2: (169,) mask for opponent 2's range (may be fractional).
        combo_weights: (169,) combo count weights.
        matrix: (169, 169) equity matrix.

    Returns:
        Float 3-way equity of h_idx (0.0 to 1.0).
        Returns 1/3 if both ranges are empty.
    """
    a = hand_vs_range_equity(h_idx, range1, combo_weights, matrix)
    b = hand_vs_range_equity(h_idx, range2, combo_weights, matrix)
    raw = a * b
    denom = raw + (1.0 - a) * b + a * (1.0 - b) + 1e-10
    return raw / denom


def eq4_vs_ranges(
    h_idx: int,
    range1: np.ndarray,
    range2: np.ndarray,
    range3: np.ndarray,
    combo_weights: np.ndarray,
    matrix: np.ndarray,
) -> float:
    """Approximate 4-way equity of hand h_idx against three ranges.

    Extends the pairwise independence approximation to 4-way pots.

    Let a = hand_vs_range_equity(h_idx, range1, ...)
        b = hand_vs_range_equity(h_idx, range2, ...)
        c = hand_vs_range_equity(h_idx, range3, ...)

    Approximate 4-way equity:
        raw = a * b * c
        equity = raw / (raw + (1-a)*b*c + a*(1-b)*c + a*b*(1-c))

    Args:
        h_idx: Index of the hero hand (0-168).
        range1: (169,) mask for opponent 1's range.
        range2: (169,) mask for opponent 2's range.
        range3: (169,) mask for opponent 3's range.
        combo_weights: (169,) combo count weights.
        matrix: (169, 169) equity matrix.

    Returns:
        Float 4-way equity of h_idx (0.0 to 1.0).
        Returns 0.25 if all ranges are empty.
    """
    a = hand_vs_range_equity(h_idx, range1, combo_weights, matrix)
    b = hand_vs_range_equity(h_idx, range2, combo_weights, matrix)
    c = hand_vs_range_equity(h_idx, range3, combo_weights, matrix)
    raw = a * b * c
    denom = (
        raw
        + (1.0 - a) * b * c
        + a * (1.0 - b) * c
        + a * b * (1.0 - c)
        + 1e-10
    )
    return raw / denom


# ---------------------------------------------------------------------------
# Vectorized multiway equity — returns (169,) arrays for all hands at once
# ---------------------------------------------------------------------------
# These are the "vec" variants used in the EV computation functions in solver.py.
# Each operates on all 169 hands simultaneously via matrix multiplication.


def hand_vs_range_equity_vec(
    matrix: np.ndarray,
    range_mask: np.ndarray,
    combo_weights: np.ndarray,
) -> np.ndarray:
    """Return combo-weighted equity of every hand against a range (vectorized).

    Computes equity for all 169 hero hands simultaneously via matrix multiply:
        weighted = range_mask * combo_weights          # (169,)
        equity   = matrix @ weighted / weighted.sum() # (169,)

    Args:
        matrix:        (169, 169) equity matrix, matrix[i][j] = equity of i vs j.
        range_mask:    (169,) array, 1.0 for hands in the opponent range (may be fractional).
        combo_weights: (169,) array of combo counts (6/4/12).

    Returns:
        (169,) float64 array: equity[i] = equity of hand i against the range.
        Returns np.full(169, 0.5) if the range is empty (denominator == 0).
    """
    weighted = range_mask * combo_weights
    denom = weighted.sum()
    if denom == 0.0:
        return np.full(169, 0.5)
    return (matrix @ weighted) / denom


def eq3_vs_ranges_vec(
    matrix: np.ndarray,
    range1_mask: np.ndarray,
    range2_mask: np.ndarray,
    combo_weights: np.ndarray,
) -> np.ndarray:
    """Return 3-way equity for every hand vs two ranges (vectorized).

    Uses the precomputed 3-way tensor (data/equity_3way.npy) when available
    for an exact weighted-average lookup; falls back to the pairwise independence
    approximation when the tensor is not present.

    **Tensor path** (exact):
        w1 = range1_mask * combo_weights  (unnormalized)
        w2 = range2_mask * combo_weights  (unnormalized)
        equity[h] = sum_{i,j} tensor[h,i,j] * w1_norm[i] * w2_norm[j]
        Computed efficiently as: (tensor @ w2_norm) @ w1_norm  →  (169,)

    **Fallback path** (pairwise independence approximation):
        a = hand_vs_range_equity_vec(matrix, range1, combo_weights)  # (169,)
        b = hand_vs_range_equity_vec(matrix, range2, combo_weights)  # (169,)
        raw = a * b
        equity = raw / (raw + (1-a)*b + a*(1-b) + 1e-10)

    Args:
        matrix:        (169, 169) equity matrix.
        range1_mask:   (169,) mask for opponent 1's range.
        range2_mask:   (169,) mask for opponent 2's range.
        combo_weights: (169,) combo count weights.

    Returns:
        (169,) float64 array: 3-way equity for each hand.
    """
    tensor = load_3way_tensor()
    if tensor is not None:
        w1 = range1_mask * combo_weights
        w2 = range2_mask * combo_weights
        denom1 = w1.sum()
        denom2 = w2.sum()
        if denom1 == 0.0 or denom2 == 0.0:
            return np.full(169, 1.0 / 3.0)
        w1_norm = w1 / denom1
        w2_norm = w2 / denom2
        # (169,169,169) @ (169,) → (169,169), then (169,169) @ (169,) → (169,)
        return (tensor @ w2_norm) @ w1_norm

    # Fallback: pairwise independence approximation
    a = hand_vs_range_equity_vec(matrix, range1_mask, combo_weights)
    b = hand_vs_range_equity_vec(matrix, range2_mask, combo_weights)
    raw = a * b
    denom = raw + (1.0 - a) * b + a * (1.0 - b) + 1e-10
    return raw / denom


def eq4_vs_ranges_vec(
    matrix: np.ndarray,
    range1_mask: np.ndarray,
    range2_mask: np.ndarray,
    range3_mask: np.ndarray,
    combo_weights: np.ndarray,
) -> np.ndarray:
    """Return approximate 4-way equity for every hand vs three ranges (vectorized).

    Uses the precomputed 3-way tensor when available for a better approximation;
    falls back to the pure pairwise independence approximation otherwise.

    **Tensor path**:
        Compute eq3(h, r1, r2) via tensor → (169,) array.
        Then combine with pairwise equity vs r3 using a 2-factor normalization:
            eq3_12 = eq3_vs_ranges_vec(tensor, r1, r2, ...)   (169,)
            eq_r3  = hand_vs_range_equity_vec(matrix, r3, ...) (169,)
            raw    = eq3_12 * eq_r3
            equity = raw / (raw + (1-eq3_12)*eq_r3 + eq3_12*(1-eq_r3) + 1e-10)

    **Fallback path** (pairwise independence approximation):
        a = hand_vs_range_equity_vec(matrix, range1, combo_weights)  # (169,)
        b = hand_vs_range_equity_vec(matrix, range2, combo_weights)  # (169,)
        c = hand_vs_range_equity_vec(matrix, range3, combo_weights)  # (169,)
        raw = a * b * c
        equity = raw / (raw + (1-a)*b*c + a*(1-b)*c + a*b*(1-c) + 1e-10)

    Args:
        matrix:        (169, 169) equity matrix.
        range1_mask:   (169,) mask for opponent 1's range.
        range2_mask:   (169,) mask for opponent 2's range.
        range3_mask:   (169,) mask for opponent 3's range.
        combo_weights: (169,) combo count weights.

    Returns:
        (169,) float64 array: approximate 4-way equity for each hand.
    """
    tensor = load_3way_tensor()
    if tensor is not None:
        # Use exact 3-way tensor for r1, r2; combine with pairwise for r3
        eq3_12 = eq3_vs_ranges_vec(matrix, range1_mask, range2_mask, combo_weights)
        eq_r3 = hand_vs_range_equity_vec(matrix, range3_mask, combo_weights)
        raw = eq3_12 * eq_r3
        denom = raw + (1.0 - eq3_12) * eq_r3 + eq3_12 * (1.0 - eq_r3) + 1e-10
        return raw / denom

    # Fallback: pairwise independence approximation
    a = hand_vs_range_equity_vec(matrix, range1_mask, combo_weights)
    b = hand_vs_range_equity_vec(matrix, range2_mask, combo_weights)
    c = hand_vs_range_equity_vec(matrix, range3_mask, combo_weights)
    raw = a * b * c
    denom = (
        raw
        + (1.0 - a) * b * c
        + a * (1.0 - b) * c
        + a * b * (1.0 - c)
        + 1e-10
    )
    return raw / denom
