"""Equity matrix loading and lookup functions for AoF solver.

All equity functions operate on the precomputed 169x169 equity matrix.
No Monte Carlo at solve time — all lookups are O(1) or vectorized numpy.
"""

import numpy as np


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
