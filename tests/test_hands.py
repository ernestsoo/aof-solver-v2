"""Tests for src/hands.py — hand representations, rankings, and range utilities."""

import numpy as np
import pytest

from src.hands import (
    ALL_HANDS,
    COMBO_WEIGHTS,
    HAND_MAP,
    combos_with_removal,
    grid_to_hand,
    hand_to_grid,
    hands_to_range_pct,
    mask_to_hands,
    parse_range,
    range_to_mask,
    top_n_percent,
)


# ---------------------------------------------------------------------------
# 1. Core constants
# ---------------------------------------------------------------------------

def test_hand_count():
    """ALL_HANDS must contain exactly 169 canonical hands."""
    assert len(ALL_HANDS) == 169


def test_combo_weights():
    """COMBO_WEIGHTS must be shape (169,) and sum to 1326."""
    assert COMBO_WEIGHTS.shape == (169,)
    assert COMBO_WEIGHTS.sum() == 1326.0


# ---------------------------------------------------------------------------
# 2. Hand indices and ranks
# ---------------------------------------------------------------------------

def test_hand_indices():
    """Key hands must have the correct canonical indices."""
    assert HAND_MAP["AA"].index == 0
    assert HAND_MAP["KK"].index == 1
    assert HAND_MAP["AKs"].index == 13
    assert HAND_MAP["AKo"].index == 91
    assert HAND_MAP["32o"].index == 168


def test_hand_ranks():
    """AA must be rank 1 (strongest), 72o must be rank 169 (weakest)."""
    assert HAND_MAP["AA"].rank == 1
    assert HAND_MAP["KK"].rank == 2
    assert HAND_MAP["72o"].rank == 169


# ---------------------------------------------------------------------------
# 3. top_n_percent
# ---------------------------------------------------------------------------

def test_top_n_percent_bounds():
    """top_n_percent(0) should return all zeros; top_n_percent(100) all ones."""
    zeros = top_n_percent(0)
    ones = top_n_percent(100)
    assert zeros.shape == (169,)
    assert ones.shape == (169,)
    assert np.all(zeros == 0.0)
    assert np.all(ones == 1.0)


def test_top_n_percent_aa_in_top30():
    """AA must be in the top 30%, 72o must not be."""
    mask = top_n_percent(30)
    assert mask[HAND_MAP["AA"].index] == 1.0
    assert mask[HAND_MAP["72o"].index] == 0.0


# ---------------------------------------------------------------------------
# 4. Grid mapping
# ---------------------------------------------------------------------------

def test_grid_roundtrip():
    """All 169 hands must round-trip through hand_to_grid -> grid_to_hand."""
    for hand in ALL_HANDS:
        row, col = hand_to_grid(hand.name)
        recovered = grid_to_hand(row, col)
        assert recovered == hand.name, (
            f"Round-trip failed for {hand.name}: got {recovered} from ({row},{col})"
        )


def test_grid_spots():
    """Key hands must map to known grid positions."""
    assert hand_to_grid("AA") == (0, 0)
    assert hand_to_grid("AKs") == (0, 1)
    assert hand_to_grid("AKo") == (1, 0)
    assert hand_to_grid("22") == (12, 12)


# ---------------------------------------------------------------------------
# 5. parse_range
# ---------------------------------------------------------------------------

def test_parse_range_pairs():
    """TT+ should expand to TT, JJ, QQ, KK, AA (5 hands)."""
    result = parse_range("TT+")
    assert result == ["TT", "JJ", "QQ", "KK", "AA"]


def test_parse_range_suited():
    """A2s+ should return 12 suited ace hands (A2s through AKs)."""
    result = parse_range("A2s+")
    assert len(result) == 12
    # All should be suited aces
    assert all(h.startswith("A") and h.endswith("s") for h in result)


def test_parse_range_offsuit():
    """KTo+ should expand to KTo, KJo, KQo."""
    result = parse_range("KTo+")
    assert result == ["KTo", "KJo", "KQo"]


def test_parse_range_empty():
    """Empty string should return empty list."""
    assert parse_range("") == []


def test_parse_range_random():
    """'random' should return all 169 hands."""
    result = parse_range("random")
    assert len(result) == 169


def test_parse_range_single():
    """Single hand name should return a one-element list."""
    assert parse_range("AKs") == ["AKs"]


# ---------------------------------------------------------------------------
# 6. range_to_mask / mask_to_hands round-trip
# ---------------------------------------------------------------------------

def test_range_mask_roundtrip():
    """range_to_mask -> mask_to_hands should recover the original hand list for all 169 hands."""
    all_names = [h.name for h in ALL_HANDS]
    mask = range_to_mask(all_names)
    recovered = mask_to_hands(mask)
    assert recovered == all_names


def test_range_mask_single():
    """range_to_mask(['AA']) should have 1.0 only at index 0, sum = 1.0."""
    mask = range_to_mask(["AA"])
    assert mask.shape == (169,)
    assert mask[0] == 1.0
    assert mask[1:].sum() == 0.0
    assert mask.sum() == 1.0


# ---------------------------------------------------------------------------
# 7. hands_to_range_pct
# ---------------------------------------------------------------------------

def test_hands_to_range_pct_aa():
    """AA alone should be ~0.452% of all combos (6/1326*100)."""
    pct = hands_to_range_pct(["AA"])
    expected = 6 / 1326 * 100
    assert abs(pct - expected) < 0.01


def test_hands_to_range_pct_all():
    """All 169 hands should total 100.0%."""
    all_names = [h.name for h in ALL_HANDS]
    pct = hands_to_range_pct(all_names)
    assert abs(pct - 100.0) < 0.001


# ---------------------------------------------------------------------------
# 8. combos_with_removal
# ---------------------------------------------------------------------------

def test_combos_with_removal_suited():
    """AKs with As blocked: AsKs is dead, 3 combos remain."""
    assert combos_with_removal("AKs", ["As"]) == 3


def test_combos_with_removal_pair():
    """AA with As blocked: AsAh, AsAd, AsAc are dead, 3 combos remain."""
    assert combos_with_removal("AA", ["As"]) == 3


def test_combos_with_removal_offsuit():
    """AKo with As blocked: AsKh, AsKd, AsKc dead, 9 combos remain."""
    assert combos_with_removal("AKo", ["As"]) == 9


def test_combos_with_removal_none():
    """AKs with no blocked cards should return 4."""
    assert combos_with_removal("AKs", []) == 4
