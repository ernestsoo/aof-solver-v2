"""Hand representations, rankings, and range utilities for AoF solver."""

from dataclasses import dataclass

import numpy as np

# Rank characters in descending order (A=highest, 2=lowest)
RANKS: list[str] = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']

# Map rank character to integer index (A=0, K=1, ..., 2=12)
RANK_INDEX: dict[str, int] = {r: i for i, r in enumerate(RANKS)}


@dataclass
class HandInfo:
    """Canonical 2-card hand (one of 169 unique starting hands)."""

    name: str          # e.g. "AKs", "TT", "87o"
    index: int         # 0-168
    hand_type: str     # "pair", "suited", "offsuit"
    combos: int        # number of suit combinations: 6 (pair), 4 (suited), 12 (offsuit)
    rank1: int         # higher card rank index (0=A, 12=2)
    rank2: int         # lower card rank index (0=A, 12=2)


def _generate_hands() -> list[HandInfo]:
    """Generate all 169 canonical hands in canonical order.

    Order: 13 pairs (AA..22), 78 suited (AKs..32s), 78 offsuit (AKo..32o).
    Within suited/offsuit: higher rank1 first, then higher rank2.
    """
    hands: list[HandInfo] = []
    idx = 0

    # Pairs: AA (r=0) -> 22 (r=12)
    for r in range(13):
        hands.append(HandInfo(
            name=RANKS[r] + RANKS[r],
            index=idx,
            hand_type="pair",
            combos=6,
            rank1=r,
            rank2=r,
        ))
        idx += 1

    # Suited: rank1 from A(0) to 3(11), rank2 from rank1+1 to 2(12)
    for r1 in range(13):
        for r2 in range(r1 + 1, 13):
            hands.append(HandInfo(
                name=RANKS[r1] + RANKS[r2] + "s",
                index=idx,
                hand_type="suited",
                combos=4,
                rank1=r1,
                rank2=r2,
            ))
            idx += 1

    # Offsuit: same traversal order as suited
    for r1 in range(13):
        for r2 in range(r1 + 1, 13):
            hands.append(HandInfo(
                name=RANKS[r1] + RANKS[r2] + "o",
                index=idx,
                hand_type="offsuit",
                combos=12,
                rank1=r1,
                rank2=r2,
            ))
            idx += 1

    return hands


# All 169 canonical hands, ordered by index 0-168
ALL_HANDS: list[HandInfo] = _generate_hands()

# Name -> HandInfo lookup
HAND_MAP: dict[str, HandInfo] = {h.name: h for h in ALL_HANDS}

# Combo weights: shape (169,), values 6/4/12 depending on hand type; sums to 1326
COMBO_WEIGHTS: np.ndarray = np.array([h.combos for h in ALL_HANDS], dtype=np.float64)

assert len(ALL_HANDS) == 169, f"Expected 169 hands, got {len(ALL_HANDS)}"
assert COMBO_WEIGHTS.sum() == 1326, f"Expected combo sum 1326, got {COMBO_WEIGHTS.sum()}"
