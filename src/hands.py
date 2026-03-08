"""Hand representations, rankings, and range utilities for AoF solver."""

from dataclasses import dataclass

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
