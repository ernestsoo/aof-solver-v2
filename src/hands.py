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
    rank: int = 0      # preflop strength rank: 1=AA (strongest), 169=72o (weakest)


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


# Standard 169-hand preflop strength ranking (rank 1 = AA = strongest, rank 169 = 72o = weakest).
# Based on equity vs random hand. Suited hands ranked above offsuit of same ranks.
# Wheel potential raises A5s/A4s/A3s/A2s slightly above A6s in many references.
HAND_RANK_ORDER: list[str] = [
    # 1-10: Premium pairs and top suited broadways
    "AA", "KK", "QQ", "AKs", "JJ", "AQs", "KQs", "AJs", "KJs", "TT",
    # 11-20: Top offsuit broadways, suited broadway connectors
    "AKo", "ATs", "QJs", "KTs", "QTs", "JTs", "99", "AQo", "A9s", "KQo",
    # 21-30: High pairs, top offsuit aces, suited broadway gaps
    "88", "AJo", "K9s", "T9s", "A8s", "KJo", "QJo", "J9s", "A7s", "ATo",
    # 31-40: Mid suited aces, 77, suited queens/kings, broadway offsuit
    "A6s", "KTo", "A5s", "77", "Q9s", "A4s", "T8s", "K8s", "A3s", "JTo",
    # 41-50: Low suited aces, 66, suited connectors, offsuit aces
    "A2s", "98s", "66", "T9o", "K7s", "Q8s", "J8s", "A9o", "T7s", "K6s",
    # 51-60: Suited connectors, 55, offsuit broadway gaps, suited kings
    "87s", "QTo", "55", "86s", "J9o", "A8o", "Q9o", "J7s", "K5s", "76s",
    # 61-70: Offsuit top pairs, suited one-gappers, 44
    "T8o", "97s", "A7o", "K4s", "T6s", "44", "96s", "A6o", "75s", "J8o",
    # 71-80: Offsuit queens/jacks, suited mid connectors, 33
    "Q8o", "K3s", "85s", "A5o", "J6s", "98o", "K2s", "74s", "A4o", "33",
    # 81-90: Suited two-gappers, offsuit mid connectors, low pairs
    "65s", "84s", "Q7s", "J5s", "A3o", "K9o", "87o", "95s", "J4s", "Q6s",
    # 91-100: 22, offsuit broadway gaps, low suited connectors
    "A2o", "64s", "22", "Q5s", "J3s", "T7o", "Q4s", "86o", "73s", "J2s",
    # 101-110: Low suited hands, offsuit connectors
    "T5s", "K8o", "Q3s", "97o", "76o", "83s", "T4s", "Q2s", "63s", "K7o",
    # 111-120: Weak suited, offsuit two-gappers, low offsuit kings
    "T3s", "75o", "96o", "93s", "K6o", "T2s", "85o", "52s", "65o", "K5o",
    # 121-130: Weak suited connectors, offsuit gaps
    "53s", "95o", "94s", "J7o", "74o", "64o", "K4o", "43s", "84o", "62s",
    # 131-140: Weak offsuit, low suited trash
    "T6o", "J6o", "K3o", "54s", "73o", "93o", "42s", "K2o", "J5o", "83o",
    # 141-150: Borderline suited, weak offsuit connectors
    "32s", "52o", "T5o", "J4o", "63o", "Q7o", "T4o", "J3o", "92s", "53o",
    # 151-160: Weak offsuit, near-unplayable
    "Q6o", "J2o", "43o", "T3o", "Q5o", "T2o", "42o", "Q4o", "54o", "32o",
    # 161-169: Trash hands, 72o dead last
    "Q3o", "82s", "Q2o", "72s", "92o", "82o", "62o", "94o", "72o",
]

# All 169 canonical hands, ordered by index 0-168
ALL_HANDS: list[HandInfo] = _generate_hands()

# Name -> HandInfo lookup
HAND_MAP: dict[str, HandInfo] = {h.name: h for h in ALL_HANDS}

# Assign preflop strength ranks from HAND_RANK_ORDER (rank 1 = strongest)
for _rank, _name in enumerate(HAND_RANK_ORDER, start=1):
    HAND_MAP[_name].rank = _rank

# Combo weights: shape (169,), values 6/4/12 depending on hand type; sums to 1326
COMBO_WEIGHTS: np.ndarray = np.array([h.combos for h in ALL_HANDS], dtype=np.float64)

assert len(ALL_HANDS) == 169, f"Expected 169 hands, got {len(ALL_HANDS)}"
assert COMBO_WEIGHTS.sum() == 1326, f"Expected combo sum 1326, got {COMBO_WEIGHTS.sum()}"
assert len(HAND_RANK_ORDER) == 169, f"HAND_RANK_ORDER has {len(HAND_RANK_ORDER)} entries, expected 169"
assert len(set(HAND_RANK_ORDER)) == 169, "HAND_RANK_ORDER contains duplicate hand names"
assert HAND_MAP["AA"].rank == 1, "AA must have rank 1"
assert HAND_MAP["72o"].rank == 169, "72o must have rank 169"


def parse_range(notation: str) -> list[str]:
    """Parse poker range notation into a list of hand names.

    Supports:
    - Empty string -> []
    - "random" -> all 169 hands in canonical order
    - Single hand: "AKs" -> ["AKs"]
    - Pair plus: "TT+" -> ["TT", "JJ", "QQ", "KK", "AA"]
    - Suited plus: "A2s+" -> ["A2s", "A3s", ..., "AKs"] (12 hands)
    - Offsuit plus: "KTo+" -> ["KTo", "KJo", "KQo"]
    - Comma-separated: "TT+, AKs" combines all tokens

    Args:
        notation: Range string, e.g. "22+, A2s+, KTo+".

    Returns:
        List of hand name strings, deduplicated, preserving encounter order.
    """
    notation = notation.strip()
    if not notation:
        return []
    if notation.lower() == "random":
        return [h.name for h in ALL_HANDS]

    result: list[str] = []
    seen: set[str] = set()

    for token in notation.split(","):
        token = token.strip()
        if not token:
            continue

        if token.endswith("+"):
            base = token[:-1]
            if len(base) == 2 and base[0] == base[1]:
                # Pair plus: "TT+" -> TT, JJ, QQ, KK, AA
                r = RANK_INDEX[base[0]]
                for i in range(r, -1, -1):
                    name = RANKS[i] + RANKS[i]
                    if name not in seen:
                        result.append(name)
                        seen.add(name)
            elif base.endswith("s"):
                # Suited plus: "A2s+" -> A2s .. AKs
                r1 = RANK_INDEX[base[0]]
                r2_start = RANK_INDEX[base[1]]
                for r2 in range(r2_start, r1, -1):
                    name = RANKS[r1] + RANKS[r2] + "s"
                    if name not in seen:
                        result.append(name)
                        seen.add(name)
            elif base.endswith("o"):
                # Offsuit plus: "KTo+" -> KTo, KJo, KQo
                r1 = RANK_INDEX[base[0]]
                r2_start = RANK_INDEX[base[1]]
                for r2 in range(r2_start, r1, -1):
                    name = RANKS[r1] + RANKS[r2] + "o"
                    if name not in seen:
                        result.append(name)
                        seen.add(name)
        else:
            if token not in seen:
                result.append(token)
                seen.add(token)

    return result


def hand_to_grid(name: str) -> tuple[int, int]:
    """Map a hand name to its (row, col) position in the 13x13 grid.

    Grid layout (row=0..12 maps A..2, same for col):
    - Pairs on the diagonal: row == col == rank1
    - Suited above the diagonal: row=rank1, col=rank2  (rank1 < rank2)
    - Offsuit below the diagonal: row=rank2, col=rank1  (row > col)

    Args:
        name: Hand name, e.g. "AKs", "TT", "87o".

    Returns:
        (row, col) tuple, both in range 0-12.
    """
    hand = HAND_MAP[name]
    if hand.hand_type == "suited":
        return (hand.rank1, hand.rank2)
    elif hand.hand_type == "offsuit":
        return (hand.rank2, hand.rank1)
    else:  # pair
        return (hand.rank1, hand.rank1)


def grid_to_hand(row: int, col: int) -> str:
    """Map a 13x13 grid position to a hand name.

    Inverse of hand_to_grid:
    - row == col -> pair (e.g. (0,0) -> "AA")
    - row < col  -> suited (e.g. (0,1) -> "AKs")
    - row > col  -> offsuit (e.g. (1,0) -> "AKo")

    Args:
        row: Row index 0-12 (0=A, 12=2).
        col: Column index 0-12 (0=A, 12=2).

    Returns:
        Hand name string, e.g. "AKs", "TT", "87o".
    """
    if row == col:
        return RANKS[row] + RANKS[col]
    elif row < col:
        return RANKS[row] + RANKS[col] + "s"
    else:
        return RANKS[col] + RANKS[row] + "o"


SUITS: list[str] = ['s', 'h', 'd', 'c']


def range_to_mask(hands: list[str]) -> np.ndarray:
    """Convert a list of hand names to a (169,) binary mask array.

    Args:
        hands: List of hand name strings, e.g. ["AA", "AKs", "AKo"].

    Returns:
        np.ndarray of shape (169,), dtype float64: 1.0 for each included hand,
        0.0 elsewhere.
    """
    mask = np.zeros(169, dtype=np.float64)
    for name in hands:
        mask[HAND_MAP[name].index] = 1.0
    return mask


def mask_to_hands(mask: np.ndarray) -> list[str]:
    """Convert a (169,) binary mask to a list of hand names.

    Inverse of range_to_mask. Returns hands in canonical index order.

    Args:
        mask: np.ndarray of shape (169,) with values 0.0 or 1.0.

    Returns:
        List of hand name strings for all positions where mask == 1.0.
    """
    return [ALL_HANDS[i].name for i in range(169) if mask[i] == 1.0]


def hands_to_range_pct(hands: list[str]) -> float:
    """Return the combo percentage of a hand list relative to all 1326 combos.

    Args:
        hands: List of hand name strings.

    Returns:
        Percentage (0.0 to 100.0). E.g. ["AA"] -> 6/1326*100 ~= 0.452.
    """
    total = sum(HAND_MAP[name].combos for name in hands)
    return total / 1326.0 * 100.0


def combos_with_removal(hand: str, blocked_cards: list[str]) -> int:
    """Return the number of combos for a hand after removing blocked specific cards.

    Enumerates all specific card combos for the hand and counts those where
    neither card appears in the blocked set. Cards are specified with suit,
    e.g. "As" (Ace of spades), "Kh" (King of hearts).

    Args:
        hand: Canonical hand name, e.g. "AKs", "AA", "AKo".
        blocked_cards: List of specific cards to treat as dead, e.g. ["As", "Kh"].

    Returns:
        Number of remaining (unblocked) combos, 0 to hand.combos.

    Examples:
        combos_with_removal("AA", ["As"]) == 3   # AsAh, AsAd, AsAc dead
        combos_with_removal("AKs", ["As"]) == 3  # AsKs dead
        combos_with_removal("AKo", ["As"]) == 9  # AsKh, AsKd, AsKc dead
    """
    blocked = set(blocked_cards)
    info = HAND_MAP[hand]
    r1 = RANKS[info.rank1]
    r2 = RANKS[info.rank2]

    count = 0
    if info.hand_type == "pair":
        # C(4,2) = 6 combos: choose 2 distinct suits for the same rank
        for i, s1 in enumerate(SUITS):
            for s2 in SUITS[i + 1:]:
                if (r1 + s1) not in blocked and (r1 + s2) not in blocked:
                    count += 1
    elif info.hand_type == "suited":
        # 4 combos: same suit for both cards
        for s in SUITS:
            if (r1 + s) not in blocked and (r2 + s) not in blocked:
                count += 1
    else:  # offsuit
        # 4*4 - 4 = 12 combos: different suits
        for s1 in SUITS:
            for s2 in SUITS:
                if s1 != s2 and (r1 + s1) not in blocked and (r2 + s2) not in blocked:
                    count += 1
    return count


def top_n_percent(pct: float) -> np.ndarray:
    """Return (169,) mask: 1.0 for hands in the top pct% by combo-weighted strength.

    Hands are ordered by preflop strength rank (rank 1=AA=strongest).
    A hand is included if the cumulative combo count of all stronger hands
    is strictly less than pct% of 1326 total combos.

    Args:
        pct: Percentage threshold, 0.0 to 100.0.
             0.0 -> all zeros (no hands included).
             100.0 -> all ones (all hands included).

    Returns:
        np.ndarray of shape (169,), dtype float64, values 0.0 or 1.0.
    """
    threshold = pct / 100.0
    mask = np.zeros(169, dtype=np.float64)
    cumulative = 0.0
    for hand in sorted(ALL_HANDS, key=lambda h: h.rank):
        if cumulative / 1326.0 < threshold:
            mask[hand.index] = 1.0
        cumulative += hand.combos
    return mask
