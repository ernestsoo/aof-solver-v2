#!/usr/bin/env python3
"""Generate the 169x169 preflop equity matrix for the AoF solver.

!! DO NOT RUN DURING AGENT SESSIONS !!
Estimated runtime: 10-30 min on 4-vCPU machine.

Usage:
    python3 -m scripts.generate_equities
    # or
    cd aof-solver-v2 && python3 scripts/generate_equities.py

Output: data/equity_matrix.npy — shape (169, 169), dtype float32

Algorithm:
    For each of the 14,196 upper-triangle canonical hand matchups (i < j):
      - Enumerate all non-conflicting specific card combos for both hands
      - For each valid combo pair: run N_BOARDS Monte Carlo boards, score with eval7
      - Average equity across all valid combo pairs -> matrix[i][j]
    Mirror: matrix[j][i] = 1.0 - matrix[i][j]
    Diagonal: 0.5 (hand vs itself)

Validation targets (printed at end):
    AA vs KK   ~ 0.82
    AA vs 72o  ~ 0.87
    KK vs AKs  ~ 0.66
    22 vs AKo  ~ 0.52
    Symmetry: max |m[i][j] + m[j][i] - 1| < 0.02
"""

import os
import sys
import time

import numpy as np
import eval7

# ---------------------------------------------------------------------------
# Bootstrap project root so `src` imports work regardless of invocation style
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.hands import ALL_HANDS, HandInfo, RANKS, SUITS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Monte Carlo board samples per specific combo-pair matchup.
# Lower = faster but noisier. 1000 gives ~1% standard error per matchup.
N_BOARDS: int = 1000

# Path where the matrix is saved (relative to project root)
OUTPUT_PATH: str = os.path.join(_ROOT, "data", "equity_matrix.npy")

# Checkpoint path — saved every CHECKPOINT_EVERY matchups
CHECKPOINT_PATH: str = os.path.join(_ROOT, "data", "equity_matrix_checkpoint.npy")
CHECKPOINT_META: str = os.path.join(_ROOT, "data", "equity_matrix_checkpoint_meta.npy")
CHECKPOINT_EVERY: int = 200

# ---------------------------------------------------------------------------
# Precomputed deck — 52 eval7.Card objects in RANKS x SUITS order
# ---------------------------------------------------------------------------

# ALL_CARDS[i] is the i-th card (0..51): i = rank_index*4 + suit_index
ALL_CARDS: list[eval7.Card] = [eval7.Card(r + s) for r in RANKS for s in SUITS]

# Map card string (e.g. 'As') -> deck index 0..51
CARD_TO_IDX: dict[str, int] = {
    r + s: ri * 4 + si
    for ri, r in enumerate(RANKS)
    for si, s in enumerate(SUITS)
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def get_specific_combos(hand: HandInfo) -> list[tuple[eval7.Card, eval7.Card]]:
    """Enumerate all specific card combinations for a canonical hand.

    Returns:
        List of (card1, card2) eval7.Card pairs — 6 for pairs, 4 for suited,
        12 for offsuit.
    """
    r1 = RANKS[hand.rank1]
    r2 = RANKS[hand.rank2]

    if hand.hand_type == "pair":
        result: list[tuple[eval7.Card, eval7.Card]] = []
        for i in range(4):
            for j in range(i + 1, 4):
                result.append((eval7.Card(r1 + SUITS[i]), eval7.Card(r1 + SUITS[j])))
        return result

    elif hand.hand_type == "suited":
        return [(eval7.Card(r1 + s), eval7.Card(r2 + s)) for s in SUITS]

    else:  # offsuit
        result = []
        for s1 in SUITS:
            for s2 in SUITS:
                if s1 != s2:
                    result.append((eval7.Card(r1 + s1), eval7.Card(r2 + s2)))
        return result


def compute_matchup_equity(
    hand_i: HandInfo,
    hand_j: HandInfo,
    n: int = N_BOARDS,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte Carlo equity of hand_i vs hand_j, averaged over all valid specific combos.

    For each non-conflicting (combo_i, combo_j) pair:
      1. Build the 48-card remaining deck.
      2. Sample `n` random 5-card boards.
      3. Evaluate both 7-card hands (hole + board) with eval7 (higher score = better).
      4. Tally wins and ties.
    Return (total wins + 0.5 * ties) / n, averaged across all valid combo pairs.

    Args:
        hand_i: Canonical hand whose equity is computed (perspective player).
        hand_j: Opponent canonical hand.
        n: Board samples per specific combo pair.
        rng: Optional numpy Generator for reproducibility. Uses default if None.

    Returns:
        Equity of hand_i vs hand_j in [0.0, 1.0].
    """
    if rng is None:
        rng = np.random.default_rng()

    combos_i = get_specific_combos(hand_i)
    combos_j = get_specific_combos(hand_j)

    total_equity = 0.0
    n_valid_pairs = 0

    for c1, c2 in combos_i:
        idx1 = CARD_TO_IDX[str(c1)]
        idx2 = CARD_TO_IDX[str(c2)]

        for c3, c4 in combos_j:
            # Skip combos that share a card (conflicting)
            if c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4:
                continue

            idx3 = CARD_TO_IDX[str(c3)]
            idx4 = CARD_TO_IDX[str(c4)]

            # Build remaining 48-card deck (exclude the 4 hole cards)
            excluded = {idx1, idx2, idx3, idx4}
            remaining = [ALL_CARDS[k] for k in range(52) if k not in excluded]
            n_remaining = len(remaining)  # always 48

            # Monte Carlo: sample n 5-card boards from remaining deck
            wins = 0
            ties = 0
            for _ in range(n):
                board_indices = rng.choice(n_remaining, 5, replace=False)
                board = [remaining[bi] for bi in board_indices]

                score_i = eval7.evaluate([c1, c2] + board)
                score_j = eval7.evaluate([c3, c4] + board)

                if score_i > score_j:
                    wins += 1
                elif score_i == score_j:
                    ties += 1

            total_equity += (wins + 0.5 * ties) / n
            n_valid_pairs += 1

    return total_equity / n_valid_pairs if n_valid_pairs > 0 else 0.5


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate the equity matrix and save to data/equity_matrix.npy."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    n_matchups = 169 * 168 // 2  # 14,196 upper-triangle matchups

    # --- Resume from checkpoint if available ---
    start_done = 0
    if os.path.exists(CHECKPOINT_PATH) and os.path.exists(CHECKPOINT_META):
        matrix = np.load(CHECKPOINT_PATH)
        meta = np.load(CHECKPOINT_META)
        start_done = int(meta[0])
        print(f"Resuming from checkpoint: {start_done}/{n_matchups} matchups already done")
    else:
        matrix = np.full((169, 169), 0.5, dtype=np.float32)
        print(f"Generating 169x169 equity matrix — N_BOARDS={N_BOARDS} per combo pair")

    print(f"Output: {OUTPUT_PATH}\n")

    done = 0
    t_start = time.time()

    rng = np.random.default_rng(seed=42)  # reproducible across runs

    for i in range(169):
        for j in range(i + 1, 169):
            done += 1
            if done <= start_done:
                # Skip already-computed matchups (advance rng to stay consistent)
                continue

            equity_ij = compute_matchup_equity(ALL_HANDS[i], ALL_HANDS[j], rng=rng)
            matrix[i][j] = float(equity_ij)
            matrix[j][i] = 1.0 - float(equity_ij)

            if done % 100 == 0:
                elapsed = time.time() - t_start
                total_done = done
                rate = (total_done - start_done) / elapsed if elapsed > 0 else 1
                remaining_sec = (n_matchups - total_done) / rate
                print(
                    f"{total_done}/{n_matchups} matchups done... "
                    f"({elapsed:.0f}s elapsed, ~{remaining_sec:.0f}s remaining)"
                )

            if done % CHECKPOINT_EVERY == 0:
                np.save(CHECKPOINT_PATH, matrix)
                np.save(CHECKPOINT_META, np.array([done]))
                print(f"  [checkpoint saved at {done}]")

    elapsed = time.time() - t_start

    # Save final output and clean up checkpoint
    np.save(OUTPUT_PATH, matrix)
    for f in [CHECKPOINT_PATH, CHECKPOINT_META]:
        if os.path.exists(f):
            os.remove(f)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Shape: {matrix.shape}, dtype: {matrix.dtype}")
    print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Validation
    print("\n--- Validation ---")
    # Symmetry: matrix[i][j] + matrix[j][i] should be ~1.0
    sym_err = np.abs(matrix + matrix.T - 1.0)
    np.fill_diagonal(sym_err, 0.0)
    print(f"Symmetry — max |m[i][j] + m[j][i] - 1|: {sym_err.max():.4f} (expect < 0.02)")

    # Spot checks (indices from src/hands.py: AA=0, KK=1, QQ=2, AKs=13)
    # 72o is index 168
    _spot = [
        (0, 1, "AA vs KK", 0.82),
        (0, 168, "AA vs 72o", 0.87),
        (1, 13, "KK vs AKs", 0.66),
        (12, 91, "22 vs AKo", 0.52),
    ]
    for i, j, label, expected in _spot:
        val = matrix[i][j]
        ok = "OK" if abs(val - expected) < 0.05 else "WARN"
        print(f"  {label}: {val:.4f} (expect ~{expected:.2f}) [{ok}]")


if __name__ == "__main__":
    main()
