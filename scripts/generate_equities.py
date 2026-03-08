#!/usr/bin/env python3
"""Generate the 169x169 preflop equity matrix for the AoF solver.

!! DO NOT RUN ON VPS DURING AGENT SESSIONS !!
Run locally on your machine for best results.

Estimated runtime (12-worker machine, N=5000):
  ~20-30 minutes

Usage:
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
import multiprocessing as mp

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
# 5000 gives ~0.4% standard error per matchup (high accuracy).
N_BOARDS: int = 5000

# Number of parallel worker processes (set to your thread count)
N_WORKERS: int = 12

# Path where the matrix is saved (relative to project root)
OUTPUT_PATH: str = os.path.join(_ROOT, "data", "equity_matrix.npy")

# Checkpoint — saved every CHECKPOINT_EVERY matchups
CHECKPOINT_PATH: str = os.path.join(_ROOT, "data", "equity_matrix_checkpoint.npy")
CHECKPOINT_META: str = os.path.join(_ROOT, "data", "equity_matrix_checkpoint_meta.npy")
CHECKPOINT_EVERY: int = 500

# ---------------------------------------------------------------------------
# Precomputed deck — 52 eval7.Card objects in RANKS x SUITS order
# ---------------------------------------------------------------------------

ALL_CARDS: list[eval7.Card] = [eval7.Card(r + s) for r in RANKS for s in SUITS]

CARD_TO_IDX: dict[str, int] = {
    r + s: ri * 4 + si
    for ri, r in enumerate(RANKS)
    for si, s in enumerate(SUITS)
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def get_specific_combos(hand: HandInfo) -> list[tuple[eval7.Card, eval7.Card]]:
    """Enumerate all specific card combinations for a canonical hand."""
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
    """Monte Carlo equity of hand_i vs hand_j, averaged over all valid specific combos."""
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
            if c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4:
                continue

            idx3 = CARD_TO_IDX[str(c3)]
            idx4 = CARD_TO_IDX[str(c4)]

            excluded = {idx1, idx2, idx3, idx4}
            remaining = [ALL_CARDS[k] for k in range(52) if k not in excluded]
            n_remaining = len(remaining)

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
# Worker function (module-level for multiprocessing pickle)
# ---------------------------------------------------------------------------

def _worker(args: tuple[int, int]) -> tuple[int, int, float]:
    """Compute one matchup equity. Called by Pool workers."""
    i, j = args
    rng = np.random.default_rng()  # each worker gets independent rng
    eq = compute_matchup_equity(ALL_HANDS[i], ALL_HANDS[j], n=N_BOARDS, rng=rng)
    return i, j, eq


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate the equity matrix using parallel workers."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    n_matchups = 169 * 168 // 2  # 14,196

    # Build full list of (i, j) pairs
    all_pairs = [(i, j) for i in range(169) for j in range(i + 1, 169)]

    # --- Resume from checkpoint if available ---
    start_done = 0
    if os.path.exists(CHECKPOINT_PATH) and os.path.exists(CHECKPOINT_META):
        matrix = np.load(CHECKPOINT_PATH)
        meta = np.load(CHECKPOINT_META)
        start_done = int(meta[0])
        remaining_pairs = all_pairs[start_done:]
        print(f"Resuming from checkpoint: {start_done}/{n_matchups} done, {len(remaining_pairs)} remaining")
    else:
        matrix = np.full((169, 169), 0.5, dtype=np.float32)
        remaining_pairs = all_pairs
        print(f"Generating 169x169 equity matrix — N_BOARDS={N_BOARDS}, N_WORKERS={N_WORKERS}")

    print(f"Output: {OUTPUT_PATH}\n")
    print(f"Starting {N_WORKERS} worker processes...\n")

    done = start_done
    t_start = time.time()

    with mp.Pool(N_WORKERS) as pool:
        for i, j, eq in pool.imap_unordered(_worker, remaining_pairs, chunksize=10):
            matrix[i][j] = float(eq)
            matrix[j][i] = 1.0 - float(eq)
            done += 1

            if done % 100 == 0:
                elapsed = time.time() - t_start
                rate = (done - start_done) / elapsed if elapsed > 0 else 1
                remaining_sec = (n_matchups - done) / rate
                print(
                    f"{done}/{n_matchups} matchups done "
                    f"({done/n_matchups*100:.1f}%) | "
                    f"elapsed {elapsed:.0f}s | ETA ~{remaining_sec/60:.1f}min"
                )

            if done % CHECKPOINT_EVERY == 0:
                np.save(CHECKPOINT_PATH, matrix)
                np.save(CHECKPOINT_META, np.array([done]))
                print(f"  [checkpoint saved at {done}]")

    elapsed = time.time() - t_start

    # Save final and clean up checkpoint
    np.save(OUTPUT_PATH, matrix)
    for f in [CHECKPOINT_PATH, CHECKPOINT_META]:
        if os.path.exists(f):
            os.remove(f)

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Shape: {matrix.shape}, dtype: {matrix.dtype}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Validation
    print("\n--- Validation ---")
    sym_err = np.abs(matrix + matrix.T - 1.0)
    np.fill_diagonal(sym_err, 0.0)
    print(f"Symmetry — max |m[i][j] + m[j][i] - 1|: {sym_err.max():.4f} (expect < 0.01)")

    for i, j, label, expected in [
        (0, 1, "AA vs KK", 0.82),
        (0, 168, "AA vs 72o", 0.87),
        (1, 13, "KK vs AKs", 0.66),
        (12, 91, "22 vs AKo", 0.52),
    ]:
        val = matrix[i][j]
        ok = "OK" if abs(val - expected) < 0.03 else "WARN"
        print(f"  {label}: {val:.4f} (expect ~{expected:.2f}) [{ok}]")


if __name__ == "__main__":
    mp.freeze_support()  # required for macOS/Windows spawn method
    main()
