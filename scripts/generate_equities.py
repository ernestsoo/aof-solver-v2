#!/usr/bin/env python3
"""Generate the 169x169 preflop equity matrix for the AoF solver.

!! DO NOT RUN ON VPS DURING AGENT SESSIONS !!
Run locally on your machine for best results.

Estimated runtime (12-worker machine, N=5000):
  ~3-5 minutes with phevaluator

Usage:
    cd aof-solver-v2 && python3 scripts/generate_equities.py

Requires:
    pip install phevaluator numpy

Output: data/equity_matrix.npy — shape (169, 169), dtype float32
"""

import os
import sys
import time
import multiprocessing as mp

import numpy as np
from phevaluator.evaluator import Card, _evaluate_cards  # integer API avoids string parsing overhead

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.hands import ALL_HANDS, HandInfo, RANKS, SUITS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BOARDS: int = 5000
N_WORKERS: int = 12

OUTPUT_PATH: str = os.path.join(_ROOT, "data", "equity_matrix.npy")
CHECKPOINT_PATH: str = os.path.join(_ROOT, "data", "equity_matrix_checkpoint.npy")
CHECKPOINT_META: str = os.path.join(_ROOT, "data", "equity_matrix_checkpoint_meta.npy")
CHECKPOINT_EVERY: int = 500

# ---------------------------------------------------------------------------
# Deck — plain strings (e.g. 'As', 'Kh') — phevaluator native format
# ---------------------------------------------------------------------------

ALL_CARDS: list[str] = [r + s for r in RANKS for s in SUITS]

CARD_TO_IDX: dict[str, int] = {
    r + s: ri * 4 + si
    for ri, r in enumerate(RANKS)
    for si, s in enumerate(SUITS)
}

# Integer card IDs for phevaluator's internal _evaluate_cards(*ints) API.
# Using integer IDs instead of string-parsed cards avoids repeated string
# parsing inside the hot Monte Carlo loop — gives ~1.5x speedup at N=5000.
CARD_TO_ID: dict[str, int] = {c: Card(c).id_ for c in ALL_CARDS}
ALL_IDS: list[int] = [CARD_TO_ID[c] for c in ALL_CARDS]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def get_specific_combos(hand: HandInfo) -> list[tuple[str, str]]:
    """Return all specific (card1, card2) string pairs for a canonical hand."""
    r1 = RANKS[hand.rank1]
    r2 = RANKS[hand.rank2]

    if hand.hand_type == "pair":
        result: list[tuple[str, str]] = []
        for i in range(4):
            for j in range(i + 1, 4):
                result.append((r1 + SUITS[i], r1 + SUITS[j]))
        return result
    elif hand.hand_type == "suited":
        return [(r1 + s, r2 + s) for s in SUITS]
    else:  # offsuit
        result = []
        for s1 in SUITS:
            for s2 in SUITS:
                if s1 != s2:
                    result.append((r1 + s1, r2 + s2))
        return result


def compute_matchup_equity(
    hand_i: HandInfo,
    hand_j: HandInfo,
    n: int = N_BOARDS,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte Carlo equity of hand_i vs hand_j. phevaluator: lower score = better."""
    if rng is None:
        rng = np.random.default_rng()

    combos_i = get_specific_combos(hand_i)
    combos_j = get_specific_combos(hand_j)

    total_equity = 0.0
    n_valid_pairs = 0

    for c1, c2 in combos_i:
        idx1 = CARD_TO_IDX[c1]
        idx2 = CARD_TO_IDX[c2]

        for c3, c4 in combos_j:
            if c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4:
                continue

            idx3 = CARD_TO_IDX[c3]
            idx4 = CARD_TO_IDX[c4]

            excluded = {idx1, idx2, idx3, idx4}
            rem_arr = np.array([ALL_IDS[k] for k in range(52) if k not in excluded])
            n_remaining = len(rem_arr)

            # Integer IDs for hole cards — avoids string parsing in hot loop
            id1, id2 = CARD_TO_ID[c1], CARD_TO_ID[c2]
            id3, id4 = CARD_TO_ID[c3], CARD_TO_ID[c4]

            # Sample all n boards at once; index into rem_arr
            all_bidx = np.stack(
                [rng.choice(n_remaining, 5, replace=False) for _ in range(n)]
            )
            boards_arr = rem_arr[all_bidx]  # (n, 5)

            # Evaluate all boards with integer API — ~1.5x faster than string API
            sc_i = np.array([_evaluate_cards(id1, id2, *row) for row in boards_arr])
            sc_j = np.array([_evaluate_cards(id3, id4, *row) for row in boards_arr])

            # lower score = better hand in phevaluator
            total_equity += (np.sum(sc_i < sc_j) + 0.5 * np.sum(sc_i == sc_j)) / n
            n_valid_pairs += 1

    return total_equity / n_valid_pairs if n_valid_pairs > 0 else 0.5


# ---------------------------------------------------------------------------
# Worker (module-level for multiprocessing pickle)
# ---------------------------------------------------------------------------

def _worker(args: tuple[int, int]) -> tuple[int, int, float]:
    i, j = args
    rng = np.random.default_rng()
    eq = compute_matchup_equity(ALL_HANDS[i], ALL_HANDS[j], n=N_BOARDS, rng=rng)
    return i, j, eq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    n_matchups = 169 * 168 // 2  # 14,196
    all_pairs = [(i, j) for i in range(169) for j in range(i + 1, 169)]

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
        print("Using phevaluator (~60M hands/sec)")

    print(f"Output: {OUTPUT_PATH}\n")
    print(f"Starting {N_WORKERS} worker processes...\n")

    done = start_done
    t_start = time.time()

    with mp.Pool(N_WORKERS) as pool:
        for i, j, eq in pool.imap_unordered(_worker, remaining_pairs, chunksize=50):
            matrix[i][j] = float(eq)
            matrix[j][i] = 1.0 - float(eq)
            done += 1

            if done % 100 == 0:
                elapsed = time.time() - t_start
                rate = (done - start_done) / elapsed if elapsed > 0 else 1
                remaining_sec = (n_matchups - done) / rate
                print(
                    f"{done}/{n_matchups} ({done/n_matchups*100:.1f}%) | "
                    f"elapsed {elapsed:.0f}s | ETA ~{remaining_sec/60:.1f}min"
                )

            if done % CHECKPOINT_EVERY == 0:
                np.save(CHECKPOINT_PATH, matrix)
                np.save(CHECKPOINT_META, np.array([done]))
                print(f"  [checkpoint at {done}]")

    elapsed = time.time() - t_start
    np.save(OUTPUT_PATH, matrix)
    for f in [CHECKPOINT_PATH, CHECKPOINT_META]:
        if os.path.exists(f):
            os.remove(f)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Validation
    print("\n--- Validation ---")
    sym_err = np.abs(matrix + matrix.T - 1.0)
    np.fill_diagonal(sym_err, 0.0)
    print(f"Symmetry max error: {sym_err.max():.4f} (expect < 0.01)")
    for i, j, label, expected in [
        (0, 1,   "AA vs KK",  0.82),
        (0, 168, "AA vs 72o", 0.87),
        (1, 13,  "KK vs AKs", 0.66),
        (12, 91, "22 vs AKo", 0.52),
    ]:
        val = matrix[i][j]
        ok = "OK" if abs(val - expected) < 0.03 else "WARN"
        print(f"  {label}: {val:.4f} (expect ~{expected:.2f}) [{ok}]")


if __name__ == "__main__":
    mp.freeze_support()
    main()
