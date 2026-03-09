#!/usr/bin/env python3
"""Generate the 169x169x169 3-way equity tensor for the AoF solver.

!! DO NOT RUN ON VPS — Run locally on your machine !!
Estimated runtime: 2-4 hours on 6-core/12-thread machine
After completion: scp data/equity_3way.npy user@vps:/path/to/aof-solver-v2/data/

Usage:
    python3 -m scripts.generate_3way_equities
    # or
    cd aof-solver-v2 && python3 scripts/generate_3way_equities.py

Output: data/equity_3way.npy — shape (169, 169, 169), dtype float32

Tensor definition:
    matrix[i, j, k] = equity of hand i in a 3-way pot vs hand j and hand k

Only the upper triangle (i < j < k) is computed directly via Monte Carlo.
All 6 permutations of each triplet are filled from those 3 equity values.
Degenerate entries (any two indices equal) are set to 0.5 by convention.

Symmetry property:
    matrix[h, op1, op2] == matrix[h, op2, op1]  (opponent order irrelevant)
    matrix[i,j,k] + matrix[j,i,k] + matrix[k,i,j] ≈ 1.0

Algorithm per triplet (i < j < k):
    - Enumerate all valid specific card combo triples (no card conflicts)
    - For each valid triple: deal N_BOARDS random 5-card boards from 46 remaining cards
    - Evaluate all three 7-card hands with eval7; credit wins/splits
    - Average equity_i, equity_j, equity_k across all valid combo triples

Checkpoint/resume:
    - Checkpoint saved every CHECKPOINT_INTERVAL triplets to data/equity_3way_checkpoint.npy
    - On startup, loads checkpoint (if present) and skips already-computed triplets
    - Resume by simply re-running the script
"""

import multiprocessing as mp
import os
import sys
import time
from itertools import combinations

import numpy as np
from phevaluator.evaluator import Card, _evaluate_cards  # integer API avoids string parsing overhead

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

# Monte Carlo board samples per valid specific combo triple.
# 500 gives acceptable accuracy (~1-2% error) with reasonable speed.
N_BOARDS: int = 500

# Number of parallel worker processes — set to your machine's thread count.
N_WORKERS: int = 6  # 6 is more stable on Windows spawn; bump to 12 on Linux/Mac

# Save a checkpoint every this many triplets so the run can be resumed.
CHECKPOINT_INTERVAL: int = 10_000

# Total number of unique triplets (i < j < k): C(169, 3) = 786,786
N_TRIPLETS: int = 169 * 168 * 167 // 6  # = 786,786

# Output paths (relative to project root)
OUTPUT_PATH: str = os.path.join(_ROOT, "data", "equity_3way.npy")
CHECKPOINT_PATH: str = os.path.join(_ROOT, "data", "equity_3way_checkpoint.npy")
# Metadata file stores [n_done, version] — version=2 means phevaluator run.
# Old eval7 checkpoints have no meta file; they are silently discarded on load.
CHECKPOINT_META: str = os.path.join(_ROOT, "data", "equity_3way_checkpoint_meta.npy")

# ---------------------------------------------------------------------------
# Precomputed deck — 52 card strings in RANKS x SUITS order (e.g. 'As', 'Kh')
# (Module-level so workers inherit it via fork without re-building.)
# ---------------------------------------------------------------------------

ALL_CARDS: list[str] = [r + s for r in RANKS for s in SUITS]

CARD_TO_IDX: dict[str, int] = {
    r + s: ri * 4 + si
    for ri, r in enumerate(RANKS)
    for si, s in enumerate(SUITS)
}

# Integer card IDs for phevaluator's internal _evaluate_cards(*ints) API.
# Using integer IDs instead of string-parsed cards avoids repeated string
# parsing inside the hot Monte Carlo loop — gives ~2x speedup.
CARD_TO_ID: dict[str, int] = {c: Card(c).id_ for c in ALL_CARDS}
ALL_IDS: list[int] = [CARD_TO_ID[c] for c in ALL_CARDS]


# ---------------------------------------------------------------------------
# Core helpers (module-level so they can be pickled by multiprocessing)
# ---------------------------------------------------------------------------

def get_specific_combos(hand: HandInfo) -> list[tuple[str, str]]:
    """Enumerate all specific card combinations for a canonical hand.

    Returns:
        List of (card1, card2) string pairs — 6 for pairs, 4 for suited,
        12 for offsuit.
    """
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


def compute_triplet_equity(
    args: tuple[int, int, int, int, int],
) -> tuple[int, int, int, float, float, float]:
    """Compute 3-way equity for canonical hand triplet (i, j, k) where i < j < k.

    Worker function — runs in a subprocess.

    Algorithm:
        For each valid specific combo triple (no shared cards across all 3 hands):
          Deal N_BOARDS random 5-card boards from the 46 remaining cards.
          Evaluate all three 7-card hands with phevaluator (lower score = better).
          Credit each player their equity share (1.0 on win, split on tie).
        Average equity_i, equity_j, equity_k across all valid combo triples.

    Args:
        args: Tuple of (i, j, k, n_boards, seed).
            i, j, k: Hand indices into ALL_HANDS (0-168), with i < j < k.
            n_boards: Number of random boards per valid combo triple.
            seed:     RNG seed for reproducibility.

    Returns:
        Tuple (i, j, k, equity_i, equity_j, equity_k).
        equity_x is in [0.0, 1.0]; the three values sum to approximately 1.0.
        Returns (i, j, k, 1/3, 1/3, 1/3) if no valid combo triple exists.
    """
    i, j, k, n_boards, seed = args
    rng = np.random.default_rng(seed)

    combos_i = get_specific_combos(ALL_HANDS[i])
    combos_j = get_specific_combos(ALL_HANDS[j])
    combos_k = get_specific_combos(ALL_HANDS[k])

    sum_eq_i = 0.0
    sum_eq_j = 0.0
    sum_eq_k = 0.0
    n_valid = 0

    for ci in combos_i:
        idx_ci = {CARD_TO_IDX[ci[0]], CARD_TO_IDX[ci[1]]}

        for cj in combos_j:
            idx_cj = {CARD_TO_IDX[cj[0]], CARD_TO_IDX[cj[1]]}
            # Skip if hand i and hand j share a card
            if idx_ci & idx_cj:
                continue

            for ck in combos_k:
                idx_ck = {CARD_TO_IDX[ck[0]], CARD_TO_IDX[ck[1]]}
                # Skip if hand k shares a card with hand i or hand j
                if (idx_ci | idx_cj) & idx_ck:
                    continue

                # Build the 46-card remaining deck (exclude 6 hole cards)
                excluded = idx_ci | idx_cj | idx_ck
                rem_arr = np.array([ALL_IDS[x] for x in range(52) if x not in excluded])
                n_rem = len(rem_arr)  # always 46

                # Integer IDs for hole cards — avoids string parsing in hot loop
                ci0, ci1 = CARD_TO_ID[ci[0]], CARD_TO_ID[ci[1]]
                cj0, cj1 = CARD_TO_ID[cj[0]], CARD_TO_ID[cj[1]]
                ck0, ck1 = CARD_TO_ID[ck[0]], CARD_TO_ID[ck[1]]

                # Sample all n_boards boards at once; index into rem_arr
                boards_arr = np.array(
                    [rem_arr[rng.choice(n_rem, 5, replace=False)] for _ in range(n_boards)]
                )  # (n_boards, 5)

                # Evaluate all boards with integer API — ~2x faster than string API
                sc_i = np.array([_evaluate_cards(ci0, ci1, *row) for row in boards_arr])
                sc_j = np.array([_evaluate_cards(cj0, cj1, *row) for row in boards_arr])
                sc_k = np.array([_evaluate_cards(ck0, ck1, *row) for row in boards_arr])

                # Vectorised win/split counting
                best = np.minimum(np.minimum(sc_i, sc_j), sc_k)
                n_best = (
                    (sc_i == best).astype(np.float64)
                    + (sc_j == best).astype(np.float64)
                    + (sc_k == best).astype(np.float64)
                )
                sum_eq_i += float(np.sum((sc_i == best) / n_best)) / n_boards
                sum_eq_j += float(np.sum((sc_j == best) / n_best)) / n_boards
                sum_eq_k += float(np.sum((sc_k == best) / n_best)) / n_boards
                n_valid += 1

    if n_valid == 0:
        return (i, j, k, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    return (
        i,
        j,
        k,
        sum_eq_i / n_valid,
        sum_eq_j / n_valid,
        sum_eq_k / n_valid,
    )


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------

def _finalize_and_save(matrix: np.ndarray, output_path: str) -> None:
    """Fill degenerate entries, save output, and print validation stats.

    Degenerate entries (any two indices equal) are set to 0.5 by convention,
    since in real poker two players cannot hold the same hole cards.

    Args:
        matrix: (169, 169, 169) float32 array, fully computed (no NaN).
        output_path: Where to save the final .npy file.
    """
    print("\nFilling degenerate entries (any two indices equal → 0.5)...")
    for x in range(169):
        for y in range(169):
            if x == y:
                # All three same: [x, x, x] — each player has 1/3 equity
                matrix[x, x, x] = 1.0 / 3.0
            else:
                # Two same, one different: [x, x, y], [x, y, x], [y, x, x]
                matrix[x, x, y] = 0.5
                matrix[x, y, x] = 0.5
                matrix[y, x, x] = 0.5

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, matrix)
    print(f"Saved to {output_path}")
    print(f"Shape: {matrix.shape}, dtype: {matrix.dtype}")

    # Validation
    print("\n--- Validation ---")

    # Sum check: matrix[i,j,k] + matrix[j,i,k] + matrix[k,i,j] ≈ 1.0
    # Sample a few triplets
    sample_triplets = [(0, 1, 2), (0, 1, 12), (13, 14, 91)]
    for (a, b, c) in sample_triplets:
        s = float(matrix[a, b, c]) + float(matrix[b, a, c]) + float(matrix[c, a, b])
        ok = "OK" if abs(s - 1.0) < 0.05 else "WARN"
        print(
            f"  Sum [{a},{b},{c}]+[{b},{a},{c}]+[{c},{a},{b}] = {s:.4f} "
            f"(expect ~1.0) [{ok}]"
        )

    # Spot check: AA (0) vs KK (1) vs QQ (2) — AA should dominate
    eq_aa = float(matrix[0, 1, 2])
    eq_kk = float(matrix[1, 0, 2])
    eq_qq = float(matrix[2, 0, 1])
    print(
        f"\n  AA vs KK vs QQ: AA={eq_aa:.4f}, KK={eq_kk:.4f}, QQ={eq_qq:.4f} "
        f"(sum={eq_aa+eq_kk+eq_qq:.4f})"
    )
    print(f"  AA dominant: {eq_aa > eq_kk > eq_qq} (expect True)")

    # Symmetry in opponent order: matrix[i,j,k] == matrix[i,k,j]
    sym_err = abs(float(matrix[0, 1, 2]) - float(matrix[0, 2, 1]))
    print(f"\n  Opponent symmetry |matrix[0,1,2] - matrix[0,2,1]|: {sym_err:.6f} (expect 0.0)")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate the 3-way equity tensor and save to data/equity_3way.npy."""
    print("=" * 70)
    print("AoF Solver — 3-way equity tensor generator")
    print("!! DO NOT RUN ON VPS — run locally on 6-core/12-thread machine !!")
    print("=" * 70)
    print(f"N_BOARDS={N_BOARDS}, N_WORKERS={N_WORKERS}")
    print(f"Total unique triplets (i < j < k): {N_TRIPLETS:,}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}\n")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load checkpoint if it exists and version matches, otherwise start fresh.
    # Version check guards against silently reusing an old eval7 checkpoint
    # (version=2 == phevaluator run).  Old checkpoints have no meta file.
    _use_checkpoint = False
    if os.path.exists(CHECKPOINT_PATH):
        if os.path.exists(CHECKPOINT_META):
            meta = np.load(CHECKPOINT_META)
            if len(meta) >= 2 and int(meta[1]) == 2:
                _use_checkpoint = True
            else:
                print(
                    "Checkpoint version mismatch "
                    f"(got version {int(meta[1]) if len(meta) >= 2 else '?'}, "
                    "expected 2) — starting fresh (old eval7 checkpoint discarded)."
                )
        else:
            print(
                "Checkpoint found but no version metadata — starting fresh "
                "(likely an old eval7 checkpoint; discarding to avoid corrupt data)."
            )

    if _use_checkpoint:
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        matrix = np.load(CHECKPOINT_PATH)
        assert matrix.shape == (169, 169, 169), f"Unexpected shape: {matrix.shape}"
        assert matrix.dtype == np.float32, f"Unexpected dtype: {matrix.dtype}"
    else:
        print("No valid checkpoint found. Starting fresh.")
        # NaN sentinel: marks triplets not yet computed
        matrix = np.full((169, 169, 169), np.nan, dtype=np.float32)

    # Build full list of upper-triangle triplets
    all_triplets = list(combinations(range(169), 3))
    assert len(all_triplets) == N_TRIPLETS

    # Skip triplets already computed (matrix[i,j,k] is not NaN)
    pending = [
        (i, j, k)
        for i, j, k in all_triplets
        if np.isnan(matrix[i, j, k])
    ]

    n_done_total = N_TRIPLETS - len(pending)
    print(f"Already computed: {n_done_total:,} / {N_TRIPLETS:,}")
    print(f"Remaining: {len(pending):,}\n")

    if not pending:
        print("All triplets computed! Finalizing output...")
        _finalize_and_save(matrix, OUTPUT_PATH)
        return

    # Build args for workers: (i, j, k, n_boards, seed)
    # Seed is deterministic per triplet for reproducibility
    args_list = [
        (i, j, k, N_BOARDS, i * 28561 + j * 169 + k)
        for i, j, k in pending
    ]

    t_start = time.time()
    n_done_session = 0
    last_checkpoint_count = n_done_total

    print(f"Starting {N_WORKERS} worker processes...\n")

    # Larger chunksize reduces IPC round-trips on Windows (spawn start method).
    chunksize = max(50, len(args_list) // (N_WORKERS * 8))
    print(f"Using chunksize={chunksize} for {len(args_list):,} remaining triplets\n")

    with mp.Pool(N_WORKERS) as pool:
        for result in pool.imap_unordered(
            compute_triplet_equity, args_list, chunksize=chunksize
        ):
            i, j, k, eq_i, eq_j, eq_k = result

            # Fill all 6 permutations (opponent order is irrelevant for equity)
            matrix[i, j, k] = eq_i
            matrix[i, k, j] = eq_i
            matrix[j, i, k] = eq_j
            matrix[j, k, i] = eq_j
            matrix[k, i, j] = eq_k
            matrix[k, j, i] = eq_k

            n_done_session += 1
            n_done_total += 1

            # Progress report every 1000 triplets
            if n_done_session % 1000 == 0:
                elapsed = time.time() - t_start
                rate = n_done_session / elapsed  # triplets/second in this session
                remaining = (N_TRIPLETS - n_done_total) / rate
                h = int(remaining // 3600)
                m = int((remaining % 3600) // 60)
                pct = 100.0 * n_done_total / N_TRIPLETS
                print(
                    f"{n_done_total:,}/{N_TRIPLETS:,} triplets done "
                    f"({pct:.1f}%) | ETA: {h}h {m}m"
                )

            # Save checkpoint every CHECKPOINT_INTERVAL new triplets
            if n_done_total - last_checkpoint_count >= CHECKPOINT_INTERVAL:
                np.save(CHECKPOINT_PATH, matrix)
                np.save(CHECKPOINT_META, np.array([n_done_total, 2]))  # version=2 phevaluator
                elapsed = time.time() - t_start
                print(
                    f"  Checkpoint saved at {n_done_total:,} triplets "
                    f"({elapsed:.0f}s elapsed)"
                )
                last_checkpoint_count = n_done_total

    # All workers done — finalize
    total_elapsed = time.time() - t_start
    print(
        f"\nAll triplets computed in {total_elapsed:.0f}s "
        f"({total_elapsed / 3600:.2f}h)"
    )

    # Remove checkpoint files now that computation is complete
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print(f"Checkpoint removed: {CHECKPOINT_PATH}")
    if os.path.exists(CHECKPOINT_META):
        os.remove(CHECKPOINT_META)
        print(f"Checkpoint meta removed: {CHECKPOINT_META}")

    _finalize_and_save(matrix, OUTPUT_PATH)


if __name__ == "__main__":
    # Guard against recursive process spawning on Windows/macOS spawn start method
    mp.freeze_support()
    main()
