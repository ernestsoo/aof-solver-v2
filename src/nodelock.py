"""Nodelock solver for 4-max All-in or Fold poker.

Nodelocking fixes one or more opponent strategies and solves for the
exploitative best response of the remaining positions.  The IBR loop is
identical to solve_nash in src/solver.py except that any strategy present
in the ``locked`` dict is never updated — it stays fixed at its given value
throughout all iterations.
"""

import numpy as np

from src.hands import top_n_percent, range_to_mask
from src.solver import (
    SolverResult,
    STRATEGY_NAMES,
    _FOLD_EV,
    _EV_FUNCTIONS,
    initial_strategies,
    best_response,
    compute_exploitability,
    ev_push_co,
    ev_push_btn_open,
    ev_push_sb_open,
    ev_call_btn_vs_co,
    ev_call_sb_vs_co,
    ev_call_sb_vs_btn,
    ev_call_sb_vs_co_btn,
    ev_call_bb_vs_sb,
    ev_call_bb_vs_btn,
    ev_call_bb_vs_co,
    ev_call_bb_vs_btn_sb,
    ev_call_bb_vs_co_sb,
    ev_call_bb_vs_co_btn,
    ev_call_bb_vs_co_btn_sb,
)


def nodelock_solve(
    equity_matrix: np.ndarray,
    combo_weights: np.ndarray,
    locked: dict[str, np.ndarray],
    max_iter: int = 500,
    tolerance: float = 0.001,
) -> SolverResult:
    """Find exploitative equilibrium with some strategies fixed (nodelocked).

    Runs the same IBR loop as solve_nash, but skips updating any strategy
    whose name appears in ``locked``.  Non-locked strategies are initialised
    via initial_strategies() and then solved to best-respond against the
    fixed ranges.

    Args:
        equity_matrix: (169, 169) float32 precomputed equity matrix.
        combo_weights: (169,) float64 combo count weights (sum = 1326).
        locked:        Dict mapping strategy name -> fixed (169,) float64
                       array.  Names must be valid STRATEGY_NAMES entries.
                       Example: {"push_co": some_array}.
        max_iter:      Maximum IBR iterations (default 500).
        tolerance:     Convergence threshold for max strategy change (default 0.001).

    Returns:
        SolverResult with final strategies (locked ones preserved), ev_table,
        iterations run, converged flag, and exploitability in bb.

    Raises:
        ValueError: If any key in ``locked`` is not a valid strategy name.
    """
    invalid = set(locked) - set(STRATEGY_NAMES)
    if invalid:
        raise ValueError(f"Unknown strategy name(s) in locked: {invalid}")

    # Initialise all strategies, then overwrite with locked values.
    strategies = initial_strategies(combo_weights)
    for name, arr in locked.items():
        strategies[name] = arr.astype(np.float64).copy()

    converged = False
    iterations = 0

    for _ in range(max_iter):
        old = {k: v.copy() for k, v in strategies.items()}

        # --- CO ---
        if "push_co" not in locked:
            strategies["push_co"] = best_response(
                ev_push_co(equity_matrix, combo_weights, strategies),
                _FOLD_EV["push_co"],
                old["push_co"],
            )

        # --- BTN ---
        if "push_btn_open" not in locked:
            strategies["push_btn_open"] = best_response(
                ev_push_btn_open(equity_matrix, combo_weights, strategies),
                _FOLD_EV["push_btn_open"],
                old["push_btn_open"],
            )
        if "call_btn_vs_co" not in locked:
            strategies["call_btn_vs_co"] = best_response(
                ev_call_btn_vs_co(equity_matrix, combo_weights, strategies),
                _FOLD_EV["call_btn_vs_co"],
                old["call_btn_vs_co"],
            )

        # --- SB ---
        if "push_sb_open" not in locked:
            strategies["push_sb_open"] = best_response(
                ev_push_sb_open(equity_matrix, combo_weights, strategies),
                _FOLD_EV["push_sb_open"],
                old["push_sb_open"],
            )
        if "call_sb_vs_co" not in locked:
            strategies["call_sb_vs_co"] = best_response(
                ev_call_sb_vs_co(equity_matrix, combo_weights, strategies),
                _FOLD_EV["call_sb_vs_co"],
                old["call_sb_vs_co"],
            )
        if "call_sb_vs_btn" not in locked:
            strategies["call_sb_vs_btn"] = best_response(
                ev_call_sb_vs_btn(equity_matrix, combo_weights, strategies),
                _FOLD_EV["call_sb_vs_btn"],
                old["call_sb_vs_btn"],
            )
        if "call_sb_vs_co_btn" not in locked:
            strategies["call_sb_vs_co_btn"] = best_response(
                ev_call_sb_vs_co_btn(equity_matrix, combo_weights, strategies),
                _FOLD_EV["call_sb_vs_co_btn"],
                old["call_sb_vs_co_btn"],
            )

        # --- BB ---
        if "call_bb_vs_sb" not in locked:
            strategies["call_bb_vs_sb"] = best_response(
                ev_call_bb_vs_sb(equity_matrix, combo_weights, strategies),
                _FOLD_EV["call_bb_vs_sb"],
                old["call_bb_vs_sb"],
            )
        if "call_bb_vs_btn" not in locked:
            strategies["call_bb_vs_btn"] = best_response(
                ev_call_bb_vs_btn(equity_matrix, combo_weights, strategies),
                _FOLD_EV["call_bb_vs_btn"],
                old["call_bb_vs_btn"],
            )
        if "call_bb_vs_co" not in locked:
            strategies["call_bb_vs_co"] = best_response(
                ev_call_bb_vs_co(equity_matrix, combo_weights, strategies),
                _FOLD_EV["call_bb_vs_co"],
                old["call_bb_vs_co"],
            )
        if "call_bb_vs_btn_sb" not in locked:
            strategies["call_bb_vs_btn_sb"] = best_response(
                ev_call_bb_vs_btn_sb(equity_matrix, combo_weights, strategies),
                _FOLD_EV["call_bb_vs_btn_sb"],
                old["call_bb_vs_btn_sb"],
            )
        if "call_bb_vs_co_sb" not in locked:
            strategies["call_bb_vs_co_sb"] = best_response(
                ev_call_bb_vs_co_sb(equity_matrix, combo_weights, strategies),
                _FOLD_EV["call_bb_vs_co_sb"],
                old["call_bb_vs_co_sb"],
            )
        if "call_bb_vs_co_btn" not in locked:
            strategies["call_bb_vs_co_btn"] = best_response(
                ev_call_bb_vs_co_btn(equity_matrix, combo_weights, strategies),
                _FOLD_EV["call_bb_vs_co_btn"],
                old["call_bb_vs_co_btn"],
            )
        if "call_bb_vs_co_btn_sb" not in locked:
            strategies["call_bb_vs_co_btn_sb"] = best_response(
                ev_call_bb_vs_co_btn_sb(equity_matrix, combo_weights, strategies),
                _FOLD_EV["call_bb_vs_co_btn_sb"],
                old["call_bb_vs_co_btn_sb"],
            )

        iterations += 1

        # Convergence check — only over non-locked strategies.
        free_names = [n for n in STRATEGY_NAMES if n not in locked]
        if free_names:
            max_change = max(
                np.max(np.abs(strategies[k] - old[k])) for k in free_names
            )
            if max_change < tolerance:
                converged = True
                break
        else:
            # All strategies locked — nothing to solve; trivially converged.
            converged = True
            break

    # Build final EV table for all 14 strategies.
    ev_table: dict[str, np.ndarray] = {
        name: _EV_FUNCTIONS[name](equity_matrix, combo_weights, strategies)
        for name in STRATEGY_NAMES
    }

    return SolverResult(
        strategies=strategies,
        ev_table=ev_table,
        iterations=iterations,
        converged=converged,
        exploitability=compute_exploitability(strategies, equity_matrix, combo_weights),
    )


# ---------------------------------------------------------------------------
# Convenience helpers for building locked strategy arrays
# ---------------------------------------------------------------------------


def lock_from_range_pct(pct: float, combo_weights: np.ndarray) -> np.ndarray:
    """Return a (169,) mask for the top pct% of hands by preflop strength.

    Thin wrapper around top_n_percent from src.hands.

    Args:
        pct:           Percentage threshold, 0.0 to 100.0.
        combo_weights: (169,) combo count weights — accepted for API symmetry
                       but not used (top_n_percent uses the global ranking).

    Returns:
        np.ndarray of shape (169,), dtype float64, values 0.0 or 1.0.
    """
    return top_n_percent(pct)


def compare_vs_nash(
    nash_result: "SolverResult",
    nodelock_result: "SolverResult",
) -> dict:
    """Compare a nodelocked result against the Nash baseline.

    For each of the 14 strategy names, computes the mean EV difference
    (nodelock minus Nash) across all 169 hands.  Also includes top-level
    exploitability scalars and their delta.

    Args:
        nash_result:      SolverResult from solve_nash (the baseline).
        nodelock_result:  SolverResult from nodelock_solve.

    Returns:
        dict with keys:
          - One key per STRATEGY_NAMES entry (str -> float), value is
            np.mean(nodelock_ev[s] - nash_ev[s]) for each strategy s.
          - "exploitability_nash"     : float — exploitability of the Nash result.
          - "exploitability_nodelock" : float — exploitability of the nodelock result.
          - "exploitability_delta"    : float — nodelock minus Nash exploitability.

    Notes:
        If a strategy is missing from one of the ev_tables (e.g. because the
        result was built without a full ev_table), the diff for that strategy
        is set to 0.0 rather than raising an error.
    """
    result: dict = {}

    for name in STRATEGY_NAMES:
        nash_ev = nash_result.ev_table.get(name)
        nl_ev = nodelock_result.ev_table.get(name)
        if nash_ev is None or nl_ev is None:
            result[name] = 0.0
        else:
            result[name] = float(np.mean(nl_ev - nash_ev))

    result["exploitability_nash"] = float(nash_result.exploitability)
    result["exploitability_nodelock"] = float(nodelock_result.exploitability)
    result["exploitability_delta"] = float(
        nodelock_result.exploitability - nash_result.exploitability
    )

    return result


def lock_from_hands(hands: list[str]) -> np.ndarray:
    """Convert a list of hand name strings to a (169,) binary mask.

    Thin wrapper around range_to_mask from src.hands.

    Args:
        hands: List of canonical hand names, e.g. ["AA", "KK", "AKs"].

    Returns:
        np.ndarray of shape (169,), dtype float64: 1.0 for each named hand,
        0.0 elsewhere.
    """
    return range_to_mask(hands)
