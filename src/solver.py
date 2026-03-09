"""Nash solver for 4-max All-in or Fold poker using Iterative Best Response (IBR).

All strategy arrays are shape (169,), float64, values 0.0–1.0.
All equity lookups use the precomputed 169x169 matrix — no Monte Carlo at solve time.
All EV computations are fully vectorized — no Python loops over hands.
"""

from dataclasses import dataclass, field

import numpy as np

from src.hands import COMBO_WEIGHTS, top_n_percent
from src.equity import (
    hand_vs_range_equity,
    hand_vs_range_equity_vec,
    eq3_vs_ranges_vec,
    eq4_vs_ranges_vec,
)


# ---------------------------------------------------------------------------
# Strategy name constants
# ---------------------------------------------------------------------------

# All 14 strategy array names in canonical order.
# Push strategies (3): CO, BTN open, SB open.
# Call strategies (11): BTN (1), SB (3), BB (7).
STRATEGY_NAMES: list[str] = [
    # Push strategies
    "push_co",
    "push_btn_open",
    "push_sb_open",
    # Call strategies — BTN
    "call_btn_vs_co",
    # Call strategies — SB
    "call_sb_vs_co",
    "call_sb_vs_btn",
    "call_sb_vs_co_btn",
    # Call strategies — BB
    "call_bb_vs_sb",
    "call_bb_vs_btn",
    "call_bb_vs_co",
    "call_bb_vs_btn_sb",
    "call_bb_vs_co_sb",
    "call_bb_vs_co_btn",
    "call_bb_vs_co_btn_sb",
]

assert len(STRATEGY_NAMES) == 14, f"Expected 14 strategy names, got {len(STRATEGY_NAMES)}"
assert len(set(STRATEGY_NAMES)) == 14, "STRATEGY_NAMES has duplicates"


# ---------------------------------------------------------------------------
# SolverResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    """Result of a Nash or nodelock solve.

    Attributes:
        strategies:      Dict mapping strategy name -> (169,) float64 array.
                         Values 0.0 (fold) to 1.0 (push/call always).
        ev_table:        Dict mapping strategy name -> (169,) EV array (bb).
                         Populated after a full solve; may be empty dict before.
        iterations:      Number of IBR iterations run.
        converged:       True if max strategy change < tolerance at termination.
        exploitability:  Total exploitability in bb (sum across all positions).
                         0.0 at Nash equilibrium.
    """

    strategies: dict[str, np.ndarray]
    ev_table: dict[str, np.ndarray] = field(default_factory=dict)
    iterations: int = 0
    converged: bool = False
    exploitability: float = 0.0


# ---------------------------------------------------------------------------
# Strategy initialization
# ---------------------------------------------------------------------------

def initial_strategies(combo_weights: np.ndarray) -> dict[str, np.ndarray]:
    """Return a dict with all 14 strategy arrays set to reasonable opening values.

    Push strategies are initialised wider than call strategies, reflecting
    the positional stealing incentive in AoF. All values are 0.0 or 1.0
    (pure strategies based on top-N% by combo-weighted preflop strength).

    Initialization percentages:
        push_co:         top 30%  (tightest push — first to act, full risk)
        push_btn_open:   top 40%  (wider — 1.5bb steal vs 2 remaining players)
        push_sb_open:    top 50%  (widest push — only BB left, cheap steal)
        call_*:          top 20%  (calls need strong hands — risk full stack)

    Args:
        combo_weights: (169,) array of combo counts (6/4/12); sum = 1326.

    Returns:
        Dict mapping each of the 14 STRATEGY_NAMES to a (169,) float64 array
        with values 0.0 or 1.0.
    """
    strategies: dict[str, np.ndarray] = {}

    # Push strategies: wider by position
    strategies["push_co"]       = top_n_percent(30.0).copy()
    strategies["push_btn_open"] = top_n_percent(40.0).copy()
    strategies["push_sb_open"]  = top_n_percent(50.0).copy()

    # All call strategies: start at top 20%
    call_init = top_n_percent(20.0)
    for name in STRATEGY_NAMES:
        if name.startswith("call_"):
            strategies[name] = call_init.copy()

    assert len(strategies) == 14, f"Expected 14 strategies, got {len(strategies)}"
    for name, arr in strategies.items():
        assert arr.shape == (169,), f"{name}: expected shape (169,), got {arr.shape}"
        assert arr.dtype == np.float64, f"{name}: expected float64, got {arr.dtype}"

    return strategies


# ---------------------------------------------------------------------------
# Fold / call probability helpers
# ---------------------------------------------------------------------------

def fold_prob(strategy: np.ndarray, combo_weights: np.ndarray) -> float:
    """Return the probability that a random hand from the range folds.

    Computes combo-weighted fold frequency:
        fold_prob = dot(1 - strategy, combo_weights) / sum(combo_weights)

    Args:
        strategy:      (169,) array, values 0.0 (always fold) to 1.0 (always push/call).
        combo_weights: (169,) combo count weights (e.g. COMBO_WEIGHTS).

    Returns:
        Float in [0.0, 1.0]. 1.0 if strategy is all zeros (always folds).
    """
    return float(np.dot(1.0 - strategy, combo_weights) / combo_weights.sum())


def call_prob(strategy: np.ndarray, combo_weights: np.ndarray) -> float:
    """Return the probability that a random hand from the range calls/pushes.

    Complement of fold_prob:
        call_prob = 1.0 - fold_prob(strategy, combo_weights)

    Args:
        strategy:      (169,) array, values 0.0 to 1.0.
        combo_weights: (169,) combo count weights.

    Returns:
        Float in [0.0, 1.0]. 0.0 if strategy is all zeros.
    """
    return 1.0 - fold_prob(strategy, combo_weights)


# ---------------------------------------------------------------------------
# EV computation — CO open push
# ---------------------------------------------------------------------------


def ev_push_co(
    equity_matrix: np.ndarray,
    combo_weights: np.ndarray,
    strategies: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the (169,) EV array for CO pushing each hand. EV(fold) = 0.

    Covers all 8 terminal nodes reached when CO pushes (Terminals 8-15 in
    CLAUDE.md). Fully vectorized — operates on all 169 hands simultaneously
    with no Python loops.

    Args:
        equity_matrix: (169, 169) float32 precomputed equity matrix.
        combo_weights: (169,) float64 combo count weights (sum = 1326).
        strategies:    Dict with all 14 strategy arrays (shape (169,) each).
                       Must contain: call_btn_vs_co, call_sb_vs_co,
                       call_bb_vs_co, call_sb_vs_co_btn, call_bb_vs_co_sb,
                       call_bb_vs_co_btn, call_bb_vs_co_btn_sb.

    Returns:
        (169,) float64 array: EV of pushing for each hand (in bb, net from
        start of hand). EV(fold) for CO = 0.0; push is correct when EV > 0.
    """
    # ------------------------------------------------------------------
    # Scalar fold/call probabilities for each opponent in each scenario
    # ------------------------------------------------------------------
    f_btn        = fold_prob(strategies["call_btn_vs_co"],      combo_weights)
    c_btn        = 1.0 - f_btn
    f_sb_co      = fold_prob(strategies["call_sb_vs_co"],       combo_weights)
    c_sb_co      = 1.0 - f_sb_co
    f_bb_co      = fold_prob(strategies["call_bb_vs_co"],       combo_weights)
    c_bb_co      = 1.0 - f_bb_co
    f_sb_co_btn  = fold_prob(strategies["call_sb_vs_co_btn"],   combo_weights)
    c_sb_co_btn  = 1.0 - f_sb_co_btn
    f_bb_co_sb   = fold_prob(strategies["call_bb_vs_co_sb"],    combo_weights)
    c_bb_co_sb   = 1.0 - f_bb_co_sb
    f_bb_co_btn  = fold_prob(strategies["call_bb_vs_co_btn"],   combo_weights)
    c_bb_co_btn  = 1.0 - f_bb_co_btn
    f_bb_co_btn_sb = fold_prob(strategies["call_bb_vs_co_btn_sb"], combo_weights)
    c_bb_co_btn_sb = 1.0 - f_bb_co_btn_sb

    # ------------------------------------------------------------------
    # Vectorized equity of each CO hand against each opponent call range
    # Shape: (169,)
    # ------------------------------------------------------------------
    eq_vs_bb  = hand_vs_range_equity_vec(equity_matrix, strategies["call_bb_vs_co"],      combo_weights)
    eq_vs_sb  = hand_vs_range_equity_vec(equity_matrix, strategies["call_sb_vs_co"],      combo_weights)
    eq_vs_btn = hand_vs_range_equity_vec(equity_matrix, strategies["call_btn_vs_co"],     combo_weights)

    # 3-way equity arrays — (169,)
    eq3_sb_bb  = eq3_vs_ranges_vec(equity_matrix, strategies["call_sb_vs_co"],     strategies["call_bb_vs_co_sb"],    combo_weights)
    eq3_btn_bb = eq3_vs_ranges_vec(equity_matrix, strategies["call_btn_vs_co"],    strategies["call_bb_vs_co_btn"],   combo_weights)
    eq3_btn_sb = eq3_vs_ranges_vec(equity_matrix, strategies["call_btn_vs_co"],    strategies["call_sb_vs_co_btn"],   combo_weights)

    # 4-way equity array — (169,)
    eq4_all    = eq4_vs_ranges_vec(equity_matrix, strategies["call_btn_vs_co"], strategies["call_sb_vs_co_btn"], strategies["call_bb_vs_co_btn_sb"], combo_weights)

    # ------------------------------------------------------------------
    # Sum EV contributions from all 8 terminal nodes
    # ------------------------------------------------------------------
    EV = np.zeros(169)

    # Terminal 8: all fold — CO steals 1.5bb (SB 0.5 + BB 1.0 already posted)
    EV += f_btn * f_sb_co * f_bb_co * 1.5

    # Terminal 9: only BB calls — pot = 20.5 (SB 0.5 dead)
    EV += f_btn * f_sb_co * c_bb_co * (eq_vs_bb * 20.5 - 10.0)

    # Terminal 10: only SB calls — pot = 21.0 (BB 1.0 dead)
    EV += f_btn * c_sb_co * f_bb_co_sb * (eq_vs_sb * 21.0 - 10.0)

    # Terminal 11: SB + BB call — pot = 30.0 (3-way, no dead money)
    EV += f_btn * c_sb_co * c_bb_co_sb * (eq3_sb_bb * 30.0 - 10.0)

    # Terminal 12: BTN calls, SB + BB fold — pot = 21.5 (both blinds dead)
    EV += c_btn * f_sb_co_btn * f_bb_co_btn * (eq_vs_btn * 21.5 - 10.0)

    # Terminal 13: BTN + BB call — pot = 30.5 (SB 0.5 dead)
    EV += c_btn * f_sb_co_btn * c_bb_co_btn * (eq3_btn_bb * 30.5 - 10.0)

    # Terminal 14: BTN + SB call — pot = 31.0 (BB 1.0 dead)
    EV += c_btn * c_sb_co_btn * f_bb_co_btn_sb * (eq3_btn_sb * 31.0 - 10.0)

    # Terminal 15: all call — pot = 40.0 (4-way, no dead money)
    EV += c_btn * c_sb_co_btn * c_bb_co_btn_sb * (eq4_all * 40.0 - 10.0)

    return EV


# ---------------------------------------------------------------------------
# EV computation — BTN open push (CO folded)
# ---------------------------------------------------------------------------


def ev_push_btn_open(
    equity_matrix: np.ndarray,
    combo_weights: np.ndarray,
    strategies: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the (169,) EV array for BTN open push (CO folded). EV(fold) = 0.

    Covers Terminals 4-7 from CLAUDE.md: BTN pushes after CO folds.
    Fully vectorized — no Python loops over hands.

    Args:
        equity_matrix: (169, 169) float32 precomputed equity matrix.
        combo_weights: (169,) float64 combo count weights (sum = 1326).
        strategies:    Dict with all 14 strategy arrays. Must contain:
                       call_sb_vs_btn, call_bb_vs_btn, call_bb_vs_btn_sb.

    Returns:
        (169,) float64 EV of BTN pushing each hand (bb, net from start).
        EV(fold) for BTN = 0.0; push is correct when EV > 0.
    """
    # Scalar fold/call probabilities
    f_sb        = fold_prob(strategies["call_sb_vs_btn"],    combo_weights)
    c_sb        = 1.0 - f_sb
    f_bb_btn    = fold_prob(strategies["call_bb_vs_btn"],    combo_weights)
    c_bb_btn    = 1.0 - f_bb_btn
    f_bb_btn_sb = fold_prob(strategies["call_bb_vs_btn_sb"], combo_weights)
    c_bb_btn_sb = 1.0 - f_bb_btn_sb

    # Vectorized equity arrays — (169,)
    eq_vs_bb    = hand_vs_range_equity_vec(equity_matrix, strategies["call_bb_vs_btn"],    combo_weights)
    eq_vs_sb    = hand_vs_range_equity_vec(equity_matrix, strategies["call_sb_vs_btn"],    combo_weights)
    eq3_sb_bb   = eq3_vs_ranges_vec(equity_matrix, strategies["call_sb_vs_btn"], strategies["call_bb_vs_btn_sb"], combo_weights)

    EV = np.zeros(169)

    # Terminal 4: BTN steals — SB+BB fold, pot=1.5
    EV += f_sb * f_bb_btn * 1.5

    # Terminal 5: BTN vs BB — SB folds, pot=20.5 (SB 0.5 dead)
    EV += f_sb * c_bb_btn * (eq_vs_bb * 20.5 - 10.0)

    # Terminal 6: BTN vs SB — BB folds, pot=21.0 (BB 1.0 dead)
    EV += c_sb * f_bb_btn_sb * (eq_vs_sb * 21.0 - 10.0)

    # Terminal 7: BTN vs SB vs BB — 3-way, pot=30.0
    EV += c_sb * c_bb_btn_sb * (eq3_sb_bb * 30.0 - 10.0)

    return EV


# ---------------------------------------------------------------------------
# EV computation — BTN call vs CO push
# ---------------------------------------------------------------------------


def ev_call_btn_vs_co(
    equity_matrix: np.ndarray,
    combo_weights: np.ndarray,
    strategies: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the (169,) EV array for BTN calling CO's push. EV(fold) = 0.

    Covers Terminals 12-15 from CLAUDE.md: CO pushed, BTN calls.
    Fully vectorized — no Python loops over hands.

    Args:
        equity_matrix: (169, 169) float32 precomputed equity matrix.
        combo_weights: (169,) float64 combo count weights (sum = 1326).
        strategies:    Dict with all 14 strategy arrays. Must contain:
                       push_co, call_sb_vs_co_btn, call_bb_vs_co_btn,
                       call_bb_vs_co_btn_sb.

    Returns:
        (169,) float64 EV of BTN calling each hand (bb, net from start).
        EV(fold) for BTN = 0.0; call is correct when EV > 0.
    """
    # Scalar fold/call probabilities
    f_sb_co_btn    = fold_prob(strategies["call_sb_vs_co_btn"],    combo_weights)
    c_sb_co_btn    = 1.0 - f_sb_co_btn
    f_bb_co_btn    = fold_prob(strategies["call_bb_vs_co_btn"],    combo_weights)
    c_bb_co_btn    = 1.0 - f_bb_co_btn
    f_bb_co_btn_sb = fold_prob(strategies["call_bb_vs_co_btn_sb"], combo_weights)
    c_bb_co_btn_sb = 1.0 - f_bb_co_btn_sb

    # Vectorized equity arrays — (169,)
    eq_vs_co      = hand_vs_range_equity_vec(equity_matrix, strategies["push_co"],            combo_weights)
    eq3_co_bb     = eq3_vs_ranges_vec(equity_matrix, strategies["push_co"], strategies["call_bb_vs_co_btn"],    combo_weights)
    eq3_co_sb     = eq3_vs_ranges_vec(equity_matrix, strategies["push_co"], strategies["call_sb_vs_co_btn"],    combo_weights)
    eq4_co_sb_bb  = eq4_vs_ranges_vec(equity_matrix, strategies["push_co"], strategies["call_sb_vs_co_btn"], strategies["call_bb_vs_co_btn_sb"], combo_weights)

    EV = np.zeros(169)

    # Terminal 12: BTN vs CO — SB+BB fold, pot=21.5 (both blinds dead)
    EV += f_sb_co_btn * f_bb_co_btn * (eq_vs_co * 21.5 - 10.0)

    # Terminal 13: BTN vs CO vs BB — SB folds, pot=30.5 (SB 0.5 dead)
    EV += f_sb_co_btn * c_bb_co_btn * (eq3_co_bb * 30.5 - 10.0)

    # Terminal 14: BTN vs CO vs SB — BB folds, pot=31.0 (BB 1.0 dead)
    EV += c_sb_co_btn * f_bb_co_btn_sb * (eq3_co_sb * 31.0 - 10.0)

    # Terminal 15: 4-way — pot=40.0
    EV += c_sb_co_btn * c_bb_co_btn_sb * (eq4_co_sb_bb * 40.0 - 10.0)

    return EV


# ---------------------------------------------------------------------------
# EV computation — SB open push (CO+BTN folded)
# ---------------------------------------------------------------------------


def ev_push_sb_open(
    equity_matrix: np.ndarray,
    combo_weights: np.ndarray,
    strategies: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the (169,) EV array for SB open push (CO+BTN folded). EV(fold) = -0.5.

    Covers Terminals 2-3 from CLAUDE.md: SB pushes after CO and BTN fold.
    Fully vectorized — no Python loops over hands.

    SB already posted 0.5bb. Net EV from start of hand:
        EV(fold) = -0.5
        EV(push) = f_bb * 1.0  +  c_bb * (eq_vs_bb * 20.0 - 10.0)

    Args:
        equity_matrix: (169, 169) float32 precomputed equity matrix.
        combo_weights: (169,) float64 combo count weights (sum = 1326).
        strategies:    Dict with all 14 strategy arrays. Must contain: call_bb_vs_sb.

    Returns:
        (169,) float64 EV of SB pushing each hand (bb, net from start).
        EV(fold) for SB = -0.5; push is correct when EV > -0.5.
    """
    f_bb   = fold_prob(strategies["call_bb_vs_sb"], combo_weights)
    c_bb   = 1.0 - f_bb

    eq_vs_bb = hand_vs_range_equity_vec(equity_matrix, strategies["call_bb_vs_sb"], combo_weights)

    EV = np.zeros(169)

    # Terminal 2: SB steals — BB folds, pot=1.5, net profit = +1.0 (SB keeps their 0.5 + wins BB's 1.0)
    EV += f_bb * 1.0

    # Terminal 3: SB vs BB — pot=20.0 (heads-up, no dead money)
    EV += c_bb * (eq_vs_bb * 20.0 - 10.0)

    return EV


# ---------------------------------------------------------------------------
# EV computation — SB call vs CO push (BTN folded)
# ---------------------------------------------------------------------------


def ev_call_sb_vs_co(
    equity_matrix: np.ndarray,
    combo_weights: np.ndarray,
    strategies: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the (169,) EV array for SB calling CO's push (BTN folded). EV(fold) = -0.5.

    Covers Terminals 10-11 from CLAUDE.md: CO pushed, BTN folded, SB calls.
    Fully vectorized — no Python loops over hands.

    Args:
        equity_matrix: (169, 169) float32 precomputed equity matrix.
        combo_weights: (169,) float64 combo count weights (sum = 1326).
        strategies:    Dict with all 14 strategy arrays. Must contain:
                       push_co, call_bb_vs_co_sb.

    Returns:
        (169,) float64 EV of SB calling each hand (bb, net from start).
        EV(fold) for SB = -0.5; call is correct when EV > -0.5.
    """
    f_bb_co_sb = fold_prob(strategies["call_bb_vs_co_sb"], combo_weights)
    c_bb_co_sb = 1.0 - f_bb_co_sb

    eq_vs_co   = hand_vs_range_equity_vec(equity_matrix, strategies["push_co"],         combo_weights)
    eq3_co_bb  = eq3_vs_ranges_vec(equity_matrix, strategies["push_co"], strategies["call_bb_vs_co_sb"], combo_weights)

    EV = np.zeros(169)

    # Terminal 10: SB vs CO — BB folds, pot=21.0 (BB 1.0 dead)
    EV += f_bb_co_sb * (eq_vs_co * 21.0 - 10.0)

    # Terminal 11: SB vs CO vs BB — 3-way, pot=30.0
    EV += c_bb_co_sb * (eq3_co_bb * 30.0 - 10.0)

    return EV


# ---------------------------------------------------------------------------
# EV computation — SB call vs BTN push (CO folded)
# ---------------------------------------------------------------------------


def ev_call_sb_vs_btn(
    equity_matrix: np.ndarray,
    combo_weights: np.ndarray,
    strategies: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the (169,) EV array for SB calling BTN's push (CO folded). EV(fold) = -0.5.

    Covers Terminals 6-7 from CLAUDE.md: BTN pushed (CO folded), SB calls.
    Fully vectorized — no Python loops over hands.

    Args:
        equity_matrix: (169, 169) float32 precomputed equity matrix.
        combo_weights: (169,) float64 combo count weights (sum = 1326).
        strategies:    Dict with all 14 strategy arrays. Must contain:
                       push_btn_open, call_bb_vs_btn_sb.

    Returns:
        (169,) float64 EV of SB calling each hand (bb, net from start).
        EV(fold) for SB = -0.5; call is correct when EV > -0.5.
    """
    f_bb_btn_sb = fold_prob(strategies["call_bb_vs_btn_sb"], combo_weights)
    c_bb_btn_sb = 1.0 - f_bb_btn_sb

    eq_vs_btn   = hand_vs_range_equity_vec(equity_matrix, strategies["push_btn_open"],       combo_weights)
    eq3_btn_bb  = eq3_vs_ranges_vec(equity_matrix, strategies["push_btn_open"], strategies["call_bb_vs_btn_sb"], combo_weights)

    EV = np.zeros(169)

    # Terminal 6: SB vs BTN — BB folds, pot=21.0 (BB 1.0 dead)
    EV += f_bb_btn_sb * (eq_vs_btn * 21.0 - 10.0)

    # Terminal 7: SB vs BTN vs BB — 3-way, pot=30.0
    EV += c_bb_btn_sb * (eq3_btn_bb * 30.0 - 10.0)

    return EV


# ---------------------------------------------------------------------------
# EV computation — SB call vs CO push + BTN call
# ---------------------------------------------------------------------------


def ev_call_sb_vs_co_btn(
    equity_matrix: np.ndarray,
    combo_weights: np.ndarray,
    strategies: dict[str, np.ndarray],
) -> np.ndarray:
    """Return (169,) EV for SB calling when CO pushed and BTN called. EV(fold) = -0.5.

    Covers Terminals 14-15 from CLAUDE.md: CO pushed, BTN called, SB calls.
    Fully vectorized — no Python loops over hands.

    Args:
        equity_matrix: (169, 169) float32 precomputed equity matrix.
        combo_weights: (169,) float64 combo count weights (sum = 1326).
        strategies:    Dict with all 14 strategy arrays. Must contain:
                       push_co, call_btn_vs_co, call_bb_vs_co_btn_sb.

    Returns:
        (169,) float64 EV of SB calling each hand (bb, net from start).
        EV(fold) for SB = -0.5; call is correct when EV > -0.5.
    """
    f_bb_co_btn_sb = fold_prob(strategies["call_bb_vs_co_btn_sb"], combo_weights)
    c_bb_co_btn_sb = 1.0 - f_bb_co_btn_sb

    eq3_co_btn    = eq3_vs_ranges_vec(equity_matrix, strategies["push_co"], strategies["call_btn_vs_co"],        combo_weights)
    eq4_co_btn_bb = eq4_vs_ranges_vec(equity_matrix, strategies["push_co"], strategies["call_btn_vs_co"], strategies["call_bb_vs_co_btn_sb"], combo_weights)

    EV = np.zeros(169)

    # Terminal 14: SB vs CO vs BTN — BB folds, pot=31.0 (BB 1.0 dead)
    EV += f_bb_co_btn_sb * (eq3_co_btn * 31.0 - 10.0)

    # Terminal 15: 4-way — pot=40.0
    EV += c_bb_co_btn_sb * (eq4_co_btn_bb * 40.0 - 10.0)

    return EV


# ---------------------------------------------------------------------------
# EV computation — BB call decisions (all 7)
# ---------------------------------------------------------------------------


def ev_call_bb_vs_sb(
    equity_matrix: np.ndarray,
    combo_weights: np.ndarray,
    strategies: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the (169,) EV array for BB calling SB's push. EV(fold) = -1.0.

    Terminal 3: SB vs BB heads-up, pot = 20.0bb (no dead money).
    Fully vectorized — no Python loops over hands.

    Args:
        equity_matrix: (169, 169) float32 precomputed equity matrix.
        combo_weights: (169,) float64 combo count weights (sum = 1326).
        strategies:    Dict with all 14 strategy arrays. Must contain: push_sb_open.

    Returns:
        (169,) float64 EV of BB calling each hand (bb, net from start).
        EV(fold) for BB = -1.0; call is correct when EV > -1.0.
    """
    eq_vs_sb = hand_vs_range_equity_vec(equity_matrix, strategies["push_sb_open"], combo_weights)
    return eq_vs_sb * 20.0 - 10.0


def ev_call_bb_vs_btn(
    equity_matrix: np.ndarray,
    combo_weights: np.ndarray,
    strategies: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the (169,) EV array for BB calling BTN's open push. EV(fold) = -1.0.

    Terminal 5: BTN vs BB heads-up, pot = 20.5bb (SB 0.5 dead).
    Fully vectorized — no Python loops over hands.

    Args:
        equity_matrix: (169, 169) float32 precomputed equity matrix.
        combo_weights: (169,) float64 combo count weights (sum = 1326).
        strategies:    Dict with all 14 strategy arrays. Must contain: push_btn_open.

    Returns:
        (169,) float64 EV of BB calling each hand (bb, net from start).
        EV(fold) for BB = -1.0; call is correct when EV > -1.0.
    """
    eq_vs_btn = hand_vs_range_equity_vec(equity_matrix, strategies["push_btn_open"], combo_weights)
    return eq_vs_btn * 20.5 - 10.0


def ev_call_bb_vs_co(
    equity_matrix: np.ndarray,
    combo_weights: np.ndarray,
    strategies: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the (169,) EV array for BB calling CO's push. EV(fold) = -1.0.

    Terminal 9: CO vs BB heads-up, pot = 20.5bb (SB 0.5 dead, BTN folded).
    Fully vectorized — no Python loops over hands.

    Args:
        equity_matrix: (169, 169) float32 precomputed equity matrix.
        combo_weights: (169,) float64 combo count weights (sum = 1326).
        strategies:    Dict with all 14 strategy arrays. Must contain: push_co.

    Returns:
        (169,) float64 EV of BB calling each hand (bb, net from start).
        EV(fold) for BB = -1.0; call is correct when EV > -1.0.
    """
    eq_vs_co = hand_vs_range_equity_vec(equity_matrix, strategies["push_co"], combo_weights)
    return eq_vs_co * 20.5 - 10.0


def ev_call_bb_vs_btn_sb(
    equity_matrix: np.ndarray,
    combo_weights: np.ndarray,
    strategies: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the (169,) EV array for BB calling when BTN pushed and SB called. EV(fold) = -1.0.

    Terminal 7: BTN vs SB vs BB 3-way, pot = 30.0bb (CO folded, no dead money).
    Fully vectorized — no Python loops over hands.

    Args:
        equity_matrix: (169, 169) float32 precomputed equity matrix.
        combo_weights: (169,) float64 combo count weights (sum = 1326).
        strategies:    Dict with all 14 strategy arrays. Must contain:
                       push_btn_open, call_sb_vs_btn.

    Returns:
        (169,) float64 EV of BB calling each hand (bb, net from start).
        EV(fold) for BB = -1.0; call is correct when EV > -1.0.
    """
    eq3_btn_sb = eq3_vs_ranges_vec(
        equity_matrix,
        strategies["push_btn_open"],
        strategies["call_sb_vs_btn"],
        combo_weights,
    )
    return eq3_btn_sb * 30.0 - 10.0


def ev_call_bb_vs_co_sb(
    equity_matrix: np.ndarray,
    combo_weights: np.ndarray,
    strategies: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the (169,) EV array for BB calling when CO pushed and SB called. EV(fold) = -1.0.

    Terminal 11: CO vs SB vs BB 3-way, pot = 30.0bb (BTN folded, no dead money).
    Fully vectorized — no Python loops over hands.

    Args:
        equity_matrix: (169, 169) float32 precomputed equity matrix.
        combo_weights: (169,) float64 combo count weights (sum = 1326).
        strategies:    Dict with all 14 strategy arrays. Must contain:
                       push_co, call_sb_vs_co.

    Returns:
        (169,) float64 EV of BB calling each hand (bb, net from start).
        EV(fold) for BB = -1.0; call is correct when EV > -1.0.
    """
    eq3_co_sb = eq3_vs_ranges_vec(
        equity_matrix,
        strategies["push_co"],
        strategies["call_sb_vs_co"],
        combo_weights,
    )
    return eq3_co_sb * 30.0 - 10.0


def ev_call_bb_vs_co_btn(
    equity_matrix: np.ndarray,
    combo_weights: np.ndarray,
    strategies: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the (169,) EV array for BB calling when CO pushed and BTN called. EV(fold) = -1.0.

    Terminal 13: CO vs BTN vs BB 3-way, pot = 30.5bb (SB 0.5 dead).
    Fully vectorized — no Python loops over hands.

    Args:
        equity_matrix: (169, 169) float32 precomputed equity matrix.
        combo_weights: (169,) float64 combo count weights (sum = 1326).
        strategies:    Dict with all 14 strategy arrays. Must contain:
                       push_co, call_btn_vs_co.

    Returns:
        (169,) float64 EV of BB calling each hand (bb, net from start).
        EV(fold) for BB = -1.0; call is correct when EV > -1.0.
    """
    eq3_co_btn = eq3_vs_ranges_vec(
        equity_matrix,
        strategies["push_co"],
        strategies["call_btn_vs_co"],
        combo_weights,
    )
    return eq3_co_btn * 30.5 - 10.0


def ev_call_bb_vs_co_btn_sb(
    equity_matrix: np.ndarray,
    combo_weights: np.ndarray,
    strategies: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the (169,) EV array for BB calling when CO pushed, BTN called, SB called. EV(fold) = -1.0.

    Terminal 15: CO vs BTN vs SB vs BB 4-way, pot = 40.0bb (no dead money).
    Fully vectorized — no Python loops over hands.

    Args:
        equity_matrix: (169, 169) float32 precomputed equity matrix.
        combo_weights: (169,) float64 combo count weights (sum = 1326).
        strategies:    Dict with all 14 strategy arrays. Must contain:
                       push_co, call_btn_vs_co, call_sb_vs_co_btn.

    Returns:
        (169,) float64 EV of BB calling each hand (bb, net from start).
        EV(fold) for BB = -1.0; call is correct when EV > -1.0.
    """
    eq4_co_btn_sb = eq4_vs_ranges_vec(
        equity_matrix,
        strategies["push_co"],
        strategies["call_btn_vs_co"],
        strategies["call_sb_vs_co_btn"],
        combo_weights,
    )
    return eq4_co_btn_sb * 40.0 - 10.0


# ---------------------------------------------------------------------------
# Exploitability helpers
# ---------------------------------------------------------------------------

# Fold EV per strategy name (net bb from start of hand).
_FOLD_EV: dict[str, float] = {
    "push_co":              0.0,
    "push_btn_open":        0.0,
    "push_sb_open":        -0.5,
    "call_btn_vs_co":       0.0,
    "call_sb_vs_co":       -0.5,
    "call_sb_vs_btn":      -0.5,
    "call_sb_vs_co_btn":   -0.5,
    "call_bb_vs_sb":       -1.0,
    "call_bb_vs_btn":      -1.0,
    "call_bb_vs_co":       -1.0,
    "call_bb_vs_btn_sb":   -1.0,
    "call_bb_vs_co_sb":    -1.0,
    "call_bb_vs_co_btn":   -1.0,
    "call_bb_vs_co_btn_sb": -1.0,
}

assert set(_FOLD_EV.keys()) == set(STRATEGY_NAMES), "FOLD_EV keys mismatch"


# ---------------------------------------------------------------------------
# Best response with damping
# ---------------------------------------------------------------------------


def best_response(
    ev_action: np.ndarray,
    ev_fold: float,
    old_strategy: np.ndarray,
    alpha: float = 0.9,
) -> np.ndarray:
    """Compute the best-response strategy with damping to prevent oscillation.

    Returns a (169,) array where 1.0 means push/call (ev_action > ev_fold)
    and 0.0 means fold. Damping blends the pure best response with the
    previous strategy to stabilise borderline hands that would otherwise
    oscillate between push and fold across iterations.

    Args:
        ev_action:    (169,) EV of taking the action (push or call) for each hand.
        ev_fold:      Scalar EV of folding (0.0 for CO/BTN; -0.5 for SB; -1.0 for BB).
        old_strategy: (169,) previous strategy array (values 0.0–1.0).
        alpha:        Damping weight on new pure best response (default 0.9).
                      0.0 = never update; 1.0 = no damping.

    Returns:
        (169,) float64 array with values in [0.0, 1.0].
    """
    pure_best = (ev_action > ev_fold).astype(np.float64)
    return alpha * pure_best + (1.0 - alpha) * old_strategy


# ---------------------------------------------------------------------------
# Exploitability computation
# ---------------------------------------------------------------------------

# Maps each strategy name to its EV function (defined above).
_EV_FUNCTIONS = {
    "push_co":              ev_push_co,
    "push_btn_open":        ev_push_btn_open,
    "push_sb_open":         ev_push_sb_open,
    "call_btn_vs_co":       ev_call_btn_vs_co,
    "call_sb_vs_co":        ev_call_sb_vs_co,
    "call_sb_vs_btn":       ev_call_sb_vs_btn,
    "call_sb_vs_co_btn":    ev_call_sb_vs_co_btn,
    "call_bb_vs_sb":        ev_call_bb_vs_sb,
    "call_bb_vs_btn":       ev_call_bb_vs_btn,
    "call_bb_vs_co":        ev_call_bb_vs_co,
    "call_bb_vs_btn_sb":    ev_call_bb_vs_btn_sb,
    "call_bb_vs_co_sb":     ev_call_bb_vs_co_sb,
    "call_bb_vs_co_btn":    ev_call_bb_vs_co_btn,
    "call_bb_vs_co_btn_sb": ev_call_bb_vs_co_btn_sb,
}


def compute_exploitability(
    strategies: dict[str, np.ndarray],
    equity_matrix: np.ndarray,
    combo_weights: np.ndarray,
) -> float:
    """Compute total exploitability of the current strategies in bb.

    For each of the 14 decision points, computes the combo-weighted gain
    available by switching to the best response. Sums these gains across all
    decision points to produce total exploitability.

    At Nash equilibrium no decision point has a profitable deviation, so
    exploitability = 0. Higher values indicate strategies farther from Nash.

    Args:
        strategies:    Dict mapping strategy name -> (169,) float64 array.
        equity_matrix: (169, 169) float32 precomputed equity matrix.
        combo_weights: (169,) float64 combo count weights (sum = 1326).

    Returns:
        Total exploitability in bb (>= 0.0). Near 0.0 at Nash equilibrium.
    """
    w = combo_weights / combo_weights.sum()
    total = 0.0

    for name in STRATEGY_NAMES:
        ev_action = _EV_FUNCTIONS[name](equity_matrix, combo_weights, strategies)
        ev_fold = _FOLD_EV[name]
        strategy = strategies[name]

        # Best response EV per hand: take the action only when it beats folding.
        br_ev = np.maximum(ev_action, ev_fold)

        # Current strategy EV per hand (mixed strategy in [0, 1]).
        current_ev = strategy * ev_action + (1.0 - strategy) * ev_fold

        # Gain from deviating to best response (always >= 0).
        gain = br_ev - current_ev

        total += float(np.dot(gain, w))

    return total


# ---------------------------------------------------------------------------
# IBR Nash solver
# ---------------------------------------------------------------------------


def solve_nash(
    equity_matrix: np.ndarray,
    combo_weights: np.ndarray,
    max_iter: int = 500,
    tolerance: float = 0.001,
) -> "SolverResult":
    """Find Nash equilibrium via Iterative Best Response (IBR).

    Each iteration updates all 14 strategy arrays in position order:
    CO → BTN → SB → BB. Convergence is declared when the maximum absolute
    strategy change across all 14 arrays drops below ``tolerance``.

    Args:
        equity_matrix: (169, 169) float32 equity matrix (from load_equity_matrix).
        combo_weights: (169,) float64 combo count weights (sum = 1326).
        max_iter:      Maximum number of IBR iterations (default 500).
        tolerance:     Convergence threshold for max strategy change (default 0.001).

    Returns:
        SolverResult with final strategies, ev_table (one array per strategy),
        iterations run, converged flag, and exploitability=0.0 (placeholder).
    """
    strategies = initial_strategies(combo_weights)
    converged = False
    iterations = 0

    for _ in range(max_iter):
        old = {k: v.copy() for k, v in strategies.items()}

        # --- CO ---
        strategies["push_co"] = best_response(
            ev_push_co(equity_matrix, combo_weights, strategies),
            0.0,
            old["push_co"],
        )

        # --- BTN ---
        strategies["push_btn_open"] = best_response(
            ev_push_btn_open(equity_matrix, combo_weights, strategies),
            0.0,
            old["push_btn_open"],
        )
        strategies["call_btn_vs_co"] = best_response(
            ev_call_btn_vs_co(equity_matrix, combo_weights, strategies),
            0.0,
            old["call_btn_vs_co"],
        )

        # --- SB ---
        strategies["push_sb_open"] = best_response(
            ev_push_sb_open(equity_matrix, combo_weights, strategies),
            -0.5,
            old["push_sb_open"],
        )
        strategies["call_sb_vs_co"] = best_response(
            ev_call_sb_vs_co(equity_matrix, combo_weights, strategies),
            -0.5,
            old["call_sb_vs_co"],
        )
        strategies["call_sb_vs_btn"] = best_response(
            ev_call_sb_vs_btn(equity_matrix, combo_weights, strategies),
            -0.5,
            old["call_sb_vs_btn"],
        )
        strategies["call_sb_vs_co_btn"] = best_response(
            ev_call_sb_vs_co_btn(equity_matrix, combo_weights, strategies),
            -0.5,
            old["call_sb_vs_co_btn"],
        )

        # --- BB ---
        strategies["call_bb_vs_sb"] = best_response(
            ev_call_bb_vs_sb(equity_matrix, combo_weights, strategies),
            -1.0,
            old["call_bb_vs_sb"],
        )
        strategies["call_bb_vs_btn"] = best_response(
            ev_call_bb_vs_btn(equity_matrix, combo_weights, strategies),
            -1.0,
            old["call_bb_vs_btn"],
        )
        strategies["call_bb_vs_co"] = best_response(
            ev_call_bb_vs_co(equity_matrix, combo_weights, strategies),
            -1.0,
            old["call_bb_vs_co"],
        )
        strategies["call_bb_vs_btn_sb"] = best_response(
            ev_call_bb_vs_btn_sb(equity_matrix, combo_weights, strategies),
            -1.0,
            old["call_bb_vs_btn_sb"],
        )
        strategies["call_bb_vs_co_sb"] = best_response(
            ev_call_bb_vs_co_sb(equity_matrix, combo_weights, strategies),
            -1.0,
            old["call_bb_vs_co_sb"],
        )
        strategies["call_bb_vs_co_btn"] = best_response(
            ev_call_bb_vs_co_btn(equity_matrix, combo_weights, strategies),
            -1.0,
            old["call_bb_vs_co_btn"],
        )
        strategies["call_bb_vs_co_btn_sb"] = best_response(
            ev_call_bb_vs_co_btn_sb(equity_matrix, combo_weights, strategies),
            -1.0,
            old["call_bb_vs_co_btn_sb"],
        )

        iterations += 1

        max_change = max(
            np.max(np.abs(strategies[k] - old[k])) for k in STRATEGY_NAMES
        )
        if max_change < tolerance:
            converged = True
            break

    # Build final EV table
    ev_table: dict[str, np.ndarray] = {
        "push_co":              ev_push_co(equity_matrix, combo_weights, strategies),
        "push_btn_open":        ev_push_btn_open(equity_matrix, combo_weights, strategies),
        "push_sb_open":         ev_push_sb_open(equity_matrix, combo_weights, strategies),
        "call_btn_vs_co":       ev_call_btn_vs_co(equity_matrix, combo_weights, strategies),
        "call_sb_vs_co":        ev_call_sb_vs_co(equity_matrix, combo_weights, strategies),
        "call_sb_vs_btn":       ev_call_sb_vs_btn(equity_matrix, combo_weights, strategies),
        "call_sb_vs_co_btn":    ev_call_sb_vs_co_btn(equity_matrix, combo_weights, strategies),
        "call_bb_vs_sb":        ev_call_bb_vs_sb(equity_matrix, combo_weights, strategies),
        "call_bb_vs_btn":       ev_call_bb_vs_btn(equity_matrix, combo_weights, strategies),
        "call_bb_vs_co":        ev_call_bb_vs_co(equity_matrix, combo_weights, strategies),
        "call_bb_vs_btn_sb":    ev_call_bb_vs_btn_sb(equity_matrix, combo_weights, strategies),
        "call_bb_vs_co_sb":     ev_call_bb_vs_co_sb(equity_matrix, combo_weights, strategies),
        "call_bb_vs_co_btn":    ev_call_bb_vs_co_btn(equity_matrix, combo_weights, strategies),
        "call_bb_vs_co_btn_sb": ev_call_bb_vs_co_btn_sb(equity_matrix, combo_weights, strategies),
    }

    return SolverResult(
        strategies=strategies,
        ev_table=ev_table,
        iterations=iterations,
        converged=converged,
        exploitability=compute_exploitability(strategies, equity_matrix, combo_weights),
    )
