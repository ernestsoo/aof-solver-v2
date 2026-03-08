"""Nash solver for 4-max All-in or Fold poker using Iterative Best Response (IBR).

All strategy arrays are shape (169,), float64, values 0.0–1.0.
All equity lookups use the precomputed 169x169 matrix — no Monte Carlo at solve time.
All EV computations are fully vectorized — no Python loops over hands.
"""

from dataclasses import dataclass, field

import numpy as np

from src.hands import COMBO_WEIGHTS, top_n_percent
from src.equity import hand_vs_range_equity


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
