# AoF Solver — All-in or Fold 4-max Push/Fold Solver

## What this is
Real-time Nash equilibrium solver for 4-max All-in or Fold poker. Computes optimal push and call ranges per position, supports nodelocking (fixing opponent strategies and solving exploitative responses). Includes a web dashboard for visualization.

---

## Game Rules — All-in or Fold (AoF)

### Setup
- **Players:** 4 (CO, BTN, SB, BB) — always exactly 4
- **Stacks:** Fixed 10bb for all players, every hand (stacks reset)
- **Blinds:** SB posts 0.5bb, BB posts 1.0bb
- **Antes:** None
- **Action:** Each player can only **shove all-in (10bb)** or **fold** — no other bet sizes, no limping, no raising

### Action Order
1. **CO** (Cutoff) acts first: push or fold
2. **BTN** (Button): if no one has pushed → push or fold; if someone pushed → call or fold
3. **SB** (Small Blind): same logic — open push if folded to, otherwise call or fold
4. **BB** (Big Blind): same logic — if everyone folds, BB wins the blinds uncontested

**Key rule:** Once any player pushes all-in, all subsequent players can only CALL (match the all-in) or FOLD. There is no re-raise because the push is already all-in.

### Pot Mathematics

All amounts in big blinds (bb). Stack = 10bb for everyone.

**Blind contributions (already in pot before action):**
- SB: 0.5bb
- BB: 1.0bb
- Starting pot: 1.5bb

**Risk calculations (additional chips a player puts in):**
| Player | When pushing | When calling | Already posted |
|--------|-------------|-------------|----------------|
| CO     | 10bb        | 10bb        | 0              |
| BTN    | 10bb        | 10bb        | 0              |
| SB     | 9.5bb more  | 9.5bb more  | 0.5bb          |
| BB     | 9.0bb more  | 9.0bb more  | 1.0bb          |

**Total contribution when in the pot:**
- CO/BTN: 10bb (all new chips)
- SB: 10bb total (0.5bb already posted + 9.5bb more)
- BB: 10bb total (1.0bb already posted + 9.0bb more)

---

## Complete Game Tree — All 16 Terminal Nodes

Every hand reaches exactly one of these 16 outcomes. The "dead money" column shows blinds posted by players who folded (this money is in the pot but those players have no equity).

### Terminal Nodes

| # | Action Sequence | Result | Pot (bb) | Dead Money | Notes |
|---|----------------|--------|----------|------------|-------|
| 1 | CO folds → BTN folds → SB folds | BB wins uncontested | 1.5 | SB: 0.5 | BB profits +0.5bb |
| 2 | CO folds → BTN folds → SB pushes → BB folds | SB steals | 1.5 | BB: 1.0 | SB profits +1.0bb |
| 3 | CO folds → BTN folds → SB pushes → BB calls | SB vs BB showdown | 20.0 | — | Heads-up, no dead money |
| 4 | CO folds → BTN pushes → SB folds → BB folds | BTN steals | 1.5 | SB: 0.5, BB: 1.0 | BTN profits +1.5bb |
| 5 | CO folds → BTN pushes → SB folds → BB calls | BTN vs BB showdown | 20.5 | SB: 0.5 | Heads-up, SB dead |
| 6 | CO folds → BTN pushes → SB calls → BB folds | BTN vs SB showdown | 21.0 | BB: 1.0 | Heads-up, BB dead |
| 7 | CO folds → BTN pushes → SB calls → BB calls | BTN vs SB vs BB | 30.0 | — | 3-way |
| 8 | CO pushes → BTN folds → SB folds → BB folds | CO steals | 1.5 | SB: 0.5, BB: 1.0 | CO profits +1.5bb |
| 9 | CO pushes → BTN folds → SB folds → BB calls | CO vs BB showdown | 20.5 | SB: 0.5 | Heads-up, SB dead |
| 10 | CO pushes → BTN folds → SB calls → BB folds | CO vs SB showdown | 21.0 | BB: 1.0 | Heads-up, BB dead |
| 11 | CO pushes → BTN folds → SB calls → BB calls | CO vs SB vs BB | 30.0 | — | 3-way (BTN folded) |
| 12 | CO pushes → BTN calls → SB folds → BB folds | CO vs BTN showdown | 21.5 | SB: 0.5, BB: 1.0 | Heads-up, both blinds dead |
| 13 | CO pushes → BTN calls → SB folds → BB calls | CO vs BTN vs BB | 30.5 | SB: 0.5 | 3-way, SB dead |
| 14 | CO pushes → BTN calls → SB calls → BB folds | CO vs BTN vs SB | 31.0 | BB: 1.0 | 3-way, BB dead |
| 15 | CO pushes → BTN calls → SB calls → BB calls | CO vs BTN vs SB vs BB | 40.0 | — | 4-way |

**Pot formula:** pot = (10bb × number_of_players_in) + dead_blinds_from_folders

---

## Decision Points Per Position

Each position faces different decision types depending on prior action.

### CO (first to act)
- **Always:** open push or fold
- **Strategies:** `push_co[h]` = probability of pushing hand h (0.0 or 1.0 in pure strategy)

### BTN
- **If CO folded:** open push or fold → `push_btn_open[h]`
- **If CO pushed:** call or fold → `call_btn_vs_co[h]`

### SB
- **If all fold to SB:** open push or fold → `push_sb_open[h]`
- **If CO pushed, BTN folded:** call or fold → `call_sb_vs_co[h]`
- **If BTN pushed (CO folded):** call or fold → `call_sb_vs_btn[h]`
- **If CO pushed, BTN called:** call or fold → `call_sb_vs_co_btn[h]`

### BB
- **If all fold to BB:** wins uncontested (no decision needed)
- **If SB pushed (others folded):** call or fold → `call_bb_vs_sb[h]`
- **If BTN pushed (CO folded, SB folded):** call or fold → `call_bb_vs_btn[h]`
- **If CO pushed (BTN folded, SB folded):** call or fold → `call_bb_vs_co[h]`
- **If BTN pushed, SB called (CO folded):** call or fold → `call_bb_vs_btn_sb[h]`
- **If CO pushed, SB called (BTN folded):** call or fold → `call_bb_vs_co_sb[h]`
- **If CO pushed, BTN called (SB folded):** call or fold → `call_bb_vs_co_btn[h]`
- **If CO pushed, BTN called, SB called:** call or fold → `call_bb_vs_co_btn_sb[h]`

### Total Strategy Arrays
- **Push strategies:** 3 arrays (CO, BTN open, SB open)
- **Call strategies:** 11 arrays (BTN: 1, SB: 3, BB: 7)
- **Total: 14 strategy arrays**, each shape (169,)

---

## EV Formulas

All EVs are in big blinds (bb) from the perspective of the acting player. Positive = profitable.

### EV Convention

Use **net profit from start of hand** consistently:

```
EV = (equity x pot) - total_chips_put_in

Where total_chips_put_in includes blinds already posted.
For each terminal:
  Player in the pot:     EV = equity x pot - 10
  Player who folded:     EV = -blind_posted (0 for CO/BTN, -0.5 for SB, -1.0 for BB)
  BB wins uncontested:   EV = +0.5 (wins SB's blind)
```

### Notation
- `eq(h, R)` = equity of hand h against range R (from 169x169 matrix, combo-weighted)
- `eq3(h, R1, R2)` = equity of hand h in 3-way pot (see Multiway Equity section)
- `f_x` = probability opponent x folds (combo-weighted across their range)
- `c_x` = probability opponent x calls = 1 - f_x
- All strategies are (169,) numpy arrays with values 0.0 to 1.0

### CO — Open Push

EV(fold) = 0

EV(push, h) = sum over all response scenarios:
```
  # Everyone folds (Terminal 8): CO wins 1.5bb
  + f_btn x f_sb|co x f_bb|co x 1.5

  # Only BB calls (Terminal 9): HU, pot=20.5
  + f_btn x f_sb|co x c_bb|co x (eq(h, bb_call_range|co) x 20.5 - 10)

  # Only SB calls (Terminal 10): HU, pot=21
  + f_btn x c_sb|co x f_bb|co_sb x (eq(h, sb_call_range|co) x 21.0 - 10)

  # SB and BB call (Terminal 11): 3-way, pot=30
  + f_btn x c_sb|co x c_bb|co_sb x (eq3(h, sb_range, bb_range) x 30.0 - 10)

  # BTN calls, SB+BB fold (Terminal 12): HU, pot=21.5
  + c_btn x f_sb|co_btn x f_bb|co_btn x (eq(h, btn_call_range) x 21.5 - 10)

  # BTN+BB call (Terminal 13): 3-way, pot=30.5
  + c_btn x f_sb|co_btn x c_bb|co_btn x (eq3(h, btn_range, bb_range) x 30.5 - 10)

  # BTN+SB call (Terminal 14): 3-way, pot=31
  + c_btn x c_sb|co_btn x f_bb|co_btn_sb x (eq3(h, btn_range, sb_range) x 31.0 - 10)

  # All call (Terminal 15): 4-way, pot=40
  + c_btn x c_sb|co_btn x c_bb|co_btn_sb x (eq4(h, btn_r, sb_r, bb_r) x 40.0 - 10)
```

### BTN — Open Push (CO folded)

EV(fold) = 0

EV(push, h):
```
  + f_sb x f_bb|btn x 1.5                              # Terminal 4: steal
  + f_sb x c_bb|btn x (eq(h, bb_range) x 20.5 - 10)    # Terminal 5: HU vs BB
  + c_sb x f_bb|btn_sb x (eq(h, sb_range) x 21.0 - 10)  # Terminal 6: HU vs SB
  + c_sb x c_bb|btn_sb x (eq3(h, sb_r, bb_r) x 30.0 - 10) # Terminal 7: 3-way
```

### BTN — Call vs CO Push

EV(fold) = 0

EV(call, h):
```
  + f_sb x f_bb x (eq(h, co_range) x 21.5 - 10)           # Terminal 12: HU vs CO
  + f_sb x c_bb x (eq3(h, co_range, bb_range) x 30.5 - 10) # Terminal 13: 3-way
  + c_sb x f_bb x (eq3(h, co_range, sb_range) x 31.0 - 10) # Terminal 14: 3-way
  + c_sb x c_bb x (eq4(h, co_r, sb_r, bb_r) x 40.0 - 10)  # Terminal 15: 4-way
```

### SB — Open Push (CO+BTN folded)

EV(fold) = -0.5

EV(push, h):
```
  + f_bb x 1.0                                    # Terminal 2: steal (net +1.0)
  + c_bb x (eq(h, bb_range) x 20.0 - 10.0)        # Terminal 3: HU vs BB
```
Note: SB risked 10 total (0.5 already posted + 9.5 more). Net EV from start of hand.

### SB — Call Decisions

SB facing a push (already posted 0.5bb):

EV(fold) = -0.5

EV(call vs CO push, BTN folded, h):
```
  + f_bb x (eq(h, co_range) x 21.0 - 10)         # Terminal 10: HU vs CO, BB dead
  + c_bb x (eq3(h, co_range, bb_range) x 30.0 - 10) # Terminal 11: 3-way
```

### BB — Call Decisions

BB always has 1.0bb already posted.

EV(fold) = -1.0

**BB vs SB push (Terminal 3):**
```
  EV(call) = eq(h, sb_push_range) x 20.0 - 10.0
```

**BB vs CO push only (Terminal 9):**
```
  EV(call) = eq(h, co_push_range) x 20.5 - 10.0
```
(pot 20.5 because SB's 0.5 is dead money)

**BB vs BTN push only (Terminal 5):**
```
  EV(call) = eq(h, btn_push_range) x 20.5 - 10.0
```

**BB vs CO+BTN (Terminal 13, 3-way):**
```
  EV(call) = eq3(h, co_range, btn_range) x 30.5 - 10.0
```

**BB vs CO+SB (Terminal 11, 3-way):**
```
  EV(call) = eq3(h, co_range, sb_range) x 30.0 - 10.0
```

**BB vs CO+BTN+SB (Terminal 15, 4-way):**
```
  EV(call) = eq4(h, co_range, btn_range, sb_range) x 40.0 - 10.0
```

---

## Multiway Equity — Pairwise Independence Approximation

### The Problem
The 169x169 equity matrix only gives heads-up equity. For 3-way and 4-way pots, we need multiway equity. Computing an exact 169x169x169 tensor would require ~4.8M entries x MC simulation = infeasible on a 2-vCPU VPS.

### The Solution

For hand h vs specific opponent hands h1, h2 in a 3-way pot:

```python
def eq3_approx(h, h1, h2, matrix):
    """Approximate 3-way equity from pairwise matrix."""
    p_h  = matrix[h][h1] * matrix[h][h2]
    p_h1 = matrix[h1][h] * matrix[h1][h2]
    p_h2 = matrix[h2][h] * matrix[h2][h1]
    total = p_h + p_h1 + p_h2
    if total == 0:
        return 1/3
    return p_h / total
```

For hand h vs ranges (vectorized):

```python
def eq3_vs_ranges(h_idx, range1, range2, combo_weights, matrix):
    """Approximate 3-way equity of hand h vs two ranges."""
    eq_vs_r1 = hand_vs_range_equity(h_idx, range1, combo_weights, matrix)
    eq_vs_r2 = hand_vs_range_equity(h_idx, range2, combo_weights, matrix)
    raw = eq_vs_r1 * eq_vs_r2
    return raw / (raw + (1-eq_vs_r1)*eq_vs_r2 + eq_vs_r1*(1-eq_vs_r2) + 1e-10)
```

**4-way:** Same principle with one more factor.

### Accuracy
- Standard approximation used in push/fold calculators
- Error typically < 2% equity, acceptable for Nash convergence
- Main inaccuracy: doesn't account for card removal between opponents
- Good enough for v2. V3 could precompute 3-way equities offline.

---

## Iterative Best Response (IBR) Algorithm

### Overview
IBR finds Nash equilibrium by iteratively computing the best response for each position while holding others fixed. For push/fold games with finite actions, IBR converges to Nash.

### Algorithm

```
Input: equity_matrix (169x169), combo_weights (169,)
Output: SolverResult with all push/call strategies

1. INITIALIZE
   push_co = top_n_percent(30%)
   push_btn_open = top_n_percent(40%)
   push_sb_open = top_n_percent(50%)
   call_* = top_n_percent(20%)

2. ITERATE (max 500 iterations)
   For each position in order [CO, BTN, SB, BB]:
     a. Compute EV(push/call, h) for ALL 169 hands at once (vectorized)
     b. Compute EV(fold, h)
     c. new_strategy = (ev_action > ev_fold).astype(float)
     d. Update this position's strategy

   CONVERGENCE: max |new - old| < 0.001 across all strategies

3. COMPUTE EXPLOITABILITY
   Sum of (EV_best_response - EV_current) across positions

4. RETURN SolverResult
```

### Vectorized Best Response (example for CO)

```python
# Probability BTN calls (scalar)
p_btn_calls = np.dot(call_btn_vs_co, combo_weights) / combo_weights.sum()

# Equity of each CO hand vs BTN calling range (169,) array
eq_vs_btn = (matrix @ (call_btn_vs_co * combo_weights)) / np.dot(call_btn_vs_co, combo_weights)

# EV for each hand pushing (169,) array
ev_push = (
    p_all_fold * 1.5 +
    p_only_bb_calls * (eq_vs_bb * 20.5 - 10) +
    # ... etc for all 8 scenarios
)

# Best response
new_push_co = (ev_push > 0).astype(float)
```

**Critical:** Every line operates on (169,) arrays. No Python for-loops.

### Convergence Notes
- Typical: 20-100 iterations
- Borderline hands may oscillate: use damping (90% new + 10% old)
- Performance target: < 5 seconds on 2-vCPU

---

## Data Structures

### Core Arrays (all numpy)

```python
ALL_HANDS: list[HandInfo]          # 169 entries, ordered by index
HAND_MAP: dict[str, HandInfo]      # name -> HandInfo
COMBO_WEIGHTS: np.ndarray          # shape (169,), values: 6/4/12, sum=1326

equity_matrix: np.ndarray          # shape (169, 169), float32
                                    # equity_matrix[i][j] + equity_matrix[j][i] ~ 1.0

# All strategy arrays: shape (169,), float64, values 0.0 to 1.0
# Push: push_co, push_btn_open, push_sb_open (3)
# Call: call_btn_vs_co, call_sb_vs_co, call_sb_vs_btn, call_sb_vs_co_btn,
#       call_bb_vs_sb, call_bb_vs_btn, call_bb_vs_co, call_bb_vs_btn_sb,
#       call_bb_vs_co_sb, call_bb_vs_co_btn, call_bb_vs_co_btn_sb (11)

@dataclass
class SolverResult:
    push_strategies: dict[str, np.ndarray]
    call_strategies: dict[str, dict[str, np.ndarray]]
    ev_table: dict[str, np.ndarray]    # position -> EV per hand (169,)
    iterations: int
    converged: bool
    exploitability: float              # in bb

@dataclass
class HandInfo:
    name: str           # "AKs", "TT", "87o"
    index: int          # 0-168
    hand_type: str      # "pair", "suited", "offsuit"
    combos: int         # 6, 4, or 12
    rank: int           # 1-169 (1=AA, 169=worst)
    rank1: int          # first card rank (0=A, 12=2)
    rank2: int          # second card rank
```

---

## Equity Matrix Generation

For each of 169x169 = 28,561 canonical hand matchups:
1. Enumerate non-conflicting specific card combos of hand i and hand j
2. For each matchup: Monte Carlo with N=5000-10000 random boards using eval7
3. Average equity across combos
4. Upper triangle only: matrix[j][i] = 1.0 - matrix[i][j]
5. Save to data/equity_matrix.npy (169x169, float32, ~114KB)

**Validation targets:**
- AA vs KK ~ 0.82
- AA vs 72o ~ 0.87
- KK vs AKs ~ 0.66
- 22 vs AKo ~ 0.52
- matrix[i][j] + matrix[j][i] ~ 1.0 (tolerance 0.02)

**Runtime:** 10-30 min on 2-vCPU. DO NOT RUN during agent sessions.

---

## Validation — Expected Nash Ranges

### Approximate Push Ranges (10bb, 4-max, no antes)
| Position | Push % | Range |
|----------|--------|-------|
| CO | 15-20% | Pairs 55+, suited Ax, KTs+, suited connectors |
| BTN open | 30-40% | Pairs 22+, most suited Ax, suited broadways, KTo+ |
| SB open | 50-65% | Very wide: most pairs, most suited, many offsuit |

### Approximate Call Ranges
| Scenario | Call % | Range |
|----------|--------|-------|
| BTN vs CO | 10-15% | 77+, ATs+, AJo+, KQs |
| BB vs SB | 30-40% | Wide: most pairs, suited aces, broadways |
| BB vs CO | 8-12% | TT+, AQs+, AKo |

### Sanity Checks
- AA, KK always pushed from every position
- 72o never pushed
- CO range < BTN open < SB open (wider by position)
- Call ranges tighten with more players in pot
- Exploitability < 0.1bb

---

## Project Structure

```
aof-solver-v2/
|-- src/
|   |-- __init__.py
|   |-- hands.py           # Hand representations, rankings, range parsing
|   |-- equity.py          # Equity matrix loading and lookup
|   |-- solver.py          # Nash solver (IBR algorithm)
|   |-- nodelock.py        # Fix ranges, re-solve exploitative
|   |-- dashboard.py       # Flask app + API endpoints
|-- data/
|   |-- equity_matrix.npy  # Precomputed 169x169 (not in git)
|-- templates/
|   |-- index.html         # Dashboard UI
|-- scripts/
|   |-- generate_equities.py
|-- tests/
|   |-- __init__.py
|   |-- conftest.py
|   |-- fixtures/
|   |   |-- tiny_equity.npy
|   |-- test_hands.py
|   |-- test_equity.py
|   |-- test_solver.py
|   |-- test_nodelock.py
|-- requirements.txt
|-- CLAUDE.md
|-- AGENTS.md
|-- PROGRESS.md
```

## Dependencies

```
eval7    # Fast poker hand eval (C extension) - equity generation only
numpy    # Matrix operations - core
flask    # Web dashboard
pytest   # Testing
```

## Resource Constraints

- 2-vCPU, 4GB RAM VPS
- Equity generation: offline, 10-30 min
- Solver: must complete in < 5 seconds
- No Monte Carlo at solve time
- No Python for-loops over 169+ elements
- Use numpy vectorization everywhere

## Future: Integration with poker_bot

This solver connects to Ernest's aof-bot (GGPoker screen capture bot):
1. aof-bot tracks opponent push/call frequencies (SQLite DB)
2. Detect deviations from Nash
3. Feed into solver's nodelock
4. Re-solve exploitatively
5. Push updated ranges to bot
6. Import: `from src.solver import solve_nash` / `from src.nodelock import nodelock_solve`
