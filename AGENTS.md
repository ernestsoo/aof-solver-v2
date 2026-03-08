# AoF Solver — Agent Task Definitions

## Overview
This document defines the tasks for OpenClaw dev agents to build the AoF Solver iteratively. Tasks are ordered by dependency — each task builds on the previous ones.

---

## Task 1: Hand Representations (`src/hands.py`)

### Goal
Build the foundational hand representation system that all other modules depend on.

### Deliverables
- `src/hands.py` with all exports listed below
- `tests/test_hands.py` with full coverage

### Requirements

**169 canonical hands:**
- 13 pairs: AA, KK, QQ, ..., 22
- 78 suited hands: AKs, AQs, ..., 32s
- 78 offsuit hands: AKo, AQo, ..., 32o
- Each hand has: `name` (str), `type` (pair/suited/offsuit), `combos` (int: 6/4/12)

**Hand ranking:**
- All 169 hands ranked 1-169 by pre-flop strength (standard ranking)
- Rank 1 = AA, Rank 2 = KK, etc.
- Include percentile: `percentile = rank / 169 * 100`

**Range operations:**
- `parse_range(notation: str) -> list[str]` — parse "22+, A2s+, KTo+" into list of hand names
- `range_to_hands(range_pct: float) -> list[str]` — top N% of hands by ranking
- `hands_to_range_pct(hands: list[str]) -> float` — what % of total combos a hand list represents
- `expand_plus_notation(hand: str) -> list[str]` — "ATs+" → ["ATs", "AJs", "AQs", "AKs"]

**Combo counting:**
- `combo_count(hand: str) -> int` — 6 for pairs, 4 for suited, 12 for offsuit
- `total_combos(hands: list[str]) -> int` — sum of combos for a list
- `combos_with_removal(hand: str, blocked_cards: list[str]) -> int` — combos accounting for card removal

**Grid mapping:**
- `hand_to_grid(hand: str) -> tuple[int, int]` — map hand to 13x13 grid position
- `grid_to_hand(row: int, col: int) -> str` — reverse mapping
- Grid layout: rows/cols indexed by rank (A=0, K=1, ..., 2=12), suited above diagonal, offsuit below, pairs on diagonal

### Tests
- Verify all 169 hands are generated with correct types and combos
- Total combos = 1326
- Range parsing: "AA" → ["AA"], "TT+" → ["TT","JJ","QQ","KK","AA"], "A2s+" → all suited aces
- Round-trip: parse → format → parse produces same result
- Grid mapping: all 169 hands map to unique grid positions
- Combo removal: AKs with As blocked = 3 combos

### Dependencies
None — this is the first task.

---

## Task 2: Equity Engine (`src/equity.py` + `scripts/generate_equities.py`)

### Goal
Precompute the 169x169 equity matrix and provide fast equity lookups.

### Deliverables
- `scripts/generate_equities.py` — generates `data/equity_matrix.npy`
- `src/equity.py` — loads matrix, provides lookup functions
- `tests/test_equity.py`
- `requirements.txt` (add eval7, numpy)

### Requirements

**Equity matrix generation (`scripts/generate_equities.py`):**
- For each pair of canonical hands (169x169 = 28,561 matchups):
  - Enumerate all possible specific card combos for both hands (accounting for card removal)
  - For each specific combo pair, evaluate equity by enumerating all 5-card boards (or Monte Carlo with 10k+ samples)
  - Weight by number of specific combos
  - Store result in matrix
- Use `eval7` for hand evaluation
- Save as `data/equity_matrix.npy` (numpy array, float32, shape 169x169)
- Print progress during generation (this takes minutes)
- Matrix property: `equity[i][j] + equity[j][i] ≈ 1.0` (verify this)
- Skip redundant computation: only compute upper triangle, mirror to lower

**Equity lookups (`src/equity.py`):**
- `load_equity_matrix() -> np.ndarray` — load from disk at module import
- `hand_vs_hand_equity(hand1: str, hand2: str) -> float` — lookup from matrix
- `hand_vs_range_equity(hand: str, range_hands: list[str]) -> float` — weighted average equity of hand vs all hands in range, weighted by combos (with card removal)
- `range_vs_range_equity(range1: list[str], range2: list[str]) -> float` — average equity of range1 vs range2

**Performance:**
- Matrix load: <100ms
- hand_vs_hand lookup: <1ms
- hand_vs_range (vs 100 hands): <10ms

### Tests
- Matrix is 169x169
- `equity[i][j] + equity[j][i]` ≈ 1.0 for all i,j (within 0.01)
- Known equities: AA vs KK ≈ 0.82, AKs vs QQ ≈ 0.46, AA vs 72o ≈ 0.88
- hand_vs_range with single hand in range equals hand_vs_hand
- Combo weighting: AKs (4 combos) weighted less than AKo (12 combos) in mixed range

### Dependencies
- Task 1 (hands.py) — needs hand names, rankings, combo counts

---

## Task 3: Nash Solver (`src/solver.py`)

### Goal
Compute Nash equilibrium push/call ranges for all positions.

### Deliverables
- `src/solver.py`
- `tests/test_solver.py`

### Requirements

**Game tree for 4-max AoF (10bb, 0.5/1 blinds):**

The pot starts with 1.5bb (0.5 SB + 1.0 BB). A push is 10bb.

Decision nodes (in order):
1. **CO**: push (10bb) or fold
2. **BTN**:
   - If CO pushed: call (10bb) or fold
   - If CO folded: push (10bb) or fold
3. **SB**:
   - Facing one or more pushes: call (10bb, but only risking 9.5bb more since 0.5bb posted) or fold
   - If all folded to SB: push (10bb, risking 9.5bb more) or fold
4. **BB**:
   - Facing one or more pushes: call (risking 9bb more since 1bb posted) or fold
   - If all folded to BB: wins pot (check, no decision needed)

**EV calculations:**

For a **pusher** (first to go all-in):
```
EV(push) = P(everyone folds) * current_pot + P(called) * [equity * total_pot - push_amount]
EV(fold) = 0  (for CO/BTN) or -blind_posted (for SB/BB)
Push if EV(push) > EV(fold)
```

For a **caller** (facing a push):
```
EV(call) = equity_vs_pusher_range * total_pot - call_cost
EV(fold) = 0 (for BTN) or -blind_posted (for SB/BB, already lost)
Call if EV(call) > EV(fold)
```

Multi-way pots (multiple pushers): calculate equity vs combined ranges.

**Solver algorithm — iterative best-response:**
1. Initialize: all positions push top 50%, call top 30%
2. For each position in order (CO → BTN → SB → BB):
   - Compute EV(push) and EV(fold) for each of 169 hands, given current other strategies
   - Set push range = all hands where EV(push) > EV(fold)
   - Similarly compute call ranges vs each possible push scenario
3. Repeat step 2 until convergence: no hand changes push/call status between iterations
4. Cap at 1000 iterations (should converge in <50)

**Output format:**
```python
@dataclass
class SolverResult:
    push_ranges: dict[str, list[str]]     # position → list of hands to push
    call_ranges: dict[str, dict[str, list[str]]]  # position → {scenario: list of hands to call}
    ev_table: dict[str, dict[str, float]] # position → {hand: EV of pushing}
    iterations: int                        # iterations to converge
    converged: bool
```

Call range scenarios:
- BTN: "vs_co_push"
- SB: "vs_co_push", "vs_btn_push", "vs_co_btn_push", "open_push"
- BB: "vs_co_push", "vs_btn_push", "vs_sb_push", "vs_co_btn_push", "vs_co_sb_push", "vs_btn_sb_push", "vs_co_btn_sb_push"

**Performance target:** Full solve in <1 second.

### Tests
- Solver converges (converged=True) within 1000 iterations
- Push ranges are reasonable: AA always pushed from every position, 72o never pushed
- CO range is tightest, BB calling range is widest (getting best odds)
- Symmetry: if we swap two equal positions, ranges should be same
- Known approximate results: CO push ~28%, BTN push ~42%, SB push ~52%, BB call vs single push ~55% (these are approximate — verify against published Nash charts)
- EV of AA > EV of KK > EV of QQ from any position

### Dependencies
- Task 1 (hands.py) — hand list, combos
- Task 2 (equity.py) — hand-vs-range equity lookups

---

## Task 4: Nodelocking (`src/nodelock.py`)

### Goal
Allow fixing player strategies and solving exploitative responses.

### Deliverables
- `src/nodelock.py`
- `tests/test_nodelock.py`

### Requirements

**Core function:**
```python
def nodelock_solve(
    locked_ranges: dict[str, list[str]],  # position → locked push range (hands)
    locked_call_ranges: dict[str, dict[str, list[str]]] = None,  # optional locked call ranges
) -> SolverResult:
```

- Same iterative best-response as Nash solver
- But skip re-solving locked positions — their ranges stay fixed
- Only optimize unlocked positions against the locked strategies

**Use cases:**
1. Lock villain BTN push range to 60%, solve optimal BB call range
2. Lock all opponents to observed frequencies, solve hero's optimal strategy
3. Lock one player tight (15%), see how it affects other positions

**Exploitability metric:**
- After nodelocked solve, compute how much EV each locked player loses vs the exploitative response compared to Nash
- `exploitability(position) = EV_at_nash - EV_at_nodelocked` (in bb)

### Tests
- Locking all positions to Nash ranges and solving returns Nash (no change)
- Locking BTN to wider range → BB call range widens
- Locking BTN to tighter range → BB call range tightens
- Locking CO to 100% push → BTN/SB/BB all call wider
- Exploitability of Nash-locked positions = 0
- Exploitability of wildly off-Nash ranges > 0

### Dependencies
- Task 3 (solver.py) — reuses solver algorithm
- Task 1, Task 2

---

## Task 5: Dashboard (`src/dashboard.py` + `templates/index.html`)

### Goal
Web UI to visualize solver results and perform nodelocking interactively.

### Deliverables
- `src/dashboard.py` — Flask app with API endpoints
- `templates/index.html` — single-page dashboard
- `static/` directory if needed for CSS/JS

### Requirements

**API Endpoints:**

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/api/solve` | GET | — | Nash solution: all push/call ranges, EVs |
| `/api/nodelock` | POST | `{locks: {position: range_pct_or_hands}}` | Nodelocked solution |
| `/api/hand_equity` | GET | `?hand=AKs&range=top30` | Equity of hand vs range |
| `/api/hand_info` | GET | `?hand=AKs` | Rank, percentile, type, combos |

**Dashboard Layout (single page, no navigation):**

**Top Section — Nash Solution:**
- 4 hand grids side by side (CO / BTN / SB / BB)
- Each grid is 13x13 showing push range
- Color: green = push, red = fold, intensity = EV magnitude
- Hover tooltip: hand name, EV(push), EV(fold), equity vs callers
- Below each grid: "Push X% (Y combos)" summary

**Middle Section — Call Ranges:**
- Dropdown to select scenario (e.g., "BTN vs CO push")
- Hand grid showing call range for selected scenario
- Hover: hand, EV(call), EV(fold), equity vs pusher range

**Bottom Section — Nodelock Panel:**
- 4 columns (CO / BTN / SB / BB)
- Each column has:
  - Range slider (0-100%)
  - Text input for custom range notation
  - Lock checkbox (checked = this position is locked)
  - Current range display (% and combo count)
- "Solve Exploitative" button
- Results: updated grids showing exploitative ranges vs Nash ranges
- Side-by-side comparison: Nash grid | Exploitative grid
- Difference highlighting: hands that changed from push→fold (red border) or fold→push (green border)

**Styling:**
- Dark theme (matches poker aesthetic)
- Same vanilla JS approach as poker_bot dashboard (no React/Vue)
- Responsive grid layout
- Hand grid cells: ~30x30px, monospace font, clear borders

### Tests
- API endpoints return valid JSON
- `/api/solve` returns all 4 positions with push ranges
- `/api/nodelock` with no locks returns same as `/api/solve`
- Dashboard loads without JS errors (manual test)

### Dependencies
- All previous tasks (1-4)

---

## Task Order & Parallelism

```
Task 1 (hands.py)          ← START HERE, no dependencies
    ↓
Task 2 (equity.py)         ← depends on Task 1
    ↓
Task 3 (solver.py)         ← depends on Task 1 + 2
    ↓
Task 4 (nodelock.py)       ← depends on Task 3
    ↓
Task 5 (dashboard.py)      ← depends on all above
```

Tasks are strictly sequential. Each task should be fully tested before moving to the next.

---

## Agent Instructions

### General rules
- Read `CLAUDE.md` first for project context and architecture
- Run `pytest tests/` after completing each task — all tests must pass
- Follow existing code patterns from previous tasks
- Keep functions focused and well-typed (use type hints)
- No unnecessary dependencies — only add what's needed
- Performance matters — this runs in real-time during poker play

### Per-task checklist
1. Read this file and `CLAUDE.md`
2. Read any dependency modules (previous tasks) to understand interfaces
3. Implement the module
4. Write tests
5. Run tests, fix failures
6. Verify no regressions in previous tests (`pytest tests/`)

### Code style
- Python 3.11+
- Type hints on all public functions
- Docstrings on public functions (one-liner is fine)
- No classes unless genuinely needed — prefer functions and dataclasses
- Constants at module top
- Imports: stdlib → third-party → local (separated by blank lines)
