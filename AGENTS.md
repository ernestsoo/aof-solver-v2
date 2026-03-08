# AoF Solver — Agent Task Definitions

## IMPORTANT — Read First
1. **Read `CLAUDE.md`** before doing anything — it has all project context
2. **Check `PROGRESS.md`** to see what's done and find your next task
3. **Update `PROGRESS.md`** immediately when you start and finish a task
4. **Update this file** (AGENTS.md) with any notes, issues, or changes to the plan
5. **DO NOT run long-running tests** (equity generation, Monte Carlo, etc.) — this runs on a 4GB CPU. Write the code and tests, but leave heavy computation for the human to run manually. Mark those tasks as "needs human testing" in PROGRESS.md
6. **One task per agent run** — pick the next unchecked task, do it, update progress, done

---

## Progress Tracking

All progress is tracked in `PROGRESS.md`. Check it before starting. Update it when done.

Format in PROGRESS.md:
- `[ ]` = not started
- `[x]` = complete
- `[!]` = needs human testing (heavy compute)
- `[~]` = in progress (agent working on it)

---

## Tasks

### Phase 1: Hand Representations (`src/hands.py`)

**1.1 — Hand constants and generation**
- Create `src/hands.py`
- Define `RANKS = ['A','K','Q','J','T','9','8','7','6','5','4','3','2']`
- Generate all 169 canonical hands: 13 pairs, 78 suited, 78 offsuit
- Each hand as a dict or namedtuple: `name`, `type` (pair/suited/offsuit), `combos` (6/4/12)
- Store as `ALL_HANDS` list and `HAND_MAP` dict (name → hand info)
- Verify total combos = 1326

**1.2 — Hand ranking**
- Add `HAND_RANKINGS` — all 169 hands ranked 1-169 by standard pre-flop strength
- Rank 1 = AA, 2 = KK, 3 = QQ, ..., 169 = worst
- Use well-known standard ranking (look up if needed, hard-code the order)
- Add `hand_rank(hand: str) -> int` and `hand_percentile(hand: str) -> float`
- Percentile = rank position as % of total combos (weighted by combo count)

**1.3 — Range parsing and operations**
- `parse_range(notation: str) -> list[str]` — parse "22+, A2s+, KTo+" into hand list
- `expand_plus_notation(hand: str) -> list[str]` — "ATs+" → ["ATs","AJs","AQs","AKs"]
- `range_to_hands(range_pct: float) -> list[str]` — top N% of hands by combo-weighted ranking
- `hands_to_range_pct(hands: list[str]) -> float` — combo % of 1326 total

**1.4 — Grid mapping**
- `hand_to_grid(hand: str) -> tuple[int, int]` — map to 13x13 grid position
- `grid_to_hand(row: int, col: int) -> str` — reverse mapping
- Grid: rows/cols by rank (A=0..2=12), suited above diagonal, offsuit below, pairs on diagonal
- All 169 hands must map to unique positions

**1.5 — Combo counting with removal**
- `combo_count(hand: str) -> int` — basic: 6/4/12
- `total_combos(hands: list[str]) -> int` — sum combos
- `combos_with_removal(hand: str, blocked_cards: list[str]) -> int` — account for dead cards

**1.6 — Tests for hands.py**
- Create `tests/test_hands.py`
- Test: 169 hands generated, total combos = 1326
- Test: ranking order (AA=1, KK=2, etc.)
- Test: parse_range("TT+") → ["TT","JJ","QQ","KK","AA"]
- Test: parse_range("A2s+") → all 12 suited aces
- Test: grid round-trip for all 169 hands
- Test: combos_with_removal("AKs", ["As"]) = 3
- **Run `pytest tests/test_hands.py`** — this is lightweight, OK to run

---

### Phase 2: Equity Engine (`src/equity.py`)

**2.1 — Equity matrix generator script**
- Create `scripts/generate_equities.py`
- Use `eval7` to compute equity for each of 169x169 canonical hand matchups
- For each matchup: enumerate specific card combos, run Monte Carlo (10k samples per matchup)
- Only compute upper triangle, mirror to lower
- Save to `data/equity_matrix.npy` as float32
- Print progress (current matchup / total)
- **DO NOT RUN THIS** — just write the script. Mark as "needs human testing"

**2.2 — Equity lookup module**
- Create `src/equity.py`
- `load_equity_matrix(path: str = "data/equity_matrix.npy") -> np.ndarray`
- `hand_vs_hand_equity(hand1: str, hand2: str) -> float` — matrix lookup
- `hand_vs_range_equity(hand: str, range_hands: list[str]) -> float` — weighted avg by combos with card removal
- `range_vs_range_equity(range1: list[str], range2: list[str]) -> float`
- Handle missing matrix file gracefully (clear error message)

**2.3 — Tests for equity.py**
- Create `tests/test_equity.py`
- Test: matrix shape is 169x169 (skip if matrix file doesn't exist)
- Test: `equity[i][j] + equity[j][i]` ≈ 1.0 (tolerance 0.02)
- Test: known equities AA vs KK ≈ 0.82 (tolerance 0.03)
- Test: hand_vs_range with single hand equals hand_vs_hand
- **DO NOT RUN** equity generation or heavy tests. Mark as "needs human testing"

---

### Phase 3: Nash Solver (`src/solver.py`)

**3.1 — EV calculation functions**
- Create `src/solver.py`
- `ev_push(hand, position, push_ranges, call_ranges, equity_fn) -> float`
- `ev_call(hand, position, pusher_range, equity_fn) -> float`
- Handle pot math correctly:
  - Pot starts at 1.5bb (0.5 SB + 1.0 BB)
  - Push = 10bb total
  - SB risking 9.5bb more (already posted 0.5)
  - BB risking 9bb more (already posted 1.0)
- See CLAUDE.md for full EV formulas

**3.2 — Iterative best-response solver**
- `solve_nash() -> SolverResult` — main solver function
- SolverResult dataclass: push_ranges, call_ranges, ev_table, iterations, converged
- Algorithm: iterate CO→BTN→SB→BB, compute best response for each, repeat until stable
- Initialize with reasonable starting ranges
- Convergence: no hand changes status between iterations, or cap at 1000
- Target: <1 second solve time

**3.3 — All call range scenarios**
- BTN: "vs_co_push" (call range), "open" (push range when CO folded)
- SB: "vs_co_push", "vs_btn_push", "vs_co_btn_push", "open_push"
- BB: "vs_co_push", "vs_btn_push", "vs_sb_push", "vs_co_btn_push", "vs_co_sb_push", "vs_btn_sb_push", "vs_co_btn_sb_push"
- Each scenario needs correct pot size and call cost calculation

**3.4 — Tests for solver.py**
- Create `tests/test_solver.py`
- Test: solver converges
- Test: AA always in push range for all positions
- Test: 72o never in push range
- Test: CO range tightest, ranges widen by position
- Test: EV(AA) > EV(KK) > EV(QQ)
- **DO NOT RUN if equity matrix not present** — tests should skip gracefully with `pytest.mark.skipif`

---

### Phase 4: Nodelocking (`src/nodelock.py`)

**4.1 — Nodelock solver**
- Create `src/nodelock.py`
- `nodelock_solve(locked_ranges: dict, locked_call_ranges: dict = None) -> SolverResult`
- Same algorithm as Nash but skip locked positions during iteration
- Accept locked ranges as hand lists or range percentages

**4.2 — Exploitability metric**
- `exploitability(position: str, nash_result: SolverResult, locked_result: SolverResult) -> float`
- Returns EV difference in bb between Nash play and nodelocked play for a position
- Positive = locked player is losing EV

**4.3 — Tests for nodelock.py**
- Create `tests/test_nodelock.py`
- Test: locking all to Nash returns Nash (no change)
- Test: wider lock → wider call ranges
- Test: tighter lock → tighter call ranges
- **Same skip logic as solver tests if matrix not present**

---

### Phase 5: Dashboard — Backend (`src/dashboard.py`)

**5.1 — Flask app and solve endpoint**
- Create `src/dashboard.py`
- Flask app with `/api/solve` GET endpoint
- Returns JSON: push_ranges, call_ranges, ev_table per position
- Precompute Nash solution at startup (cache it)

**5.2 — Nodelock endpoint**
- `/api/nodelock` POST endpoint
- Body: `{locks: {"BTN": 45, "CO": "22+,A2s+"}}` — accepts % or notation
- Returns nodelocked solution JSON

**5.3 — Utility endpoints**
- `/api/hand_equity?hand=AKs&vs=top30` — equity lookup
- `/api/hand_info?hand=AKs` — rank, percentile, combos, type

---

### Phase 6: Dashboard — Frontend (`templates/index.html`)

**6.1 — Page layout and Nash grids**
- Create `templates/index.html`
- Dark theme, single page
- Top section: 4 hand grids side by side (CO/BTN/SB/BB)
- Each grid 13x13 showing push range
- Color: green = push, red = fold, intensity by EV
- Below each grid: "Push X% (Y combos)" summary
- Fetch from `/api/solve` on page load

**6.2 — Call range viewer**
- Middle section: dropdown to select scenario
- Shows call range as 13x13 grid
- Populate scenarios from solve result
- Hover tooltip: hand, EV(call), EV(fold)

**6.3 — Nodelock panel**
- Bottom section: 4 columns (CO/BTN/SB/BB)
- Each: range slider (0-100%), text input, lock checkbox
- "Solve Exploitative" button → POST to `/api/nodelock`
- Show results as updated grids

**6.4 — Range comparison view**
- Side-by-side: Nash grid vs exploitative grid
- Difference highlighting: green border = added to range, red border = removed
- EV change summary per position

**6.5 — Hover tooltips and polish**
- Hover on any grid cell: hand name, EV(push), EV(fold), equity
- Grid cell styling: monospace, ~30px cells, clear borders
- Responsive layout
- Loading states during solve

---

## Task Dependency Map

```
1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6
                                  ↓
                          2.1 → 2.2 → 2.3
                                        ↓
                                3.1 → 3.2 → 3.3 → 3.4
                                                    ↓
                                            4.1 → 4.2 → 4.3
                                                          ↓
                                                  5.1 → 5.2 → 5.3
                                                                ↓
                                                  6.1 → 6.2 → 6.3 → 6.4 → 6.5
```

---

## Agent Instructions

### Before you start
1. Read `CLAUDE.md` for full project context
2. Read `PROGRESS.md` to find your next task
3. Read source files from completed tasks to understand interfaces

### When working
- Do ONE task per agent run
- Write clean code with type hints
- Follow patterns from previous tasks
- Don't over-engineer — keep it simple

### When done
- Update `PROGRESS.md`: mark your task `[x]` or `[!]` (needs human testing)
- Add any notes about decisions, issues, or gotchas
- If you had to deviate from the spec, note WHY in PROGRESS.md
- **Update AGENTS.md** if you discovered anything that affects future tasks

### Resource constraints (4GB CPU machine)
- **DO NOT** run equity matrix generation (takes minutes + lots of RAM)
- **DO NOT** run Monte Carlo simulations
- **OK to run**: pytest on lightweight tests, basic imports, syntax checks
- When in doubt, skip the test run and mark "needs human testing"

### Code style
- Python 3.11+
- Type hints on all public functions
- Docstrings on public functions (one-liner is fine)
- Prefer functions and dataclasses over classes
- Constants at module top
- Imports: stdlib → third-party → local (separated by blank lines)
