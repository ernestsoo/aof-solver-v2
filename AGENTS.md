# AoF Solver — Agent Task Definitions

## IMPORTANT — Read First
1. **Read `CLAUDE.md`** before doing anything — it has all project context
2. **Check `PROGRESS.md`** to see what's done and find your next task
3. **Update `PROGRESS.md`** immediately when you start and finish a task
4. **Update this file** (AGENTS.md) with any notes, issues, or changes to the plan
5. **One task per agent run** — pick the next unchecked task, do it, update progress, done
6. **DO NOT commit or push** — Billy (the orchestrating agent) handles all git operations

---

## Resource Constraints — CRITICAL

This runs on a **2-vCPU, 4GB RAM VPS**. Respect these limits:

### What you CAN run (< 30 seconds)
- `pytest` on unit tests that don't need the equity matrix
- Importing modules and checking they work
- Small data structure operations (generating 169 hands, grid mappings)
- Tests with mocked/stubbed equity data

### What you CANNOT run — mark as `[!]` (needs human testing)
- Equity matrix generation (`scripts/generate_equities.py`) — takes 30-60 min
- Any Monte Carlo simulation with > 1000 samples
- Full solver convergence runs (even with precomputed equity)
- Flask server startup + integration tests
- Anything that loops over all 1326 × 1326 hand combinations

### Test Rules
- Every test file must have: `pytest.mark.skipif(not EQUITY_MATRIX_EXISTS, reason="equity matrix not generated")`
- Tests that need the equity matrix must skip gracefully, not fail
- Write a `conftest.py` with the skip fixture so all test files share it
- If a test takes > 5 seconds, it's too heavy — split it or mark it `[!]`

---

## Lessons From v1 — DO NOT REPEAT THESE MISTAKES

### ❌ v1 Problem: Monte Carlo equity during solve time
The v1 solver computed equity via MC sampling on every iteration. Each best-response computation called MC equity for 1326 hands × opponent combos. Result: ~80 seconds per iteration, solver never finished.

### ✅ v2 Solution: Precomputed 169×169 equity matrix
- Generate a 169×169 canonical hand equity matrix ONCE (offline, human runs it)
- Solver looks up equity from the matrix — O(1) per lookup
- Hand-vs-range equity = weighted average of matrix lookups (fast numpy dot product)
- **No MC during solve time. Ever.**

### ❌ v1 Problem: Python for-loops over 1326 combos
Best response computation iterated over every combo in Python. Slow.

### ✅ v2 Solution: Work at the 169-hand level, vectorize with numpy
- Strategies are 169-element arrays (one per canonical hand type), not 1326
- EV computation uses numpy matrix multiplication: `ev = equity_matrix @ (opponent_range * pot_sizes)`
- No Python for-loops in the hot path
- Combo weighting (6/4/12) is a separate vector, applied via element-wise multiply

### ❌ v1 Problem: Complex game tree object model
v1 built a tree of node objects with recursive traversal. Over-engineered for a push/fold game.

### ✅ v2 Solution: Flat scenario-based EV
- No tree objects. Each decision point is a function that computes EV directly
- Action sequences are enumerated as flat scenarios (CO pushes, BTN calls, etc.)
- Each scenario: who pushed, who called, pot size, equity lookup
- Total scenarios: ~16 terminal nodes, easily enumerable

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
- Each hand as a dataclass or namedtuple: `name`, `type` (pair/suited/offsuit), `combos` (6/4/12)
- Store as `ALL_HANDS: list` (ordered by index 0-168) and `HAND_MAP: dict` (name → hand info)
- Verify total combos = 1326
- **Performance note:** This is the foundation. Use simple list/dict, not complex objects. The 169-hand index is used everywhere downstream.

**1.2 — Hand ranking**
- Add `HAND_RANKINGS` — all 169 hands ranked 1-169 by standard pre-flop strength
- Rank 1 = AA, 2 = KK, 3 = QQ, ..., 169 = worst (use well-known standard ranking, hard-code the order)
- Add `hand_rank(hand: str) -> int` and `hand_percentile(hand: str) -> float`
- Percentile = cumulative combo percentage (e.g., AA = 6/1326 = 0.45%, top 5% = ~first 66 combos)
- Add `top_n_percent(pct: float) -> list[str]` — returns hands in the top N% by combo-weighted ranking

**1.3 — Range parsing and operations**
- `parse_range(notation: str) -> list[str]` — parse "22+, A2s+, KTo+" into hand list
- `expand_plus_notation(hand: str) -> list[str]` — "ATs+" → ["ATs","AJs","AQs","AKs"]
- `range_to_hands(range_pct: float) -> list[str]` — top N% of hands by combo-weighted ranking
- `hands_to_range_pct(hands: list[str]) -> float` — combo % of 1326 total
- Handle edge cases: "random" = 100%, empty = 0%, single hand "AKs" = just that hand

**1.4 — Grid mapping**
- `hand_to_grid(hand: str) -> tuple[int, int]` — map to 13x13 grid position
- `grid_to_hand(row: int, col: int) -> str` — reverse mapping
- Grid layout: rows/cols by rank (A=0..2=12), suited above diagonal, offsuit below, pairs on diagonal
- All 169 hands must map to unique positions and round-trip correctly

**1.5 — Combo counting with removal**
- `combo_count(hand: str) -> int` — basic: 6/4/12
- `total_combos(hands: list[str]) -> int` — sum combos
- `combos_with_removal(hand: str, blocked_cards: list[str]) -> int` — account for dead cards
- `COMBO_WEIGHTS: np.ndarray` — shape (169,), each entry = combo count for that canonical hand. Used for weighted equity calculations.
- **This array is critical for solver performance** — vectorized operations use it everywhere

**1.6 — Tests for hands.py**
- Create `tests/test_hands.py`
- Create `tests/conftest.py` with shared fixtures and skip markers
- Test: 169 hands generated, total combos = 1326
- Test: ranking order (AA=1, KK=2, etc.)
- Test: parse_range("TT+") → ["TT","JJ","QQ","KK","AA"]
- Test: parse_range("A2s+") → all 12 suited aces
- Test: grid round-trip for all 169 hands
- Test: combos_with_removal("AKs", ["As"]) = 3
- Test: COMBO_WEIGHTS sums to 1326
- **Run `pytest tests/test_hands.py -v`** — this is lightweight, OK to run (< 5s)

---

### Phase 2: Equity Engine (`src/equity.py`)

**2.1 — Equity matrix generator script**
- Create `scripts/generate_equities.py`
- Use `eval7` to compute equity for each of 169×169 canonical hand matchups
- For each matchup: enumerate representative card combos, run Monte Carlo or exact enumeration
- Only compute upper triangle (i < j), mirror to lower: `equity[j][i] = 1.0 - equity[i][j]`
- Diagonal (hand vs itself): set to 0.5 (only relevant when cards don't conflict)
- Save to `data/equity_matrix.npy` as float32 (169×169 = ~114KB)
- Print progress every 100 matchups
- Estimated runtime: 10-30 minutes on 2-vCPU. **DO NOT RUN.** Mark as `[!]`
- **Key design:** This matrix is the ONLY equity source for the solver. No MC at solve time.

**2.2 — Equity lookup module**
- Create `src/equity.py`
- `load_equity_matrix(path: str = "data/equity_matrix.npy") -> np.ndarray` — returns (169, 169) float32
- `hand_vs_hand_equity(hand1: str, hand2: str, matrix: np.ndarray) -> float` — matrix[idx1][idx2]
- `hand_vs_range_equity(hand_idx: int, range_mask: np.ndarray, combo_weights: np.ndarray, matrix: np.ndarray) -> float`
  - **This must be vectorized:** `np.dot(matrix[hand_idx] * range_mask, combo_weights) / np.dot(range_mask, combo_weights)`
  - `range_mask` is a (169,) boolean/float array. `combo_weights` is from hands.py.
  - No Python for-loops. This function is called thousands of times during a solve.
- `range_vs_range_equity(range1: np.ndarray, range2: np.ndarray, combo_weights: np.ndarray, matrix: np.ndarray) -> float`
  - Double-weighted average, also vectorized
- Handle missing matrix file: raise `FileNotFoundError` with message "Run scripts/generate_equities.py first"

**2.3 — Tests for equity.py**
- Create `tests/test_equity.py`
- **All tests that need the real matrix must skip if `data/equity_matrix.npy` doesn't exist**
- Create a `tests/fixtures/tiny_equity.npy` — a small 169×169 matrix with known values for a few hands (AA, KK, QQ, 72o) and zeros elsewhere. Tests use this fixture.
- Test: load_equity_matrix returns (169, 169) shape
- Test: `equity[i][j] + equity[j][i]` ≈ 1.0 (with fixture data)
- Test: hand_vs_range_equity with single hand equals hand_vs_hand
- Test: vectorized equity matches manual loop (correctness check on fixture)
- **Run `pytest tests/test_equity.py -v`** — OK to run with fixture data (< 5s)
- **DO NOT run generate_equities.py**

---

### Phase 3: Nash Solver (`src/solver.py`)

**3.1 — EV calculation functions**
- Create `src/solver.py`
- All strategies represented as `np.ndarray` shape (169,) — values 0.0 to 1.0 (probability of pushing/calling)
- `ev_push_fold(hand_idx, position, strategies, call_strategies, equity_matrix, combo_weights) -> tuple[float, float]`
  - Returns (ev_fold, ev_push) for a hand at a position
  - ev_fold: 0 for CO/BTN, -0.5 for SB (posted), -1.0 for BB (posted)
  - ev_push: must account for all possible caller combinations
- Pot math (CRITICAL — get this right):
  - Base pot: 1.5bb (0.5 SB + 1.0 BB)
  - Effective stacks: 10bb for all
  - SB pushes: risks 9.5bb more (already posted 0.5)
  - BB calls: risks 9.0bb more (already posted 1.0)
  - When multiple players push: pot = 10bb × num_players_in + dead blinds from folders
- **Implementation must be vectorized:** compute ev_push for all 169 hands at once using matrix ops, not a for-loop over hands
- See CLAUDE.md for detailed EV formulas per position and scenario

**3.2 — Iterative best-response solver**
- `solve_nash(equity_matrix, combo_weights, max_iter=500, tolerance=0.001) -> SolverResult`
- SolverResult dataclass: `push_strategies: dict[str, np.ndarray]`, `call_strategies: dict[str, dict[str, np.ndarray]]`, `ev_table: dict`, `iterations: int`, `converged: bool`, `exploitability: float`
- Algorithm:
  1. Initialize: all positions push top 50% (by ranking)
  2. For each position (CO→BTN→SB→BB): compute best response given others' current strategies
  3. Best response: for each hand, if EV(push) > EV(fold), push probability = 1.0, else 0.0
  4. Repeat until convergence: max strategy change < tolerance across all positions
  5. Cap at max_iter
- **Performance target: full solve in < 5 seconds on 2-vCPU** (with precomputed equity matrix)
- If this target isn't met, profile and report where time goes — don't try to optimize blind

**3.3 — All call range scenarios**
- Each position has multiple call scenarios depending on who pushed before them:
- **BTN:** `vs_co` (CO pushed), `open` (CO folded, BTN decides to push)
- **SB:** `vs_co`, `vs_btn`, `vs_co_btn`, `open` (all fold to SB)
- **BB:** `vs_co`, `vs_btn`, `vs_sb`, `vs_co_btn`, `vs_co_sb`, `vs_btn_sb`, `vs_co_btn_sb`, `check` (all fold to BB — wins pot)
- Each scenario needs correct pot size and risk calculation
- Store as a dict: `call_strategies[position][scenario] = np.ndarray(169,)`

**3.4 — Tests for solver.py**
- Create `tests/test_solver.py`
- **Must skip if equity matrix not present** (use conftest fixture)
- With fixture data (tiny matrix):
  - Test: EV calculation returns sensible values (ev_push > ev_fold for AA)
  - Test: ev_fold is correct per position (0, 0, -0.5, -1.0)
  - Test: pot math is correct for each scenario
- With real matrix (skip if missing):
  - Test: solver converges within max_iter
  - Test: AA always in push range for all positions
  - Test: 72o never in push range for any position
  - Test: CO range ⊂ BTN range ⊂ SB range (ranges widen by position)
  - Test: exploitability < 0.1 bb
- **DO NOT run solver tests if equity matrix missing** — they will skip automatically
- **Run `pytest tests/test_solver.py -v`** — OK if using fixture, skip gracefully otherwise

---

### Phase 4: Nodelocking (`src/nodelock.py`)

**4.1 — Nodelock solver**
- Create `src/nodelock.py`
- `nodelock_solve(equity_matrix, combo_weights, locked: dict[str, np.ndarray], max_iter=500) -> SolverResult`
- `locked` dict: position → fixed push strategy array (169,). Locked positions are NOT updated during iteration.
- Same IBR algorithm as Nash but skip locked positions
- Accept locked ranges as: hand list, range percentage, or raw numpy array
- Helper: `lock_from_range(range_pct: float) -> np.ndarray` — convert % to push array
- Helper: `lock_from_hands(hands: list[str]) -> np.ndarray` — convert hand list to push array

**4.2 — Exploitability metric**
- `exploitability(push_strategies, call_strategies, equity_matrix, combo_weights) -> float`
- Returns total exploitability in bb (sum of max(ev_best_response - ev_current) across all positions)
- For Nash solution, should be ≈ 0
- For nodelocked solution, measures how far locked player deviates from equilibrium

**4.3 — Tests for nodelock.py**
- Create `tests/test_nodelock.py`
- **Must skip if equity matrix not present**
- With real matrix (skip if missing):
  - Test: locking all to Nash returns Nash result (no change)
  - Test: locking one position wider → opponents call tighter
  - Test: locking one position tighter → opponents push wider
  - Test: exploitability(nash_result) ≈ 0
  - Test: exploitability(nodelocked_result) > 0 when lock deviates from Nash
- **Mark all as `[!]`** — needs human testing with real equity matrix

---

### Phase 5: Dashboard — Backend (`src/dashboard.py`)

**5.1 — Flask app and solve endpoint**
- Create `src/dashboard.py`
- Flask app with `/api/solve` GET endpoint
- Load equity matrix at startup. If missing, return 503 with message.
- Precompute Nash solution at startup (cache in memory)
- Return JSON: push_strategies, call_strategies, ev_table per position
- All arrays serialized as: `{hand_name: probability}` dict (not raw numpy)

**5.2 — Nodelock endpoint**
- `/api/nodelock` POST endpoint
- Body: `{"locks": {"BTN": 45, "CO": "22+,A2s+"}}` — accepts % (float) or range notation (string)
- Parse locks, run nodelock_solve, return result JSON
- Include both Nash and nodelocked results for comparison

**5.3 — Utility endpoints**
- `/api/hand_equity?hand=AKs&vs=top30` — equity lookup against a range
- `/api/hand_info?hand=AKs` — rank, percentile, combos, type
- `/api/range?notation=22%2B,A2s%2B` — expand notation to hand list
- **Mark all dashboard tasks as `[!]`** — needs human testing (Flask startup + equity matrix)

---

### Phase 6: Dashboard — Frontend (`templates/index.html`)

**6.1 — Page layout and Nash grids**
- Create `templates/index.html`
- Dark theme, single page, vanilla JS (no build tools)
- Top section: 4 hand grids side by side (CO/BTN/SB/BB)
- Each grid 13×13 showing push range
- Color: green = push, red = fold, intensity by EV magnitude
- Below each grid: "Push X% (Y combos)" summary
- Fetch from `/api/solve` on page load

**6.2 — Call range viewer**
- Middle section: dropdown to select position and scenario
- Shows call range as 13×13 grid
- Populate scenarios from solve result
- Hover tooltip: hand name, EV(call), EV(fold), equity vs pusher range

**6.3 — Nodelock panel**
- Bottom section: 4 columns (CO/BTN/SB/BB)
- Each: range slider (0-100%), text input for notation, lock checkbox
- "Solve Exploitative" button → POST to `/api/nodelock`
- Show results as updated grids replacing the Nash grids
- Show EV difference per position (how much locked player is losing)

**6.4 — Range comparison view**
- Toggle: Nash vs exploitative side-by-side
- Difference highlighting: green border = added to range vs Nash, red border = removed
- EV change summary per position in bb

**6.5 — Hover tooltips and polish**
- Hover on any grid cell: hand name, type, combos, EV(push), EV(fold), equity
- Grid cell styling: monospace, ~30px cells, clear borders
- Loading spinner during solve/nodelock
- Error states (matrix missing, solve failed)
- **All Phase 6 tasks are `[!]`** — needs human testing with running Flask server

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
3. Read source files from completed tasks to understand existing interfaces

### When working
- Do ONE task per agent run
- Write clean code with type hints
- Follow patterns from previous tasks
- Don't over-engineer — keep it simple
- **Use numpy arrays (169,) for all strategies and weights — never loop over 1326 combos in Python**

### When done
- Update `PROGRESS.md`: mark your task `[x]` or `[!]` (needs human testing)
- Add any notes about decisions, issues, or gotchas to PROGRESS.md Notes section
- If you had to deviate from the spec, explain WHY
- **DO NOT commit or push** — Billy handles all git operations

### Performance checklist (ask yourself before finishing)
- [ ] Did I use numpy vectorization instead of Python for-loops for any computation over 169+ elements?
- [ ] Did I avoid any Monte Carlo or random sampling in code that runs during solve time?
- [ ] Did I use the precomputed equity matrix for all equity lookups?
- [ ] Would this code complete in < 5 seconds for a full solve on a 2-vCPU machine?
- [ ] Did I mark heavy-compute tasks as `[!]` in PROGRESS.md?

### Code style
- Python 3.11+
- Type hints on all public functions
- Docstrings on public functions (one-liner is fine)
- Prefer functions and dataclasses over classes
- Constants at module top
- Imports: stdlib → third-party → local (separated by blank line)
- numpy arrays for any numeric computation over hands/ranges
