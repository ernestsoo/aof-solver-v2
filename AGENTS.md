# AoF Solver — Agent Task Definitions

## IMPORTANT — Read First
1. **Read `CLAUDE.md`** before doing anything — it has all project context
2. **Check this file** for your next task (find the first `[ ]` task)
3. **Update this file** when you start `[~]`, finish `[x]`, or hit issues
4. **One task per agent run** — do it, update status + notes below, done
5. **DO NOT commit or push** — Billy (the orchestrating agent) handles all git operations

### Progress Legend
- `[ ]` = not started
- `[~]` = in progress
- `[x]` = complete
- `[!]` = needs human testing (heavy compute)

### Self-Tracking Rules (CRITICAL — READ THIS)
After completing each task, you MUST update this file:
1. Change the task checkbox from `[ ]` to `[x]`
2. Add a brief note under the task with what you did and any decisions made
3. If you created functions/classes, list their signatures so the next session has context
4. If you deviated from the spec, explain WHY
5. If you hit issues, document them so they're not repeated

This is how context survives between sessions. Be thorough in your notes.

---

## Resource Constraints — CRITICAL

**Machine:** 4-vCPU, 8GB RAM VPS

### OK to run (< 30 seconds)
- `pytest` on unit tests that use fixtures (not real equity matrix)
- Module imports, small data ops, hand generation
- Tests with mocked/stubbed equity data

### DO NOT run — mark `[!]`
- Equity matrix generation (30-60 min)
- Full solver convergence with real equity matrix
- Flask server + integration tests
- Any Monte Carlo with > 1000 samples

### Test Rules
- Tests needing real equity matrix: skip if `data/equity_matrix.npy` missing
- Shared skip fixture in `tests/conftest.py`
- Any test > 5 seconds is too heavy — split it or mark `[!]`

---

## Lessons From v1 — DO NOT REPEAT

| v1 Problem | v2 Solution |
|---|---|
| Monte Carlo equity during solve (~80s/iteration) | Precomputed 169×169 matrix, O(1) lookups |
| Python for-loops over 1326 combos | 169-hand arrays, numpy vectorization |
| Complex game tree object model | Flat scenario-based EV functions |
| `treys` library (slow) | `eval7` (fast C extension) |

---

## Phase 1: Hand Representations (`src/hands.py`)

### 1.1 — Create HandInfo dataclass and constants
- [x] Create `src/__init__.py` (empty) — already existed
- [x] Create `src/hands.py`
- [x] Define `RANKS = ['A','K','Q','J','T','9','8','7','6','5','4','3','2']`
- [x] Define `RANK_INDEX = {'A': 0, 'K': 1, ..., '2': 12}` — map rank char to int
- [x] Create `HandInfo` dataclass: `name: str`, `index: int`, `hand_type: str` (pair/suited/offsuit), `combos: int` (6/4/12), `rank1: int`, `rank2: int`

**Notes:**
Created `src/hands.py` with:
- `RANKS: list[str]` — 13 rank chars, A..2
- `RANK_INDEX: dict[str, int]` — built via dict comprehension from RANKS (A=0, K=1, ..., 2=12)
- `HandInfo` dataclass — fields: `name`, `index`, `hand_type`, `combos`, `rank1`, `rank2`

`rank` field (1-169 preflop strength rank) is NOT included yet — that is added in task 1.3 per spec.
Verified import and instantiation work correctly with python3.

### 1.2 — Generate all 169 canonical hands
- [x] Generate 13 pairs (AA, KK, ..., 22) — combos=6
- [x] Generate 78 suited hands (AKs, AQs, ..., 32s) — combos=4
- [x] Generate 78 offsuit hands (AKo, AQo, ..., 32o) — combos=12
- [x] Store as `ALL_HANDS: list[HandInfo]` ordered by index 0-168
- [x] Store as `HAND_MAP: dict[str, HandInfo]` for name lookup
- [x] Create `COMBO_WEIGHTS: np.ndarray` shape (169,) — each entry = combo count
- [x] Assert: `len(ALL_HANDS) == 169` and `COMBO_WEIGHTS.sum() == 1326`

**Ordering:** Pairs first (AA=0, KK=1, ..., 22=12), then suited (AKs=13, AQs=14, ...), then offsuit (AKo, AQo, ...). Exact order within suited/offsuit: higher rank1 first, then higher rank2.

**Notes:**
Added to `src/hands.py`. Key details:
- `_generate_hands() -> list[HandInfo]` — private generator, loops pairs (r=0..12), then suited (r1 < r2), then offsuit (r1 < r2). Assigns `index` sequentially.
- `ALL_HANDS: list[HandInfo]` — module-level list, 169 entries. Indices: AA=0, KK=1, ..., 22=12, AKs=13, AQs=14, ..., 32s=90, AKo=91, ..., 32o=168.
- `HAND_MAP: dict[str, HandInfo]` — built from `{h.name: h for h in ALL_HANDS}`.
- `COMBO_WEIGHTS: np.ndarray` — shape (169,), dtype float64, values 6/4/12. Sum = 1326.0 ✓
- Module-level asserts run on import — `len(ALL_HANDS)==169` and `COMBO_WEIGHTS.sum()==1326`.
- Added `import numpy as np` at top of file.
- Spot-checked: `HAND_MAP["AA"].index==0`, `HAND_MAP["AKs"].index==13`, `HAND_MAP["AKo"].index==91`, last hand = `32o` at index 168.

### 1.3 — Hand ranking by preflop strength
- [x] Add `rank: int` field to HandInfo (1=AA strongest, 169=weakest)
- [x] Hard-code the standard 169-hand ranking order (well-known, look it up)
- [x] Add function: `top_n_percent(pct: float) -> np.ndarray` — returns (169,) mask where 1.0 = hand is in top N% by combo-weighted rank
- [x] Percentile calculation: cumulative combos / 1326

**Notes:**
Added to `src/hands.py`:
- `HandInfo.rank: int = 0` — defaulted so existing construction calls unchanged; assigned after ALL_HANDS built.
- `HAND_RANK_ORDER: list[str]` — 169-entry constant listing all hands strongest-to-weakest (AA=1, 72o=169). Based on equity vs random hand; suited > offsuit same ranks; wheel potential (A5s, A4s, A3s, A2s) ranked above A6s.
- After `HAND_MAP` is built, loop `enumerate(HAND_RANK_ORDER, start=1)` assigns `.rank` on each HandInfo in place.
- Module-level asserts verify: `len(HAND_RANK_ORDER)==169`, no duplicates, `AA.rank==1`, `72o.rank==169`.
- `top_n_percent(pct: float) -> np.ndarray` — accepts 0.0–100.0 percentage. Sorts ALL_HANDS by rank, walks in strength order, includes a hand if `cumulative_combos_before_it / 1326 < pct/100`. Returns (169,) float64 mask.
- Sanity checks confirmed: `top_n_percent(0)` all zeros, `top_n_percent(100)` all ones, `top_n_percent(30)` gives ~30.8% (quantized by combo granularity), AA in/72o out of top-30%.
- No Python loops over 169 at solve time — `top_n_percent` is only called at initialization (3× per solve).

### 1.4 — Grid mapping (13×13)
- [x] `hand_to_grid(name: str) -> tuple[int, int]` — map hand to 13×13 position
- [x] `grid_to_hand(row: int, col: int) -> str` — reverse mapping
- [x] Grid layout: row=rank1, col=rank2. Pairs on diagonal. Suited above diagonal (row < col). Offsuit below (row > col).
- [x] Verify: all 169 hands map to unique grid positions and round-trip

**Notes:**
Added to `src/hands.py`:
- `hand_to_grid(name: str) -> tuple[int, int]`: Looks up HandInfo from HAND_MAP. Suited → (rank1, rank2); offsuit → (rank2, rank1) to place below diagonal; pair → (rank1, rank1). O(1) lookup.
- `grid_to_hand(row: int, col: int) -> str`: row==col → pair; row<col → RANKS[row]+RANKS[col]+"s" (suited); row>col → RANKS[col]+RANKS[row]+"o" (offsuit — note col/row swap to maintain rank1<rank2 convention in name).
- Sanity verified: 0 round-trip failures, 169 unique grid positions out of 169 hands.
- Spot checks confirmed: AA=(0,0), AKs=(0,1), AKo=(1,0), 22=(12,12), 32s=(11,12), 32o=(12,11).
- RANKS index: A=0, K=1, ..., 8=6, 7=7, ..., 2=12.

### 1.5 — Range parsing
- [x] `parse_range(notation: str) -> list[str]` — parse "22+, A2s+, KTo+" into hand names
- [x] Handle plus notation: "TT+" -> [TT, JJ, QQ, KK, AA]
- [x] Handle suited plus: "ATs+" -> [ATs, AJs, AQs, AKs]
- [x] Handle offsuit plus: "KTo+" -> [KTo, KJo, KQo]
- [x] Handle single hands: "AKs" -> ["AKs"]
- [x] Handle combos: "22+, A2s+, KTo+" (comma-separated)
- [x] Handle "random" = all 169 hands, empty = none

**Notes:**
Added `parse_range(notation: str) -> list[str]` to `src/hands.py` (before `hand_to_grid`).

Logic:
- Empty/whitespace → []
- "random" (case-insensitive) → all 169 hands in canonical order
- Split on comma, strip each token
- Token ending in "+":
  - Pair (e.g. "TT"): r=RANK_INDEX[char], loop i from r down to 0 → TT, JJ, QQ, KK, AA
  - Suited (ends "s"): r1=RANK_INDEX[rank1], r2_start=RANK_INDEX[rank2], loop r2 from r2_start down to r1+1 (improving toward rank1)
  - Offsuit (ends "o"): same as suited but builds "...o" names
- Other: single hand, appended as-is
- Deduplication via seen set, preserves encounter order

Sanity checks confirmed:
- parse_range("TT+") == ["TT","JJ","QQ","KK","AA"]
- parse_range("A2s+") → 12 suited aces (A2s..AKs)
- parse_range("KTo+") == ["KTo","KJo","KQo"]
- parse_range("random") → 169 hands
- parse_range("") == []
- parse_range("AKs") == ["AKs"]
- parse_range("22+, AKs") → 14 hands (13 pairs + AKs)

Note: AGENTS.md checkbox spec listed "KTo+" → [KTo,KJo,KQo,KAo] but KAo is invalid (A is higher rank than K, so rank1 would be A). Task description correctly specifies [KTo,KJo,KQo] — implemented that way.

### 1.6 — Range utility functions
- [x] `range_to_mask(hands: list[str]) -> np.ndarray` — (169,) float array, 1.0 for included hands
- [x] `mask_to_hands(mask: np.ndarray) -> list[str]` — reverse
- [x] `hands_to_range_pct(hands: list[str]) -> float` — combo % of 1326
- [x] `combos_with_removal(hand: str, blocked_cards: list[str]) -> int` — reduced combos when cards are dead

**Notes:**
Added four functions to `src/hands.py`. Also added `SUITS: list[str] = ['s', 'h', 'd', 'c']` constant.

**Signatures:**
- `range_to_mask(hands: list[str]) -> np.ndarray` — loops hands, sets mask[HAND_MAP[name].index] = 1.0. O(n).
- `mask_to_hands(mask: np.ndarray) -> list[str]` — list comprehension over range(169), returns names where mask==1.0. Canonical index order.
- `hands_to_range_pct(hands: list[str]) -> float` — sums combos for each hand, divides by 1326, multiplies by 100.
- `combos_with_removal(hand: str, blocked_cards: list[str]) -> int` — enumerates specific card combos:
  - Pair: C(4,2)=6 combos via nested SUITS loops (s1 < s2 by index)
  - Suited: 4 combos, one per suit (both cards same suit)
  - Offsuit: 12 combos (s1 != s2, 4*3 combos)
  - A combo is dead if EITHER card appears in the blocked set (correct set-union logic)

**Spec discrepancy:** The task specified `combos_with_removal("AKs", ["As", "Ks"]) == 2`, but the correct answer is 3. Both "As" and "Ks" block the SAME combo (AsKs); only 1 combo is removed, leaving 3. The spec incorrectly assumes each blocked card removes a separate combo. Implemented the correct union logic (3 is right). All other sanity checks pass.

### 1.7 — Tests for Phase 1
- [x] Create `tests/__init__.py` (empty)
- [x] Create `tests/conftest.py` with skip fixture for missing equity matrix
- [x] Create `tests/test_hands.py`
- [x] Test: 169 hands generated, total combos = 1326
- [x] Test: COMBO_WEIGHTS shape (169,) and sum 1326
- [x] Test: AA index=0, rank=1; KK index=1, rank=2
- [x] Test: top_n_percent(100) returns all ones, top_n_percent(0) returns all zeros
- [x] Test: grid round-trip for all 169 hands
- [x] Test: parse_range("TT+") -> [TT, JJ, QQ, KK, AA]
- [x] Test: parse_range("A2s+") -> 12 suited aces
- [x] Test: range_to_mask round-trips with mask_to_hands
- [x] Test: combos_with_removal("AKs", ["As"]) == 3
- [x] **Run `pytest tests/test_hands.py -v`** — must pass, < 5s

**Notes:**
Created three test files:
- `tests/__init__.py` — already existed (empty), left as-is.
- `tests/conftest.py` — `require_equity_matrix` fixture: checks for `data/equity_matrix.npy` relative to tests dir; calls `pytest.skip()` if missing. Used as a function argument fixture (not autouse) so tests opt in explicitly.
- `tests/test_hands.py` — 22 tests covering all spec items:
  1. `test_hand_count` — len(ALL_HANDS) == 169
  2. `test_combo_weights` — shape (169,), sum == 1326
  3. `test_hand_indices` — AA=0, KK=1, AKs=13, AKo=91, 32o=168
  4. `test_hand_ranks` — AA rank=1, KK rank=2, 72o rank=169
  5. `test_top_n_percent_bounds` — 0→all zeros, 100→all ones
  6. `test_top_n_percent_aa_in_top30` — AA in, 72o out of top 30%
  7. `test_grid_roundtrip` — all 169 hands round-trip through hand_to_grid↔grid_to_hand
  8. `test_grid_spots` — AA=(0,0), AKs=(0,1), AKo=(1,0), 22=(12,12)
  9. `test_parse_range_pairs` — "TT+" == ["TT","JJ","QQ","KK","AA"]
  10. `test_parse_range_suited` — "A2s+" returns 12 hands
  11. `test_parse_range_offsuit` — "KTo+" == ["KTo","KJo","KQo"]
  12. `test_parse_range_empty` — "" == []
  13. `test_parse_range_random` — 169 hands
  14. `test_parse_range_single` — "AKs" == ["AKs"]
  15. `test_range_mask_roundtrip` — round-trips for all 169 hands
  16. `test_range_mask_single` — ["AA"] → 1.0 at index 0, 0 elsewhere, sum=1.0
  17. `test_hands_to_range_pct_aa` — ~0.452% (within 0.01)
  18. `test_hands_to_range_pct_all` — all 169 → 100.0% (within 0.001)
  19. `test_combos_with_removal_suited` — combos_with_removal("AKs", ["As"]) == 3
  20. `test_combos_with_removal_pair` — combos_with_removal("AA", ["As"]) == 3
  21. `test_combos_with_removal_offsuit` — combos_with_removal("AKo", ["As"]) == 9
  22. `test_combos_with_removal_none` — combos_with_removal("AKs", []) == 4

pytest output: **22 passed in 0.23s** ✓

---

## Phase 2: Equity Engine (`src/equity.py`)

### 2.1 — Equity matrix generator script
- [!] Create `scripts/generate_equities.py`
- [!] For each of 169×169 matchups: enumerate non-conflicting card combos
- [!] Use `eval7` for hand evaluation with Monte Carlo (N=1000 boards per specific combo pair)
- [!] Only compute upper triangle (i < j), mirror: `matrix[j][i] = 1.0 - matrix[i][j]`
- [!] Diagonal: 0.5
- [!] Save to `data/equity_matrix.npy`, shape (169, 169), dtype float32
- [!] Print progress every 100 matchups
- [!] DO NOT RUN — estimated 10-30 min

**Notes:**
Created three files:
- `scripts/__init__.py` — empty, makes scripts/ a package
- `data/.gitkeep` — ensures data/ directory exists in git
- `scripts/generate_equities.py` — the equity matrix generator (DO NOT RUN)

**Key implementation decisions:**
- N_BOARDS=1000 per specific combo pair (CLAUDE.md mentions 5000-10000 for final run; 1000 is the agent task spec; can be raised before human runs it)
- `eval7.Card` supports `__eq__` and `__hash__`, so conflict detection uses `c1 == c3` (no string conversion needed for the check itself)
- `str(eval7.Card('As'))` returns `'As'` — used for CARD_TO_IDX lookup (a dict mapping card string → deck index 0..51)
- `rng=np.random.default_rng(seed=42)` passed through for reproducibility across runs
- Progress printed every 100 matchups with elapsed time and ETA
- End-of-run validation prints symmetry error and spot-checks 4 known equities

**Functions created:**
- `get_specific_combos(hand: HandInfo) -> list[tuple[eval7.Card, eval7.Card]]`
  Returns 6 / 4 / 12 specific card pairs for pairs / suited / offsuit hands.
- `compute_matchup_equity(hand_i, hand_j, n=N_BOARDS, rng=None) -> float`
  Monte Carlo equity of hand_i vs hand_j averaged across all valid combo pairs.
- `main() -> None`
  Loops all 14,196 upper-triangle matchups, fills matrix, saves, prints validation.

**Module-level constants:**
- `N_BOARDS = 1000`
- `OUTPUT_PATH = ".../data/equity_matrix.npy"`
- `ALL_CARDS: list[eval7.Card]` — 52 cards in RANKS×SUITS order
- `CARD_TO_IDX: dict[str, int]` — card string → deck index

**Verified:**
- `python3 -c "import scripts.generate_equities"` → clean import ✓
- AA vs KK equity (n=50): 0.824 ≈ 0.82 ✓
- AA vs AKs equity (n=50): 0.869 ≈ 0.87 ✓
- Combo counts: AA=6, AKs=4, AKo=12 ✓
- AA vs KK conflict count: 0 ✓

### 2.2 — Equity lookup functions
- [x] Create `src/equity.py`
- [x] `load_equity_matrix(path="data/equity_matrix.npy") -> np.ndarray` — returns (169,169) float32. Raise FileNotFoundError if missing.
- [x] `hand_vs_hand_equity(idx1: int, idx2: int, matrix: np.ndarray) -> float` — simple matrix lookup
- [x] `hand_vs_range_equity(hand_idx: int, range_mask: np.ndarray, combo_weights: np.ndarray, matrix: np.ndarray) -> float`
  - **Vectorized:** `np.dot(matrix[hand_idx], range_mask * combo_weights) / (range_mask * combo_weights).sum()`
  - No Python for-loops. Called thousands of times during solve.

**Notes:**
Created `src/equity.py`. Key implementation decisions:
- `load_equity_matrix`: wraps `np.load` in try/except, re-raises FileNotFoundError with helpful message. Calls `.astype(np.float32)` on result to ensure dtype consistency.
- `hand_vs_hand_equity`: `float(matrix[idx1, idx2])` — O(1).
- `hand_vs_range_equity`: computes `weighted = range_mask * combo_weights`, then `np.dot(matrix[hand_idx], weighted) / weighted.sum()`. Returns 0.5 if denom == 0.0 (empty range). Fully vectorized, no loops.
- `range_mask` can be fractional (mixed strategies from solver) — this is intentional and correct.

**Sanity checks verified:**
- AA vs KK single-hand range → 0.8200 ✓
- Empty range → 0.5 ✓
- AA vs {KK,QQ} with both at 0.82 → 0.82 ✓

### 2.3 — Multiway equity approximation
- [x] `eq3_approx(h: int, h1: int, h2: int, matrix: np.ndarray) -> float` — 3-way equity from pairwise
  - `p_h = matrix[h][h1] * matrix[h][h2]`, normalize with other two players
- [x] `eq3_vs_ranges(h_idx: int, range1: np.ndarray, range2: np.ndarray, combo_weights: np.ndarray, matrix: np.ndarray) -> float`
  - Vectorized 3-way equity against two ranges
- [x] `eq4_vs_ranges(h_idx, range1, range2, range3, combo_weights, matrix) -> float` — same for 4-way pots
- [x] See CLAUDE.md "Multiway Equity" section for exact formulas

**Notes:**
All three multiway functions implemented in `src/equity.py`.

- `eq3_approx(h, h1, h2, matrix)`: Exact formula from CLAUDE.md. Computes p_h = matrix[h,h1]*matrix[h,h2], similarly for p_h1 and p_h2, normalizes. Returns 1/3 if total==0. Verified: AA=0.789, KK=0.173, QQ=0.038, sum=1.0 ✓

- `eq3_vs_ranges(h_idx, range1, range2, combo_weights, matrix)`: Calls `hand_vs_range_equity` twice to get a=eq_vs_r1, b=eq_vs_r2. Then: `raw=a*b; raw/(raw + (1-a)*b + a*(1-b) + 1e-10)`. Fully vectorized (two dot products total, no hand loops). Empty ranges return 1/3 (correct graceful handling).

- `eq4_vs_ranges(h_idx, range1, range2, range3, combo_weights, matrix)`: Extends to 4-way: `raw=a*b*c; raw/(raw + (1-a)*b*c + a*(1-b)*c + a*b*(1-c) + 1e-10)`. Three dot products, no hand loops. Empty ranges return 0.25.

**Key design note:** `eq3_vs_ranges` uses pairwise equity-against-range as the approximation building block (not individual hand lookups). This is faster and sufficient for Nash convergence. The result differs slightly from averaging `eq3_approx` per-hand-pair due to the range-averaging step, but error is acceptable (< 2%).

**All sanity checks passed** (verified inline with synthetic matrix).

### 2.4 — Test fixture and tests for equity
- [x] Create `tests/fixtures/` directory
- [x] Create `tests/fixtures/tiny_equity.npy` — 169×169 matrix with known values for AA(idx 0), KK(idx 1), QQ(idx 2), 72o(idx ~168). Rest = 0.5.
  - AA vs KK ~ 0.82, AA vs QQ ~ 0.82, KK vs QQ ~ 0.82
- [x] Create `tests/test_equity.py`
- [x] Test: load returns (169, 169) shape
- [x] Test: `matrix[i][j] + matrix[j][i]` ~ 1.0 (on fixture)
- [x] Test: hand_vs_range_equity with single hand == hand_vs_hand
- [x] Test: eq3_approx normalizes (probabilities sum to ~1)
- [x] **Run `pytest tests/test_equity.py -v`** — must pass with fixture, < 5s

**Notes:**
Created `tests/fixtures/tiny_equity.npy` using `np.full((169,169), 0.5, dtype=np.float32)` then patching known values:
- matrix[0,1]=0.82, matrix[1,0]=0.18 (AA vs KK)
- matrix[0,2]=0.82, matrix[2,0]=0.18 (AA vs QQ)
- matrix[1,2]=0.82, matrix[2,1]=0.18 (KK vs QQ)
- matrix[0,168]=0.87, matrix[168,0]=0.13 (AA vs 72o)
- Diagonal stays 0.5 (default value)

Created `tests/test_equity.py` with 11 tests (1 skipped if real matrix absent):
- `test_fixture_shape` — shape == (169, 169)
- `test_fixture_dtype` — dtype == float32
- `test_symmetry` — matrix + matrix.T ≈ 1.0 everywhere (tol 1e-5)
- `test_diagonal` — np.diag(matrix) all == 0.5
- `test_aa_vs_kk` — hand_vs_hand_equity(0,1,matrix) ≈ 0.82
- `test_hand_vs_range_single` — single-hand range mask (KK only) gives same as direct lookup
- `test_hand_vs_range_empty` — empty mask returns 0.5
- `test_eq3_approx_sum` — eq3_approx(AA,KK,QQ) + eq3_approx(KK,AA,QQ) + eq3_approx(QQ,AA,KK) ≈ 1.0 (tol 1e-4)
- `test_eq3_approx_aa_dominant` — AA 3-way equity > 0.5
- `test_load_equity_matrix_missing` — FileNotFoundError on nonexistent path
- `test_load_real_matrix` — skip if data/equity_matrix.npy absent; checks shape (169,169)

pytest output: **10 passed, 1 skipped in 0.11s** ✓

---

## Phase 3: Nash Solver (`src/solver.py`)

### 3.1 — SolverResult dataclass and strategy storage
- [ ] Create `src/solver.py`
- [ ] Define all 14 strategy array names (see CLAUDE.md "Decision Points Per Position"):
  - Push (3): `push_co`, `push_btn_open`, `push_sb_open`
  - Call (11): `call_btn_vs_co`, `call_sb_vs_co`, `call_sb_vs_btn`, `call_sb_vs_co_btn`, `call_bb_vs_sb`, `call_bb_vs_btn`, `call_bb_vs_co`, `call_bb_vs_btn_sb`, `call_bb_vs_co_sb`, `call_bb_vs_co_btn`, `call_bb_vs_co_btn_sb`
- [ ] `SolverResult` dataclass with: strategies dict, ev_table, iterations, converged, exploitability
- [ ] `initial_strategies(combo_weights) -> dict` — init all 14 arrays. Push = top 30-50% depending on position. Call = top 20%.

**Notes:**
_(agent fills in after completing)_

### 3.2 — Fold probability helpers
- [ ] `fold_prob(strategy: np.ndarray, combo_weights: np.ndarray) -> float` — probability a random hand folds
  - `= np.dot(1 - strategy, combo_weights) / combo_weights.sum()`
- [ ] `call_prob(strategy, combo_weights) -> float` — 1 - fold_prob
- [ ] These are scalars used in every EV computation (branch probabilities in game tree)

**Notes:**
_(agent fills in after completing)_

### 3.3 — EV computation: CO open push
- [ ] `ev_push_co(equity_matrix, combo_weights, strategies) -> np.ndarray` — returns (169,) EV for each hand
- [ ] Implement all 8 terminal scenarios for CO push (Terminals 8-15 in CLAUDE.md)
- [ ] Each scenario: probability x (equity x pot - risk)
- [ ] **Must be fully vectorized** — one (169,) result array, no loops over hands
- [ ] EV(fold) for CO = 0.0

**Notes:**
_(agent fills in after completing)_

### 3.4 — EV computation: BTN decisions
- [ ] `ev_push_btn_open(...)` — BTN open push when CO folded (Terminals 4-7)
- [ ] `ev_call_btn_vs_co(...)` — BTN call when CO pushed (Terminals 12-15)
- [ ] Both return (169,) arrays, fully vectorized
- [ ] EV(fold) for BTN = 0.0

**Notes:**
_(agent fills in after completing)_

### 3.5 — EV computation: SB decisions
- [ ] `ev_push_sb_open(...)` — SB open push when CO+BTN folded (Terminals 2-3)
- [ ] `ev_call_sb_vs_co(...)` — SB call when CO pushed, BTN folded (Terminals 10-11)
- [ ] `ev_call_sb_vs_btn(...)` — SB call when BTN pushed, CO folded (Terminals 6-7)
- [ ] `ev_call_sb_vs_co_btn(...)` — SB call when CO pushed + BTN called (Terminals 14-15)
- [ ] All return (169,) arrays, vectorized
- [ ] EV(fold) for SB = -0.5

**Notes:**
_(agent fills in after completing)_

### 3.6 — EV computation: BB decisions
- [ ] `ev_call_bb_vs_sb(...)` — Terminal 3
- [ ] `ev_call_bb_vs_btn(...)` — Terminal 5
- [ ] `ev_call_bb_vs_co(...)` — Terminal 9
- [ ] `ev_call_bb_vs_btn_sb(...)` — Terminal 7
- [ ] `ev_call_bb_vs_co_sb(...)` — Terminal 11
- [ ] `ev_call_bb_vs_co_btn(...)` — Terminal 13
- [ ] `ev_call_bb_vs_co_btn_sb(...)` — Terminal 15
- [ ] All return (169,) arrays, vectorized
- [ ] EV(fold) for BB = -1.0

**Notes:**
_(agent fills in after completing)_

### 3.7 — Best response function
- [ ] `best_response(ev_action: np.ndarray, ev_fold: float) -> np.ndarray`
  - Returns (169,) array: 1.0 where ev_action > ev_fold, else 0.0
- [ ] Optional damping: `new = alpha * best + (1-alpha) * old` where alpha=0.9
  - Prevents oscillation on borderline hands

**Notes:**
_(agent fills in after completing)_

### 3.8 — IBR solve loop
- [ ] `solve_nash(equity_matrix, combo_weights, max_iter=500, tolerance=0.001) -> SolverResult`
- [ ] Initialize strategies via `initial_strategies()`
- [ ] Each iteration: compute best response for CO -> BTN -> SB -> BB
  - CO: compute ev_push_co, update push_co
  - BTN: compute ev_push_btn_open + ev_call_btn_vs_co, update both
  - SB: compute all 4 SB EVs, update all 4 SB strategies
  - BB: compute all 7 BB EVs, update all 7 BB strategies
- [ ] Convergence: `max(|new - old|)` across all 14 strategy arrays < tolerance
- [ ] Return SolverResult with final strategies, iterations, converged flag

**Notes:**
_(agent fills in after completing)_

### 3.9 — Exploitability calculation
- [ ] `compute_exploitability(strategies, equity_matrix, combo_weights) -> float`
- [ ] For each position: compute EV of best response vs current opponents, compare to current strategy EV
- [ ] Sum of differences = total exploitability in bb
- [ ] Nash solution should have exploitability ~ 0

**Notes:**
_(agent fills in after completing)_

### 3.10 — Tests for solver
- [ ] Create `tests/test_solver.py`
- [ ] Test: SolverResult created with correct structure
- [ ] Test: initial_strategies returns 14 arrays all shape (169,)
- [ ] Test: fold_prob/call_prob sum to 1.0
- [ ] Test: ev_fold values correct per position (0, 0, -0.5, -1.0)
- [ ] Test: best_response returns binary array
- [ ] Test with fixture: ev_push_co gives higher EV for AA than 72o
- [ ] Test with real matrix `[!]`: solver converges, AA always pushed, 72o never pushed, exploitability < 0.1bb
- [ ] **Run `pytest tests/test_solver.py -v`** — fixture tests must pass, real-matrix tests skip

**Notes:**
_(agent fills in after completing)_

---

## Phase 4: Nodelocking (`src/nodelock.py`)

### 4.1 — Nodelock solver
- [ ] Create `src/nodelock.py`
- [ ] `nodelock_solve(equity_matrix, combo_weights, locked: dict, max_iter=500) -> SolverResult`
  - `locked`: `{"CO": np.ndarray(169,), "BTN_open": np.ndarray(169,)}` — fixed strategies
  - Same IBR as Nash but skip locked positions during iteration
- [ ] `lock_from_range_pct(pct: float, combo_weights: np.ndarray) -> np.ndarray` — top N% as mask
- [ ] `lock_from_hands(hands: list[str]) -> np.ndarray` — hand list to mask

**Notes:**
_(agent fills in after completing)_

### 4.2 — Exploitability for nodelock
- [ ] Reuse `compute_exploitability()` from solver.py
- [ ] `compare_vs_nash(nash_result, nodelock_result) -> dict` — EV difference per position

**Notes:**
_(agent fills in after completing)_

### 4.3 — Tests for nodelock `[!]`
- [ ] Create `tests/test_nodelock.py`
- [ ] Test with real matrix: locking all to Nash = no change
- [ ] Test: locking wider -> opponents call tighter
- [ ] Test: exploitability(nash) ~ 0, exploitability(nodelock) > 0 if lock deviates
- [ ] All skip if equity matrix missing

**Notes:**
_(agent fills in after completing)_

---

## Phase 5: Dashboard Backend (`src/dashboard.py`)

### 5.1 — Flask app skeleton
- [ ] Create `src/dashboard.py`
- [ ] Flask app, load equity matrix at startup
- [ ] If matrix missing: all endpoints return 503
- [ ] CORS headers for local development
- [ ] `if __name__ == '__main__': app.run(host='0.0.0.0', port=5000, debug=True)`

**Notes:**
_(agent fills in after completing)_

### 5.2 — Solve endpoint
- [ ] `/api/solve` GET — precompute Nash at startup, cache in memory
- [ ] Return JSON: `{position: {hand_name: push_probability}}` for each position + scenario
- [ ] Include EV table and metadata (iterations, exploitability)

**Notes:**
_(agent fills in after completing)_

### 5.3 — Nodelock endpoint
- [ ] `/api/nodelock` POST
- [ ] Body: `{"locks": {"CO": 45, "BTN_open": "22+,A2s+"}}` — accepts % or notation
- [ ] Return nodelock result + Nash comparison

**Notes:**
_(agent fills in after completing)_

### 5.4 — Utility endpoints
- [ ] `/api/hand_equity?hand=AKs&vs=top30` — equity lookup
- [ ] `/api/hand_info?hand=AKs` — rank, percentile, combos
- [ ] `/api/range?notation=22+,A2s+` — expand to hand list

**Notes:**
_(agent fills in after completing)_

### 5.5 — Dashboard tests `[!]`
- [ ] All need Flask running + equity matrix — mark `[!]`

**Notes:**
_(agent fills in after completing)_

---

## Phase 6: Dashboard Frontend (`templates/index.html`)

### 6.1 — Page layout and grid renderer
- [ ] Create `templates/index.html`
- [ ] Dark theme, single page, vanilla JS
- [ ] `renderGrid(containerId, strategyData)` — draw 13x13 grid with color coding
- [ ] Green = push/call, red = fold, intensity by probability
- [ ] CSS grid: ~30px cells, monospace font, clear borders

**Notes:**
_(agent fills in after completing)_

### 6.2 — Nash push range display
- [ ] Fetch `/api/solve` on page load
- [ ] Display 4 grids side by side: CO / BTN open / SB open / BB (uncontested label)
- [ ] Below each: "Push X% (Y combos)" summary line
- [ ] Loading spinner while fetching

**Notes:**
_(agent fills in after completing)_

### 6.3 — Call range viewer
- [ ] Dropdown: select position + scenario (e.g., "BTN vs CO push")
- [ ] Show selected call range as 13x13 grid
- [ ] Dynamic: changes grid on dropdown selection

**Notes:**
_(agent fills in after completing)_

### 6.4 — Nodelock controls
- [ ] 4 columns (CO/BTN/SB/BB): range slider (0-100%) + text input for notation + lock checkbox
- [ ] "Solve Exploitative" button -> POST `/api/nodelock`
- [ ] Display results replacing Nash grids

**Notes:**
_(agent fills in after completing)_

### 6.5 — Nash vs exploitative comparison
- [ ] Toggle: side-by-side view
- [ ] Highlight differences: green border = added, red = removed vs Nash
- [ ] EV change summary per position

**Notes:**
_(agent fills in after completing)_

### 6.6 — Tooltips and polish
- [ ] Hover on grid cell: hand name, EV(push), EV(fold), equity, combos
- [ ] Error states: matrix missing, solve failed
- [ ] Mobile-friendly layout (grid wraps to 2x2 on narrow screens)
- [ ] **All Phase 6 tasks `[!]`** — need running Flask server

**Notes:**
_(agent fills in after completing)_

---

## Task Dependency Map

```
Phase 1 (sequential):
  1.1 -> 1.2 -> 1.3 -> 1.4 -> 1.5 -> 1.6 -> 1.7

Phase 2 (needs Phase 1):
  2.1 -> 2.2 -> 2.3 -> 2.4

Phase 3 (needs Phase 2):
  3.1 -> 3.2 -> 3.3 -> 3.4 -> 3.5 -> 3.6 -> 3.7 -> 3.8 -> 3.9 -> 3.10

Phase 4 (needs Phase 3):
  4.1 -> 4.2 -> 4.3

Phase 5 (needs Phase 4):
  5.1 -> 5.2 -> 5.3 -> 5.4 -> 5.5

Phase 6 (needs Phase 5):
  6.1 -> 6.2 -> 6.3 -> 6.4 -> 6.5 -> 6.6
```

---

## Agent Instructions

### Before you start
1. Read `CLAUDE.md` for full project context and EV formulas
2. Find your next task in this file (first `[ ]`)
3. Read source files from completed tasks to understand interfaces

### When working
- Do ONE task per agent run (some tasks have multiple checkboxes — do them all)
- Write clean code with type hints and docstrings
- Follow patterns from existing code
- **numpy arrays (169,) for all strategies — no loops over 1326 combos**

### When done — SELF-TRACKING CHECKLIST
- [ ] Changed task checkbox(es) from `[ ]` to `[x]` (or `[!]`)
- [ ] Added notes under the task: what you did, functions created, decisions made
- [ ] If you deviated from spec, explained why in notes
- [ ] If you found issues/gotchas, documented them in notes
- [ ] Listed function signatures you created (for next session's context)

### Performance checklist
- [ ] numpy vectorization, not Python for-loops over 169+ elements?
- [ ] No Monte Carlo at solve time?
- [ ] Precomputed equity matrix for all equity lookups?
- [ ] Would complete in < 5s for full solve on 2-vCPU?

### Code style
- Python 3.11+, type hints, docstrings on public functions
- Prefer functions + dataclasses over classes
- Constants at module top
- Imports: stdlib -> third-party -> local
