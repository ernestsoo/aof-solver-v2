# AoF Solver — All-in or Fold 4-max Push/Fold Solver

## What this is
Real-time Nash equilibrium solver for 4-max All-in or Fold poker. Computes optimal push and call ranges per position, supports nodelocking (fixing opponent strategies and solving exploitative responses). Includes a web dashboard for visualization.

## Game rules — All-in or Fold (AoF)
- 4-max (4 players): CO, BTN, SB, BB
- Fixed 10bb stacks for all players, every hand
- Standard blinds: SB posts 0.5bb, BB posts 1bb
- No antes
- Each player can only **shove all-in (10bb)** or **fold** — no other bet sizes
- Action order: CO → BTN → SB → BB
- BB closes action; if everyone folds to BB, BB wins the blinds
- If one or more players shove, later positions can call or fold (no re-raise since it's already all-in)

## Project structure
```
aof-solver/
├── src/
│   ├── equity.py          # Precompute & load 169x169 equity matrix
│   ├── solver.py          # Nash equilibrium solver (iterative best-response)
│   ├── nodelock.py        # Fix ranges, re-solve exploitative response
│   ├── hands.py           # Hand representations, rankings, range parsing
│   └── dashboard.py       # Flask app + API endpoints
├── data/
│   └── equity_matrix.npy  # Precomputed equities (generated once)
├── templates/
│   └── index.html         # Dashboard UI
├── scripts/
│   └── generate_equities.py  # One-time equity matrix generation
├── tests/
│   ├── test_equity.py
│   ├── test_solver.py
│   ├── test_nodelock.py
│   └── test_hands.py
├── requirements.txt
├── CLAUDE.md              # This file — project instructions
└── AGENTS.md              # OpenClaw agent task definitions
```

## Architecture

### Equity Engine (`src/equity.py`)
- Precomputed 169x169 equity matrix: equity of each canonical hand vs each other canonical hand
- Canonical hands: 169 types (13 pairs + 78 suited + 78 offsuit)
- Each matchup is weighted by number of combos (e.g., AKs vs QQ accounts for suit removal)
- Matrix is symmetric: `equity[A][B] = 1 - equity[B][A]`
- Stored as numpy array in `data/equity_matrix.npy` (~230KB)
- Hand-vs-range equity: weighted average of hand-vs-each-hand-in-range equities
- Generation uses `eval7` library to enumerate all possible boards (or Monte Carlo with high sample count)

### Hand Representations (`src/hands.py`)
- 169 canonical hands ordered by strength (from charts/standard ranking)
- Each hand: name ("AKs"), type (pair/suited/offsuit), combos count (6/4/12)
- Range representation: ordered list of hands with inclusion flag, or top-N% notation
- Range parsing: "22+, A2s+, KTo+" → set of canonical hands
- Combo counting with card removal (when computing equity vs specific hands)

### Nash Solver (`src/solver.py`)
- Algorithm: iterative best-response
  1. Start with initial guess (e.g., push top 50% from each position)
  2. For each position, compute best response given other positions' current strategies
  3. Repeat until convergence (strategy changes < epsilon)
- Must handle multiple decision nodes:
  - CO: push or fold (first to act)
  - BTN: facing CO push → call or fold; CO folded → push or fold
  - SB: facing various scenarios → call or fold; or push if folded to
  - BB: facing various scenarios → call or fold; or check if all fold
- Each "push range" is a set of hands; each "call range" depends on who pushed
- Convergence target: <0.1% range change between iterations
- Performance target: solve in <1 second

### Nodelocking (`src/nodelock.py`)
- Lock any player's strategy to a fixed range
- Re-solve remaining players' optimal responses
- Same iterative best-response but skip locked players
- Support locking multiple players simultaneously
- Use case: "BTN pushes 45% instead of Nash 28% — what's my optimal call range from BB?"

### Dashboard (`src/dashboard.py`)
- Flask web app with single-page dashboard
- **Solver panel**: position selector, solve button, displays push/call ranges as 13x13 grid
- **Hand grid**: 13x13 heatmap (suited top-right, offsuit bottom-left, pairs diagonal)
  - Color intensity = included in range
  - Hover tooltip: hand name, EV, equity vs calling range
- **Nodelock panel**: per-position range slider (0-100%) or text input for custom range
  - Lock checkbox per position
  - Re-solve button → updates all unlocked positions
- **Range comparison**: Nash vs nodelocked side-by-side grids
- **EV table**: per-hand EV for push vs fold, sortable
- All computation happens server-side via API endpoints, dashboard is pure display

### API Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/solve` | POST | Solve Nash equilibrium, returns all ranges |
| `/api/nodelock` | POST | Solve with locked positions |
| `/api/equity` | GET | Hand vs range equity lookup |
| `/api/hand_info` | GET | Hand ranking, combos, type |
| `/api/range` | GET | Expand range notation to hand list |

## Key design decisions
- Stack size is always 10bb — no need to parameterize
- Blinds are always 0.5bb/1bb — no antes
- 4-max only (CO/BTN/SB/BB)
- Equity matrix is precomputed once, loaded at startup
- All solving happens in-memory with numpy — no external solver needed
- Dashboard uses same Flask + vanilla JS pattern as the poker_bot dashboard

## Running
```bash
# Install deps
pip install -r requirements.txt

# Generate equity matrix (one-time, takes a few minutes)
python scripts/generate_equities.py

# Run dashboard
python src/dashboard.py

# Run tests
pytest tests/
```

## Future integration with poker_bot
This solver will eventually replace the static `charts.py` in `poker_bot/`. Integration points:
- `poker_bot` tracks opponent push frequencies from hand history
- Feeds frequencies into solver's nodelock to get exploitative ranges
- Solver returns optimal action for each hand in real-time
- Import: `from aof_solver import solve, nodelock`

## Dependencies
- `eval7` — fast poker hand evaluation (for equity matrix generation)
- `numpy` — matrix operations
- `flask` — web dashboard
- `pytest` — testing
