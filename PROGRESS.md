# AoF Solver — Progress Tracker

**Legend:** `[ ]` not started | `[~]` in progress | `[x]` complete | `[!]` needs human testing

---

## Phase 1: Hand Representations
- [ ] 1.1 — Hand constants and generation (`src/hands.py`)
- [ ] 1.2 — Hand ranking (1-169 standard order)
- [ ] 1.3 — Range parsing and operations
- [ ] 1.4 — Grid mapping (13x13)
- [ ] 1.5 — Combo counting with card removal
- [ ] 1.6 — Tests for hands.py

## Phase 2: Equity Engine
- [ ] 2.1 — Equity matrix generator script (`scripts/generate_equities.py`)
- [ ] 2.2 — Equity lookup module (`src/equity.py`)
- [ ] 2.3 — Tests for equity.py

## Phase 3: Nash Solver
- [ ] 3.1 — EV calculation functions (`src/solver.py`)
- [ ] 3.2 — Iterative best-response solver
- [ ] 3.3 — All call range scenarios
- [ ] 3.4 — Tests for solver.py

## Phase 4: Nodelocking
- [ ] 4.1 — Nodelock solver (`src/nodelock.py`)
- [ ] 4.2 — Exploitability metric
- [ ] 4.3 — Tests for nodelock.py

## Phase 5: Dashboard Backend
- [ ] 5.1 — Flask app + solve endpoint (`src/dashboard.py`)
- [ ] 5.2 — Nodelock endpoint
- [ ] 5.3 — Utility endpoints

## Phase 6: Dashboard Frontend
- [ ] 6.1 — Page layout + Nash grids (`templates/index.html`)
- [ ] 6.2 — Call range viewer
- [ ] 6.3 — Nodelock panel
- [ ] 6.4 — Range comparison view
- [ ] 6.5 — Hover tooltips and polish

---

## Notes
_(Agents: add notes here about decisions, issues, deviations from spec)_
