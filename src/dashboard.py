"""Flask dashboard backend for AoF Nash solver.

Loads the equity matrix at startup. If the matrix is missing, all
data-dependent endpoints return 503 Service Unavailable. CORS headers
are added to every response for local development.
"""

import os

import numpy as np
from flask import Flask, jsonify, render_template, request

from src.equity import load_equity_matrix
from src.hands import (
    COMBO_WEIGHTS,
    HAND_MAP,
    HAND_NAMES,
    mask_to_hands,
    parse_range,
    range_to_mask,
    top_n_percent,
)
from src.nodelock import compare_vs_nash, lock_from_range_pct, nodelock_solve
from src.solver import STRATEGY_NAMES, SolverResult, solve_nash

# ---------------------------------------------------------------------------
# App and matrix initialisation
# ---------------------------------------------------------------------------

app = Flask(__name__)

_MATRIX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "equity_matrix.npy")

try:
    equity_matrix: np.ndarray | None = load_equity_matrix(_MATRIX_PATH)
    matrix_loaded: bool = True
except FileNotFoundError:
    equity_matrix = None
    matrix_loaded = False

# Pre-compute Nash solution at startup (only when matrix is available).
_nash_result: SolverResult | None = None
if matrix_loaded and equity_matrix is not None:
    try:
        _nash_result = solve_nash(equity_matrix, COMBO_WEIGHTS)
    except Exception:
        _nash_result = None

# ---------------------------------------------------------------------------
# CORS — add headers to every response
# ---------------------------------------------------------------------------

@app.after_request
def add_cors_headers(response):
    """Add permissive CORS headers for local development."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.route("/api/<path:_>", methods=["OPTIONS"])
def options_handler(_):
    """Handle pre-flight OPTIONS requests for all /api/* routes."""
    return jsonify({}), 200


# ---------------------------------------------------------------------------
# Helper — 503 when matrix is missing
# ---------------------------------------------------------------------------

def _matrix_unavailable():
    """Return a 503 JSON response when the equity matrix is not loaded."""
    return (
        jsonify({"error": "Equity matrix not loaded", "matrix_loaded": False}),
        503,
    )


# ---------------------------------------------------------------------------
# Root page
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Serve the dashboard SPA."""
    return render_template("index.html")


# Health endpoint (always available)
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    """GET /api/health — liveness check.

    Returns:
        200 JSON: {"status": "ok", "matrix_loaded": <bool>}
    """
    return jsonify({"status": "ok", "matrix_loaded": matrix_loaded}), 200


# ---------------------------------------------------------------------------
# Stub routes for tasks 5.2 – 5.4  (return 501 until implemented)
# ---------------------------------------------------------------------------

@app.route("/api/solve", methods=["GET"])
def solve():
    """GET /api/solve — return cached Nash solution.

    Returns:
        200 JSON: {strategies, ev_table, metadata} where each strategy/ev_table
                  entry maps the 169 canonical hand names to float values.
        503 JSON: if equity matrix is not loaded.
        500 JSON: if solver failed at startup.
    """
    if not matrix_loaded:
        return _matrix_unavailable()
    if _nash_result is None:
        return jsonify({"error": "Solve failed", "detail": "Solver did not produce a result"}), 500

    strategies = {
        name: dict(zip(HAND_NAMES, arr.tolist()))
        for name, arr in _nash_result.strategies.items()
    }
    ev_table = {
        name: dict(zip(HAND_NAMES, arr.tolist()))
        for name, arr in _nash_result.ev_table.items()
    }
    return jsonify({
        "strategies": strategies,
        "ev_table": ev_table,
        "metadata": {
            "iterations": _nash_result.iterations,
            "converged": _nash_result.converged,
            "exploitability": _nash_result.exploitability,
        },
    }), 200


# Mapping from friendly short names to canonical strategy names.
# Also accepts strategy names directly (push_co, call_bb_vs_sb, etc.).
_LOCK_KEY_MAP: dict[str, str] = {
    "CO":             "push_co",
    "BTN_open":       "push_btn_open",
    "SB_open":        "push_sb_open",
    "BTN_vs_CO":      "call_btn_vs_co",
    "SB_vs_CO":       "call_sb_vs_co",
    "SB_vs_BTN":      "call_sb_vs_btn",
    "SB_vs_CO_BTN":   "call_sb_vs_co_btn",
    "BB_vs_SB":       "call_bb_vs_sb",
    "BB_vs_BTN":      "call_bb_vs_btn",
    "BB_vs_CO":       "call_bb_vs_co",
    "BB_vs_BTN_SB":   "call_bb_vs_btn_sb",
    "BB_vs_CO_SB":    "call_bb_vs_co_sb",
    "BB_vs_CO_BTN":   "call_bb_vs_co_btn",
    "BB_vs_CO_BTN_SB":"call_bb_vs_co_btn_sb",
}


@app.route("/api/nodelock", methods=["POST"])
def nodelock():
    """POST /api/nodelock — run exploitative nodelock solve.

    Body: {"locks": {"CO": 45, "BTN_open": "22+,A2s+"}}

    Each lock value is either:
    - A number (0–100): treated as push % — top-N% of hands by strength.
    - A string: parsed as range notation (e.g. "22+,A2s+").

    Lock keys are friendly short names (e.g. "CO", "BTN_open") or direct
    strategy names (e.g. "push_co", "call_bb_vs_sb").

    Returns:
        200 JSON: {strategies, ev_table, metadata, comparison}
        400 JSON: if body is missing, lacks "locks", or locks is empty.
        500 JSON: if Nash result is not ready or solve fails.
        503 JSON: if equity matrix is not loaded.
    """
    if not matrix_loaded:
        return _matrix_unavailable()
    if _nash_result is None:
        return jsonify({"error": "Nash result not ready", "detail": "Solver did not produce a result at startup"}), 500

    body = request.get_json(silent=True)
    if body is None or "locks" not in body or not body["locks"]:
        return jsonify({"error": "Bad request", "detail": "Body must contain a non-empty 'locks' object"}), 400

    locked_strategies: dict[str, np.ndarray] = {}
    for key, value in body["locks"].items():
        strategy_name = _LOCK_KEY_MAP.get(key, key)
        if strategy_name not in STRATEGY_NAMES:
            return jsonify({"error": "Bad request", "detail": f"Unknown lock key: {key!r}"}), 400

        if isinstance(value, (int, float)):
            pct = float(value)
            if not (0.0 <= pct <= 100.0):
                return jsonify({"error": "Bad request", "detail": f"Percentage must be 0–100, got {pct}"}), 400
            arr = lock_from_range_pct(pct, COMBO_WEIGHTS)
        elif isinstance(value, str):
            try:
                hands = parse_range(value)
                if not hands:
                    return jsonify({"error": "Bad request", "detail": f"Range notation {value!r} matched no hands"}), 400
                arr = range_to_mask(hands)
            except (KeyError, Exception) as exc:
                return jsonify({"error": "Bad request", "detail": f"Invalid range notation {value!r}: {exc}"}), 400
        else:
            return jsonify({"error": "Bad request", "detail": f"Lock value must be a number or string, got {type(value).__name__}"}), 400

        locked_strategies[strategy_name] = arr

    try:
        nl_result = nodelock_solve(equity_matrix, COMBO_WEIGHTS, locked_strategies)
    except ValueError as exc:
        return jsonify({"error": "Bad request", "detail": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "Solve failed", "detail": str(exc)}), 500

    comparison = compare_vs_nash(_nash_result, nl_result)

    strategies = {
        name: dict(zip(HAND_NAMES, arr.tolist()))
        for name, arr in nl_result.strategies.items()
    }
    ev_table = {
        name: dict(zip(HAND_NAMES, arr.tolist()))
        for name, arr in nl_result.ev_table.items()
    }
    return jsonify({
        "strategies": strategies,
        "ev_table": ev_table,
        "metadata": {
            "iterations": nl_result.iterations,
            "converged": nl_result.converged,
            "exploitability": nl_result.exploitability,
        },
        "comparison": comparison,
    }), 200


def _parse_vs_param(vs: str) -> tuple[np.ndarray, list[str]]:
    """Parse a vs= query param into (mask, hand_names).

    Accepts:
    - "topN" or "topN%" (e.g. "top30", "top30%"): top-N% by strength.
    - Range notation (e.g. "22+,A2s+"): parsed via parse_range.

    Raises ValueError on invalid input.
    """
    vs = vs.strip()
    low = vs.lower()
    if low.startswith("top"):
        num_str = low[3:].rstrip("%").strip()
        try:
            pct = float(num_str)
        except ValueError:
            raise ValueError(f"Invalid top-N notation: {vs!r}")
        if not (0.0 <= pct <= 100.0):
            raise ValueError(f"Percentage must be 0–100, got {pct}")
        mask = top_n_percent(pct)
        hands = mask_to_hands(mask)
        return mask, hands
    else:
        hands = parse_range(vs)
        mask = range_to_mask(hands)
        return mask, hands


@app.route("/api/hand_equity", methods=["GET"])
def hand_equity():
    """GET /api/hand_equity?hand=AKs&vs=top30 — equity of one hand vs a range.

    Returns:
        200 JSON: {"hand", "vs_range", "equity", "vs_count"}
        400 JSON: missing/invalid params.
        503 JSON: matrix not loaded.
    """
    if not matrix_loaded:
        return _matrix_unavailable()

    hand_name = request.args.get("hand", "").strip()
    vs_param = request.args.get("vs", "").strip()

    if not hand_name:
        return jsonify({"error": "Bad request", "detail": "Missing 'hand' param"}), 400
    if hand_name not in HAND_MAP:
        return jsonify({"error": "Bad request", "detail": f"Unknown hand: {hand_name!r}"}), 400
    if not vs_param:
        return jsonify({"error": "Bad request", "detail": "Missing 'vs' param"}), 400

    try:
        vs_mask, vs_hands = _parse_vs_param(vs_param)
    except ValueError as exc:
        return jsonify({"error": "Bad request", "detail": str(exc)}), 400

    if not vs_hands:
        return jsonify({"error": "Bad request", "detail": f"'vs' param matched no hands: {vs_param!r}"}), 400

    hand_idx = HAND_MAP[hand_name].index
    weighted = vs_mask * COMBO_WEIGHTS
    total = float(weighted.sum())
    if total == 0.0:
        equity = 0.0
    else:
        equity = float(np.dot(equity_matrix[hand_idx], weighted) / total)

    return jsonify({
        "hand": hand_name,
        "vs_range": vs_hands,
        "equity": round(equity, 6),
        "vs_count": len(vs_hands),
    }), 200


@app.route("/api/hand_info", methods=["GET"])
def hand_info():
    """GET /api/hand_info?hand=AKs — rank, percentile, and combo info.

    Returns:
        200 JSON: {"hand", "rank", "percentile", "combos", "hand_type"}
        400 JSON: missing/unknown hand.
        503 JSON: matrix not loaded (consistency guard).
    """
    if not matrix_loaded:
        return _matrix_unavailable()

    hand_name = request.args.get("hand", "").strip()
    if not hand_name:
        return jsonify({"error": "Bad request", "detail": "Missing 'hand' param"}), 400
    if hand_name not in HAND_MAP:
        return jsonify({"error": "Bad request", "detail": f"Unknown hand: {hand_name!r}"}), 400

    info = HAND_MAP[hand_name]
    percentile = round((169 - info.rank) / 168 * 100, 2)
    return jsonify({
        "hand": hand_name,
        "rank": info.rank,
        "percentile": percentile,
        "combos": info.combos,
        "hand_type": info.hand_type,
    }), 200


@app.route("/api/range", methods=["GET"])
def range_expand():
    """GET /api/range?notation=22+,A2s+ — expand range notation to hand list.

    Does not require the equity matrix.

    Returns:
        200 JSON: {"notation", "hands", "count", "combo_count"}
        400 JSON: missing or empty notation.
    """
    notation = request.args.get("notation", "").strip()
    if not notation:
        return jsonify({"error": "Bad request", "detail": "Missing 'notation' param"}), 400

    try:
        hands = parse_range(notation)
    except Exception as exc:
        return jsonify({"error": "Bad request", "detail": f"Invalid notation: {exc}"}), 400

    if not hands:
        return jsonify({"error": "Bad request", "detail": f"Notation matched no hands: {notation!r}"}), 400

    combo_count = sum(HAND_MAP[h].combos for h in hands)
    return jsonify({
        "notation": notation,
        "hands": hands,
        "count": len(hands),
        "combo_count": combo_count,
    }), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
