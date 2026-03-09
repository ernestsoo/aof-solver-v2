"""Flask dashboard backend for AoF Nash solver.

Loads the equity matrix at startup. If the matrix is missing, all
data-dependent endpoints return 503 Service Unavailable. CORS headers
are added to every response for local development.
"""

import os

import numpy as np
from flask import Flask, jsonify, request

from src.equity import load_equity_matrix
from src.hands import COMBO_WEIGHTS, HAND_NAMES
from src.solver import SolverResult, solve_nash

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


@app.route("/api/nodelock", methods=["POST"])
def nodelock():
    """POST /api/nodelock — run exploitative nodelock solve.

    Body: {"locks": {"CO": 45, "BTN_open": "22+,A2s+"}}
    Stub: returns 501 Not Implemented (task 5.3).
    """
    if not matrix_loaded:
        return _matrix_unavailable()
    return jsonify({"error": "Not implemented"}), 501


@app.route("/api/hand_equity", methods=["GET"])
def hand_equity():
    """GET /api/hand_equity?hand=AKs&vs=top30 — equity lookup.

    Stub: returns 501 Not Implemented (task 5.4).
    """
    if not matrix_loaded:
        return _matrix_unavailable()
    return jsonify({"error": "Not implemented"}), 501


@app.route("/api/hand_info", methods=["GET"])
def hand_info():
    """GET /api/hand_info?hand=AKs — rank, percentile, combos.

    Stub: returns 501 Not Implemented (task 5.4).
    """
    if not matrix_loaded:
        return _matrix_unavailable()
    return jsonify({"error": "Not implemented"}), 501


@app.route("/api/range", methods=["GET"])
def range_expand():
    """GET /api/range?notation=22+,A2s+ — expand range notation to hand list.

    Stub: returns 501 Not Implemented (task 5.4).
    """
    if not matrix_loaded:
        return _matrix_unavailable()
    return jsonify({"error": "Not implemented"}), 501


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
