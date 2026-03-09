"""Tests for src/dashboard.py Flask app skeleton.

All tests use Flask's test client — no server is started.
Patches src.dashboard.matrix_loaded to simulate missing/present matrix.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.dashboard as dash
from src.hands import HAND_NAMES
from src.solver import SolverResult

# Use the module's test client directly; patch module-level booleans per test.
app = dash.app
app.config["TESTING"] = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Test client with matrix_loaded=True (default state)."""
    with patch("src.dashboard.matrix_loaded", True):
        yield app.test_client()


@pytest.fixture
def client_no_matrix():
    """Test client with matrix_loaded=False (simulate missing matrix)."""
    with patch("src.dashboard.matrix_loaded", False):
        yield app.test_client()


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_json_structure(self, client):
        resp = client.get("/api/health")
        data = json.loads(resp.data)
        assert "status" in data
        assert "matrix_loaded" in data

    def test_health_status_ok(self, client):
        resp = client.get("/api/health")
        data = json.loads(resp.data)
        assert data["status"] == "ok"

    def test_health_matrix_loaded_true_when_present(self, client):
        resp = client.get("/api/health")
        data = json.loads(resp.data)
        assert data["matrix_loaded"] is True

    def test_health_matrix_loaded_false_when_missing(self, client_no_matrix):
        resp = client_no_matrix.get("/api/health")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["matrix_loaded"] is False

    def test_health_always_available_even_without_matrix(self, client_no_matrix):
        """Health endpoint must NOT return 503 even when matrix is missing."""
        resp = client_no_matrix.get("/api/health")
        assert resp.status_code != 503


# ---------------------------------------------------------------------------
# 503 when matrix missing
# ---------------------------------------------------------------------------

class TestMatrixMissing503:
    @pytest.mark.parametrize("url,method", [
        ("/api/solve", "GET"),
        ("/api/nodelock", "POST"),
        ("/api/hand_equity", "GET"),
        ("/api/hand_info", "GET"),
        ("/api/range", "GET"),
    ])
    def test_returns_503_when_matrix_missing(self, client_no_matrix, url, method):
        if method == "GET":
            resp = client_no_matrix.get(url)
        else:
            resp = client_no_matrix.post(url, json={})
        assert resp.status_code == 503

    @pytest.mark.parametrize("url,method", [
        ("/api/solve", "GET"),
        ("/api/nodelock", "POST"),
        ("/api/hand_equity", "GET"),
        ("/api/hand_info", "GET"),
        ("/api/range", "GET"),
    ])
    def test_503_body_contains_error_key(self, client_no_matrix, url, method):
        if method == "GET":
            resp = client_no_matrix.get(url)
        else:
            resp = client_no_matrix.post(url, json={})
        data = json.loads(resp.data)
        assert "error" in data

    @pytest.mark.parametrize("url,method", [
        ("/api/solve", "GET"),
        ("/api/nodelock", "POST"),
        ("/api/hand_equity", "GET"),
        ("/api/hand_info", "GET"),
        ("/api/range", "GET"),
    ])
    def test_503_body_has_matrix_loaded_false(self, client_no_matrix, url, method):
        if method == "GET":
            resp = client_no_matrix.get(url)
        else:
            resp = client_no_matrix.post(url, json={})
        data = json.loads(resp.data)
        assert data.get("matrix_loaded") is False


# ---------------------------------------------------------------------------
# Stub endpoints return 501  (excludes /api/solve — now implemented)
# ---------------------------------------------------------------------------

class TestStubEndpoints501:
    @pytest.mark.parametrize("url,method", [
        ("/api/nodelock", "POST"),
        ("/api/hand_equity", "GET"),
        ("/api/hand_info", "GET"),
        ("/api/range", "GET"),
    ])
    def test_stubs_return_501_when_matrix_present(self, client, url, method):
        if method == "GET":
            resp = client.get(url)
        else:
            resp = client.post(url, json={})
        assert resp.status_code == 501


# ---------------------------------------------------------------------------
# /api/solve — task 5.2
# ---------------------------------------------------------------------------

def _make_nash_result() -> SolverResult:
    """Build a minimal SolverResult with fake strategy and EV arrays."""
    strategies = {name: np.zeros(169) for name in [
        "push_co", "push_btn_open", "push_sb_open",
        "call_btn_vs_co",
        "call_sb_vs_co", "call_sb_vs_btn", "call_sb_vs_co_btn",
        "call_bb_vs_sb", "call_bb_vs_btn", "call_bb_vs_co",
        "call_bb_vs_btn_sb", "call_bb_vs_co_sb", "call_bb_vs_co_btn",
        "call_bb_vs_co_btn_sb",
    ]}
    ev_table = {name: np.zeros(169) for name in strategies}
    return SolverResult(
        strategies=strategies,
        ev_table=ev_table,
        iterations=42,
        converged=True,
        exploitability=0.0012,
    )


class TestSolveEndpoint:
    @pytest.fixture
    def client_with_nash(self):
        result = _make_nash_result()
        with patch("src.dashboard.matrix_loaded", True), \
             patch("src.dashboard._nash_result", result):
            yield app.test_client()

    @pytest.fixture
    def client_solve_failed(self):
        with patch("src.dashboard.matrix_loaded", True), \
             patch("src.dashboard._nash_result", None):
            yield app.test_client()

    def test_solve_returns_200(self, client_with_nash):
        resp = client_with_nash.get("/api/solve")
        assert resp.status_code == 200

    def test_solve_returns_503_when_no_matrix(self, client_no_matrix):
        resp = client_no_matrix.get("/api/solve")
        assert resp.status_code == 503

    def test_solve_returns_500_when_solver_failed(self, client_solve_failed):
        resp = client_solve_failed.get("/api/solve")
        assert resp.status_code == 500
        data = json.loads(resp.data)
        assert data.get("error") == "Solve failed"

    def test_solve_response_has_strategies_key(self, client_with_nash):
        data = json.loads(client_with_nash.get("/api/solve").data)
        assert "strategies" in data

    def test_solve_response_has_ev_table_key(self, client_with_nash):
        data = json.loads(client_with_nash.get("/api/solve").data)
        assert "ev_table" in data

    def test_solve_response_has_metadata_key(self, client_with_nash):
        data = json.loads(client_with_nash.get("/api/solve").data)
        assert "metadata" in data

    def test_solve_metadata_keys(self, client_with_nash):
        data = json.loads(client_with_nash.get("/api/solve").data)
        meta = data["metadata"]
        assert "iterations" in meta
        assert "converged" in meta
        assert "exploitability" in meta

    def test_solve_metadata_values(self, client_with_nash):
        data = json.loads(client_with_nash.get("/api/solve").data)
        meta = data["metadata"]
        assert meta["iterations"] == 42
        assert meta["converged"] is True
        assert abs(meta["exploitability"] - 0.0012) < 1e-9

    def test_solve_strategies_contain_all_14_names(self, client_with_nash):
        data = json.loads(client_with_nash.get("/api/solve").data)
        strategies = data["strategies"]
        assert len(strategies) == 14

    def test_solve_strategy_has_169_hands(self, client_with_nash):
        data = json.loads(client_with_nash.get("/api/solve").data)
        push_co = data["strategies"]["push_co"]
        assert len(push_co) == 169

    def test_solve_strategy_hand_keys_are_strings(self, client_with_nash):
        data = json.loads(client_with_nash.get("/api/solve").data)
        push_co = data["strategies"]["push_co"]
        assert all(isinstance(k, str) for k in push_co.keys())

    def test_solve_strategy_values_are_floats(self, client_with_nash):
        data = json.loads(client_with_nash.get("/api/solve").data)
        push_co = data["strategies"]["push_co"]
        assert all(isinstance(v, float) for v in push_co.values())

    def test_solve_strategy_uses_canonical_hand_names(self, client_with_nash):
        data = json.loads(client_with_nash.get("/api/solve").data)
        push_co = data["strategies"]["push_co"]
        assert "AA" in push_co
        assert "72o" in push_co
        # All 169 canonical names present
        assert set(push_co.keys()) == set(HAND_NAMES)

    def test_solve_ev_table_structure_mirrors_strategies(self, client_with_nash):
        data = json.loads(client_with_nash.get("/api/solve").data)
        assert set(data["ev_table"].keys()) == set(data["strategies"].keys())


# ---------------------------------------------------------------------------
# CORS headers
# ---------------------------------------------------------------------------

class TestCorsHeaders:
    def test_cors_header_on_health(self, client):
        resp = client.get("/api/health")
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_cors_allow_methods_present(self, client):
        resp = client.get("/api/health")
        methods = resp.headers.get("Access-Control-Allow-Methods", "")
        assert "GET" in methods
        assert "POST" in methods

    def test_cors_header_on_503_response(self, client_no_matrix):
        resp = client_no_matrix.get("/api/solve")
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_cors_header_on_501_response(self, client):
        resp = client.get("/api/solve")
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
