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
    # /api/range does NOT guard on matrix — excluded from these tests.
    @pytest.mark.parametrize("url,method", [
        ("/api/solve", "GET"),
        ("/api/nodelock", "POST"),
        ("/api/hand_equity", "GET"),
        ("/api/hand_info", "GET"),
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
    ])
    def test_503_body_has_matrix_loaded_false(self, client_no_matrix, url, method):
        if method == "GET":
            resp = client_no_matrix.get(url)
        else:
            resp = client_no_matrix.post(url, json={})
        data = json.loads(resp.data)
        assert data.get("matrix_loaded") is False




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


# ---------------------------------------------------------------------------
# /api/nodelock — task 5.3
# ---------------------------------------------------------------------------

def _make_nodelock_result() -> SolverResult:
    """Minimal nodelock SolverResult with non-zero exploitability."""
    strategies = {name: np.zeros(169) for name in [
        "push_co", "push_btn_open", "push_sb_open",
        "call_btn_vs_co",
        "call_sb_vs_co", "call_sb_vs_btn", "call_sb_vs_co_btn",
        "call_bb_vs_sb", "call_bb_vs_btn", "call_bb_vs_co",
        "call_bb_vs_btn_sb", "call_bb_vs_co_sb", "call_bb_vs_co_btn",
        "call_bb_vs_co_btn_sb",
    ]}
    ev_table = {name: np.ones(169) * 0.1 for name in strategies}
    return SolverResult(
        strategies=strategies,
        ev_table=ev_table,
        iterations=10,
        converged=True,
        exploitability=0.05,
    )


class TestNodelockEndpoint:
    @pytest.fixture
    def client_ready(self):
        """Matrix loaded, Nash result available."""
        nash = _make_nash_result()
        with patch("src.dashboard.matrix_loaded", True), \
             patch("src.dashboard._nash_result", nash):
            yield app.test_client()

    @pytest.fixture
    def client_nash_none(self):
        """Matrix loaded but Nash result is None."""
        with patch("src.dashboard.matrix_loaded", True), \
             patch("src.dashboard._nash_result", None):
            yield app.test_client()

    # --- 503 when matrix missing ---

    def test_503_when_no_matrix(self, client_no_matrix):
        resp = client_no_matrix.post("/api/nodelock", json={"locks": {"CO": 45}})
        assert resp.status_code == 503

    # --- 500 when nash result not ready ---

    def test_500_when_nash_none(self, client_nash_none):
        with patch("src.dashboard.nodelock_solve", return_value=_make_nodelock_result()):
            resp = client_nash_none.post("/api/nodelock", json={"locks": {"CO": 45}})
        assert resp.status_code == 500
        data = json.loads(resp.data)
        assert "error" in data

    # --- 400 bad requests ---

    def test_400_missing_body(self, client_ready):
        resp = client_ready.post("/api/nodelock", data="not json",
                                 content_type="application/json")
        assert resp.status_code == 400

    def test_400_no_locks_key(self, client_ready):
        resp = client_ready.post("/api/nodelock", json={"foo": "bar"})
        assert resp.status_code == 400

    def test_400_empty_locks(self, client_ready):
        resp = client_ready.post("/api/nodelock", json={"locks": {}})
        assert resp.status_code == 400

    def test_400_unknown_lock_key(self, client_ready):
        resp = client_ready.post("/api/nodelock", json={"locks": {"UNKNOWN_POS": 45}})
        assert resp.status_code == 400

    def test_400_invalid_range_notation(self, client_ready):
        resp = client_ready.post("/api/nodelock", json={"locks": {"CO": "BADHAND+++"}})
        assert resp.status_code == 400

    def test_400_pct_out_of_range(self, client_ready):
        resp = client_ready.post("/api/nodelock", json={"locks": {"CO": 150.0}})
        assert resp.status_code == 400

    def test_400_invalid_value_type(self, client_ready):
        resp = client_ready.post("/api/nodelock", json={"locks": {"CO": [1, 2, 3]}})
        assert resp.status_code == 400

    # --- 200 success (mocked nodelock_solve + compare_vs_nash) ---

    @pytest.fixture
    def client_mocked_solve(self):
        """Patch nodelock_solve and compare_vs_nash to avoid heavy compute."""
        nash = _make_nash_result()
        nl = _make_nodelock_result()
        comparison = {
            "push_co": 0.01, "exploitability_nash": 0.0012,
            "exploitability_nodelock": 0.05, "exploitability_delta": 0.0488,
        }
        with patch("src.dashboard.matrix_loaded", True), \
             patch("src.dashboard._nash_result", nash), \
             patch("src.dashboard.nodelock_solve", return_value=nl), \
             patch("src.dashboard.compare_vs_nash", return_value=comparison):
            yield app.test_client()

    def test_200_with_pct_lock(self, client_mocked_solve):
        resp = client_mocked_solve.post("/api/nodelock", json={"locks": {"CO": 45}})
        assert resp.status_code == 200

    def test_200_with_range_notation_lock(self, client_mocked_solve):
        resp = client_mocked_solve.post("/api/nodelock",
                                        json={"locks": {"BTN_open": "22+,A2s+"}})
        assert resp.status_code == 200

    def test_200_with_direct_strategy_name(self, client_mocked_solve):
        resp = client_mocked_solve.post("/api/nodelock",
                                        json={"locks": {"push_co": 30}})
        assert resp.status_code == 200

    def test_response_has_strategies_key(self, client_mocked_solve):
        data = json.loads(client_mocked_solve.post(
            "/api/nodelock", json={"locks": {"CO": 20}}).data)
        assert "strategies" in data

    def test_response_has_ev_table_key(self, client_mocked_solve):
        data = json.loads(client_mocked_solve.post(
            "/api/nodelock", json={"locks": {"CO": 20}}).data)
        assert "ev_table" in data

    def test_response_has_metadata_key(self, client_mocked_solve):
        data = json.loads(client_mocked_solve.post(
            "/api/nodelock", json={"locks": {"CO": 20}}).data)
        assert "metadata" in data

    def test_response_has_comparison_key(self, client_mocked_solve):
        data = json.loads(client_mocked_solve.post(
            "/api/nodelock", json={"locks": {"CO": 20}}).data)
        assert "comparison" in data

    def test_metadata_keys(self, client_mocked_solve):
        data = json.loads(client_mocked_solve.post(
            "/api/nodelock", json={"locks": {"CO": 20}}).data)
        meta = data["metadata"]
        assert "iterations" in meta
        assert "converged" in meta
        assert "exploitability" in meta

    def test_metadata_values(self, client_mocked_solve):
        data = json.loads(client_mocked_solve.post(
            "/api/nodelock", json={"locks": {"CO": 20}}).data)
        meta = data["metadata"]
        assert meta["iterations"] == 10
        assert meta["converged"] is True
        assert abs(meta["exploitability"] - 0.05) < 1e-9

    def test_strategies_has_14_keys(self, client_mocked_solve):
        data = json.loads(client_mocked_solve.post(
            "/api/nodelock", json={"locks": {"CO": 20}}).data)
        assert len(data["strategies"]) == 14

    def test_strategy_has_169_hands(self, client_mocked_solve):
        data = json.loads(client_mocked_solve.post(
            "/api/nodelock", json={"locks": {"CO": 20}}).data)
        assert len(data["strategies"]["push_co"]) == 169

    def test_strategy_uses_canonical_hand_names(self, client_mocked_solve):
        data = json.loads(client_mocked_solve.post(
            "/api/nodelock", json={"locks": {"CO": 20}}).data)
        push_co = data["strategies"]["push_co"]
        assert "AA" in push_co
        assert "72o" in push_co
        assert set(push_co.keys()) == set(HAND_NAMES)

    def test_comparison_exploitability_keys(self, client_mocked_solve):
        data = json.loads(client_mocked_solve.post(
            "/api/nodelock", json={"locks": {"CO": 20}}).data)
        cmp = data["comparison"]
        assert "exploitability_nash" in cmp
        assert "exploitability_nodelock" in cmp
        assert "exploitability_delta" in cmp

    def test_comparison_values(self, client_mocked_solve):
        data = json.loads(client_mocked_solve.post(
            "/api/nodelock", json={"locks": {"CO": 20}}).data)
        cmp = data["comparison"]
        assert abs(cmp["exploitability_nash"] - 0.0012) < 1e-9
        assert abs(cmp["exploitability_nodelock"] - 0.05) < 1e-9
        assert abs(cmp["exploitability_delta"] - 0.0488) < 1e-9


# ---------------------------------------------------------------------------
# /api/hand_equity — task 5.4
# ---------------------------------------------------------------------------

# Fake 169x169 equity matrix: 0.6 everywhere, diagonal 0 (hand vs itself).
_FAKE_MATRIX = np.full((169, 169), 0.6, dtype=np.float32)
np.fill_diagonal(_FAKE_MATRIX, 0.0)


class TestHandEquityEndpoint:
    @pytest.fixture
    def client_eq(self):
        with patch("src.dashboard.matrix_loaded", True), \
             patch("src.dashboard.equity_matrix", _FAKE_MATRIX):
            yield app.test_client()

    # --- 503 ---

    def test_503_when_no_matrix(self, client_no_matrix):
        resp = client_no_matrix.get("/api/hand_equity?hand=AKs&vs=top30")
        assert resp.status_code == 503

    # --- 400 bad params ---

    def test_400_missing_hand(self, client_eq):
        resp = client_eq.get("/api/hand_equity?vs=top30")
        assert resp.status_code == 400

    def test_400_missing_vs(self, client_eq):
        resp = client_eq.get("/api/hand_equity?hand=AKs")
        assert resp.status_code == 400

    def test_400_unknown_hand(self, client_eq):
        resp = client_eq.get("/api/hand_equity?hand=BADHAND&vs=top30")
        assert resp.status_code == 400

    def test_400_invalid_topn(self, client_eq):
        resp = client_eq.get("/api/hand_equity?hand=AKs&vs=topXYZ")
        assert resp.status_code == 400

    def test_400_empty_range(self, client_eq):
        # top0 produces empty range
        resp = client_eq.get("/api/hand_equity?hand=AKs&vs=top0")
        assert resp.status_code == 400

    # --- 200 success ---

    def test_200_topn(self, client_eq):
        resp = client_eq.get("/api/hand_equity?hand=AKs&vs=top30")
        assert resp.status_code == 200

    def test_200_range_notation(self, client_eq):
        resp = client_eq.get("/api/hand_equity?hand=AA&vs=22%2B")  # "22+" URL-encoded
        assert resp.status_code == 200

    def test_response_has_required_keys(self, client_eq):
        data = json.loads(client_eq.get("/api/hand_equity?hand=AKs&vs=top30").data)
        assert "hand" in data
        assert "vs_range" in data
        assert "equity" in data
        assert "vs_count" in data

    def test_hand_field_matches_param(self, client_eq):
        data = json.loads(client_eq.get("/api/hand_equity?hand=AKs&vs=top30").data)
        assert data["hand"] == "AKs"

    def test_equity_is_float(self, client_eq):
        data = json.loads(client_eq.get("/api/hand_equity?hand=AKs&vs=top30").data)
        assert isinstance(data["equity"], float)

    def test_vs_count_matches_vs_range_length(self, client_eq):
        data = json.loads(client_eq.get("/api/hand_equity?hand=AKs&vs=top30").data)
        assert data["vs_count"] == len(data["vs_range"])

    def test_vs_range_is_list_of_strings(self, client_eq):
        data = json.loads(client_eq.get("/api/hand_equity?hand=AKs&vs=top30").data)
        assert isinstance(data["vs_range"], list)
        assert all(isinstance(h, str) for h in data["vs_range"])

    def test_equity_in_valid_range(self, client_eq):
        data = json.loads(client_eq.get("/api/hand_equity?hand=AKs&vs=top30").data)
        assert 0.0 <= data["equity"] <= 1.0

    def test_topn_percent_notation(self, client_eq):
        resp = client_eq.get("/api/hand_equity?hand=AKs&vs=top30%25")  # "top30%"
        assert resp.status_code == 200

    def test_range_notation_vs(self, client_eq):
        resp = client_eq.get("/api/hand_equity?hand=AKs&vs=TT%2B%2CAKs")  # "TT+,AKs"
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/hand_info — task 5.4
# ---------------------------------------------------------------------------

class TestHandInfoEndpoint:
    @pytest.fixture
    def client_hi(self):
        with patch("src.dashboard.matrix_loaded", True):
            yield app.test_client()

    # --- 503 ---

    def test_503_when_no_matrix(self, client_no_matrix):
        resp = client_no_matrix.get("/api/hand_info?hand=AKs")
        assert resp.status_code == 503

    # --- 400 ---

    def test_400_missing_hand(self, client_hi):
        resp = client_hi.get("/api/hand_info")
        assert resp.status_code == 400

    def test_400_unknown_hand(self, client_hi):
        resp = client_hi.get("/api/hand_info?hand=ZZZZ")
        assert resp.status_code == 400

    # --- 200 ---

    def test_200_known_hand(self, client_hi):
        resp = client_hi.get("/api/hand_info?hand=AA")
        assert resp.status_code == 200

    def test_response_has_required_keys(self, client_hi):
        data = json.loads(client_hi.get("/api/hand_info?hand=AA").data)
        for key in ("hand", "rank", "percentile", "combos", "hand_type"):
            assert key in data

    def test_aa_rank_is_1(self, client_hi):
        data = json.loads(client_hi.get("/api/hand_info?hand=AA").data)
        assert data["rank"] == 1

    def test_72o_rank_is_169(self, client_hi):
        data = json.loads(client_hi.get("/api/hand_info?hand=72o").data)
        assert data["rank"] == 169

    def test_aa_percentile_is_100(self, client_hi):
        data = json.loads(client_hi.get("/api/hand_info?hand=AA").data)
        assert abs(data["percentile"] - 100.0) < 0.01

    def test_72o_percentile_is_0(self, client_hi):
        data = json.loads(client_hi.get("/api/hand_info?hand=72o").data)
        assert abs(data["percentile"] - 0.0) < 0.01

    def test_pair_combos_is_6(self, client_hi):
        data = json.loads(client_hi.get("/api/hand_info?hand=AA").data)
        assert data["combos"] == 6
        assert data["hand_type"] == "pair"

    def test_suited_combos_is_4(self, client_hi):
        data = json.loads(client_hi.get("/api/hand_info?hand=AKs").data)
        assert data["combos"] == 4
        assert data["hand_type"] == "suited"

    def test_offsuit_combos_is_12(self, client_hi):
        data = json.loads(client_hi.get("/api/hand_info?hand=AKo").data)
        assert data["combos"] == 12
        assert data["hand_type"] == "offsuit"

    def test_hand_field_matches_param(self, client_hi):
        data = json.loads(client_hi.get("/api/hand_info?hand=TT").data)
        assert data["hand"] == "TT"


# ---------------------------------------------------------------------------
# /api/range — task 5.4
# ---------------------------------------------------------------------------

class TestRangeEndpoint:
    # No matrix guard — use plain client fixture (matrix_loaded=True)
    # but also verify it works with client_no_matrix.

    def test_no_503_when_matrix_missing(self, client_no_matrix):
        resp = client_no_matrix.get("/api/range?notation=AA")
        assert resp.status_code != 503

    # --- 400 ---

    def test_400_missing_notation(self, client):
        resp = client.get("/api/range")
        assert resp.status_code == 400

    def test_400_empty_notation(self, client):
        resp = client.get("/api/range?notation=")
        assert resp.status_code == 400

    # --- 200 ---

    def test_200_single_hand(self, client):
        resp = client.get("/api/range?notation=AA")
        assert resp.status_code == 200

    def test_200_pair_plus(self, client):
        resp = client.get("/api/range?notation=TT%2B")  # "TT+"
        assert resp.status_code == 200

    def test_200_complex_notation(self, client):
        resp = client.get("/api/range?notation=22%2B%2CA2s%2B")  # "22+,A2s+"
        assert resp.status_code == 200

    def test_response_has_required_keys(self, client):
        data = json.loads(client.get("/api/range?notation=AA").data)
        for key in ("notation", "hands", "count", "combo_count"):
            assert key in data

    def test_notation_field_preserved(self, client):
        data = json.loads(client.get("/api/range?notation=AA").data)
        assert data["notation"] == "AA"

    def test_hands_is_list_of_strings(self, client):
        data = json.loads(client.get("/api/range?notation=AA").data)
        assert isinstance(data["hands"], list)
        assert all(isinstance(h, str) for h in data["hands"])

    def test_count_matches_hands_length(self, client):
        data = json.loads(client.get("/api/range?notation=TT%2B").data)
        assert data["count"] == len(data["hands"])

    def test_aa_single_hand(self, client):
        data = json.loads(client.get("/api/range?notation=AA").data)
        assert data["hands"] == ["AA"]
        assert data["count"] == 1
        assert data["combo_count"] == 6

    def test_pair_plus_includes_all_pairs(self, client):
        data = json.loads(client.get("/api/range?notation=22%2B").data)
        assert "AA" in data["hands"]
        assert "22" in data["hands"]
        assert data["count"] == 13

    def test_combo_count_is_integer(self, client):
        data = json.loads(client.get("/api/range?notation=AA").data)
        assert isinstance(data["combo_count"], int)

    def test_random_returns_169_hands(self, client):
        data = json.loads(client.get("/api/range?notation=random").data)
        assert data["count"] == 169
        assert data["combo_count"] == 1326
