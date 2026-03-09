"""Tests for src/dashboard.py Flask app skeleton.

All tests use Flask's test client — no server is started.
Patches src.dashboard.matrix_loaded to simulate missing/present matrix.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest

import src.dashboard as dash

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
# Stub endpoints return 501
# ---------------------------------------------------------------------------

class TestStubEndpoints501:
    @pytest.mark.parametrize("url,method", [
        ("/api/solve", "GET"),
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
