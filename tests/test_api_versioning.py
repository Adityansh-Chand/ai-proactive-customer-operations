"""API versioning: /v1 is the contract, the bare path is a deprecated alias.

Both are served by one set of handlers. These tests exist because "we added
versioning" is easy to claim and easy to half-do -- a router mounted but not
reachable, or an alias that quietly diverges from the versioned route.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.server import API_VERSION, app  # noqa: E402

client = TestClient(app)

# Endpoints that must exist under /v1. Infrastructure routes (/health, /metrics,
# /version) are deliberately NOT versioned: they describe the process, not the
# API, and a monitoring system should not have to follow an API version bump.
VERSIONED = ['/events', '/decide', '/events/incident', '/proactive/outreach']
UNVERSIONED_BY_DESIGN = ["/health", "/metrics", "/version"]


def registered():
    return {route.path for route in app.routes if hasattr(route, "path")}


def test_every_data_endpoint_is_served_under_v1():
    paths = registered()
    for endpoint in VERSIONED:
        assert f"/{API_VERSION}{endpoint}" in paths, f"{endpoint} missing under /v1"


def test_unversioned_alias_still_serves_existing_consumers():
    """Removing the alias is a breaking change; it is a deprecation path."""
    paths = registered()
    for endpoint in VERSIONED:
        assert endpoint in paths, f"{endpoint} alias removed -- that breaks consumers"


def test_infrastructure_endpoints_are_not_versioned():
    paths = registered()
    for endpoint in UNVERSIONED_BY_DESIGN:
        assert endpoint in paths
        assert f"/{API_VERSION}{endpoint}" not in paths, (
            f"{endpoint} should not be versioned -- monitoring must not chase API versions"
        )


def test_version_endpoint_declares_what_is_supported():
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert body["current"] == API_VERSION
    assert API_VERSION in body["supported"]
    assert body["unversioned_alias"]["status"] == "deprecated"


def test_health_is_reachable_without_a_key():
    """Probes must not need credentials."""
    assert client.get("/health").status_code == 200
