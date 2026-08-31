"""API versioning: /v1 is the contract, the bare path is a deprecated alias.

These tests issue real requests rather than inspecting `app.routes`. The first
version did inspect it and passed locally while failing in CI: Starlette 1.6 no
longer lists included-router routes on that attribute, though the endpoints
themselves serve perfectly. Asserting on an internal attribute tested the
framework's bookkeeping instead of whether the API works.

A route that exists answers something -- 200, 401, 405, 422. Only a route that
does not exist answers 404, so that is what these assert on.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.server import API_VERSION, app  # noqa: E402

client = TestClient(app)

# (method, path) for every data endpoint that must live under /v1.
VERSIONED = [('GET', '/events'), ('POST', '/decide'), ('POST', '/events/incident'), ('GET', '/proactive/outreach')]

# Infrastructure routes are deliberately NOT versioned: they describe the
# process, not the API, and a monitoring system should not have to follow an API
# version bump to keep scraping.
UNVERSIONED_BY_DESIGN = ["/health", "/metrics", "/version"]


def reaches(method, path):
    """True when the route exists, whatever it then decides to answer."""
    response = client.request(method, path, json={})
    return response.status_code != 404


@pytest.mark.parametrize("method,path", VERSIONED)
def test_every_data_endpoint_is_served_under_v1(method, path):
    assert reaches(method, f"/{API_VERSION}{path}"), f"{path} missing under /v1"


@pytest.mark.parametrize("method,path", VERSIONED)
def test_unversioned_alias_still_serves_existing_consumers(method, path):
    """Removing the alias is a breaking change; it is a deprecation path."""
    assert reaches(method, path), f"{path} alias removed -- that breaks consumers"


@pytest.mark.parametrize("path", UNVERSIONED_BY_DESIGN)
def test_infrastructure_endpoints_are_not_versioned(path):
    assert reaches("GET", path)
    assert not reaches("GET", f"/{API_VERSION}{path}"), (
        f"{path} should not be versioned -- monitoring must not chase API versions"
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
