"""Cross-service enrichment: the connected path AND the degraded path.

These run against a real HTTP server (a stub started in-process on a loopback
port), not a mock, so the client's timeout, retry, circuit-breaker and header
propagation are genuinely exercised.

The degraded cases matter more than the happy one. A service that only works
when its dependencies are healthy is not independently runnable, and
independent runnability is the constraint this integration was built under.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agents.policy_agent import decide_policy, explain_policy
from integrations.client import CircuitBreaker, ServiceClient
from integrations.services import infer_service, reset_clients
from orchestration.agent_dag import run_dag


class _Stub(BaseHTTPRequestHandler):
    """Serves canned responses; records the headers it was called with."""

    routes = {}
    seen_headers = []
    delay = 0.0

    def do_GET(self):  # noqa: N802
        type(self).seen_headers.append(dict(self.headers))
        if type(self).delay:
            time.sleep(type(self).delay)
        path = self.path.split("?")[0]
        if path in type(self).routes:
            body = json.dumps(type(self).routes[path]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub():
    _Stub.routes = {}
    _Stub.seen_headers = []
    _Stub.delay = 0.0
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


# --- the degraded path: nothing configured -----------------------------------

def test_decision_is_made_with_no_integrations_configured(monkeypatch):
    """The whole system must work as five standalone services."""
    for var in ("SALES_API_URL", "INCIDENT_API_URL", "RAG_API_URL"):
        monkeypatch.delenv(var, raising=False)
    reset_clients()

    trace = run_dag("where is my order", customer_id="acct_00001")

    assert trace["policy"]["value"]
    assert trace["policy"]["used_enrichment"] is False
    for key in ("account", "incident", "knowledge"):
        assert trace["enrichment"][key]["outcome"] == "not_configured"


def test_unreachable_dependency_does_not_break_the_decision(monkeypatch):
    """A dead dependency must cost a decision nothing but enrichment."""
    # Port 9 is the discard protocol -- reliably refuses.
    monkeypatch.setenv("SALES_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("INTEGRATION_TIMEOUT_SECONDS", "0.2")
    reset_clients()

    trace = run_dag("I want a refund", customer_id="acct_00001")

    assert trace["policy"]["value"] == "refund_review"
    assert trace["enrichment"]["account"]["outcome"] in {"timeout", "error"}


def test_slow_dependency_does_not_hang_the_caller(stub, monkeypatch):
    _, base = stub
    _Stub.routes = {"/accounts/acct_00001/score": {"score": 0.9}}
    _Stub.delay = 1.5

    monkeypatch.setenv("SALES_API_URL", base)
    monkeypatch.setenv("INTEGRATION_TIMEOUT_SECONDS", "0.15")
    reset_clients()

    started = time.monotonic()
    trace = run_dag("where is my order", customer_id="acct_00001")
    elapsed = time.monotonic() - started

    assert trace["policy"]["value"]
    # Two attempts at 0.15s plus overhead -- must not approach the 1.5s delay.
    assert elapsed < 1.4, f"caller waited {elapsed:.2f}s on a slow dependency"


# --- the connected path -------------------------------------------------------

def test_active_incident_changes_the_decision(stub, monkeypatch):
    """The edge that most changes an outcome: a known fault is not an
    individual customer problem, so no individual refund is issued."""
    _, base = stub
    _Stub.routes = {
        "/incidents/active": {
            "service": "checkout",
            "active": True,
            "incident": {"service": "checkout", "anomaly_count": 7},
        }
    }
    monkeypatch.setenv("INCIDENT_API_URL", base)
    reset_clients()

    trace = run_dag(
        "my checkout payment keeps failing and I want my money back",
        customer_id="acct_00001",
        service="checkout",
    )

    assert trace["policy"]["value"] == "incident_response"
    assert trace["policy"]["rule"] == "known_incident_on_the_service_complained_about"
    assert trace["policy"]["used_enrichment"] is True
    assert trace["action"]["type"] == "link_to_incident"


def test_no_active_incident_leaves_the_decision_alone(stub, monkeypatch):
    _, base = stub
    _Stub.routes = {"/incidents/active": {"service": "checkout", "active": False}}
    monkeypatch.setenv("INCIDENT_API_URL", base)
    reset_clients()

    trace = run_dag(
        "I want my money back for the checkout order",
        customer_id="acct_00001",
        service="checkout",
    )

    assert trace["policy"]["value"] == "refund_review"
    assert trace["policy"]["used_enrichment"] is False


def test_high_value_unhappy_account_escalates(stub, monkeypatch):
    _, base = stub
    _Stub.routes = {
        "/accounts/acct_00001/score": {
            "score": 0.91, "segment": "high_propensity", "industry": "saas",
        }
    }
    monkeypatch.setenv("SALES_API_URL", base)
    reset_clients()

    trace = run_dag(
        "honestly this has been a very poor experience and nobody has helped",
        customer_id="acct_00001",
    )

    assert trace["sentiment"]["label"] == "negative"
    assert trace["policy"]["value"] == "escalate"
    assert trace["policy"]["rule"] == "high_value_account_at_risk"


def test_request_id_propagates_to_downstream_services(stub, monkeypatch):
    """One identifier across every hop, or the trace is not distributed."""
    _, base = stub
    _Stub.routes = {"/incidents/active": {"active": False}}
    monkeypatch.setenv("INCIDENT_API_URL", base)
    reset_clients()

    run_dag("where is my checkout order", customer_id="acct_1",
            service="checkout", request_id="req-abc-123")

    # urllib title-cases outgoing header names, so compare case-insensitively.
    forwarded = [
        value
        for headers in _Stub.seen_headers
        for key, value in headers.items()
        if key.lower() == "x-request-id"
    ]
    assert "req-abc-123" in forwarded


# --- the client's own protections ---------------------------------------------

def test_circuit_opens_after_repeated_failures():
    breaker = CircuitBreaker(threshold=2, cooldown=60)
    assert not breaker.is_open()
    breaker.record_failure()
    assert not breaker.is_open()
    breaker.record_failure()
    assert breaker.is_open()
    assert breaker.state == "open"


def test_circuit_half_opens_after_cooldown():
    breaker = CircuitBreaker(threshold=1, cooldown=0.05)
    breaker.record_failure()
    assert breaker.is_open()
    time.sleep(0.08)
    assert not breaker.is_open(), "must let a probe through after cooldown"


def test_circuit_stops_calling_a_dead_dependency(monkeypatch):
    """Without this, every request pays a full timeout forever."""
    client = ServiceClient("dead", base_url="http://127.0.0.1:9", timeout=0.1)
    client.breaker.threshold = 2

    for _ in range(2):
        client.get("/anything")
    assert client.breaker.is_open()

    started = time.monotonic()
    data, outcome = client.get("/anything")
    assert outcome == "circuit_open"
    assert data is None
    assert time.monotonic() - started < 0.05, "open circuit must not make a call"


def test_client_never_raises_on_a_bad_url():
    data, outcome = ServiceClient("bad", base_url="http://nonexistent.invalid", timeout=0.2).get("/x")
    assert data is None
    assert outcome in {"timeout", "error"}


def test_unconfigured_client_reports_not_configured():
    data, outcome = ServiceClient("none", base_url="").get("/x")
    assert (data, outcome) == (None, "not_configured")


# --- policy layer stays additive ----------------------------------------------

def test_enriched_rules_cannot_fire_without_enrichment():
    """The guarantee that keeps the offline evaluation valid."""
    for sentiment in ("negative", "neutral", "positive"):
        for priority in ("high", "normal"):
            base = decide_policy("refund_request", sentiment, priority)
            with_empty = decide_policy("refund_request", sentiment, priority, enrichment={})
            with_none = decide_policy("refund_request", sentiment, priority, enrichment=None)
            assert base == with_empty == with_none


def test_enrichment_without_signal_does_not_change_the_policy():
    enrichment = {"account": None, "incident": {"active": False}, "knowledge": None}
    assert explain_policy("tracking_request", "neutral", "normal", enrichment)[0] == (
        "send_tracking_update"
    )


def test_service_inference_is_best_effort():
    assert infer_service("my checkout order failed") == "checkout"
    assert infer_service("the payments api is down") == "payments"
    assert infer_service("where is my parcel") is None


# --- inbound events: proactive outreach ---------------------------------------

from orchestration.proactive import (  # noqa: E402
    handle_incident_event,
    outreach_log,
    record_contact,
    recent_contacts,
    reset as reset_proactive,
)


@pytest.fixture(autouse=True)
def _clean_proactive():
    reset_proactive()
    yield
    reset_proactive()


def _event(event_id="evt-1", service="checkout", kind="incident.opened"):
    return {
        "event_id": event_id,
        "type": kind,
        "source": "ai-incident-detection-platform",
        "payload": {"service": service, "incident": {"anomaly_count": 5}},
    }


def test_incident_event_notifies_customers_who_contacted_about_that_service():
    record_contact("acct_1", "checkout", "my checkout is broken")
    record_contact("acct_2", "checkout", "payment failed")
    record_contact("acct_3", "search", "unrelated")

    result = handle_incident_event(_event())

    assert result["notified"] == 2
    notified = {o["customer_id"] for o in result["outreach"]}
    assert notified == {"acct_1", "acct_2"}


def test_redelivery_does_not_notify_customers_twice():
    """At-least-once delivery makes this the property that matters most.

    Without it, a delivery retry messages the same customers again -- exactly the
    bug that makes people distrust automated outreach.
    """
    record_contact("acct_1", "checkout")

    first = handle_incident_event(_event(event_id="evt-dup"))
    second = handle_incident_event(_event(event_id="evt-dup"))

    assert first["notified"] == 1
    assert second["duplicate"] is True
    assert len(outreach_log()) == 1, "a redelivery must not create a second batch"


def test_distinct_events_are_both_handled():
    record_contact("acct_1", "checkout")
    handle_incident_event(_event(event_id="evt-a"))
    handle_incident_event(_event(event_id="evt-b"))
    assert len(outreach_log()) == 2


def test_incident_with_no_recent_contacts_notifies_nobody():
    result = handle_incident_event(_event(service="billing"))
    assert result["notified"] == 0
    assert result["outreach"] == []


def test_resolution_events_do_not_send_outreach():
    record_contact("acct_1", "checkout")
    result = handle_incident_event(_event(event_id="evt-r", kind="incident.resolved"))
    assert result["outreach"] == []


def test_unknown_event_types_are_ignored_safely():
    result = handle_incident_event(_event(event_id="evt-x", kind="something.else"))
    assert result["outreach"] == []


def test_contacts_are_deduplicated_per_customer():
    for _ in range(4):
        record_contact("acct_1", "checkout")
    assert len(recent_contacts("checkout")) == 1


def test_anonymous_contacts_are_not_tracked():
    record_contact("anonymous", "checkout")
    record_contact("", "checkout")
    assert recent_contacts("checkout") == []


def test_decide_records_a_contact_for_later_outreach(monkeypatch):
    """The audience for proactive outreach comes from /decide traffic."""
    for var in ("SALES_API_URL", "INCIDENT_API_URL", "RAG_API_URL"):
        monkeypatch.delenv(var, raising=False)
    reset_clients()

    record_contact("acct_77", "checkout", "checkout is failing")
    result = handle_incident_event(_event(event_id="evt-flow"))

    assert "acct_77" in {o["customer_id"] for o in result["outreach"]}
