"""Configured downstream services for decision enrichment.

Each is optional. Unset means "this service does not exist as far as we are
concerned", and the DAG runs exactly as it does standalone.

    SALES_API_URL       propensity for the account the message came from
    INCIDENT_API_URL    whether the service being complained about is degraded
    RAG_API_URL         a grounding passage for the response
    INTEGRATION_API_KEY  X-API-Key sent to all three, when they require one
"""
import os
import re

from integrations.client import NOT_CONFIGURED, OK, ServiceClient

# Kept module-level so the circuit breakers persist across requests -- a breaker
# rebuilt per request would never actually open.
_clients = {}


def _client(name, env_var):
    if name not in _clients:
        _clients[name] = ServiceClient(
            name,
            base_url=os.getenv(env_var, ""),
            api_key=os.getenv("INTEGRATION_API_KEY") or None,
        )
    return _clients[name]


def sales_client():
    return _client("sales", "SALES_API_URL")


def incident_client():
    return _client("incident", "INCIDENT_API_URL")


def rag_client():
    return _client("rag", "RAG_API_URL")


def reset_clients():
    """Test hook -- re-reads environment and clears breaker state."""
    _clients.clear()


def integration_status():
    return {
        "sales": sales_client().status(),
        "incident": incident_client().status(),
        "rag": rag_client().status(),
    }


def fetch_account_propensity(account_id, request_id=None):
    """Propensity for an account, or None if sales is unavailable."""
    if not sales_client().configured:
        return None, NOT_CONFIGURED
    if not account_id or account_id == "anonymous":
        return None, "no_account_id"

    data, outcome = sales_client().get(
        f"/accounts/{account_id}/score", request_id=request_id
    )
    if outcome != OK or not isinstance(data, dict):
        return None, outcome
    return {
        "score": data.get("score"),
        "segment": data.get("segment"),
        "industry": data.get("industry"),
    }, OK


def fetch_incident_status(service, request_id=None):
    """Whether `service` is currently in an incident, or None if unavailable."""
    # Configuration status is the more useful signal, so it is reported first:
    # "not_configured" tells a reader the edge is off, "no_service" tells them
    # the edge is on but this message named no service.
    if not incident_client().configured:
        return None, NOT_CONFIGURED
    if not service:
        return None, "no_service"

    data, outcome = incident_client().get(
        "/incidents/active", request_id=request_id, params={"service": service}
    )
    if outcome != OK or not isinstance(data, dict):
        return None, outcome
    return {
        "service": service,
        "active": bool(data.get("active")),
        "incident": data.get("incident"),
    }, OK


def fetch_grounding(query, request_id=None, max_chars=400):
    """A supporting passage for the response, or None if retrieval is unavailable."""
    if not rag_client().configured:
        return None, NOT_CONFIGURED
    if not query:
        return None, "no_query"

    data, outcome = rag_client().get(
        "/query", request_id=request_id, params={"q": query}
    )
    if outcome != OK or not isinstance(data, dict):
        return None, outcome

    response = data.get("response") or {}
    sources = response.get("sources") or []
    return {
        "answer": (response.get("answer") or "")[:max_chars],
        "groundedness": response.get("groundedness"),
        "top_source": sources[0].get("doc_id") if sources else None,
        "retriever": response.get("retriever"),
    }, OK


# Service names are mentioned in customer messages in lowercase far more often
# than not; this is a deliberately dumb scan, and the request may always name the
# service explicitly instead.
_KNOWN_SERVICES = ("checkout", "payments", "search", "identity", "billing")


def infer_service(message):
    """Best-effort service mention. Explicit `service` on the request wins."""
    lowered = (message or "").lower()
    for service in _KNOWN_SERVICES:
        if re.search(rf"\b{service}\b", lowered):
            return service
    return None
