
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from monitoring.drift import DriftMonitor
from monitoring.metrics import metrics
from agents.classifiers import model_metadata
from integrations.services import integration_status
from orchestration.agent_dag import run_dag
from orchestration.proactive import (
    handle_incident_event,
    outreach_log,
    record_contact,
    status as proactive_status,
)
from utils.security import current_request_id, request_id_middleware, require_api_key
from utils.storage import recent_events, save_event

app = FastAPI(title="AI Proactive Customer Operations", version="1.0.0")
app.middleware("http")(request_id_middleware)

API_VERSION = "v1"

# Data endpoints live on a router so they can be served at BOTH /v1/... and the
# original unversioned paths from a single definition. Without a version prefix
# there is no way to change a response shape without breaking every consumer on
# the same deploy -- the contract checks in the portfolio repo detect that
# breakage, they do not prevent it.
#
# The unversioned alias is kept because consumers already call it. It is the
# deprecation path, not a permanent second interface.
api = APIRouter()

# The mix of intents this router is predicting, against the mix seen at training
# time. If the share of refund requests doubles, something changed -- and unlike
# classifier confidence, that signal does not swing on phrasing the model happens
# to have memorised.
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "models" / "artifacts"
drift_monitor = DriftMonitor.from_file(
    ARTIFACT_DIR / "drift_reference.json", name="intent_mix"
)


class IncidentEvent(BaseModel):
    """An event pushed by the incident detection platform."""

    event_id: str = Field(..., min_length=1, max_length=200)
    type: str = Field(..., min_length=1, max_length=100)
    source: str = Field("", max_length=200)
    payload: dict = Field(default_factory=dict)


class DecisionRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    customer_id: str = "anonymous"
    # CRM context. Priority is derived from these rather than read out of the
    # message text, so the same words route differently for an enterprise
    # account near its SLA than for a free account with days to spare.
    account_tier: str = Field("standard", pattern="^(free|standard|enterprise)$")
    # Which service the contact is about. Explicit beats the message-text scan.
    service: str | None = None
    created_at: str | None = None
    sla_due_at: str | None = None
    metadata: dict = Field(default_factory=dict)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    metrics.increment("http_errors_total")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "path": str(request.url.path)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    metrics.increment("validation_errors_total")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request",
            "details": exc.errors(),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    metrics.increment("unhandled_errors_total")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "path": str(request.url.path)},
    )


@app.get("/")
def health():
    return {"status": "running"}


@app.get("/health")
def health_check():
    """Health plus the identity of the classifiers actually loaded."""
    return {
        "status": "running",
        "model": model_metadata(),
        "integrations": integration_status(),
        "proactive": proactive_status(),
    }


@app.get("/metrics")
def metrics_endpoint():
    return metrics.snapshot()


@api.get("/events", dependencies=[Depends(require_api_key)])
def events(limit: int = 20, request_id: str | None = None):
    """Recent events, optionally narrowed to one request id.

    `request_id` is what makes this endpoint a trace source rather than a log
    tail: the portfolio's scripts/trace.py asks all five services the same
    question and joins the answers into one timeline.
    """
    return {"events": recent_events(limit=min(limit, 100), request_id=request_id)}


@api.post("/decide", dependencies=[Depends(require_api_key)])
def decide(request: DecisionRequest, http_request: Request):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    metrics.increment("decisions_total")
    result = run_dag(
        request.message,
        customer_id=request.customer_id,
        account_tier=request.account_tier,
        created_at=request.created_at,
        sla_due_at=request.sla_due_at,
        metadata=request.metadata,
        service=request.service,
        request_id=current_request_id(http_request),
    )
    # Remember who contacted us about what, so an incident on that service can
    # be pushed back to them proactively.
    record_contact(request.customer_id, request.service, request.message)

    predicted_intent = (result.get("intent") or {}).get("label")
    if predicted_intent:
        drift_monitor.observe(predicted_intent)

    save_event("customer_decision", result, current_request_id(http_request))
    return result


@api.post("/events/incident", dependencies=[Depends(require_api_key)])
def receive_incident_event(event: IncidentEvent, http_request: Request):
    """Receive an incident event and decide proactive outreach.

    Delivery from the incident platform is at-least-once, so this endpoint is
    idempotent: the same event_id twice produces one round of outreach and the
    response says `duplicate`. Without that, a delivery retry would message the
    same customers again.
    """
    metrics.increment("incident_events_total")
    result = handle_incident_event(event.model_dump())
    result["request_id"] = current_request_id(http_request)
    save_event("proactive_outreach", {
        "event_id": event.event_id, "type": event.type,
        "service": result.get("service"), "notified": result.get("notified", 0),
        "duplicate": result.get("duplicate", False),
    }, result["request_id"])
    return result


@api.get("/proactive/outreach", dependencies=[Depends(require_api_key)])
def proactive_outreach():
    """Outreach batches generated from incident events."""
    return {"status": proactive_status(), "batches": outreach_log()}


@api.get("/drift", dependencies=[Depends(require_api_key)])
def drift():
    """Is the mix of intents still what this router was fitted on?"""
    return drift_monitor.report()


@app.get("/version")
def version():
    """What this service speaks, so a consumer can check rather than assume."""
    return {
        "service": "ai-proactive-customer-operations",
        "current": API_VERSION,
        "supported": [API_VERSION],
        "unversioned_alias": {
            "status": "deprecated",
            "note": ("the same endpoints are served without a /v1 prefix for "
                     "consumers that predate versioning; new callers should use "
                     f"/{API_VERSION}"),
        },
    }


# Mounted twice, one set of handlers. The alias is hidden from the schema so the
# generated docs show one interface rather than two identical ones.
app.include_router(api, prefix=f"/{API_VERSION}")
app.include_router(api, include_in_schema=False)
