
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from monitoring.metrics import metrics
from agents.classifiers import model_metadata
from orchestration.agent_dag import run_dag
from utils.security import request_id_middleware, require_api_key
from utils.storage import recent_events, save_event

app = FastAPI(title="AI Proactive Customer Operations", version="1.0.0")
app.middleware("http")(request_id_middleware)


class DecisionRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    customer_id: str = "anonymous"
    # CRM context. Priority is derived from these rather than read out of the
    # message text, so the same words route differently for an enterprise
    # account near its SLA than for a free account with days to spare.
    account_tier: str = Field("standard", pattern="^(free|standard|enterprise)$")
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
    return {"status": "running", "model": model_metadata()}


@app.get("/metrics")
def metrics_endpoint():
    return metrics.snapshot()


@app.get("/events", dependencies=[Depends(require_api_key)])
def events(limit: int = 20):
    return {"events": recent_events(limit=min(limit, 100))}


@app.post("/decide", dependencies=[Depends(require_api_key)])
def decide(request: DecisionRequest):
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
    )
    save_event("customer_decision", result)
    return result
