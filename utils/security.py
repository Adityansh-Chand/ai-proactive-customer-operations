import os
import uuid

from fastapi import Header, HTTPException, Request


def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.getenv("API_KEY")
    if not expected:
        return

    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def request_id_middleware(request: Request, call_next):
    """Accept an inbound request id or mint one, and make it readable downstream.

    Storing it on `request.state` is what turns a per-service id into a
    distributed one: a handler can read it and forward it on the calls it makes,
    so a single identifier follows one decision across every service that
    contributed to it. Without this the id is generated, echoed back, and lost.
    """
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


def current_request_id(request: Request) -> str | None:
    """Request id for this request, if the middleware has run."""
    return getattr(request.state, "request_id", None)
