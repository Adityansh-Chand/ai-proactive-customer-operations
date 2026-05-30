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
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response
