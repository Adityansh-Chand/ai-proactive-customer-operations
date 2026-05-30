
from fastapi import FastAPI
from pydantic import BaseModel, Field

from orchestration.agent_dag import run_dag

app = FastAPI()


class DecisionRequest(BaseModel):
    message: str = Field(..., min_length=1)
    customer_id: str = "anonymous"
    metadata: dict = Field(default_factory=dict)


@app.get("/")
def health():
    return {"status": "running"}


@app.get("/health")
def health_check():
    return {"status": "running"}


@app.post("/decide")
def decide(request: DecisionRequest):
    return run_dag(request.message, customer_id=request.customer_id, metadata=request.metadata)
