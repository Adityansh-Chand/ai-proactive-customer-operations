# AI Proactive Customer Operations

Multi-agent customer operations workflow that routes a customer message through
planning, intent routing, sentiment analysis, policy selection, and action
execution.

## Agent DAG

```mermaid
flowchart TD
  User --> PlannerAgent
  PlannerAgent --> RouterAgent
  RouterAgent --> SentimentAgent
  SentimentAgent --> PolicyAgent
  PolicyAgent --> ActionAgent
  ActionAgent --> Tools
```

## API

- `GET /health`
- `GET /metrics`
- `GET /events` protected when `API_KEY` is set
- `POST /decide`

See `DEMO.md` for terminal demo steps, curl commands, and sample request/response files.

Example:

```json
{
  "message": "My package is delayed but I just need help",
  "customer_id": "cust_002"
}
```

Set `API_KEY` to require `X-API-Key` on decision/event endpoints.
Set `APP_DB_PATH` to control the SQLite event database location.

## Run

```bash
pip install -r requirements.txt
python -m pytest -q
python evaluation/evaluate.py
uvicorn api.server:app --reload --port 8000
```

With the server running, use a second terminal for the smoke check:

```bash
python scripts/smoke_test.py
```

Docker:

```bash
cp .env.example .env
docker compose up --build
```

Kubernetes manifests live in `k8s/deployment.yaml` and include probes, resource
limits, a Service, and a PVC for the SQLite event store. The default manifest
uses one replica because SQLite is the default event store.

Dockerfile, Docker Compose, and Kubernetes configuration are validated by static
inspection/YAML parsing in this workspace. Runtime container and cluster
validation remains a CI or cloud-environment step.

## Reviewer Status

- Purpose: customer operations workflow for routing messages through policy and action decisions.
- Quickstart: run tests/eval, start `uvicorn api.server:app --reload --port 8000`, then run `python scripts/smoke_test.py`.
- Demo path: use `DEMO.md` for curl examples and sample request/response files.
- Deployment status: local tests and smoke tests pass; Docker/Compose/Kubernetes config is present; Docker image builds are validated in CI; cloud deployment is pending.
- Remaining gaps: live CRM integration, production policy sources, managed auth/secrets, observability, cloud deployment, and production data governance.
- Portfolio index: https://github.com/Adityansh-Chand/ai-engineering-portfolio

## Highlights

- Explicit trace for planner, route, sentiment, policy, and action.
- Tool-backed actions for tickets, credits, refunds, tracking updates, and knowledge responses.
- Domain sample data with expected policy/action labels.
- Evaluation script for policy/action accuracy.
- SQLite event audit trail for decision traces.
- GitHub Actions CI for tests, eval, and container build.
- Production data contract in `datasets/production_schema.json`.

## License

MIT
