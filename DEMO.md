# Demo

This demo shows the customer operations service turning a customer message into
a routed decision, policy, action, metrics, and audit event.

## Run Locally

Terminal 1:

```bash
pip install -r requirements.txt
uvicorn api.server:app --reload --port 8000
```

Terminal 2:

```bash
python scripts/smoke_test.py
```

To demo protected endpoints, start with an API key:

```bash
API_KEY=demo-key uvicorn api.server:app --reload --port 8000
```

## Curl Walkthrough

Root:

```bash
curl http://localhost:8000/
```

Health:

```bash
curl http://localhost:8000/health
```

Metrics:

```bash
curl http://localhost:8000/metrics
```

Decision:

```bash
curl -X POST http://localhost:8000/decide \
  -H "Content-Type: application/json" \
  -d @examples/requests/decide.json
```

Events when `API_KEY` is set:

```bash
curl http://localhost:8000/events \
  -H "X-API-Key: demo-key"
```

Protected decision when `API_KEY` is set:

```bash
curl -X POST http://localhost:8000/decide \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-key" \
  -d @examples/requests/decide.json
```

## Sample Files

- Request: `examples/requests/decide.json`
- Responses: `examples/responses/root.json`, `health.json`, `metrics.json`, `decide.json`, `events.json`
