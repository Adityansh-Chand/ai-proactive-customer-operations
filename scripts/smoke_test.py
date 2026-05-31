import json
import os
import urllib.request


BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")


def request(path, method="GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def main():
    checks = [
        request("/health"),
        request("/metrics"),
        request(
            "/decide",
            method="POST",
            payload={"message": "My package is delayed", "customer_id": "cust_smoke"},
        ),
    ]

    for status, body in checks:
        assert status == 200, body

    print("smoke test passed")


if __name__ == "__main__":
    main()
