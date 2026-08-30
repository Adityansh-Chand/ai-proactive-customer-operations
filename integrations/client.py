"""Resilient HTTP client for optional cross-service enrichment.

The rule this file exists to enforce: **an enrichment may improve a decision, but
it may never prevent one.** Every call here is optional. If the dependency is
unset, slow, broken, or returning nonsense, the caller gets `None` and a reason
string, and carries on with the decision it would have made alone.

That is what keeps these services independently runnable. Clone one repo, set no
integration URLs, and it behaves exactly as it does today.

Three protections, in order of how often they matter:

1. **Timeout.** Short and explicit. An enrichment that takes longer than the
   decision is worth is not worth waiting for.
2. **Bounded retry.** One retry, for the transient case. Never more: retrying a
   dependency that is genuinely down multiplies the damage.
3. **Circuit breaker.** After repeated failures the client stops calling for a
   cooldown period. Without this, a dead dependency costs every single request a
   full timeout, and a slow dependency takes the caller down with it.

Request IDs propagate outward via `X-Request-ID`, so one identifier follows a
decision across every service that contributed to it.
"""
import json
import os
import time
import urllib.error
import urllib.request

# Read at construction time rather than import time. As module constants these
# could never be reconfigured by anything that set the environment after import
# -- including tests, which is how the omission was caught.
DEFAULT_TIMEOUT_SECONDS = 2.0  # deliberately short: a side quest, not the request
DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_COOLDOWN_SECONDS = 30.0


def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default

# Outcome codes recorded in the decision trace, so a reader can always tell
# whether an enrichment was live, skipped, or failed -- and why.
OK = "ok"
NOT_CONFIGURED = "not_configured"
TIMEOUT = "timeout"
CIRCUIT_OPEN = "circuit_open"
ERROR = "error"


class CircuitBreaker:
    """Stops calling a dependency that keeps failing.

    Half-open by default after the cooldown: the next call is allowed through,
    and its result decides whether the circuit closes or re-opens.
    """

    def __init__(self, threshold=None, cooldown=None):
        self.threshold = (
            threshold if threshold is not None
            else _env_int("INTEGRATION_FAILURE_THRESHOLD", DEFAULT_FAILURE_THRESHOLD)
        )
        self.cooldown = (
            cooldown if cooldown is not None
            else _env_float("INTEGRATION_COOLDOWN_SECONDS", DEFAULT_COOLDOWN_SECONDS)
        )
        self.failures = 0
        self.opened_at = None

    def is_open(self):
        if self.opened_at is None:
            return False
        if time.monotonic() - self.opened_at >= self.cooldown:
            self.opened_at = None  # half-open: let one call through
            self.failures = 0
            return False
        return True

    def record_success(self):
        self.failures = 0
        self.opened_at = None

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.monotonic()

    @property
    def state(self):
        if self.is_open():
            return "open"
        return "closed" if self.failures == 0 else "degraded"


class ServiceClient:
    """One configured downstream service. Never raises to the caller."""

    def __init__(self, name, base_url=None, timeout=None, api_key=None):
        self.name = name
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = (
            timeout if timeout is not None
            else _env_float("INTEGRATION_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        )
        self.api_key = api_key
        self.breaker = CircuitBreaker()

    @property
    def configured(self):
        return bool(self.base_url)

    def _headers(self, request_id):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if request_id:
            # The whole point of propagation: one id across every hop.
            headers["X-Request-ID"] = request_id
        return headers

    def _call(self, method, path, request_id=None, payload=None, params=None):
        if not self.configured:
            return None, NOT_CONFIGURED
        if self.breaker.is_open():
            return None, CIRCUIT_OPEN

        url = f"{self.base_url}{path}"
        if params:
            from urllib.parse import urlencode

            url = f"{url}?{urlencode(params)}"

        body = None if payload is None else json.dumps(payload).encode("utf-8")

        last = ERROR
        for attempt in range(2):  # one try, one retry
            request = urllib.request.Request(
                url, data=body, headers=self._headers(request_id), method=method
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                self.breaker.record_success()
                return data, OK
            except urllib.error.HTTPError as error:
                # A 4xx is the dependency telling us the request was wrong.
                # Retrying will not help, and it is not a health signal.
                if 400 <= error.code < 500:
                    self.breaker.record_success()
                    return None, ERROR
                last = ERROR
            except TimeoutError:
                last = TIMEOUT
            except OSError as error:
                last = TIMEOUT if "timed out" in str(error).lower() else ERROR
            except (ValueError, json.JSONDecodeError):
                self.breaker.record_failure()
                return None, ERROR  # malformed body; retrying gains nothing

            if attempt == 0:
                time.sleep(0.05)

        self.breaker.record_failure()
        return None, last

    def get(self, path, request_id=None, params=None):
        return self._call("GET", path, request_id=request_id, params=params)

    def post(self, path, payload, request_id=None):
        return self._call("POST", path, request_id=request_id, payload=payload)

    def status(self):
        return {
            "configured": self.configured,
            "base_url": self.base_url or None,
            "timeout_seconds": self.timeout,
            "circuit": self.breaker.state,
        }
