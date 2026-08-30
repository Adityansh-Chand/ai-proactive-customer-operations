"""Provider-agnostic optional LLM backend.

No vendor is hardcoded. The operator chooses via environment variables, and the
callers in this repo never learn which provider answered.

    LLM_PROVIDER   openai_compatible | anthropic | none    (default: none)
    LLM_API_KEY    the operator's key
    LLM_MODEL      model identifier, operator-supplied
    LLM_BASE_URL   endpoint override; enables self-hosted and gateway backends

`openai_compatible` is the primary adapter because that request shape is spoken
natively or through a compatibility endpoint by most hosted providers and local
runtimes (Ollama, vLLM, LM Studio, and gateways). Pointing LLM_BASE_URL at a
local runtime is the intended way to run this without a paid key.

Default is `none`: the fitted classifiers run, nothing leaves the machine, and CI
stays reproducible. **Every metric reported in this repository comes from the
classifier path, never the LLM path.** Numbers produced through a model would
depend on a vendor, a model version and a sampling temperature, and no reviewer
could reproduce them.
"""
import json
import os
import urllib.error
import urllib.request

SUPPORTED = ("none", "openai_compatible", "anthropic")

DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
TIMEOUT_SECONDS = 30


class LLMError(RuntimeError):
    pass


def provider() -> str:
    name = os.getenv("LLM_PROVIDER", "none").strip().lower()
    if name not in SUPPORTED:
        raise LLMError(
            f"unsupported LLM_PROVIDER {name!r}; supported values: {', '.join(SUPPORTED)}"
        )
    return name


def is_enabled() -> bool:
    return provider() != "none" and bool(os.getenv("LLM_API_KEY"))


def _post(url, payload, headers):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise LLMError(f"{error.code} from provider: {error.read()[:300]!r}") from error
    except Exception as error:  # network, timeout, malformed JSON
        raise LLMError(str(error)) from error


def complete(system: str, user: str, max_tokens: int = 512) -> str:
    """Single-turn completion. Raises LLMError; callers fall back deterministically."""
    name = provider()
    if name == "none":
        raise LLMError("LLM_PROVIDER is 'none'")

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise LLMError("LLM_API_KEY is not set")
    model = os.getenv("LLM_MODEL")
    if not model:
        raise LLMError("LLM_MODEL is not set")

    if name == "openai_compatible":
        base = os.getenv("LLM_BASE_URL", DEFAULT_OPENAI_BASE).rstrip("/")
        body = _post(
            f"{base}/chat/completions",
            {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            {"Authorization": f"Bearer {api_key}"},
        )
        return body["choices"][0]["message"]["content"]

    # anthropic: native Messages API -- system is top-level, not a message role.
    base = os.getenv("LLM_BASE_URL", DEFAULT_ANTHROPIC_BASE).rstrip("/")
    body = _post(
        f"{base}/messages",
        {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
    )
    return "".join(block.get("text", "") for block in body.get("content", []))


def status() -> dict:
    """Reportable state, safe to expose -- never includes the key itself."""
    try:
        name = provider()
    except LLMError as error:
        return {"provider": "invalid", "enabled": False, "error": str(error)}
    return {
        "provider": name,
        "enabled": is_enabled(),
        "model": os.getenv("LLM_MODEL") if name != "none" else None,
        "note": "all reported metrics come from the fitted classifier path",
    }
