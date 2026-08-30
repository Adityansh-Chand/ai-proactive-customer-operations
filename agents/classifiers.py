"""Fitted intent and sentiment classifiers.

These replace two keyword lookups. `route_request` matched the substrings
"refund", "delay" and "track"; `sentiment_agent` matched a 12-word set, so
"this is the third time I have written and nobody replies" scored neutral.

Both artifacts are produced by `training/train.py`. If they are missing this
raises rather than silently reverting to keyword matching -- a quiet fallback to
the thing being replaced is how the old behaviour would creep back.

An optional LLM path is available through the provider-agnostic seam in
`llm/client.py`. It is off by default and no reported metric uses it.
"""
import json
from functools import lru_cache
from pathlib import Path

from llm import client as llm

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "models" / "artifacts"
INTENT_PATH = ARTIFACT_DIR / "intent_classifier.joblib"
SENTIMENT_PATH = ARTIFACT_DIR / "sentiment_classifier.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

_MISSING = (
    "Classifier artifacts not found in models/artifacts/.\n"
    "Train them first:\n"
    "    python training/generate_corpus.py\n"
    "    python training/train.py"
)

INTENT_PROMPT = (
    "Classify the customer message into exactly one of: refund_request, "
    "delivery_issue, tracking_request, complaint, general_question. "
    "Reply with the label only."
)
SENTIMENT_PROMPT = (
    "Classify the sentiment of the customer message as exactly one of: "
    "negative, neutral, positive. Reply with the label only."
)


@lru_cache(maxsize=1)
def load_artifacts():
    for path in (INTENT_PATH, SENTIMENT_PATH, METRICS_PATH):
        if not path.exists():
            raise FileNotFoundError(_MISSING)

    import joblib

    metadata = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return joblib.load(INTENT_PATH), joblib.load(SENTIMENT_PATH), metadata


def _labels(component):
    _, _, metadata = load_artifacts()
    return metadata["components"][component]["labels"]


def _classify(text, component, prompt):
    """Classifier by default; LLM only when explicitly configured and reachable."""
    intent_model, sentiment_model, _ = load_artifacts()
    model = intent_model if component == "intent" else sentiment_model

    if llm.is_enabled():
        try:
            answer = llm.complete(prompt, text).strip().lower()
            for label in _labels(component):
                if label in answer:
                    return label, f"llm:{llm.provider()}"
        except llm.LLMError:
            pass  # fall through to the fitted classifier

    return str(model.predict([text])[0]), "classifier"


def classify_intent(text):
    return _classify(text, "intent", INTENT_PROMPT)


def classify_sentiment(text):
    return _classify(text, "sentiment", SENTIMENT_PROMPT)


def confidence(text, component):
    """Max predicted probability -- usable for routing low-confidence cases to review."""
    intent_model, sentiment_model, _ = load_artifacts()
    model = intent_model if component == "intent" else sentiment_model
    return float(max(model.predict_proba([text])[0]))


def model_metadata():
    _, _, metadata = load_artifacts()
    return {
        "model_type": metadata["model_type"],
        "data_source": metadata["data_source"],
        "architecture": metadata["architecture"],
        "split": metadata["split"],
        "intent_macro_f1": metadata["components"]["intent"]["test_macro_f1"],
        "sentiment_macro_f1": metadata["components"]["sentiment"]["test_macro_f1"],
        "end_to_end_policy_accuracy": metadata["end_to_end"]["policy_accuracy"],
        "llm": llm.status(),
    }
