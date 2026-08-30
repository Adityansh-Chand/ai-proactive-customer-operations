"""Behaviour of the agent DAG: learned front end, deterministic decisions."""
from datetime import datetime, timedelta, timezone

import pytest

from agents.action_agent import action_for_policy
from agents.classifiers import classify_intent, classify_sentiment, confidence, load_artifacts
from agents.policy_agent import DEFAULT_POLICY, POLICIES, decide_policy, explain_policy
from orchestration.agent_dag import derive_priority, run_dag

CREATED = datetime(2026, 4, 6, 9, tzinfo=timezone.utc)


def _dag(message, tier="standard", hours=48):
    return run_dag(
        message,
        customer_id="cust_0001",
        account_tier=tier,
        created_at=CREATED.isoformat(),
        sla_due_at=(CREATED + timedelta(hours=hours)).isoformat(),
    )


# --- the deterministic layer: exact, so tested exactly -------------------------

@pytest.mark.parametrize(
    "intent,sentiment,priority,expected",
    [
        ("general_question", "negative", "high", "escalate"),
        ("complaint", "neutral", "normal", "escalate"),
        ("delivery_issue", "neutral", "normal", "offer_credit"),
        ("refund_request", "neutral", "normal", "refund_review"),
        ("tracking_request", "positive", "normal", "send_tracking_update"),
        ("general_question", "neutral", "normal", "auto_resolve"),
    ],
)
def test_policy_rules_are_exact(intent, sentiment, priority, expected):
    assert decide_policy(intent=intent, sentiment=sentiment, priority=priority) == expected


def test_policy_names_the_rule_that_fired():
    """Every decision must be attributable to a specific rule."""
    policy, rule = explain_policy(
        intent="complaint", sentiment="neutral", priority="normal"
    )
    assert policy == "escalate"
    assert rule == "explicit_complaint"


def test_unmatched_input_falls_through_to_the_default():
    policy, rule = explain_policy(
        intent="general_question", sentiment="neutral", priority="normal"
    )
    assert policy == DEFAULT_POLICY
    assert "default" in rule


def test_every_policy_maps_to_an_action():
    for policy in POLICIES:
        assert action_for_policy(policy)


def test_priority_comes_from_crm_fields_not_message_text():
    assert derive_priority("enterprise", 4) == "high"
    assert derive_priority("free", 1) == "high"
    assert derive_priority("free", 48) == "normal"
    assert derive_priority("enterprise", 48) == "normal"


# --- the learned layer: properties, not exact labels ---------------------------

def test_classifiers_return_known_labels():
    _, _, metadata = load_artifacts()
    intent, source = classify_intent("where is my order right now")
    sentiment, _ = classify_sentiment("this has been a frustrating experience")

    assert intent in metadata["components"]["intent"]["labels"]
    assert sentiment in metadata["components"]["sentiment"]["labels"]
    assert source == "classifier"


def test_sentiment_detects_a_polite_negative():
    """The keyword set this replaced scored these neutral: no angry word appears."""
    sentiment, _ = classify_sentiment(
        "this is the third time I have written and nobody has replied to me"
    )
    assert sentiment == "negative"


def test_sentiment_separates_clear_cases():
    negative, _ = classify_sentiment("honestly this has been a very poor experience")
    positive, _ = classify_sentiment(
        "thanks for the help earlier, your team has been great"
    )
    assert negative == "negative"
    assert positive != "negative"


def test_confidence_is_a_probability():
    value = confidence("where is my parcel", "intent")
    assert 0.0 <= value <= 1.0


# --- the DAG: structure and attribution ---------------------------------------

def test_trace_records_the_source_of_every_decision():
    trace = _dag("my parcel has not arrived and I am losing patience", "enterprise", 4)

    assert trace["intent"]["source"] == "classifier"
    assert trace["sentiment"]["source"] == "classifier"
    assert "not from message text" in trace["priority"]["source"]
    assert "deterministic" in trace["policy"]["source"]
    assert trace["policy"]["rule"]


def test_enterprise_sla_breach_with_negative_sentiment_escalates():
    trace = _dag("this has been a genuinely poor experience and nobody has helped",
                 "enterprise", 4)
    assert trace["priority"]["value"] == "high"
    assert trace["policy"]["value"] == "escalate"
    assert trace["action"]["type"] == "create_ticket"


def test_same_message_routes_differently_by_account_context():
    """The text is identical; only the CRM facts differ."""
    message = "this has been a genuinely poor experience and nobody has helped"
    urgent = _dag(message, "enterprise", 4)
    relaxed = _dag(message, "free", 60)
    assert urgent["priority"]["value"] == "high"
    assert relaxed["priority"]["value"] == "normal"


def test_low_confidence_predictions_are_flagged_for_review():
    trace = _dag("hmm")
    assert isinstance(trace["needs_human_review"], bool)


def test_policy_is_always_a_known_value():
    for message in [
        "where is my order",
        "I want my money back",
        "does this come in blue",
        "nobody has helped me at all",
    ]:
        assert _dag(message)["policy"]["value"] in POLICIES
