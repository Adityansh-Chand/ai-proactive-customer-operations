"""Deterministic policy layer.

This is intentionally NOT learned. A policy that decides refunds and escalations
should be auditable, explainable to a compliance reviewer, and changeable without
retraining. The machine learning sits in front of it -- intent and sentiment are
classified from free text -- and the decision itself stays rules.

Rules are ordered; the first match wins. `explain_policy` returns which rule
fired, so every decision in a trace is attributable to a specific line here.
"""

POLICIES = [
    "escalate",
    "offer_credit",
    "refund_review",
    "send_tracking_update",
    "auto_resolve",
]

RULES = [
    ("high_priority_unhappy_customer",
     lambda intent, sentiment, priority: priority == "high" and sentiment == "negative",
     "escalate"),
    ("explicit_complaint",
     lambda intent, sentiment, priority: intent == "complaint",
     "escalate"),
    ("delivery_problem_warrants_goodwill",
     lambda intent, sentiment, priority: intent == "delivery_issue",
     "offer_credit"),
    ("refund_requested",
     lambda intent, sentiment, priority: intent == "refund_request",
     "refund_review"),
    ("tracking_requested",
     lambda intent, sentiment, priority: intent == "tracking_request",
     "send_tracking_update"),
]

DEFAULT_POLICY = "auto_resolve"


def decide_policy(intent, sentiment, priority):
    for _, predicate, policy in RULES:
        if predicate(intent, sentiment, priority):
            return policy
    return DEFAULT_POLICY


def explain_policy(intent, sentiment, priority):
    """Return (policy, rule_name) so a trace can name the rule that fired."""
    for name, predicate, policy in RULES:
        if predicate(intent, sentiment, priority):
            return policy, name
    return DEFAULT_POLICY, "default_no_rule_matched"


def policy_agent(context):
    """Backwards-compatible entry point taking the context dict."""
    return decide_policy(
        intent=context.get("intent_label", context.get("route", "general_question")),
        sentiment=context.get("sentiment", "neutral"),
        priority=context.get("priority", "normal"),
    )
