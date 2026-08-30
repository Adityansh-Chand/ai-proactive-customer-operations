"""Deterministic policy layer.

This is intentionally NOT learned. A policy that decides refunds and escalations
should be auditable, explainable to a compliance reviewer, and changeable without
retraining. The machine learning sits in front of it -- intent and sentiment are
classified from free text -- and the decision itself stays rules.

Rules are ordered; the first match wins. `explain_policy` returns which rule
fired, so every decision in a trace is attributable to a specific line here.
"""

POLICIES = [
    "incident_response",
    "escalate",
    "offer_credit",
    "refund_review",
    "send_tracking_update",
    "auto_resolve",
]

# Rules that fire only when cross-service enrichment is present. They are kept
# separate from the base rules on purpose: with no integration configured,
# `enrichment` is None, none of these can fire, and `decide_policy` behaves
# exactly as it does standalone. That is what keeps the offline evaluation and
# the committed metrics valid -- integration adds capability without moving the
# ground the model was measured on.
ENRICHED_RULES = [
    (
        "known_incident_on_the_service_complained_about",
        lambda intent, sentiment, priority, e: bool(
            (e.get("incident") or {}).get("active")
        ),
        "incident_response",
    ),
    (
        "high_value_account_at_risk",
        lambda intent, sentiment, priority, e: (
            sentiment == "negative"
            and (e.get("account") or {}).get("segment") == "high_propensity"
        ),
        "escalate",
    ),
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


def decide_policy(intent, sentiment, priority, enrichment=None):
    return explain_policy(intent, sentiment, priority, enrichment)[0]


def explain_policy(intent, sentiment, priority, enrichment=None):
    """Return (policy, rule_name) so a trace can name the rule that fired.

    Enriched rules are checked first and only when enrichment is supplied. An
    active incident on the service being complained about outranks everything
    else: that is not an individual customer problem, and treating it as one
    means issuing refunds one at a time for a fault already being fixed.
    """
    if enrichment:
        for name, predicate, policy in ENRICHED_RULES:
            if predicate(intent, sentiment, priority, enrichment):
                return policy, name

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
