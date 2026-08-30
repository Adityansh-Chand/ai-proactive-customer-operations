"""The agent DAG: learned classification in front, deterministic decisions behind.

    message ─┬─> intent classifier    (learned)
             └─> sentiment classifier (learned)
    CRM fields ─> priority            (derived, never read from the text)
                       │
                       └─> policy rules (deterministic, auditable)
                                  └─> action mapping (deterministic)

The split is deliberate. Classification of free text is exactly where a model
earns its place; deciding whether to issue a refund is exactly where one does
not. Every trace records which component produced each value and which policy
rule fired, so a decision can be audited end to end.
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime

from agents.action_agent import action_agent
from agents.classifiers import classify_intent, classify_sentiment, confidence
from agents.planner_agent import planner_agent
from agents.policy_agent import explain_policy

# Below this, the trace flags the decision for human review rather than trusting it.
LOW_CONFIDENCE = 0.45


@dataclass
class CustomerContext:
    intent: str
    customer_id: str = "anonymous"
    intent_label: str = "general_question"
    sentiment: str = "neutral"
    priority: str = "normal"
    account_tier: str = "standard"
    hours_to_sla: float = 72.0
    metadata: dict = field(default_factory=dict)


def derive_priority(account_tier, hours_to_sla):
    """Priority from CRM facts, not from keyword-sniffing the message.

    Kept identical to the generator's rule so training and serving agree.
    """
    if account_tier == "enterprise" and hours_to_sla <= 8:
        return "high"
    if hours_to_sla <= 2:
        return "high"
    return "normal"


def _hours_to_sla(created_at, sla_due_at):
    if not created_at or not sla_due_at:
        return 72.0
    try:
        created = datetime.fromisoformat(created_at)
        due = datetime.fromisoformat(sla_due_at)
        return max((due - created).total_seconds() / 3600.0, 0.0)
    except ValueError:
        return 72.0


def run_dag(
    user_input,
    customer_id="anonymous",
    account_tier="standard",
    created_at=None,
    sla_due_at=None,
    metadata=None,
):
    metadata = metadata or {}
    trace = {}

    trace["planner"] = planner_agent(user_input)

    intent_label, intent_source = classify_intent(user_input)
    sentiment_label, sentiment_source = classify_sentiment(user_input)

    intent_confidence = confidence(user_input, "intent")
    sentiment_confidence = confidence(user_input, "sentiment")

    trace["intent"] = {
        "label": intent_label,
        "source": intent_source,
        "confidence": round(intent_confidence, 4),
    }
    trace["sentiment"] = {
        "label": sentiment_label,
        "source": sentiment_source,
        "confidence": round(sentiment_confidence, 4),
    }

    hours = _hours_to_sla(created_at, sla_due_at)
    priority = derive_priority(account_tier, hours)
    trace["priority"] = {
        "value": priority,
        "source": "derived from account_tier and hours_to_sla (not from message text)",
        "hours_to_sla": round(hours, 2),
    }

    context = CustomerContext(
        intent=user_input,
        customer_id=customer_id,
        intent_label=intent_label,
        sentiment=sentiment_label,
        priority=priority,
        account_tier=account_tier,
        hours_to_sla=hours,
        metadata=metadata,
    )
    context_dict = asdict(context)
    trace["context"] = context_dict

    policy, rule = explain_policy(
        intent=intent_label, sentiment=sentiment_label, priority=priority
    )
    trace["policy"] = {
        "value": policy,
        "rule": rule,
        "source": "deterministic rules in agents/policy_agent.py",
    }
    trace["action"] = action_agent(policy, context_dict)

    # Surfaced rather than hidden: a decision built on a coin-flip classification
    # should be visible as such to whoever reads the trace.
    trace["needs_human_review"] = bool(
        min(intent_confidence, sentiment_confidence) < LOW_CONFIDENCE
    )

    # Backwards-compatible flat keys for older callers.
    trace["route"] = intent_label
    return trace
