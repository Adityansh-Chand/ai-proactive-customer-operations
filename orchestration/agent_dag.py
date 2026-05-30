from dataclasses import asdict, dataclass, field

from agents.planner_agent import planner_agent
from agents.router_agent import route_request
from agents.sentiment_agent import sentiment_agent
from agents.policy_agent import policy_agent
from agents.action_agent import action_agent
from agents.memory import save


@dataclass
class CustomerContext:
    intent: str
    customer_id: str = "anonymous"
    route: str = "support_agent"
    sentiment: str = "neutral"
    priority: str = "normal"
    delay_risk: bool = False
    metadata: dict = field(default_factory=dict)


def _has_delay_risk(text):
    text = text.lower()
    return any(term in text for term in ["delay", "late", "not arrived", "missed delivery"])


def run_dag(user_input, customer_id="anonymous", metadata=None):
    metadata = metadata or {}
    trace = {}

    trace["planner"] = planner_agent(user_input)
    trace["route"] = route_request(user_input)
    trace["sentiment"] = sentiment_agent(user_input)

    context = CustomerContext(
        intent=user_input,
        customer_id=customer_id,
        route=trace["route"],
        sentiment=trace["sentiment"],
        priority=trace["planner"]["priority"],
        delay_risk=_has_delay_risk(user_input),
        metadata=metadata,
    )

    context_dict = asdict(context)
    trace["context"] = context_dict
    trace["policy"] = policy_agent(context_dict)
    trace["action"] = action_agent(trace["policy"], context_dict)

    save({"customer_id": customer_id, "policy": trace["policy"], "action": trace["action"]["type"]})
    return trace
