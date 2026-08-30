"""Map a policy decision onto a concrete tool call."""
from tools.actions import (
    apply_credit,
    create_ticket,
    link_to_incident,
    queue_refund_review,
    send_knowledge_response,
    send_tracking_update,
)

POLICY_TO_ACTION = {
    "incident_response": "link_to_incident",
    "escalate": "create_ticket",
    "offer_credit": "apply_credit",
    "refund_review": "refund_review",
    "send_tracking_update": "send_tracking_update",
    "auto_resolve": "knowledge_response",
}


def action_for_policy(policy):
    """The action type a policy produces, without executing anything.

    Used by the corpus generator and the evaluator, which need the label but
    must not trigger side effects.
    """
    return POLICY_TO_ACTION.get(policy, "knowledge_response")


def action_agent(policy, context=None):
    context = context or {}
    user = context.get("customer_id", "anonymous")
    issue = context.get("intent", "")

    if policy == "incident_response":
        return link_to_incident(user, context.get("incident"))
    if policy == "escalate":
        return create_ticket(issue)
    if policy == "offer_credit":
        return apply_credit(user)
    if policy == "refund_review":
        return queue_refund_review(user, issue)
    if policy == "send_tracking_update":
        return send_tracking_update(user)
    return send_knowledge_response(user)
