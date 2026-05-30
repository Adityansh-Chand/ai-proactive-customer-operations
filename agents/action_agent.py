from tools.actions import (
    apply_credit,
    create_ticket,
    queue_refund_review,
    send_knowledge_response,
    send_tracking_update,
)


def action_agent(policy, context=None):
    context = context or {}
    user = context.get("customer_id", "anonymous")
    issue = context.get("intent", "")

    if policy == "escalate":
        return create_ticket(issue)

    if policy == "offer_credit":
        return apply_credit(user)

    if policy == "refund_review":
        return queue_refund_review(user, issue)

    if policy == "send_tracking_update":
        return send_tracking_update(user)

    return send_knowledge_response(user)
