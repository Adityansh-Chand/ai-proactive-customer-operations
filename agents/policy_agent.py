def policy_agent(context):
    if context["sentiment"] == "negative" or context["priority"] == "high":
        return "escalate"

    if context["route"] == "delivery_agent" and context.get("delay_risk"):
        return "offer_credit"

    if context["route"] == "returns_agent":
        return "refund_review"

    if context["route"] == "tracking_agent":
        return "send_tracking_update"

    return "auto_resolve"
