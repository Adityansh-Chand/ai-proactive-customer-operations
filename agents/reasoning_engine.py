def decide(context):

    if context["sentiment"]=="negative":
        return "escalate"

    if context.get("delay_risk"):
        return "offer_credit"

    return "auto_resolve"
