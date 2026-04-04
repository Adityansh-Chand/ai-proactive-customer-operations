def decide(context):

    if context["sentiment"]=="negative":
        return "escalate"

    if context["delay_risk"]:
        return "offer_credit"

    return "auto_resolve"
