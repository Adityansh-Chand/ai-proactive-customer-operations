def policy_agent(context):
    if context["sentiment"]=="negative":
        return "escalate"
    return "auto_resolve"
