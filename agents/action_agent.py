def action_agent(policy):
    if policy=="escalate":
        return "create_ticket"
    return "knowledge_response"
