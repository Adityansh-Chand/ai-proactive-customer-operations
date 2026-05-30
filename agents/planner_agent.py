def planner_agent(user_input):
    tasks = ["route", "sentiment", "policy", "action"]
    priority = "high" if any(term in user_input.lower() for term in ["urgent", "angry", "complaint", "cancel"]) else "normal"

    return {
        "tasks": tasks,
        "priority": priority,
    }
