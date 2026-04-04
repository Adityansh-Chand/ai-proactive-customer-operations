
def route_request(message):

    message = message.lower()

    if "refund" in message:
        return "returns_agent"

    if "delay" in message:
        return "delivery_agent"

    if "track" in message:
        return "tracking_agent"

    return "complaint_agent"
