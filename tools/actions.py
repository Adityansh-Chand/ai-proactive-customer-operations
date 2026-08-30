
def create_ticket(issue):
    return {"type": "create_ticket", "status": "created", "issue": issue}

def apply_credit(user):
    return {"type": "apply_credit", "status": "applied", "user": user, "amount": 10}


def queue_refund_review(user, issue):
    return {"type": "refund_review", "status": "queued", "user": user, "issue": issue}


def send_tracking_update(user):
    return {"type": "send_tracking_update", "status": "sent", "user": user}


def send_knowledge_response(user):
    return {"type": "knowledge_response", "status": "sent", "user": user}


def link_to_incident(user, incident=None):
    """Attach the contact to a known incident instead of treating it as isolated.

    The point of this action is what it does NOT do: no individual refund, no
    ticket in the general queue. The customer is told their issue is a known
    fault already being worked, and is added to the notify list for it.
    """
    incident = incident or {}
    return {
        "type": "link_to_incident",
        "status": "linked",
        "user": user,
        "service": incident.get("service"),
        "anomaly_count": incident.get("anomaly_count"),
        "note": "known incident; customer added to the incident notification list",
    }
