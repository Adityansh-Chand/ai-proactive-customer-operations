
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
