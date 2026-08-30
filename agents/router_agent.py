"""Intent routing.

Was three substring checks ("refund", "delay", "track") with a catch-all. Now the
fitted classifier from `training/train.py`; see `agents/classifiers.py`.
"""
from agents.classifiers import classify_intent


def route_request(message):
    label, _ = classify_intent(message)
    return label
