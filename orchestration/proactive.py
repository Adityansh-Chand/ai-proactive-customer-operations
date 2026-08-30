"""Proactive outreach: acting on an incident before customers write in again.

Every other path in this service is reactive -- a customer writes, a decision is
made. This is the one that makes the service's name true: the incident platform
pushes an event, and the customers who recently contacted us about that service
get told, unprompted, that their problem is a known fault being worked on.

Two things make this honest rather than decorative:

**Idempotency.** Delivery is at-least-once, so the same event will sometimes
arrive twice. Handlers that are not idempotent turn a retry into a second
outreach to the same customer -- exactly the kind of bug that makes people
distrust automated messaging. Events are deduplicated by `event_id`.

**The audience comes from data this service actually holds.** A recent-contact
registry is populated by `/decide` itself: who wrote in, about which service, and
when. There is no customer database here, and inventing one would be pretending.
Customers who never contacted us are not reachable by this service, and the
README says so.
"""
import os
import threading
import time
from collections import OrderedDict, defaultdict, deque

# How far back a contact still counts as "affected" by a new incident.
CONTACT_WINDOW_SECONDS = float(os.getenv("CONTACT_WINDOW_SECONDS", "86400"))
MAX_CONTACTS_PER_SERVICE = 500
MAX_SEEN_EVENTS = 2000
MAX_OUTREACH_LOG = 200

_lock = threading.Lock()
_contacts = defaultdict(lambda: deque(maxlen=MAX_CONTACTS_PER_SERVICE))
_seen_events = OrderedDict()  # event_id -> result, for idempotency
_outreach_log = deque(maxlen=MAX_OUTREACH_LOG)


def record_contact(customer_id, service, message=None):
    """Remember that this customer contacted us about this service."""
    if not service or not customer_id or customer_id == "anonymous":
        return
    with _lock:
        _contacts[service].append(
            {"customer_id": customer_id, "at": time.time(),
             "message": (message or "")[:160]}
        )


def recent_contacts(service, window=CONTACT_WINDOW_SECONDS):
    """Distinct customers who contacted us about `service` inside the window."""
    cutoff = time.time() - window
    with _lock:
        entries = [c for c in _contacts.get(service, ()) if c["at"] >= cutoff]

    seen, unique = set(), []
    for entry in reversed(entries):  # most recent contact per customer wins
        if entry["customer_id"] in seen:
            continue
        seen.add(entry["customer_id"])
        unique.append(entry)
    return unique


def already_handled(event_id):
    with _lock:
        return _seen_events.get(event_id)


def _remember(event_id, result):
    with _lock:
        _seen_events[event_id] = result
        while len(_seen_events) > MAX_SEEN_EVENTS:
            _seen_events.popitem(last=False)


def handle_incident_event(event):
    """Process an incident event. Safe to call repeatedly with the same event."""
    event_id = event.get("event_id")
    event_type = event.get("type")
    payload = event.get("payload") or {}
    service = payload.get("service")

    if event_id:
        previous = already_handled(event_id)
        if previous is not None:
            # The whole point of idempotency: a redelivery must not produce a
            # second round of outreach to the same customers.
            return {**previous, "duplicate": True}

    if event_type == "incident.resolved":
        result = {
            "event_type": event_type, "service": service,
            "outreach": [], "note": "recovery noted; no outreach sent",
        }
        if event_id:
            _remember(event_id, result)
        return result

    if event_type != "incident.opened" or not service:
        result = {"event_type": event_type, "service": service,
                  "outreach": [], "note": "no action for this event type"}
        if event_id:
            _remember(event_id, result)
        return result

    incident = payload.get("incident") or {}
    affected = recent_contacts(service)
    outreach = [
        {
            "customer_id": contact["customer_id"],
            "service": service,
            "action": "notify_known_incident",
            "message": (
                f"We've identified a fault affecting {service} and are working on "
                "it. Your recent report is linked to it -- no further action needed."
            ),
        }
        for contact in affected
    ]

    result = {
        "event_type": event_type,
        "service": service,
        "incident": incident,
        "outreach": outreach,
        "notified": len(outreach),
        "audience": (
            "customers who contacted us about this service within "
            f"{CONTACT_WINDOW_SECONDS:.0f}s. This service holds no customer "
            "database; anyone who never wrote in is not reachable from here."
        ),
    }
    if event_id:
        _remember(event_id, result)
    with _lock:
        _outreach_log.append({"at": time.time(), **result})
    return result


def outreach_log():
    with _lock:
        return list(_outreach_log)


def status():
    with _lock:
        return {
            "services_with_contacts": {s: len(c) for s, c in _contacts.items()},
            "events_seen": len(_seen_events),
            "outreach_batches": len(_outreach_log),
            "delivery": "at-least-once inbound; handler deduplicates by event_id",
        }


def reset():
    """Test hook."""
    with _lock:
        _contacts.clear()
        _seen_events.clear()
        _outreach_log.clear()
