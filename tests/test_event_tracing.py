"""Events are stored with the request id that produced them.

This is the half that was missing from distributed tracing here: the id already
crossed service boundaries, but nothing recorded it next to the event, so there
was nothing to join on. The portfolio's scripts/trace.py depends entirely on
this behaviour.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A fresh database per test, so ordering between tests cannot matter."""
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "events.sqlite3"))
    import importlib

    import utils.storage as storage
    importlib.reload(storage)
    return storage


def test_event_keeps_its_request_id(store):
    store.save_event("thing_happened", {"a": 1}, "req-1")
    events = store.recent_events()
    assert events[0]["request_id"] == "req-1"


def test_events_can_be_filtered_to_one_request(store):
    store.save_event("first", {"n": 1}, "req-1")
    store.save_event("second", {"n": 2}, "req-2")
    store.save_event("third", {"n": 3}, "req-1")

    matched = store.recent_events(request_id="req-1")
    assert {event["event_type"] for event in matched} == {"first", "third"}
    assert all(event["request_id"] == "req-1" for event in matched)


def test_filtering_an_unknown_request_returns_nothing(store):
    """An empty trace must be empty, not a fallback to everything."""
    store.save_event("first", {"n": 1}, "req-1")
    assert store.recent_events(request_id="nope") == []


def test_events_without_a_request_id_are_still_stored(store):
    """Not every event comes from an HTTP request; those must not be lost."""
    store.save_event("background_thing", {"n": 1})
    events = store.recent_events()
    assert len(events) == 1
    assert events[0]["request_id"] is None


def test_a_pre_existing_database_is_migrated_not_dropped(tmp_path, monkeypatch):
    """An event store written before request ids existed must survive the upgrade.

    Recreating the table would be the easy fix and would silently discard
    everything already recorded.
    """
    import sqlite3

    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE events (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               event_type TEXT NOT NULL,
               payload TEXT NOT NULL,
               created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"""
    )
    connection.execute(
        "INSERT INTO events (event_type, payload) VALUES ('old', '{}')"
    )
    connection.commit()
    connection.close()

    monkeypatch.setenv("APP_DB_PATH", str(path))
    import importlib

    import utils.storage as storage
    importlib.reload(storage)

    events = storage.recent_events()
    assert [event["event_type"] for event in events] == ["old"]
    assert events[0]["request_id"] is None

    storage.save_event("new", {}, "req-9")
    assert storage.recent_events(request_id="req-9")[0]["event_type"] == "new"


def test_events_endpoint_accepts_a_request_id(store):
    """The HTTP surface trace.py actually calls."""
    import importlib

    import api.server as server
    importlib.reload(server)

    client = TestClient(server.app)
    response = client.get("/v1/events", params={"request_id": "req-absent"})
    assert response.status_code == 200
    assert response.json()["events"] == []
