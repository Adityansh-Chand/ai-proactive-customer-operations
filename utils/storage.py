"""SQLite event store.

Events carry the request id that produced them. That is what makes a trace
joinable: the id already propagates across service boundaries via
`utils/security.py`, but until it was stored alongside the event there was
nothing to join, and reconstructing one decision meant reading five services'
logs by eye.
"""
import json
import os
import sqlite3
from pathlib import Path


def _db_path():
    default_path = Path(__file__).resolve().parents[1] / "data" / "app.sqlite3"
    return Path(os.getenv("APP_DB_PATH", default_path))


def _connect():
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            request_id TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # Databases created before request ids were stored are migrated in place
    # rather than dropped: an event store people have been writing to is not
    # something to recreate because a column was added.
    columns = {row[1] for row in connection.execute("PRAGMA table_info(events)")}
    if "request_id" not in columns:
        connection.execute("ALTER TABLE events ADD COLUMN request_id TEXT")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS events_request_id ON events (request_id)"
    )
    return connection


def save_event(event_type, payload, request_id=None):
    with _connect() as connection:
        connection.execute(
            "INSERT INTO events (event_type, payload, request_id) VALUES (?, ?, ?)",
            (event_type, json.dumps(payload, default=str), request_id),
        )


def recent_events(limit=20, request_id=None):
    """Most recent events, optionally only those from one request.

    Filtering in SQL rather than in the caller matters: a trace lookup for an old
    request would otherwise have to page through everything since.
    """
    query = "SELECT event_type, payload, request_id, created_at FROM events"
    parameters = []
    if request_id:
        query += " WHERE request_id = ?"
        parameters.append(request_id)
    query += " ORDER BY id DESC LIMIT ?"
    parameters.append(limit)

    with _connect() as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [
        {
            "event_type": event_type,
            "payload": json.loads(payload),
            "request_id": stored_request_id,
            "created_at": created_at,
        }
        for event_type, payload, stored_request_id, created_at in rows
    ]
