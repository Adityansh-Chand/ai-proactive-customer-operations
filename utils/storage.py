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
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return connection


def save_event(event_type, payload):
    with _connect() as connection:
        connection.execute(
            "INSERT INTO events (event_type, payload) VALUES (?, ?)",
            (event_type, json.dumps(payload, default=str)),
        )


def recent_events(limit=20):
    with _connect() as connection:
        rows = connection.execute(
            "SELECT event_type, payload, created_at FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    return [
        {
            "event_type": event_type,
            "payload": json.loads(payload),
            "created_at": created_at,
        }
        for event_type, payload, created_at in rows
    ]
