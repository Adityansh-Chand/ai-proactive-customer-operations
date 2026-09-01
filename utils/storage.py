"""SQLite event store.

Events carry the request id that produced them. That is what makes a trace
joinable: the id already propagates across service boundaries via
`utils/security.py`, but until it was stored alongside the event there was
nothing to join, and reconstructing one decision meant reading five services'
logs by eye.

**One connection per thread, reused, in WAL mode.** The original version opened
a connection per write and re-ran the schema statements on each one. That is
invisible with a single worker and is the ceiling on throughput with several:
`scripts/scale_test.py` in the portfolio repository measured it flat at about
116 writes/second across four processes no matter how many workers were added,
which is the same band as the retrieval endpoint's measured peak. The service
could not outrun its own event log.

Neither half of the fix is worth making alone, which is the part that is not
obvious. Enabling WAL on its own measured **0.55-0.80x** -- slower than the
default rollback journal -- because a connection opened per write pays WAL's
per-connection index setup and checkpointing and collects none of its benefit.
Connection reuse alone is about 2x. Together they measured **8.2-9.5x**. See
ADR-009 in `ai-engineering-portfolio`.
"""
import json
import os
import sqlite3
import threading
from pathlib import Path

# Thread-local rather than one connection behind a global lock: under WAL a
# reader does not block the writer, and a single shared connection would put
# `recent_events` behind the same lock as `save_event` for no reason.
_state = threading.local()


def _db_path():
    default_path = Path(__file__).resolve().parents[1] / "data" / "app.sqlite3"
    return Path(os.getenv("APP_DB_PATH", default_path))


def _connect():
    """The calling thread's connection, opened once.

    Keyed on the resolved path, so repointing `APP_DB_PATH` opens a new
    connection rather than continuing to write to the previous database. Without
    that, caching the connection would turn every test that redirects the store
    into a test that silently asserts against the wrong file -- a correctness
    bug introduced in the name of throughput.
    """
    path = _db_path()
    connection = getattr(_state, "connection", None)
    if connection is not None and getattr(_state, "path", None) == path:
        return connection
    if connection is not None:
        connection.close()

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    # Set before anything opens a transaction -- `journal_mode` cannot be
    # changed from inside one.
    connection.execute("PRAGMA journal_mode=WAL")
    # Wait for the write lock instead of failing when several workers commit at
    # once. Python's default is already five seconds; it is stated here because
    # it is load-bearing under concurrency rather than incidental.
    connection.execute("PRAGMA busy_timeout=5000")
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
    connection.commit()

    _state.connection = connection
    _state.path = path
    return connection


def save_event(event_type, payload, request_id=None):
    connection = _connect()
    # `with connection` is a transaction, not a close: it commits on success and
    # rolls back on an exception, and the connection stays open for the next
    # call. That is what makes the reuse above safe to combine with the original
    # call sites unchanged.
    with connection:
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

    connection = _connect()
    with connection:
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
