"""Event logger — the forensic backbone of CogWorm.

Every message, state change, defense decision, and signal observation is
recorded here with full lineage metadata.  This is the data that lets an
investigator answer: Where did the signal enter?  Which path did it take?
Where did it disappear?  Which defense blocked it?
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from cogworm.core.models import Event, EventType


class EventLogger:
    """Append-only structured event log backed by SQLite."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else None
        self._events: list[Event] = []
        self._conn: sqlite3.Connection | None = None

        if self._db_path:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._init_db()

    def _init_db(self) -> None:
        assert self._conn is not None
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                cycle INTEGER,
                source_agent TEXT,
                dest_agent TEXT,
                message_id TEXT,
                parent_message_id TEXT,
                model TEXT,
                topology TEXT,
                hop_number INTEGER,
                signal_fidelity REAL,
                agent_state TEXT,
                defense_id TEXT,
                defense_decision TEXT,
                details TEXT
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiment
            ON events(experiment_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_type
            ON events(event_type)
        """)
        self._conn.commit()

    # ── Logging ──────────────────────────────────────────────────────

    def log(self, event: Event) -> None:
        """Record a single event."""
        self._events.append(event)
        if self._conn:
            self._conn.execute(
                """INSERT INTO events VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )""",
                (
                    event.id,
                    event.experiment_id,
                    event.event_type.value,
                    event.timestamp.isoformat(),
                    event.cycle,
                    event.source_agent,
                    event.dest_agent,
                    event.message_id,
                    event.parent_message_id,
                    event.model,
                    event.topology,
                    event.hop_number,
                    event.signal_fidelity,
                    event.agent_state,
                    event.defense_id,
                    event.defense_decision,
                    json.dumps(event.details) if event.details else None,
                ),
            )
            self._conn.commit()

    def log_simple(
        self,
        experiment_id: str,
        event_type: EventType,
        cycle: int = 0,
        **kwargs: Any,
    ) -> Event:
        """Convenience: create and log an event in one call."""
        event = Event(
            experiment_id=experiment_id,
            event_type=event_type,
            cycle=cycle,
            **kwargs,
        )
        self.log(event)
        return event

    # ── Queries ──────────────────────────────────────────────────────

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def for_experiment(self, experiment_id: str) -> list[Event]:
        return [e for e in self._events if e.experiment_id == experiment_id]

    def by_type(self, event_type: EventType) -> list[Event]:
        return [e for e in self._events if e.event_type == event_type]

    def message_lineage(self, message_id: str) -> list[Event]:
        """Trace the full lineage of a message through the network."""
        chain: list[Event] = []
        current = message_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            for e in self._events:
                if e.message_id == current:
                    chain.append(e)
                    if e.parent_message_id:
                        current = e.parent_message_id
                    else:
                        current = None
                    break
            else:
                break
        return list(reversed(chain))

    def propagation_path(self, experiment_id: str) -> list[dict]:
        """Build the signal propagation path as a list of hops."""
        path = []
        for e in self.for_experiment(experiment_id):
            if e.event_type in (
                EventType.SIGNAL_INJECTED,
                EventType.SIGNAL_ACTIVATED,
                EventType.SIGNAL_DECAYED,
                EventType.DEFENSE_BLOCKED,
            ):
                path.append({
                    "hop": e.hop_number,
                    "source": e.source_agent,
                    "dest": e.dest_agent,
                    "event": e.event_type.value,
                    "fidelity": e.signal_fidelity,
                    "defense": e.defense_id,
                    "cycle": e.cycle,
                })
        return path

    # ── Export ────────────────────────────────────────────────────────

    def export_jsonl(self, path: str | Path) -> int:
        """Export all events as newline-delimited JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for e in self._events:
                f.write(e.model_dump_json() + "\n")
        return len(self._events)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
