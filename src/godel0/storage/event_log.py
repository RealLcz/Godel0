"""Event logging for observability."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from pathlib import Path

from .jsonl import append_jsonl


def log_event(
    events_path: Path,
    event: str,
    run_id: str = "",
    node_id: str = "",
    parent_node_id: str = "",
    payload: Optional[Any] = None,
) -> None:
    """Append a structured event to the event log."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "node_id": node_id,
        "parent_node_id": parent_node_id,
        "event": event,
        "payload": payload or {},
    }
    append_jsonl(events_path, record)
