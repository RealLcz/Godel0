"""Shared schema types used across Godel0."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def safe_str(val: Any, max_len: int = 10000) -> str:
    """Safely convert to string with truncation."""
    s = str(val)
    if len(s) > max_len:
        return s[:max_len] + "...<truncated>"
    return s
