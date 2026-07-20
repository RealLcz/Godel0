"""TrajectoryRecord schema for agent execution traces."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field

from .common import utc_now


class TrajectoryRecord(BaseModel):
    trajectory_id: str
    node_id: str
    role: Literal["solver", "proposer", "self_evolve"]

    task_id: Optional[str] = None
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: Optional[datetime] = None

    model: str = ""
    seed: int = 0
    resolved: Optional[bool] = None
    termination_reason: str = ""

    events_path: str = ""
    final_patch_path: Optional[str] = None
    final_answer_path: Optional[str] = None

    tool_usage_counts: Dict[str, int] = Field(default_factory=dict)
    token_usage: Dict[str, int] = Field(default_factory=dict)
    wall_time_sec: float = 0.0


class TrajectoryEvent(BaseModel):
    """A single event in a trajectory, stored as JSONL."""
    type: str
    step: int = 0
    timestamp: str = ""
    content: Optional[str] = None
    tool: Optional[str] = None
    input: Optional[dict] = None
    output: Optional[str] = None
    path: Optional[str] = None
    reason: Optional[str] = None
