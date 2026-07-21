"""NodeRecord schema for evolution tree nodes."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from .common import utc_now


class NodeStatus(str, Enum):
    CANDIDATE = "candidate"
    TOOL_GATE_FAILED = "tool_gate_failed"
    LEVEL1_FAILED = "level1_failed"
    PROPOSER_FAILED = "proposer_failed"
    COMPLETE = "complete"
    REJECTED = "rejected"


class NodeRecord(BaseModel):
    node_id: str
    parent_node_id: Optional[str] = None
    code_commit: str
    code_ref: str
    status: NodeStatus = NodeStatus.CANDIDATE

    mutation_manifest_path: Optional[str] = None

    parent_task_batch_id: Optional[str] = None
    generated_task_batch_id: Optional[str] = None

    level1_result_path: Optional[str] = None
    level2_result_path: Optional[str] = None

    retention_rate: Optional[float] = None
    frontier_accuracy: Optional[float] = None

    solver_score: Optional[float] = None
    proposer_score: Optional[float] = None
    node_score: Optional[float] = None
    solved_task_count: Optional[int] = None

    # BUG-11: HGM-style Thompson Sampling utility measurements. Each value is
    # 1.0 for a solved Level2 task and 0.0 for an unresolved Level2 task. The
    # Beta(alpha=1+successes, beta=1+failures) posterior over these outcomes
    # drives parent selection. Only trusted Level2 outcomes should be appended.
    utility_measures: List[float] = Field(default_factory=list)
    evaluated_task_ids: List[str] = Field(default_factory=list)
    # BUG-12: HGM-style quality gate. When False the node completed but its
    # proposer batch failed the quality gate (valid yield / causal ablation /
    # difficulty / batch completeness) and it must not be selected as a parent.
    selection_eligible: bool = True

    created_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = None

    def is_eligible_parent(self, min_solved: int = 3) -> bool:
        if self.status != NodeStatus.COMPLETE:
            return False
        # BUG-12: honor the explicit HGM quality gate when set. Legacy nodes
        # without selection_eligible default to True and fall through to the
        # score-based checks below.
        if not self.selection_eligible:
            return False
        if self.node_score is None or self.node_score <= 0:
            return False
        if self.generated_task_batch_id is None:
            return False
        if self.retention_rate is None or self.retention_rate < 0:
            return False
        if self.solved_task_count is not None and self.solved_task_count < min_solved:
            return False
        return True
