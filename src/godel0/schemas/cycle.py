"""Cycle summary and evidence bundle schemas."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class CycleStage(str, Enum):
    ROOT_BOOTSTRAP = "root_bootstrap"
    LEVEL1_FAILED = "level1_failed"
    PROPOSER_FAILED = "proposer_failed"
    LEVEL2_COMPLETE = "level2_complete"


class NodeCycleSummary(BaseModel):
    node_id: str
    parent_node_id: Optional[str] = None
    stage_reached: CycleStage = CycleStage.ROOT_BOOTSTRAP

    level1_retention: Optional[float] = None
    forgotten_task_ids: List[str] = Field(default_factory=list)
    newly_solved_parent_task_ids: List[str] = Field(default_factory=list)

    proposer_requested_tasks: int = 0
    proposer_generated_candidates: int = 0
    proposer_accepted_tasks: int = 0
    proposer_valid_yield: Optional[float] = None
    proposer_rejection_distribution: Dict[str, int] = Field(default_factory=dict)
    proposer_operator_distribution: Dict[str, int] = Field(default_factory=dict)

    level2_accuracy: Optional[float] = None
    level2_solved_task_ids: List[str] = Field(default_factory=list)
    level2_failed_task_ids: List[str] = Field(default_factory=list)

    solver_special_stats: Dict[str, float] = Field(default_factory=dict)
    proposer_special_stats: Dict[str, float] = Field(default_factory=dict)
    tool_usage_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    solver_score: Optional[float] = None
    proposer_score: Optional[float] = None
    node_score: Optional[float] = None


class AlertSource(str, Enum):
    SOLVER = "solver"
    PROPOSER = "proposer"
    SHARED = "shared"


class AlertPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"


class SpecialAlert(BaseModel):
    alert_id: str
    alert_type: str
    source: AlertSource
    priority: AlertPriority = AlertPriority.NORMAL

    triggered: bool = False
    severity: float = 0.0
    confidence: float = 0.0

    metric_name: str = ""
    observed_value: float = 0.0
    threshold: Optional[float] = None

    evidence_ids: List[str] = Field(default_factory=list)
    recommended_attention: str = ""


class EvidenceItem(BaseModel):
    evidence_id: str
    evidence_type: Literal[
        "solver_trajectory",
        "proposer_candidate",
        "proposer_batch_summary",
        "tool_incident",
        "success_contrast",
        "evaluation_diff",
    ]
    source_stage: str = ""
    summary: str = ""
    # BUG-22: raw representative text (up to ~8k chars for primary failures,
    # ~2-4k for supporting/contrast). The diagnoser reads this directly
    # instead of only the 500-char ``summary``.
    raw_text: Optional[str] = None
    raw_excerpt_path: Optional[str] = None
    token_estimate: int = 0
    importance: float = 0.0


class CycleEvidenceBundle(BaseModel):
    node_id: str
    cycle_summary_path: str = ""
    special_alerts: List[SpecialAlert] = Field(default_factory=list)
    items: List[EvidenceItem] = Field(default_factory=list)
    total_token_estimate: int = 0
    truncated: bool = False
