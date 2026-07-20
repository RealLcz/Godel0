"""EvaluationOutcome schema for solver evaluation results."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class EvaluationOutcome(BaseModel):
    node_id: str
    task_id: str
    level: Literal[1, 2]

    resolved: bool
    patch_path: Optional[str] = None
    trajectory_id: str

    test_summary_path: str = ""
    runtime_sec: float = 0.0
    error_type: Optional[str] = None


class Level1Result(BaseModel):
    parent_node_id: str
    child_node_id: str
    evaluated_task_ids: list[str]
    parent_solved_task_ids: list[str]
    child_retained_task_ids: list[str]
    child_forgotten_task_ids: list[str]
    child_newly_solved_task_ids: list[str]
    retention_rate: float
    threshold: float
    passed: bool


class Level2Result(BaseModel):
    node_id: str
    task_batch_id: str
    evaluated_task_ids: list[str]
    solved_task_ids: list[str]
    failed_task_ids: list[str]
    accuracy: float
    outcomes: list[EvaluationOutcome] = []


class CandidateValidationReport(BaseModel):
    candidate_id: str
    passed: bool

    patch_applied: bool = False
    source_only: bool = False
    clean_passed_tests: list[str] = []
    bugged_failed_tests: list[str] = []
    bugged_passed_tests: list[str] = []
    f2p_tests: list[str] = []
    p2p_tests: list[str] = []
    reverse_restored: bool = False

    syntax_valid: bool = False
    import_valid: bool = False
    timeout_valid: bool = False
    safety_valid: bool = False
    duplicate_valid: bool = False
    relevance_valid: bool = False

    rejection_reasons: list[str] = []
