"""EvaluationOutcome schema for solver evaluation results."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


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
    # P1-4: which of N solver_rollouts this outcome belongs to (0-indexed).
    rollout_index: int = 0


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

    # P0-7 / P0-8 / P0-6: trusted causal ablation. Only Trusted Validator
    # results may set these; proposer-declared causal_ablation metadata is
    # advisory.
    trusted_causal_ablation_pass: bool = True
    repair_one_file_results: dict = Field(default_factory=dict)
    # Clean + only File_i → contract fails. This is the true independent
    # activity signal (leave-one-out "still fail" is necessity, not activity).
    isolated_file_triggers: dict = Field(default_factory=dict)
    independently_active_file_count: int = 0

    syntax_valid: bool = False
    import_valid: bool = False
    timeout_valid: bool = False
    safety_valid: bool = False
    duplicate_valid: bool = False
    relevance_valid: bool = False

    rejection_reasons: list[str] = []
    # P1-3: structured stage codes emitted by validator / causal ablation.
    # Aggregators must count these — not substring-match rejection_reasons.
    failure_stages: list[str] = []
