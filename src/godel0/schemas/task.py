"""TaskRecord schema for generated tasks."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from .common import utc_now


class TaskRecord(BaseModel):
    task_id: str
    batch_id: str
    proposer_node_id: str

    repo_id: str
    base_commit: str
    repo_image: Optional[str] = None

    source_trajectory_ids: List[str] = Field(default_factory=list)
    failure_signature_id: str = ""

    # Task source metadata (Phase 6 / P0-12 provenance).
    source_node: str = ""
    source_trajectory: str = ""
    source_type: str = ""
    # P0-12: explicit provenance fields (preferred over the shorter aliases).
    source_node_id: str = ""
    source_trajectory_id: str = ""
    source_task_id: str = ""
    source_failure_stage: str = ""

    bug_strategy: str
    bug_patch_path: str
    oracle_patch_path: Optional[str] = None
    setup_patch_path: Optional[str] = None

    problem_statement_path: str

    f2p_tests: List[str] = Field(default_factory=list)
    baseline_test_command: str
    solver_test_command: Optional[str] = None
    failing_test_output_path: Optional[str] = None

    modified_files: List[str] = Field(default_factory=list)
    modified_entities: List[str] = Field(default_factory=list)
    patch_lines_added: int = 0
    patch_lines_deleted: int = 0

    execution_valid: bool = False
    trajectory_relevant: bool = False
    safety_valid: bool = False
    duplicate_valid: bool = False

    content_hash: str = ""
    created_at: datetime = Field(default_factory=utc_now)
