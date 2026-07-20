from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class BugConstraints(BaseModel):
    min_modified_files: int = 1
    max_modified_files: int = 1
    max_modified_lines: int = 20
    allow_test_edits: bool = False
    require_syntax_valid: bool = True
    desired_behavior: str = ""
    generation_timeout_sec: int = 3600
    context_file_budget: int = 10
    min_mutation_sites: int = 2
    max_mutation_sites: int = 8
    require_generated_tests: bool = False


class FailureSignature(BaseModel):
    signature_id: str
    source_solver_node_id: str = ""
    source_task_id: str = ""
    source_trajectory_id: str = ""
    failure_stage: Literal[
        "localization",
        "reproduction",
        "patch_generation",
        "validation",
        "tool_use",
        "context_management",
    ] = "patch_generation"
    root_cause: str = ""
    target_capability: str = ""
    code_patterns: List[str] = Field(default_factory=list)
    behavior_pattern: dict = Field(default_factory=dict)
    preferred_operators: List[str] = Field(default_factory=list)
    transfer_mode: Literal[
        "same_repo_nearby",
        "cross_repo_homologous",
    ] = "same_repo_nearby"
    forbidden_copy_features: List[str] = Field(default_factory=list)


class BugGenerationPlan(BaseModel):
    plan_id: str
    source_trajectory_ids: List[str] = Field(default_factory=list)
    failure_signature: Optional[FailureSignature] = None
    target_repo_id: str = ""
    target_base_commit: str = ""
    target_file: str = ""
    target_symbol: str = ""
    target_files: List[str] = Field(default_factory=list)
    target_symbols: List[str] = Field(default_factory=list)
    strategy: Literal[
        "lm_modify",
        "lm_rewrite",
        "procedural",
        "combine",
        "pr_mirror",
        "pr_replay",
        "repo_agent",
        "repo_chain",
    ] = "procedural"
    operator: Optional[str] = None
    constraints: BugConstraints = Field(default_factory=BugConstraints)
    rationale: str = ""
    reference_commit: str = ""
    reference_parent: str = ""
    reference_patch: str = ""
    reference_patch_path: str = ""
    task_blueprint: Dict[str, Any] = Field(default_factory=dict)
    model: str = ""
    seed: int = 0


class CodeTarget(BaseModel):
    repo_id: str
    file_path: str
    symbol_name: str
    symbol_type: str = "function"
    line_start: int = 0
    line_end: int = 0
    source: str = ""
    has_test_coverage: bool = False
    novelty_score: float = 0.0


__all__ = [
    "BugConstraints",
    "FailureSignature",
    "BugGenerationPlan",
    "CodeTarget",
]
