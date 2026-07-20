"""Diagnosis schema for cycle diagnosis output."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class CycleDiagnosis(BaseModel):
    node_id: str

    primary_root_cause: str
    selected_alert_id: Optional[str] = None
    source_stages: List[
        Literal["solver", "proposer", "validation", "tools", "runtime"]
    ] = Field(default_factory=list)

    recommended_edit_scopes: List[
        Literal[
            "coding_agent",
            "solver_prompt",
            "proposer_prompt",
            "proposer_logic",
            "tools",
            "llm_withtools",
            "utils",
            "requirements",
        ]
    ] = Field(default_factory=list)

    evidence_ids: List[str] = Field(default_factory=list)
    expected_effects: dict[str, str] = Field(default_factory=dict)
    non_goals: List[str] = Field(default_factory=list)
    validation_plan: List[str] = Field(default_factory=list)

    problem_statement: str = ""
    override_reason: Optional[str] = None
