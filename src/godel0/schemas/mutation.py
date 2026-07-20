"""Mutation manifest schema for self-evolution changes."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ToolEdit(BaseModel):
    tool_name: str
    action: Literal["added", "modified", "deleted"]
    file_path: str
    description: str = ""


class MutationManifest(BaseModel):
    parent_node_id: str
    child_node_id: str

    trigger_type: Literal[
        "solver_failure",
        "proposer_failure",
        "shared_tool_failure",
        "joint",
    ]

    source_ids: List[str] = Field(default_factory=list)
    diagnosed_problem_statement: str = ""

    changed_files: List[str] = Field(default_factory=list)

    changed_scopes: List[
        Literal[
            "runtime",
            "solver",
            "proposer",
            "self_evolve",
            "tools",
            "tool_runtime",
            "requirements",
        ]
    ] = Field(default_factory=list)

    tool_edits: List[ToolEdit] = Field(default_factory=list)
    summary: str = ""
