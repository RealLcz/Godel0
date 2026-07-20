"""Mutation manifest builder."""

from __future__ import annotations

from pathlib import Path
from typing import List

from ..git.patch import extract_changed_files
from ..git.repository import diff_vs_commit
from ..schemas.mutation import MutationManifest, ToolEdit


def build_mutation_manifest(
    parent_node_id: str,
    child_node_id: str,
    worktree_path: Path,
    base_commit: str,
    diagnosis_problem_statement: str = "",
    trigger_type: str = "joint",
) -> MutationManifest:
    """Build a mutation manifest from the actual git diff."""
    patch = diff_vs_commit(worktree_path, base_commit)
    changed = extract_changed_files(patch)

    scopes = []
    tool_edits = []
    for f in changed:
        if f.startswith("tools/"):
            if "tools" not in scopes:
                scopes.append("tools")
            tool_edits.append(ToolEdit(
                tool_name=f,
                action="modified",
                file_path=f,
            ))
        elif f.startswith("proposer/"):
            if "proposer" not in scopes:
                scopes.append("proposer")
        elif f.startswith("swesmith/"):
            if "proposer" not in scopes:
                scopes.append("proposer")
        elif f.startswith("coding_agent") or f.startswith("llm"):
            if "runtime" not in scopes:
                scopes.append("runtime")
        elif f.startswith("prompts/"):
            if "solver" not in scopes:
                scopes.append("solver")
        elif f.startswith("utils/"):
            if "runtime" not in scopes:
                scopes.append("runtime")

    return MutationManifest(
        parent_node_id=parent_node_id,
        child_node_id=child_node_id,
        trigger_type=trigger_type,
        source_ids=[],
        diagnosed_problem_statement=diagnosis_problem_statement,
        changed_files=changed,
        changed_scopes=scopes,
        tool_edits=tool_edits,
        summary=f"Changed {len(changed)} files",
    )
