"""Workspace manager for creating and cleaning isolated workspaces."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from ..errors import WorkspaceError


class WorkspaceManager:
    """Manages scratch workspaces for solver, proposer, and validation runs."""

    def __init__(self, scratch_root: Path):
        self.scratch_root = Path(scratch_root)
        self.scratch_root.mkdir(parents=True, exist_ok=True)

    def create_workspace(
        self,
        run_id: str,
        phase: str,
        node_id: str,
        task_id: Optional[str] = None,
    ) -> Path:
        """Create a workspace directory: scratch/<run_id>/<phase>/<node_id>/<task_id>/"""
        parts = [self.scratch_root, run_id, phase, node_id]
        if task_id:
            parts.append(task_id)
        ws = Path(*parts)
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    def clean_workspace(self, workspace: Path) -> None:
        """Remove a workspace directory."""
        workspace = Path(workspace)
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)

    def create_solver_workspace(
        self, run_id: str, node_id: str, task_id: str
    ) -> Path:
        return self.create_workspace(run_id, "solver", node_id, task_id)

    def create_proposer_workspace(
        self, run_id: str, node_id: str
    ) -> Path:
        return self.create_workspace(run_id, "proposer", node_id)

    def create_validator_workspace(
        self, run_id: str, node_id: str, candidate_id: str
    ) -> Path:
        return self.create_workspace(run_id, "validator", node_id, candidate_id)

    def create_self_evolve_workspace(
        self, run_id: str, node_id: str
    ) -> Path:
        return self.create_workspace(run_id, "self_evolve", node_id)
