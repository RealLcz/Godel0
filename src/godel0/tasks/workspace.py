"""Task workspace management."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from ..errors import WorkspaceError
from ..git.repository import apply_patch, commit, init_repo, reset_to_commit


class TaskWorkspace:
    """Manages a workspace for a specific task evaluation."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def setup_bugged_repo(
        self,
        source_repo: Path,
        base_commit: str,
        bug_patch: str,
        task_id: str,
    ) -> Path:
        """Set up a bugged repository for solver evaluation.

        1. Copy clean repo.
        2. Reset to base commit.
        3. Apply bug patch.
        """
        workspace = self.workspace_root / task_id / "bugged_repo"
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.copytree(source_repo, workspace)

        try:
            reset_to_commit(workspace, base_commit)
        except Exception:
            pass

        if bug_patch:
            apply_patch(workspace, bug_patch)

        return workspace

    def setup_clean_repo(
        self,
        source_repo: Path,
        base_commit: str,
        task_id: str,
    ) -> Path:
        """Set up a clean repository for validation."""
        workspace = self.workspace_root / task_id / "clean_repo"
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.copytree(source_repo, workspace)

        try:
            reset_to_commit(workspace, base_commit)
        except Exception:
            pass

        return workspace

    def seal_bugged_snapshot(self, repo: Path) -> str:
        """Replace inherited history with one bugged-tree commit.

        Generated tasks start from a known-clean repository and apply a bug
        patch.  Exposing that inherited Git history lets a solver recover the
        oracle with ``git show HEAD^``.  The solver only needs a baseline for
        producing its patch, so give it an isolated one-commit repository.
        """
        repo = Path(repo)
        git_dir = repo / ".git"
        if git_dir.is_dir():
            shutil.rmtree(git_dir)
        elif git_dir.exists():
            git_dir.unlink()
        init_repo(repo)
        return commit(repo, "bugged task snapshot")

    def cleanup(self, task_id: str) -> None:
        """Clean up workspace for a task."""
        workspace = self.workspace_root / task_id
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)
