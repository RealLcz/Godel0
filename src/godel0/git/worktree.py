"""Git worktree management for node code isolation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from ..errors import WorkspaceError
from .repository import run_git, create_ref, get_head_sha


NODE_REF_PREFIX = "refs/godel0/nodes"


class NodeWorktree:
    """Context manager for a git worktree at a specific commit."""

    def __init__(
        self,
        agent_repo: Path,
        scratch_root: Path,
        node_id: str,
        base_commit: str,
    ):
        self.agent_repo = Path(agent_repo).resolve()
        self.scratch_root = Path(scratch_root).resolve()
        self.node_id = node_id
        self.base_commit = base_commit
        self.worktree_path: Optional[Path] = None

    def __enter__(self) -> Path:
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        self.worktree_path = self.scratch_root / f"worktree_{self.node_id}"

        if self.worktree_path.exists():
            shutil.rmtree(self.worktree_path)

        result = run_git(
            self.agent_repo,
            "worktree", "add", "--detach",
            str(self.worktree_path),
            self.base_commit,
        )
        if result.returncode != 0:
            raise WorkspaceError(
                f"Failed to create worktree: {result.stderr}"
            )
        return self.worktree_path

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.worktree_path and self.worktree_path.exists():
            try:
                run_git(self.agent_repo, "worktree", "remove", "--force", str(self.worktree_path))
            except Exception:
                shutil.rmtree(self.worktree_path, ignore_errors=True)


def commit_child(
    agent_repo: Path,
    worktree_path: Path,
    node_id: str,
    message: str,
) -> str:
    """Commit changes in a worktree and create a node ref."""
    run_git(
        worktree_path,
        "add",
        "-A",
        "--",
        ".",
        ":(exclude)**/__pycache__/**",
        ":(exclude)**/.pytest_cache/**",
        ":(exclude)**/*.pyc",
        ":(exclude)**/*.pyo",
    )
    run_git(worktree_path, "commit", "--allow-empty", "-m", message)
    sha = get_head_sha(worktree_path)
    ref = f"{NODE_REF_PREFIX}/{node_id}"
    create_ref(agent_repo, ref, sha)
    return sha
