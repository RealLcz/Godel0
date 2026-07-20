"""Git repository operations for agent code versioning."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from ..errors import GitRefError, WorkspaceError


def _is_transient_untracked_path(path: str) -> bool:
    """Return whether an untracked path is a runtime/cache artifact."""
    parts = Path(path).parts
    return (
        "__pycache__" in parts
        or ".pytest_cache" in parts
        or path.endswith((".pyc", ".pyo"))
    )


def run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command in the given repo."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise WorkspaceError(
            f"git {' '.join(args)} failed (exit {result.returncode}): {result.stderr}"
        )
    return result


def init_repo(repo_path: Path) -> None:
    """Initialize a new git repository."""
    repo_path = Path(repo_path)
    repo_path.mkdir(parents=True, exist_ok=True)
    run_git(repo_path, "init")
    ensure_local_identity(repo_path)


def ensure_local_identity(repo_path: Path) -> None:
    """Ensure commits work in clean environments without global git config."""
    email = run_git(repo_path, "config", "--get", "user.email", check=False)
    if email.returncode != 0 or not email.stdout.strip():
        run_git(repo_path, "config", "user.email", "godel0@example.invalid")

    name = run_git(repo_path, "config", "--get", "user.name", check=False)
    if name.returncode != 0 or not name.stdout.strip():
        run_git(repo_path, "config", "user.name", "Godel0")


def add_all(repo_path: Path) -> None:
    """Stage all changes."""
    run_git(repo_path, "add", "-A")


def commit(repo_path: Path, message: str) -> str:
    """Create a commit and return the SHA."""
    ensure_local_identity(repo_path)
    run_git(repo_path, "add", "-A")
    result = run_git(repo_path, "commit", "--allow-empty", "-m", message)
    sha_result = run_git(repo_path, "rev-parse", "HEAD")
    return sha_result.stdout.strip()


def get_head_sha(repo_path: Path) -> str:
    """Get the current HEAD commit SHA."""
    result = run_git(repo_path, "rev-parse", "HEAD")
    return result.stdout.strip()


def diff_commits(repo_path: Path, base: str, head: str = "HEAD") -> str:
    """Get diff between two commits."""
    result = run_git(repo_path, "diff", base, head)
    return result.stdout


def diff_vs_commit(repo_path: Path, commit: str) -> str:
    """Get diff of working tree versus a commit (including untracked)."""
    diff_result = run_git(repo_path, "diff", commit)
    diff_output = diff_result.stdout

    untracked_result = run_git(repo_path, "ls-files", "--others", "--exclude-standard")
    untracked_files = untracked_result.stdout.splitlines()

    for f in untracked_files:
        if _is_transient_untracked_path(f):
            continue
        devnull = "/dev/null"
        result = subprocess.run(
            ["git", "diff", "--no-index", devnull, f],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        diff_output += result.stdout

    return diff_output


def apply_patch(repo_path: Path, patch: str) -> bool:
    """Apply a patch to the repository."""
    result = subprocess.run(
        ["git", "-C", str(repo_path), "apply", "--reject", "-"],
        input=patch,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def reverse_patch(repo_path: Path, patch: str) -> bool:
    """Reverse-apply a patch."""
    result = subprocess.run(
        ["git", "-C", str(repo_path), "apply", "--reverse", "-"],
        input=patch,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def reset_to_commit(repo_path: Path, commit: str) -> None:
    """Hard reset to a commit and clean untracked files."""
    run_git(repo_path, "reset", "--hard", commit)
    run_git(repo_path, "clean", "-fd")


def checkout(repo_path: Path, commit: str) -> None:
    """Checkout a specific commit (detached HEAD)."""
    run_git(repo_path, "checkout", commit)


def create_ref(repo_path: Path, ref: str, sha: str) -> None:
    """Create or update a git ref."""
    run_git(repo_path, "update-ref", ref, sha)


def get_ref(repo_path: Path, ref: str) -> Optional[str]:
    """Get the SHA for a git ref, or None if it doesn't exist."""
    result = run_git(repo_path, "rev-parse", "--verify", ref, check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def list_changed_files(repo_path: Path, base_commit: str) -> list[str]:
    """List files changed since base_commit."""
    result = run_git(repo_path, "diff", "--name-only", base_commit)
    return [f for f in result.stdout.splitlines() if f.strip()]
