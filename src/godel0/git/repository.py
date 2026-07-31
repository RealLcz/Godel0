"""Git repository operations for agent code versioning."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from ..errors import GitRefError, WorkspaceError

TRANSIENT_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

TRANSIENT_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".backup",
    ".bak",
)


def is_runtime_artifact(path: str) -> bool:
    """Return whether a path is a runtime/cache artifact that must not enter patches."""
    normalized = path.replace("\\", "/")
    parts = normalized.split("/")
    return (
        any(part in TRANSIENT_DIR_NAMES for part in parts)
        or normalized.endswith(TRANSIENT_SUFFIXES)
    )


def _is_transient_untracked_path(path: str) -> bool:
    """Backward-compatible alias for :func:`is_runtime_artifact`."""
    return is_runtime_artifact(path)


def restore_runtime_artifacts(repo_path: Path, base_commit: str) -> None:
    """Remove runtime-only changes before constructing an evolution patch."""
    repo_path = Path(repo_path)

    tracked = run_git(repo_path, "ls-files", "-z").stdout.split("\0")
    tracked_runtime = [
        path for path in tracked if path and is_runtime_artifact(path)
    ]

    if tracked_runtime:
        run_git(
            repo_path,
            "restore",
            "--source",
            base_commit,
            "--worktree",
            "--",
            *tracked_runtime,
        )

    untracked = run_git(
        repo_path,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    ).stdout.split("\0")

    for relative_path in untracked:
        if not relative_path or not is_runtime_artifact(relative_path):
            continue

        path = repo_path / relative_path
        # Prefer removing the whole transient directory (e.g. __pycache__).
        transient_root = None
        for parent in [path, *path.parents]:
            if parent == repo_path:
                break
            if parent.name in TRANSIENT_DIR_NAMES:
                transient_root = parent
        if transient_root is not None and transient_root.exists():
            shutil.rmtree(transient_root, ignore_errors=True)
            continue

        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def remove_runtime_artifacts_from_tree(repo_root: Path) -> None:
    """Delete runtime artifact files/dirs under an agent tree before root commit."""
    repo_root = Path(repo_root)
    if not repo_root.is_dir():
        return

    for path in sorted(repo_root.rglob("*"), reverse=True):
        try:
            rel = path.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        if not is_runtime_artifact(rel):
            continue
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def assert_no_tracked_runtime_artifacts(repo_path: Path) -> None:
    """Raise if tracked files include runtime artifacts."""
    tracked = run_git(repo_path, "ls-files").stdout.splitlines()
    bad = [path for path in tracked if path and is_runtime_artifact(path)]
    if bad:
        raise WorkspaceError(
            "Tracked runtime artifacts must be removed before root commit: "
            + ", ".join(bad[:20])
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
        if is_runtime_artifact(f):
            continue
        # Skip tracked paths that somehow appear as untracked runtime noise.
        if not f:
            continue
        result = subprocess.run(
            ["git", "diff", "--no-index", "/dev/null", f],
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
