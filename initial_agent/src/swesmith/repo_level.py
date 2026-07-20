from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, List, Optional

from .patch_utils import count_modified_lines, extract_changed_files


class RepositoryWorkspaceError(RuntimeError):
    pass


class RepositoryWorkspace:
    """Temporary clone pinned to a repository commit.

    A shared clone keeps large repository setup cheap while avoiding writes to
    the source repository's worktree or git metadata.
    """

    def __init__(
        self,
        source_repo: str,
        base_commit: str = "",
        parent_dir: Optional[str] = None,
        prefix: str = "swesmith_repo_",
    ) -> None:
        self.source_repo = os.path.abspath(source_repo)
        self.base_commit = base_commit or "HEAD"
        self.parent_dir = parent_dir
        self.prefix = prefix
        self.temp_root = ""
        self.path = ""

    def __enter__(self) -> str:
        if not os.path.isdir(self.source_repo):
            raise RepositoryWorkspaceError(
                f"Repository does not exist: {self.source_repo}"
            )
        if self.parent_dir:
            os.makedirs(self.parent_dir, exist_ok=True)
        self.temp_root = tempfile.mkdtemp(
            prefix=self.prefix,
            dir=self.parent_dir or None,
        )
        self.path = os.path.join(self.temp_root, "repo")

        clone = subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--shared",
                "--no-checkout",
                self.source_repo,
                self.path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if clone.returncode != 0:
            self._cleanup()
            raise RepositoryWorkspaceError(clone.stderr.strip() or "git clone failed")

        checkout = run_git(self.path, "checkout", "--quiet", "--detach", self.base_commit)
        if checkout.returncode != 0:
            message = checkout.stderr.strip() or f"cannot checkout {self.base_commit}"
            self._cleanup()
            raise RepositoryWorkspaceError(message)
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        if self.temp_root:
            shutil.rmtree(self.temp_root, ignore_errors=True)
        self.path = ""
        self.temp_root = ""


def run_git(
    repo_dir: str,
    *args: str,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", repo_dir, *args],
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )


def repository_diff(repo_dir: str, base_commit: str = "HEAD") -> str:
    untracked = run_git(
        repo_dir,
        "ls-files",
        "--others",
        "--exclude-standard",
    )
    paths = [line for line in untracked.stdout.splitlines() if line]
    if paths:
        run_git(repo_dir, "add", "-N", "--", *paths)
    result = run_git(
        repo_dir,
        "diff",
        "--binary",
        "--no-ext-diff",
        base_commit,
        "--",
    )
    return result.stdout if result.returncode == 0 else ""


def apply_repository_patch(repo_dir: str, patch: str, reverse: bool = False) -> bool:
    if not patch.strip():
        return False
    args = ["apply", "--whitespace=nowarn"]
    if reverse:
        args.append("--reverse")
    check = run_git(repo_dir, *args, "--check", "-", input_text=patch)
    if check.returncode != 0:
        return False
    applied = run_git(repo_dir, *args, "-", input_text=patch)
    return applied.returncode == 0


def extract_patch_from_response(response: str) -> str:
    if not response:
        return ""
    marker = response.find("diff --git ")
    if marker < 0:
        return ""
    patch = response[marker:]
    fence = patch.find("\n```")
    if fence >= 0:
        patch = patch[:fence]
    return patch.rstrip() + "\n"


def split_patch_by_file(patch: str) -> List[tuple[str, str]]:
    blocks: List[tuple[str, str]] = []
    current: List[str] = []
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git ") and current:
            block = "".join(current)
            files = extract_changed_files(block)
            if files:
                blocks.append((files[0], block))
            current = []
        if line.startswith("diff --git ") or current:
            current.append(line)
    if current:
        block = "".join(current)
        files = extract_changed_files(block)
        if files:
            blocks.append((files[0], block))
    return blocks


def filter_patch(
    patch: str,
    include_files: Optional[Iterable[str]] = None,
    allow_test_edits: bool = False,
) -> str:
    includes = {_normalize_path(path) for path in include_files or [] if path}
    selected: List[str] = []
    for path, block in split_patch_by_file(patch):
        normalized = _normalize_path(path)
        if includes and normalized not in includes:
            continue
        if not allow_test_edits and is_test_path(normalized):
            continue
        selected.append(block if block.endswith("\n") else block + "\n")
    return "".join(selected)


@dataclass(frozen=True)
class RepositoryPatchSummary:
    valid: bool
    changed_files: List[str]
    modified_lines: int
    rejection_reason: str = ""


def validate_repository_patch(
    patch: str,
    constraints: Any,
    *,
    require_multiple_files: bool = True,
) -> RepositoryPatchSummary:
    changed_files = extract_changed_files(patch)
    modified_lines = count_modified_lines(patch)
    min_files = int(getattr(constraints, "min_modified_files", 1) or 1)
    if require_multiple_files:
        min_files = max(2, min_files)
    max_files = int(getattr(constraints, "max_modified_files", 1) or 1)
    max_lines = int(getattr(constraints, "max_modified_lines", 20) or 20)

    if not patch.strip():
        return RepositoryPatchSummary(False, [], 0, "empty_patch")
    if max_files < min_files:
        return RepositoryPatchSummary(
            False,
            changed_files,
            modified_lines,
            "invalid_modified_file_constraints",
        )
    if len(changed_files) < min_files:
        return RepositoryPatchSummary(
            False,
            changed_files,
            modified_lines,
            "too_few_modified_files",
        )
    if len(changed_files) > max_files:
        return RepositoryPatchSummary(
            False,
            changed_files,
            modified_lines,
            "too_many_modified_files",
        )
    if modified_lines > max_lines:
        return RepositoryPatchSummary(
            False,
            changed_files,
            modified_lines,
            "too_many_modified_lines",
        )
    if any(not is_safe_repo_path(path) for path in changed_files):
        return RepositoryPatchSummary(
            False,
            changed_files,
            modified_lines,
            "unsafe_modified_path",
        )
    allow_tests = bool(getattr(constraints, "allow_test_edits", False))
    if not allow_tests and any(is_test_path(path) for path in changed_files):
        return RepositoryPatchSummary(
            False,
            changed_files,
            modified_lines,
            "modifies_test_files",
        )
    return RepositoryPatchSummary(True, changed_files, modified_lines)


def declared_target_files(plan: Any) -> List[str]:
    values = list(getattr(plan, "target_files", None) or [])
    primary = str(getattr(plan, "target_file", "") or "")
    if primary:
        values.insert(0, primary)
    result: List[str] = []
    for value in values:
        normalized = _normalize_path(value)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def declared_target_symbols(plan: Any) -> List[str]:
    values = list(getattr(plan, "target_symbols", None) or [])
    primary = str(getattr(plan, "target_symbol", "") or "")
    if primary:
        values.insert(0, primary)
    return list(dict.fromkeys(value for value in values if value))


def repository_path(repo_spec: Any, fallback: str = "") -> str:
    return str(
        getattr(repo_spec, "repo_path", "")
        or getattr(repo_spec, "repo_dir", "")
        or fallback
    )


def is_safe_repo_path(path: str) -> bool:
    normalized = _normalize_path(path)
    pure = PurePosixPath(normalized)
    return bool(normalized) and not pure.is_absolute() and ".." not in pure.parts and ".git" not in pure.parts


def is_test_path(path: str) -> bool:
    normalized = _normalize_path(path).lower()
    parts = PurePosixPath(normalized).parts
    name = parts[-1] if parts else ""
    return (
        "test" in parts
        or "tests" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def _normalize_path(path: str) -> str:
    normalized = str(path).replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("a/") or normalized.startswith("b/"):
        normalized = normalized[2:]
    return normalized


__all__ = [
    "RepositoryPatchSummary",
    "RepositoryWorkspace",
    "RepositoryWorkspaceError",
    "apply_repository_patch",
    "declared_target_files",
    "declared_target_symbols",
    "extract_patch_from_response",
    "filter_patch",
    "is_test_path",
    "repository_path",
    "repository_diff",
    "run_git",
    "split_patch_by_file",
    "validate_repository_patch",
]
