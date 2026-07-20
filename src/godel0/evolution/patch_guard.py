"""Patch guard: validates self-evolution diffs for safety."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from ..errors import PatchGuardError
from ..constants import ALLOWED_PATCH_PREFIXES, FORBIDDEN_PATCH_PATTERNS
from ..git.patch import extract_changed_files


@dataclass
class PatchGuardReport:
    passed: bool
    allowed_files: List[str] = field(default_factory=list)
    rejected_files: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


class PatchGuard:
    """Validates that a self-evolution patch only modifies allowed files."""

    def __init__(self, allowed_prefixes: Tuple[str, ...] = ALLOWED_PATCH_PREFIXES):
        self.allowed_prefixes = allowed_prefixes

    def check(self, patch: str) -> PatchGuardReport:
        """Check if a patch is allowed."""
        report = PatchGuardReport(passed=True)
        changed = extract_changed_files(patch)

        for f in changed:
            if self._is_forbidden(f):
                report.rejected_files.append(f)
                report.reasons.append(f"Forbidden pattern in: {f}")
                report.passed = False
                continue
            if self._is_allowed(f):
                report.allowed_files.append(f)
            else:
                report.rejected_files.append(f)
                report.reasons.append(f"Not in allowed prefixes: {f}")
                report.passed = False

        if not changed:
            report.passed = False
            report.reasons.append("Empty patch")

        return report

    def check_worktree(self, worktree_path: Path, base_commit: str) -> PatchGuardReport:
        """Check the diff in a worktree against base commit."""
        from ..git.repository import diff_vs_commit
        patch = diff_vs_commit(worktree_path, base_commit)
        return self.check(patch)

    def _is_allowed(self, filepath: str) -> bool:
        for prefix in self.allowed_prefixes:
            if filepath == prefix or filepath.startswith(prefix):
                return True
        return False

    def _is_forbidden(self, filepath: str) -> bool:
        for pattern in FORBIDDEN_PATCH_PATTERNS:
            if pattern in filepath:
                return True
        if os.path.isabs(filepath):
            return True
        if filepath.startswith(".git"):
            return True
        return False
