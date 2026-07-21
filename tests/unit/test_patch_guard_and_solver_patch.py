"""Unit tests for PatchGuard protection of proposer transport schema (BUG-25)
and solver patch persistence (BUG-07)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from godel0.evolution.patch_guard import PatchGuard
from godel0.constants import FORBIDDEN_PATCH_PATTERNS


def _make_patch(changed_files: list[str]) -> str:
    """Build a minimal unified diff that touches the given files."""
    parts = []
    for f in changed_files:
        parts.append(f"diff --git a/{f} b/{f}")
        parts.append(f"--- a/{f}")
        parts.append(f"+++ b/{f}")
        parts.append("@@ -1 +1 @@")
        parts.append("-old")
        parts.append("+new")
    return "\n".join(parts) + "\n"


class TestPatchGuardProtectsProposerTransport:
    """BUG-25: proposer/request.py and proposer/schemas.py must be forbidden."""

    def test_request_py_is_in_forbidden_patterns(self):
        assert "proposer/request.py" in FORBIDDEN_PATCH_PATTERNS

    def test_schemas_py_is_in_forbidden_patterns(self):
        assert "proposer/schemas.py" in FORBIDDEN_PATCH_PATTERNS

    def test_guard_rejects_request_py_patch(self):
        guard = PatchGuard()
        patch = _make_patch(["proposer/request.py"])
        report = guard.check(patch)
        assert not report.passed
        assert "proposer/request.py" in report.rejected_files

    def test_guard_rejects_schemas_py_patch(self):
        guard = PatchGuard()
        patch = _make_patch(["proposer/schemas.py"])
        report = guard.check(patch)
        assert not report.passed
        assert "proposer/schemas.py" in report.rejected_files

    def test_guard_allows_other_proposer_files(self):
        """Only the transport schema files are forbidden; other proposer/
        files (e.g. proposer/runner.py) remain self-editable."""
        guard = PatchGuard()
        patch = _make_patch(["proposer/runner.py"])
        report = guard.check(patch)
        assert "proposer/runner.py" not in report.rejected_files
        assert "proposer/runner.py" in report.allowed_files

    def test_guard_rejects_parent_dir_traversal(self):
        guard = PatchGuard()
        patch = _make_patch(["../etc/passwd"])
        report = guard.check(patch)
        assert not report.passed

    def test_guard_rejects_git_paths(self):
        guard = PatchGuard()
        patch = _make_patch([".git/config"])
        report = guard.check(patch)
        assert not report.passed


class TestSolverPatchPersistence:
    """BUG-07: solver patch must be persisted to model_patch.diff and
    outcome.patch_path set."""

    def test_patch_path_attribute_exists_on_outcome(self):
        from godel0.schemas.evaluation import EvaluationOutcome

        outcome = EvaluationOutcome(
            node_id="n1",
            task_id="T1",
            level=2,
            resolved=True,
            trajectory_id="tr1",
            patch_path="/scratch/run1/model_patch.diff",
        )
        assert outcome.patch_path == "/scratch/run1/model_patch.diff"

    def test_patch_path_defaults_to_none(self):
        from godel0.schemas.evaluation import EvaluationOutcome

        outcome = EvaluationOutcome(
            node_id="n1",
            task_id="T1",
            level=2,
            resolved=False,
            trajectory_id="tr1",
        )
        assert outcome.patch_path is None
