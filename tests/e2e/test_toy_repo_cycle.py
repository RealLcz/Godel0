"""E2E test: full toy-repo evolution cycle.

Tests the complete flow:
1. Create toy repo with clamp() function
2. Apply a procedural mutation to create a bug
3. Validate the bug candidate (F2P)
4. Create a task from the validated candidate
5. Run solver evaluation (simulated)
6. Compute scores (a, b, ab)
7. Build cycle summary
8. Run diagnosis
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "initial_agent" / "src"))

from godel0.git.repository import init_repo, commit, get_head_sha, reset_to_commit, apply_patch, diff_vs_commit
from godel0.git.patch import extract_changed_files, count_patch_lines, is_source_only
from godel0.proposer_trusted.candidate_validator import CandidateValidator
from godel0.proposer_trusted.safety import check_safety
from godel0.proposer_trusted.duplicate_detector import DuplicateDetector
from godel0.proposer_trusted.task_committer import TaskCommitter
from godel0.tasks.store import TaskStore
from godel0.evaluation.level1 import Level1Evaluator
from godel0.evaluation.level2 import Level2Evaluator
from godel0.controller.scorer import compute_scores
from godel0.evolution.cycle_builder import NodeCycleBuilder
from godel0.evolution.special_detectors import CompositeSpecialDetector
from godel0.evolution.evidence_selector import CycleEvidenceSelector
from godel0.evolution.diagnose import CycleDiagnoser
from godel0.schemas.node import NodeRecord, NodeStatus
from godel0.schemas.evaluation import EvaluationOutcome

from tests.fixtures.toy_repo import create_toy_repo, get_toy_repo_commit, run_toy_tests


CLAMP_SOURCE = '''"""Toy module for testing."""


def clamp(x, low, high):
    """Clamp x to [low, high] range."""
    return max(low, min(x, high))
'''

CLAMP_TESTS = '''"""Tests for toy module."""
from toy_module import clamp


def test_clamp_normal():
    assert clamp(5, 0, 10) == 5


def test_clamp_lower():
    assert clamp(-5, 0, 10) == 0


def test_clamp_upper():
    assert clamp(15, 0, 10) == 10
'''


def create_buggy_clamp(repo_path: Path) -> str:
    """Create a bug in the clamp function by changing the comparison.

    Bug: change `max(low, min(x, high))` to `max(low, min(x, low))`
    This makes the upper boundary test fail.
    """
    source_file = repo_path / "toy_module.py"
    source = source_file.read_text()

    buggy_source = source.replace("min(x, high)", "min(x, low)")

    source_file.write_text(buggy_source)

    result = subprocess.run(
        ["git", "-C", str(repo_path), "diff"],
        capture_output=True,
        text=True,
    )
    patch = result.stdout

    source_file.write_text(source)
    reset_to_commit(repo_path, get_head_sha(repo_path))

    return patch


@pytest.fixture
def toy_repo_e2e(tmp_path):
    """Create a toy repo for E2E testing."""
    repo_dir = tmp_path / "toy_repo"
    repo_dir.mkdir()

    (repo_dir / "toy_module.py").write_text(CLAMP_SOURCE)
    (repo_dir / "test_toy.py").write_text(CLAMP_TESTS)

    subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(repo_dir), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo_dir), capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=str(repo_dir), capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=str(repo_dir), capture_output=True)

    commit_sha = get_toy_repo_commit(repo_dir)

    tests = run_toy_tests(repo_dir)
    assert tests["passed"], f"Clean repo tests should pass: {tests['stdout']}"

    return {
        "path": repo_dir,
        "commit": commit_sha,
    }


class TestE2EToyRepo:
    def test_full_cycle(self, toy_repo_e2e, tmp_path):
        """Test the complete evolution cycle on a toy repo."""
        repo_path = toy_repo_e2e["path"]
        base_commit = toy_repo_e2e["commit"]

        print("\n=== Step 1: Generate bug candidate ===")
        bug_patch = create_buggy_clamp(repo_path)
        assert bug_patch, "Bug patch should not be empty"
        assert "min(x, low)" in bug_patch
        print(f"Bug patch generated: {len(bug_patch)} chars")

        print("\n=== Step 2: Validate candidate ===")
        validator = CandidateValidator(
            workspace_root=tmp_path / "validator",
            test_timeout_sec=30,
        )
        report = validator.validate(
            candidate_patch=bug_patch,
            repo_path=repo_path,
            base_commit=base_commit,
            test_command="python -m pytest test_toy.py -v",
            candidate_id="cand_001",
        )

        print(f"Validation: passed={report.passed}, f2p={report.f2p_tests}")
        print(f"  patch_applied={report.patch_applied}, syntax_valid={report.syntax_valid}")
        print(f"  reverse_restored={report.reverse_restored}")
        assert report.patch_applied, "Patch should apply"
        assert report.syntax_valid, "Patched code should be syntactically valid"

        print("\n=== Step 3: Check safety ===")
        is_safe, safety_reasons = check_safety(bug_patch)
        assert is_safe, f"Patch should be safe: {safety_reasons}"

        print("\n=== Step 4: Check duplicate ===")
        detector = DuplicateDetector()
        is_unique = detector.check(bug_patch, "toy/repo", "toy_module.py", "clamp", "change_constant")
        assert is_unique, "First candidate should be unique"

        print("\n=== Step 5: Commit task to TaskStore ===")
        task_store = TaskStore(tmp_path / "task_store")
        committer = TaskCommitter(task_store)
        task = committer.commit_task(
            batch_id="batch_001",
            proposer_node_id="root",
            repo_id="toy/repo",
            base_commit=base_commit,
            bug_strategy="procedural",
            bug_patch=bug_patch,
            problem_statement="The clamp function does not properly handle upper boundary values.",
            f2p_tests=report.f2p_tests,
            baseline_test_command="python -m pytest test_toy.py -v",
        )
        assert task.task_id
        assert task_store.exists(task.task_id)
        print(f"Task committed: {task.task_id}")

        print("\n=== Step 6: Simulate solver evaluation ===")
        solver_outcomes_level1 = [
            EvaluationOutcome(
                node_id="child",
                task_id=task.task_id,
                level=1,
                resolved=True,
                trajectory_id="tr_1",
            )
        ]

        print("\n=== Step 7: Level 1 evaluation ===")
        level1_eval = Level1Evaluator(regression_threshold=0.8)
        level1_result = level1_eval.compute_retention(
            [task.task_id],
            solver_outcomes_level1,
        )
        assert level1_result.passed
        print(f"Level 1: retention={level1_result.retention_rate:.2f}, passed={level1_result.passed}")

        print("\n=== Step 8: Level 2 evaluation ===")
        solver_outcomes_level2 = [
            EvaluationOutcome(
                node_id="child",
                task_id=task.task_id,
                level=2,
                resolved=True,
                trajectory_id="tr_2",
            )
        ]
        level2_eval = Level2Evaluator()
        level2_result = level2_eval.compute_accuracy(
            "child", "batch_001", solver_outcomes_level2
        )
        assert level2_result.accuracy == 1.0
        print(f"Level 2: accuracy={level2_result.accuracy:.2f}")

        print("\n=== Step 9: Compute scores ===")
        scores = compute_scores(
            retention_rate=level1_result.retention_rate,
            frontier_accuracy=level2_result.accuracy,
            regression_weight=0.5,
        )
        print(f"Scores: a={scores.solver_score:.4f}, b={scores.proposer_score:.4f}, ab={scores.node_score:.4f}")
        assert scores.node_score >= 0

        print("\n=== Step 10: Build cycle summary ===")
        node = NodeRecord(
            node_id="child_001",
            parent_node_id="root",
            code_commit="child_sha",
            code_ref="refs/godel0/nodes/child_001",
            status=NodeStatus.COMPLETE,
            retention_rate=scores.retention_rate,
            frontier_accuracy=scores.frontier_accuracy,
            solver_score=scores.solver_score,
            proposer_score=scores.proposer_score,
            node_score=scores.node_score,
        )
        builder = NodeCycleBuilder()
        summary = builder.build(
            node,
            level1=level1_result,
            proposer_stats={"requested": 1, "generated": 1, "accepted": 1},
            level2=level2_result,
        )
        assert summary.stage_reached.value == "level2_complete"
        print(f"Cycle summary: stage={summary.stage_reached}")

        print("\n=== Step 11: Detect special alerts ===")
        detector = CompositeSpecialDetector()
        alerts = detector.detect(summary)
        print(f"Alerts: {len(alerts)}")

        print("\n=== Step 12: Select evidence ===")
        selector = CycleEvidenceSelector()
        evidence = selector.select(summary, alerts)
        print(f"Evidence items: {len(evidence.items)}")

        print("\n=== Step 13: Diagnose ===")
        diagnoser = CycleDiagnoser()
        diagnosis = diagnoser.diagnose(node.node_id, summary, evidence)
        assert diagnosis.primary_root_cause
        print(f"Diagnosis: {diagnosis.primary_root_cause}")
        print(f"  scopes: {diagnosis.recommended_edit_scopes}")

        print("\n=== E2E Test Complete ===")
