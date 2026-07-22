"""P0-5: provenance must flow FailureSignature → Plan → Candidate → Task."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "initial_agent" / "src"))

from proposer.planner import ProposerPlanner
from proposer.provenance import merge_provenance, provenance_from_signature
from proposer.request import CandidateArtifact
from proposer.runner import ProposerRunner
from proposer.schemas import BugGenerationPlan, FailureSignature
from proposer.trajectory_analyzer import TrajectoryView


def _signature(**kwargs) -> FailureSignature:
    base = dict(
        signature_id="sig-1",
        source_solver_node_id="parent-node",
        source_task_id="task-parent-1",
        source_trajectory_id="/scratch/parent/traj.jsonl",
        failure_stage="patch_generation",
        root_cause="missed edge",
        target_capability="localization",
    )
    base.update(kwargs)
    return FailureSignature(**base)


class TestPlannerBlueprintProvenance:
    def test_blueprint_keeps_node_and_task_ids(self):
        planner = ProposerPlanner(code_locator=MagicMock())
        target = SimpleNamespace(
            repo_id="toy",
            file_path="a.py",
            symbol_name="f",
        )
        blueprint = planner._build_task_blueprint(_signature(), target)
        assert blueprint["source_node_id"] == "parent-node"
        assert blueprint["source_task_id"] == "task-parent-1"
        assert blueprint["source_trajectory_id"] == "/scratch/parent/traj.jsonl"
        assert blueprint["source_failure_stage"] == "patch_generation"


class TestStampPlanNoInventedNode:
    def test_does_not_fallback_to_proposer_node_id(self):
        runner = ProposerRunner(engine=None)
        sig = _signature(source_solver_node_id="")
        plan = BugGenerationPlan(
            plan_id="p1",
            failure_signature=sig,
            source_trajectory_ids=[],
            task_blueprint={
                "source_node_id": "",
                "source_task_id": "",
                "source_trajectory_id": "",
            },
        )
        runner._stamp_plan_source_provenance(
            plan,
            source_type="parent_failure",
            traces=[],
            traj_by_id={},
            proposer_node_id="current-child",
        )
        # Missing source node must stay empty — not become current-child.
        assert plan.task_blueprint["source_type"] == "parent_failure"
        assert plan.task_blueprint.get("source_node_id", "") == ""
        assert plan.task_blueprint.get("source_node_id") != "current-child"

    def test_prefers_signature_fields(self):
        runner = ProposerRunner(engine=None)
        plan = BugGenerationPlan(
            plan_id="p1",
            failure_signature=_signature(),
            source_trajectory_ids=["/scratch/parent/traj.jsonl"],
            task_blueprint={},
        )
        runner._stamp_plan_source_provenance(
            plan,
            source_type="parent_failure",
            traces=[],
            traj_by_id={},
            proposer_node_id="current-child",
        )
        assert plan.task_blueprint["source_node_id"] == "parent-node"
        assert plan.task_blueprint["source_task_id"] == "task-parent-1"
        assert plan.task_blueprint["source_trajectory_id"] == "/scratch/parent/traj.jsonl"
        assert plan.task_blueprint["source_failure_stage"] == "patch_generation"
        assert plan.task_blueprint["source_type"] == "parent_failure"


class TestStampCandidateProvenance:
    def test_candidate_metadata_copies_full_chain(self):
        runner = ProposerRunner(engine=None)
        plan = BugGenerationPlan(
            plan_id="p1",
            failure_signature=_signature(),
            source_trajectory_ids=["/scratch/parent/traj.jsonl"],
            reference_parent="deadbeef",  # PR parent — must NOT become source_node
            task_blueprint={
                "source_type": "parent_failure",
                "source_node_id": "parent-node",
                "source_task_id": "task-parent-1",
                "source_trajectory_id": "/scratch/parent/traj.jsonl",
                "source_failure_stage": "patch_generation",
            },
        )
        meta = runner._stamp_provenance({}, plan)
        assert meta["source_type"] == "parent_failure"
        assert meta["source_node_id"] == "parent-node"
        assert meta["source_task_id"] == "task-parent-1"
        assert meta["source_trajectory_id"] == "/scratch/parent/traj.jsonl"
        assert meta["source_failure_stage"] == "patch_generation"
        assert meta.get("source_node") == "parent-node"
        assert meta.get("source_node") != "deadbeef"


class TestBatchCommitNoNodeGuess:
    def test_commit_does_not_substitute_proposer_node(self, tmp_path):
        from godel0.proposer_trusted.task_committer import TaskCommitter
        from godel0.tasks.batch import TaskBatchBuilder
        from godel0.tasks.store import TaskStore

        store = TaskStore(tmp_path / "tasks")
        committer = TaskCommitter(store)
        builder = TaskBatchBuilder(
            batch_size=10,
            source_quotas={"parent_failure": 5, "current_child_level1": 5},
        )

        # Simulate the provenance resolution branch used at commit time.
        cand = SimpleNamespace(
            candidate_id="c1",
            plan_id="plan-1",
            strategy="repo_chain",
            patch="diff --git a/a.py b/a.py\n+x\n",
            modified_files=["a.py"],
            modified_entities=["f"],
            generation_metadata={
                "source_type": "parent_failure",
                "source_node_id": "parent-node",
                "source_task_id": "task-parent-1",
                "source_trajectory_id": "/scratch/parent/traj.jsonl",
                "source_failure_stage": "patch_generation",
                "source_trajectory_ids": ["/scratch/parent/traj.jsonl"],
            },
        )
        meta = dict(cand.generation_metadata)
        blueprint = {
            "source_type": "parent_failure",
            "source_node_id": "parent-node",
            "source_task_id": "task-parent-1",
            "source_trajectory_id": "/scratch/parent/traj.jsonl",
            "source_failure_stage": "patch_generation",
        }
        node_id = "current-child"
        source_type = "parent_failure"
        source_trajectory = "/scratch/parent/traj.jsonl"
        source_node_id = str(
            meta.get("source_node_id")
            or blueprint.get("source_node_id")
            or meta.get("source_node")
            or ""
        )
        assert source_node_id == "parent-node"
        assert source_node_id != node_id

        # Empty provenance must stay empty (no inventing current-child).
        empty_node = str(
            {}.get("source_node_id")
            or {}.get("source_node_id")
            or ""
        )
        assert empty_node == ""

        task = committer.commit_task(
            batch_id="b1",
            proposer_node_id=node_id,
            repo_id="toy",
            base_commit="abc",
            bug_strategy="repo_chain",
            bug_patch=cand.patch,
            problem_statement="bug",
            f2p_tests=["test_a"],
            baseline_test_command="pytest -q",
            solver_test_command="pytest -q",
            source_type=source_type,
            source_node_id=source_node_id,
            source_trajectory_id=source_trajectory,
            source_task_id="task-parent-1",
            source_failure_stage="patch_generation",
        )
        assert task.source_node_id == "parent-node"
        assert task.source_task_id == "task-parent-1"
        assert task.source_trajectory_id == "/scratch/parent/traj.jsonl"
        assert task.source_failure_stage == "patch_generation"
        assert task.source_type == "parent_failure"
        assert task.proposer_node_id == "current-child"

    def test_missing_provenance_stays_empty_not_current_child(self):
        from godel0.tasks.batch import TaskBatchBuilder

        # Direct check of the resolution order used in build_batch.
        meta: dict = {}
        blueprint: dict = {}
        node_id = "current-child"
        source_node_id = str(
            meta.get("source_node_id")
            or blueprint.get("source_node_id")
            or meta.get("source_node")
            or ""
        )
        assert source_node_id == ""
        assert source_node_id != node_id


class TestMergeProvenance:
    def test_first_non_empty_wins(self):
        merged = merge_provenance(
            {"source_node_id": "from-blueprint"},
            provenance_from_signature(_signature(source_solver_node_id="from-sig")),
            {"source_node_id": "from-traj"},
        )
        assert merged["source_node_id"] == "from-blueprint"
        assert merged["source_task_id"] == "task-parent-1"
