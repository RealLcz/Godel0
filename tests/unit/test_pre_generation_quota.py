"""P0-4: ProposerRunner must consume generation_quotas *before* generation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "initial_agent" / "src"))

from proposer.request import ProposerRequest, CandidateArtifact
from proposer.runner import ProposerRunner
from proposer.schemas import BugGenerationPlan, FailureSignature
from proposer.trajectory_analyzer import TrajectoryView


def _sig(sig_id: str, traj_id: str, node_id: str = "", task_id: str = "") -> FailureSignature:
    return FailureSignature(
        signature_id=sig_id,
        source_trajectory_id=traj_id,
        source_solver_node_id=node_id,
        source_task_id=task_id,
        failure_stage="patch_generation",
        root_cause="test",
        target_capability="test",
    )


def _plan(plan_id: str, sig: FailureSignature) -> BugGenerationPlan:
    return BugGenerationPlan(
        plan_id=plan_id,
        source_trajectory_ids=[sig.source_trajectory_id] if sig.source_trajectory_id else [],
        failure_signature=sig,
        target_repo_id="repo",
        target_file="a.py",
        target_symbol="f",
        strategy="procedural",
        task_blueprint={},
    )


class TestRemainingGenerationQuotas:
    def test_remaining_is_nominal_minus_committed(self):
        from godel0.tasks.batch import TaskBatchBuilder

        rem = TaskBatchBuilder._remaining_generation_quotas(
            quotas={"parent_failure": 5, "current_child_level1": 5},
            source_counts={"parent_failure": 2, "current_child_level1": 1},
            remaining_tasks=7,
            parent_failure_trajectories=["p"],
            current_child_level1_trajectories=["c"],
        )
        assert rem == {"parent_failure": 3, "current_child_level1": 4}

    def test_donation_gives_slack_to_available_source(self):
        from godel0.tasks.batch import TaskBatchBuilder

        rem = TaskBatchBuilder._remaining_generation_quotas(
            quotas={"parent_failure": 5, "current_child_level1": 5},
            source_counts={"parent_failure": 5, "current_child_level1": 3},
            remaining_tasks=2,
            parent_failure_trajectories=["p"],
            current_child_level1_trajectories=[],  # child exhausted
        )
        # rem_c = 2, rem_p = 0, slack = 0 → no donation needed
        assert rem == {"parent_failure": 0, "current_child_level1": 2}

        rem2 = TaskBatchBuilder._remaining_generation_quotas(
            quotas={"parent_failure": 5, "current_child_level1": 5},
            source_counts={"parent_failure": 5, "current_child_level1": 5},
            remaining_tasks=2,
            parent_failure_trajectories=["p"],
            current_child_level1_trajectories=["c"],
        )
        # Both at quota but batch still needs 2 → donate to parent (both full).
        assert rem2["parent_failure"] == 2
        assert rem2["current_child_level1"] == 0


class TestCreateQuotaPlans:
    def test_create_plans_called_separately_with_quotas(self, tmp_path):
        runner = ProposerRunner(engine=None)
        parent_calls = []
        child_calls = []

        def fake_create_plans(signatures, repo_index, base_commit="", max_plans=10):
            plans = []
            for i, sig in enumerate(signatures):
                if len(plans) >= max_plans:
                    break
                plans.append(_plan(f"plan-{sig.signature_id}-{i}", sig))
            # Record which bucket by looking at first signature traj id.
            if signatures and signatures[0].source_trajectory_id.startswith("parent"):
                parent_calls.append(max_plans)
            else:
                child_calls.append(max_plans)
            return plans

        runner.planner = SimpleNamespace(create_plans=fake_create_plans)
        parent_sigs = [_sig(f"ps{i}", f"parent-{i}", "parent-node", f"pt{i}") for i in range(8)]
        child_sigs = [_sig(f"cs{i}", f"child-{i}", "child-node", f"ct{i}") for i in range(8)]
        parent_traces = [
            TrajectoryView(
                trajectory_id=f"parent-{i}",
                raw_path=str(tmp_path / f"parent-{i}.jsonl"),
                node_id="parent-node",
                task_id=f"pt{i}",
            )
            for i in range(8)
        ]
        child_traces = [
            TrajectoryView(
                trajectory_id=f"child-{i}",
                raw_path=str(tmp_path / f"child-{i}.jsonl"),
                node_id="child-node",
                task_id=f"ct{i}",
            )
            for i in range(8)
        ]
        repo_index = SimpleNamespace(base_commit="abc", repo_id="repo")
        request = ProposerRequest(
            node_id="child-node",
            run_id="r1",
            agent_code_dir="",
            repo_pool_dir="",
            task_store_dir="",
            output_dir=str(tmp_path),
            target_batch_size=10,
            generation_quotas={"parent_failure": 5, "current_child_level1": 5},
        )

        plans = runner._create_quota_plans(
            request=request,
            repo_index=repo_index,
            parent_sigs=parent_sigs,
            child_sigs=child_sigs,
            parent_traces=parent_traces,
            child_traces=child_traces,
            parent_quota=5,
            child_quota=5,
            feedbacks=[],
        )

        assert parent_calls == [5]
        assert child_calls == [5]
        assert len(plans) == 10
        parent_plans = [p for p in plans if p.task_blueprint.get("source_type") == "parent_failure"]
        child_plans = [
            p for p in plans if p.task_blueprint.get("source_type") == "current_child_level1"
        ]
        assert len(parent_plans) == 5
        assert len(child_plans) == 5
        for p in parent_plans:
            assert p.task_blueprint["source_node_id"] == "parent-node"
            assert p.task_blueprint["source_trajectory_id"]
            assert p.task_blueprint["source_task_id"]
        for p in child_plans:
            assert p.task_blueprint["source_node_id"] == "child-node"
            assert p.task_blueprint["source_type"] == "current_child_level1"

    def test_zero_child_quota_skips_child_create_plans(self, tmp_path):
        runner = ProposerRunner(engine=None)
        calls = []

        def fake_create_plans(signatures, repo_index, base_commit="", max_plans=10):
            calls.append((len(signatures), max_plans))
            return [_plan(f"p-{i}", signatures[i % len(signatures)]) for i in range(max_plans)]

        runner.planner = SimpleNamespace(create_plans=fake_create_plans)
        parent_sigs = [_sig("ps0", "parent-0", "pn", "pt0")]
        child_sigs = [_sig("cs0", "child-0", "cn", "ct0")]
        plans = runner._create_quota_plans(
            request=ProposerRequest(
                node_id="n",
                run_id="r",
                agent_code_dir="",
                repo_pool_dir="",
                task_store_dir="",
                output_dir=str(tmp_path),
            ),
            repo_index=SimpleNamespace(base_commit="abc"),
            parent_sigs=parent_sigs,
            child_sigs=child_sigs,
            parent_traces=[
                TrajectoryView(trajectory_id="parent-0", raw_path="/p.jsonl", node_id="pn", task_id="pt0")
            ],
            child_traces=[
                TrajectoryView(trajectory_id="child-0", raw_path="/c.jsonl", node_id="cn", task_id="ct0")
            ],
            parent_quota=10,
            child_quota=0,
            feedbacks=[],
        )
        assert calls == [(1, 10)]
        assert len(plans) == 10
        assert all(p.task_blueprint["source_type"] == "parent_failure" for p in plans)


class TestGenerateBatchConsumesQuotas:
    def test_generate_batch_uses_split_quotas_not_union_target(self, tmp_path):
        """Regression: must not create_plans(signatures_union, max_plans=10)."""
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"
        parent_path.write_text("{}\n", encoding="utf-8")
        child_path.write_text("{}\n", encoding="utf-8")

        runner = ProposerRunner(engine=None)
        create_calls = []

        parent_sig = _sig("ps", "parent-traj", "parent-node", "pt")
        child_sig = _sig("cs", "child-traj", "child-node", "ct")

        runner._load_trajectories_from_paths = lambda paths: [  # type: ignore[method-assign]
            TrajectoryView(
                trajectory_id="parent-traj" if "parent" in str(paths) else "child-traj",
                raw_path=str(paths[0]) if paths else "",
                node_id="parent-node" if "parent" in str(paths) else "child-node",
                task_id="pt" if "parent" in str(paths) else "ct",
            )
        ] if paths else []
        # More precise load:
        def load_paths(paths):
            out = []
            for p in paths:
                if "parent" in str(p):
                    out.append(
                        TrajectoryView(
                            trajectory_id="parent-traj",
                            raw_path=str(p),
                            node_id="parent-node",
                            task_id="pt",
                        )
                    )
                else:
                    out.append(
                        TrajectoryView(
                            trajectory_id="child-traj",
                            raw_path=str(p),
                            node_id="child-node",
                            task_id="ct",
                        )
                    )
            return out

        runner._load_trajectories_from_paths = load_paths  # type: ignore[method-assign]
        runner._load_outcomes_for_traces = lambda traces: []  # type: ignore[method-assign]
        runner.trajectory_analyzer = SimpleNamespace(
            extract_signatures=lambda traces, outcomes: (
                [parent_sig] if traces and traces[0].trajectory_id.startswith("parent") else [child_sig]
            )
        )
        runner._build_repo_index = lambda request: SimpleNamespace(  # type: ignore[method-assign]
            base_commit="abc", repo_id="repo"
        )
        runner.feedback_processor = SimpleNamespace(
            load_feedback=lambda d: [],
            partition=lambda cands, fbs: {"accepted": [], "rejected": []},
        )
        runner.planner = SimpleNamespace(
            configure_strategy_policy=lambda *a, **k: None,
            create_plans=lambda signatures, repo_index, base_commit="", max_plans=10: (
                create_calls.append({"n_sigs": len(signatures), "max_plans": max_plans})
                or [
                    _plan(f"plan-{signatures[0].signature_id}-{i}", signatures[0])
                    for i in range(max_plans)
                ]
            ),
        )
        generated_plan_ids = []

        def fake_generate(request, plans, repo_index):
            generated_plan_ids.extend([p.plan_id for p in plans])
            return []

        runner._generate_candidates = fake_generate  # type: ignore[method-assign]
        runner._write_candidates = lambda *a, **k: None  # type: ignore[method-assign]
        runner._stamp_repo_chain_constraints = lambda plan: None  # type: ignore[method-assign]

        request = ProposerRequest(
            node_id="child-node",
            run_id="r1",
            agent_code_dir="",
            repo_pool_dir="",
            task_store_dir="",
            output_dir=str(tmp_path / "out"),
            target_batch_size=10,
            max_candidates=50,
            parent_failure_trajectories=[str(parent_path)],
            current_child_level1_trajectories=[str(child_path)],
            solver_trajectories=[str(parent_path), str(child_path)],
            generation_quotas={"parent_failure": 5, "current_child_level1": 5},
            allow_human_curated_data=True,
        )
        (tmp_path / "out").mkdir(parents=True, exist_ok=True)

        result = runner.generate_batch(request)
        assert result.error is None or result.error == ""
        # Two separate create_plans calls with max_plans=5 each — NOT one with 10.
        assert create_calls == [
            {"n_sigs": 1, "max_plans": 5},
            {"n_sigs": 1, "max_plans": 5},
        ]
        assert len(generated_plan_ids) == 10
        assert len(result.plans) == 10
        sources = [p.get("task_blueprint", {}).get("source_type") for p in result.plans]
        assert sources.count("parent_failure") == 5
        assert sources.count("current_child_level1") == 5


class TestStampProvenanceFromBlueprint:
    def test_stamp_copies_source_fields(self):
        runner = ProposerRunner(engine=None)
        plan = BugGenerationPlan(
            plan_id="p1",
            source_trajectory_ids=["/traj.jsonl"],
            task_blueprint={
                "source_type": "parent_failure",
                "source_node_id": "parent-node",
                "source_task_id": "t1",
                "source_trajectory_id": "/traj.jsonl",
                "source_failure_stage": "solver",
            },
        )
        meta = runner._stamp_provenance({}, plan)
        assert meta["source_type"] == "parent_failure"
        assert meta["source_node_id"] == "parent-node"
        assert meta["source_task_id"] == "t1"
        assert meta["source_trajectory_id"] == "/traj.jsonl"
        assert "/traj.jsonl" in meta["source_trajectory_ids"]
