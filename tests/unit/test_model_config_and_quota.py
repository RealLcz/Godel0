"""Unit tests for model config explicitness (BUG-24) and 5+5 quota (BUG-08/09)."""
from __future__ import annotations

import pytest

from godel0.config import ModelConfig, _build_subconfig


class TestModelConfigExplicit:
    def test_default_has_all_four_roles(self):
        # BUG-24: every chat() call must use an explicit model. Even when all
        # roles share the same model, each is recorded explicitly.
        cfg = ModelConfig()
        assert cfg.solver_model
        assert cfg.proposer_model
        assert cfg.diagnose_model
        assert cfg.self_improve_model

    def test_agent_model_alias_resolves_to_solver_model(self):
        cfg = ModelConfig(solver_model="gpt-4")
        # Backward-compatible alias.
        assert cfg.agent_model == "gpt-4"

    def test_legacy_agent_model_maps_to_solver_model(self):
        # BUG-24: legacy configs that set only agent_model should map onto
        # solver_model and propagate to the other roles.
        cfg = _build_subconfig("models", {"agent_model": "gpt-4"})
        assert cfg.solver_model == "gpt-4"
        assert cfg.proposer_model == "gpt-4"
        assert cfg.self_improve_model == "gpt-4"

    def test_explicit_roles_are_respected(self):
        cfg = _build_subconfig("models", {
            "solver_model": "gpt-4",
            "proposer_model": "llama",
            "diagnose_model": "claude",
            "self_improve_model": "mistral",
        })
        assert cfg.solver_model == "gpt-4"
        assert cfg.proposer_model == "llama"
        assert cfg.diagnose_model == "claude"
        assert cfg.self_improve_model == "mistral"

    def test_explicit_roles_not_overwritten_by_solver(self):
        cfg = _build_subconfig("models", {
            "solver_model": "gpt-4",
            "proposer_model": "llama",
        })
        # proposer_model should remain "llama", not be overwritten by solver.
        assert cfg.proposer_model == "llama"
        # But unset roles (diagnose, self_improve) default to solver.
        assert cfg.diagnose_model == "gpt-4"
        assert cfg.self_improve_model == "gpt-4"


class TestQuotaEnforcement:
    def test_accepts_source_returns_true_when_below_quota(self):
        from godel0.tasks.batch import TaskBatchBuilder

        builder = TaskBatchBuilder(
            batch_size=10,
            source_quotas={"parent_failure": 5, "current_child_level1": 5},
        )
        counts = {"parent_failure": 0, "current_child_level1": 0, "bootstrap": 0}
        assert builder._accepts_source(
            "parent_failure", counts, builder.source_quotas, 10
        ) is True

    def test_accepts_source_returns_false_when_quota_full_and_other_not_full(self):
        from godel0.tasks.batch import TaskBatchBuilder

        builder = TaskBatchBuilder(
            batch_size=10,
            source_quotas={"parent_failure": 5, "current_child_level1": 5},
        )
        counts = {"parent_failure": 5, "current_child_level1": 0, "bootstrap": 0}
        # parent_failure is full (5), current_child_level1 is not (0), so
        # fallback is NOT allowed yet.
        assert builder._accepts_source(
            "parent_failure", counts, builder.source_quotas, 10
        ) is False

    def test_accepts_source_returns_true_for_fallback_when_other_full(self):
        from godel0.tasks.batch import TaskBatchBuilder

        builder = TaskBatchBuilder(
            batch_size=12,
            source_quotas={"parent_failure": 5, "current_child_level1": 5},
        )
        # Both sources are at their nominal quota but batch is not full (10<12).
        counts = {"parent_failure": 5, "current_child_level1": 5, "bootstrap": 0}
        # parent_failure can be accepted as fallback since other is full and
        # the batch still has room (10 < 12).
        assert builder._accepts_source(
            "parent_failure", counts, builder.source_quotas, 12
        ) is True

    def test_accepts_source_returns_false_when_batch_complete(self):
        from godel0.tasks.batch import TaskBatchBuilder

        builder = TaskBatchBuilder(
            batch_size=10,
            source_quotas={"parent_failure": 5, "current_child_level1": 5},
        )
        counts = {"parent_failure": 5, "current_child_level1": 5, "bootstrap": 0}
        # batch_size is 10 and total is 10 -> full, no fallback accepted.
        assert builder._accepts_source(
            "parent_failure", counts, builder.source_quotas, 10
        ) is False

    def test_bootstrap_always_accepted(self):
        from godel0.tasks.batch import TaskBatchBuilder

        builder = TaskBatchBuilder(batch_size=10)
        counts = {"parent_failure": 10, "current_child_level1": 10, "bootstrap": 0}
        assert builder._accepts_source(
            "bootstrap", counts, builder.source_quotas, 10
        ) is True


class TestClassifySourceV2:
    def test_returns_bootstrap_when_no_trajectories(self):
        from godel0.tasks.batch import TaskBatchBuilder
        from types import SimpleNamespace

        builder = TaskBatchBuilder()
        cand = SimpleNamespace(
            generation_metadata={"source_trajectory_ids": []},
        )
        source_type, traj = builder._classify_source_v2(
            cand, [], [], [], bootstrap=True
        )
        assert source_type == "bootstrap"
        assert traj == ""

    def test_classifies_parent_failure(self):
        from godel0.tasks.batch import TaskBatchBuilder
        from types import SimpleNamespace

        builder = TaskBatchBuilder()
        parent_traj = "/scratch/run1/solver/parent_node/trajectory.jsonl"
        cand = SimpleNamespace(
            generation_metadata={"source_trajectory_ids": [parent_traj]},
        )
        source_type, traj = builder._classify_source_v2(
            cand,
            parent_failure_trajectories=[parent_traj],
            current_child_level1_trajectories=[],
            parent_task_ids=[],
        )
        assert source_type == "parent_failure"
        assert traj == parent_traj

    def test_classifies_current_child_level1(self):
        from godel0.tasks.batch import TaskBatchBuilder
        from types import SimpleNamespace

        builder = TaskBatchBuilder()
        child_traj = "/scratch/run1/solver/child_node/trajectory.jsonl"
        cand = SimpleNamespace(
            generation_metadata={"source_trajectory_ids": [child_traj]},
        )
        source_type, traj = builder._classify_source_v2(
            cand,
            parent_failure_trajectories=[],
            current_child_level1_trajectories=[child_traj],
            parent_task_ids=[],
        )
        assert source_type == "current_child_level1"
        assert traj == child_traj
