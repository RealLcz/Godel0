"""Integration test: tool sharing across nodes."""

from __future__ import annotations

import pytest
import subprocess
from pathlib import Path

from godel0.git.repository import init_repo, commit, get_head_sha
from godel0.git.worktree import NodeWorktree, commit_child
from godel0.git.node_refs import create_node_ref, get_node_sha, node_exists


class TestToolSharing:
    def test_child_can_add_tool(self, tmp_path):
        """Child self-edit can add a new tool that the parent doesn't have."""
        agent_repo = tmp_path / "agent_repo"
        agent_repo.mkdir()

        init_repo(agent_repo)
        (agent_repo / "coding_agent.py").write_text("# solver")
        (agent_repo / "tools").mkdir()
        (agent_repo / "tools" / "__init__.py").write_text("")
        (agent_repo / "tools" / "bash.py").write_text("# bash tool")
        parent_sha = commit(agent_repo, "root agent")

        create_node_ref(agent_repo, "root", parent_sha)

        with NodeWorktree(agent_repo, tmp_path / "scratch", "child_1", parent_sha) as worktree:
            new_tool = worktree / "tools" / "smart_view.py"
            new_tool.write_text("# smart view tool")

            child_sha = commit_child(agent_repo, worktree, "child_1", "child with new tool")

        assert node_exists(agent_repo, "child_1")

        with NodeWorktree(agent_repo, tmp_path / "scratch2", "verify_child", child_sha) as worktree:
            assert (worktree / "tools" / "smart_view.py").exists()

        with NodeWorktree(agent_repo, tmp_path / "scratch3", "verify_parent", parent_sha) as worktree:
            assert not (worktree / "tools" / "smart_view.py").exists()


class TestSecretIsolation:
    def test_solver_cannot_access_private(self, tmp_path):
        """Solver workspace should not have access to proposer private inputs."""
        from godel0.tasks.store import TaskStore, TaskArtifacts
        from godel0.schemas.task import TaskRecord

        store = TaskStore(tmp_path / "task_store")
        record = TaskRecord(
            task_id="secret_task",
            batch_id="batch",
            proposer_node_id="node",
            repo_id="repo",
            base_commit="abc",
            bug_strategy="procedural",
            bug_patch_path="",
            problem_statement_path="",
            baseline_test_command="pytest",
        )
        artifacts = TaskArtifacts(
            problem_statement="public problem",
            bug_patch="public patch",
            f2p_tests=["secret_test_1", "secret_test_2"],
            generation_context={"secret": "data"},
        )
        store.put(record, artifacts)

        public_dir = tmp_path / "public"
        store.materialize_public("secret_task", public_dir)
        assert not (public_dir / "f2p_tests.json").exists()
        assert not (public_dir / "generation_context.json").exists()

        private_dir = tmp_path / "private"
        store.materialize_private("secret_task", private_dir)
        assert (private_dir / "f2p_tests.json").exists()
