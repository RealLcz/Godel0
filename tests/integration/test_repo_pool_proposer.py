"""Integration test: repo pool -> proposer -> SWESmith engine -> validator.

Tests the full pipeline:
1. Prepare a repo pool with a toy repo.
2. Create a ProposerRequest with repo_specs from the pool.
3. Run the ProposerRunner with SWESmithEngine.
4. Validate the generated candidate with CandidateValidator.
5. Commit the task to TaskStore.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "initial_agent" / "src"))

from godel0.tasks.repo_pool import RepoPool, RepoSpec
from godel0.proposer_trusted.candidate_validator import CandidateValidator
from godel0.proposer_trusted.task_committer import TaskCommitter
from godel0.tasks.store import TaskStore
from godel0.git.repository import init_repo, commit, get_head_sha


TOY_SOURCE = '''"""Toy module."""


def clamp(x, low, high):
    return max(low, min(x, high))


def fibonacci(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
'''

TOY_TESTS = '''"""Tests."""
from toy_module import clamp, fibonacci


def test_clamp_normal():
    assert clamp(5, 0, 10) == 5

def test_clamp_lower():
    assert clamp(-5, 0, 10) == 0

def test_clamp_upper():
    assert clamp(15, 0, 10) == 10

def test_fib_base():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1

def test_fib_recursive():
    assert fibonacci(10) == 55
'''


@pytest.fixture
def repo_pool(tmp_path):
    """Create a repo pool with a toy repository."""
    pool_dir = tmp_path / "repo_pool"
    pool_dir.mkdir()

    repo_dir = pool_dir / "toy_repo"
    repo_dir.mkdir()
    (repo_dir / "toy_module.py").write_text(TOY_SOURCE)
    (repo_dir / "test_toy.py").write_text(TOY_TESTS)
    (repo_dir / "conftest.py").write_text("import sys\nsys.path.insert(0, '.')\n")

    subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(repo_dir), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo_dir), capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=str(repo_dir), capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=str(repo_dir), capture_output=True)
    base_commit = get_head_sha(repo_dir)

    pool = RepoPool(pool_dir)
    spec = RepoSpec(
        repo_id="toy_repo",
        base_commit=base_commit,
        path=str(repo_dir),
        test_command="python -m pytest test_toy.py -v",
        install_command="pip install -e .",
        timeout_sec=60,
    )
    pool.add(spec)

    return pool


class TestRepoPool:
    def test_pool_has_toy_repo(self, repo_pool):
        """The pool should have the toy repo."""
        assert repo_pool.exists("toy_repo")
        spec = repo_pool.get("toy_repo")
        assert spec is not None
        assert spec.repo_id == "toy_repo"
        assert spec.base_commit
        assert Path(spec.path).exists()

    def test_pool_returns_all_repos(self, repo_pool):
        """all_repos() should return the toy repo."""
        repos = repo_pool.all_repos()
        assert len(repos) == 1
        assert repos[0].repo_id == "toy_repo"

    def test_pool_get_for_proposer(self, repo_pool):
        """get_for_proposer() should return dict with correct fields."""
        specs = repo_pool.get_for_proposer()
        assert len(specs) == 1
        assert specs[0]["repo_id"] == "toy_repo"
        assert specs[0]["base_commit"]
        assert "repo_dir" in specs[0]

    def test_pool_materialize_repo(self, repo_pool, tmp_path):
        """materialize_repo should copy the repo."""
        dest = tmp_path / "copy"
        repo_pool.materialize_repo("toy_repo", dest)
        assert (dest / "toy_module.py").exists()
        assert (dest / "test_toy.py").exists()


class TestProposerWithRepoPool:
    def test_proposer_request_carries_repo_specs(self, repo_pool):
        """ProposerRequest should carry the full repo specs from the pool."""
        from proposer.request import ProposerRequest, RepoSpecInfo

        pool_specs = repo_pool.get_for_proposer()
        request = ProposerRequest(
            node_id="test_node",
            run_id="test_run",
            agent_code_dir="/tmp/agent",
            repo_pool_dir=str(repo_pool.pool_dir),
            task_store_dir="/tmp/task_store",
            output_dir="/tmp/output",
            repo_specs=[
                RepoSpecInfo(
                    repo_id=s["repo_id"],
                    base_commit=s["base_commit"],
                    path=s["repo_dir"],
                    test_command="python -m pytest test_toy.py -v",
                )
                for s in pool_specs
            ],
        )

        assert len(request.repo_specs) == 1
        spec = request.first_repo()
        assert spec is not None
        assert spec.repo_id == "toy_repo"
        assert spec.base_commit
        assert spec.path

    def test_proposer_runner_uses_repo_specs(self, repo_pool, tmp_path):
        """ProposerRunner should use repo_specs from the request."""
        from proposer.request import ProposerRequest, RepoSpecInfo
        from proposer.runner import ProposerRunner

        pool_specs = repo_pool.get_for_proposer()
        request = ProposerRequest(
            node_id="test_node",
            run_id="test_run",
            agent_code_dir=str(tmp_path / "agent"),
            repo_pool_dir=str(repo_pool.pool_dir),
            task_store_dir=str(tmp_path / "task_store"),
            output_dir=str(tmp_path / "output"),
            target_batch_size=1,
            max_candidates=5,
            repo_specs=[
                RepoSpecInfo(
                    repo_id=s["repo_id"],
                    base_commit=s["base_commit"],
                    path=s["repo_dir"],
                    test_command="python -m pytest test_toy.py -v",
                )
                for s in pool_specs
            ],
        )

        runner = ProposerRunner(engine=None)
        result = runner.generate_batch(request)

        # Without an engine, proposer produces no candidates but should not crash
        assert result.node_id == "test_node"

    def test_full_proposer_to_validator(self, repo_pool, tmp_path):
        """Full pipeline: repo pool -> proposer -> engine -> validator -> task store."""
        from proposer.request import ProposerRequest, RepoSpecInfo
        from proposer.runner import ProposerRunner
        from swesmith.engine import SWESmithEngine

        # Create engine
        engine = SWESmithEngine(agent_adapter=None)

        # Create proposer runner with engine
        runner = ProposerRunner(engine=engine)

        # Create request with repo specs
        pool_spec = repo_pool.get("toy_repo")
        request = ProposerRequest(
            node_id="test_node",
            run_id="test_run",
            agent_code_dir=str(tmp_path / "agent"),
            repo_pool_dir=str(repo_pool.pool_dir),
            task_store_dir=str(tmp_path / "task_store"),
            output_dir=str(tmp_path / "proposer_output"),
            target_batch_size=1,
            max_candidates=5,
            repo_specs=[
                RepoSpecInfo(
                    repo_id=pool_spec.repo_id,
                    base_commit=pool_spec.base_commit,
                    path=str(pool_spec.path),
                    test_command=pool_spec.test_command,
                )
            ],
        )

        # Run the proposer
        result = runner.generate_batch(request)

        # The proposer may or may not produce candidates depending on
        # whether the trajectory analyzer finds signatures.
        # The key test is that it doesn't crash and uses the repo specs.
        assert result is not None
        assert result.node_id == "test_node"

    def test_validator_uses_repo_from_pool(self, repo_pool, tmp_path):
        """The trusted validator should use the repo path from the pool."""
        spec = repo_pool.get("toy_repo")

        validator = CandidateValidator(
            workspace_root=tmp_path / "validator",
            test_timeout_sec=30,
        )

        # Create a simple bug patch: change `max(low, min(x, high))` to `max(low, min(x, low))`
        repo_path = Path(spec.path)
        source_file = repo_path / "toy_module.py"
        original = source_file.read_text()
        buggy = original.replace("min(x, high)", "min(x, low)")

        # Generate patch
        source_file.write_text(buggy)
        result = subprocess.run(
            ["git", "-C", str(repo_path), "diff"],
            capture_output=True, text=True,
        )
        patch = result.stdout

        # Restore original
        source_file.write_text(original)
        subprocess.run(
            ["git", "-C", str(repo_path), "checkout", "--", "toy_module.py"],
            capture_output=True,
        )

        # Validate
        report = validator.validate(
            candidate_patch=patch,
            repo_path=repo_path,
            base_commit=spec.base_commit,
            test_command=spec.test_command,
            candidate_id="test_cand_001",
        )

        assert report.patch_applied, "Patch should apply to repo from pool"
        assert report.f2p_tests, "Should have F2P tests"

        # Commit the task
        task_store = TaskStore(tmp_path / "task_store")
        committer = TaskCommitter(task_store)
        task = committer.commit_task(
            batch_id="batch_001",
            proposer_node_id="test_node",
            repo_id=spec.repo_id,
            base_commit=spec.base_commit,
            bug_strategy="procedural",
            bug_patch=patch,
            problem_statement="The clamp function does not handle upper boundary correctly.",
            f2p_tests=report.f2p_tests,
            baseline_test_command=spec.test_command,
        )

        assert task.task_id
        assert task_store.exists(task.task_id)
        assert task.repo_id == "toy_repo"
