"""Regression tests for proposer/SWE-smith candidate contract mapping."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "initial_agent" / "src"))

from proposer.runner import ProposerRunner
from proposer.schemas import BugGenerationPlan
from proposer.code_locator import RepoSpec
from godel0.schemas.evaluation import CandidateValidationReport
from godel0.tasks.batch import TaskBatchBuilder


class RawSwesmithCandidate:
    def to_dict(self):
        return {
            "candidate_id": "cand",
            "plan_id": "plan",
            "strategy": "procedural",
            "operator": "change_operator",
            "target_file": "module.py",
            "target_symbol": "func",
            "bug_patch": "diff --git a/module.py b/module.py\n+bug",
        }


def test_runner_coerces_swesmith_candidate_fields():
    runner = ProposerRunner(engine=None)
    plan = BugGenerationPlan(
        plan_id="plan",
        target_repo_id="repo",
        target_base_commit="abc",
        target_file="fallback.py",
        target_symbol="fallback",
    )
    repo = RepoSpec(repo_id="repo", repo_dir="/repo", base_commit="abc")

    candidate = runner._coerce_candidate(RawSwesmithCandidate(), plan, repo)

    assert candidate.patch == "diff --git a/module.py b/module.py\n+bug"
    assert candidate.file_path == "module.py"
    assert candidate.symbol_name == "func"
    assert candidate.operator == "change_operator"


class RawRepoCandidate:
    def to_dict(self):
        return {
            "candidate_id": "repo-cand",
            "plan_id": "repo-plan",
            "strategy": "repo_agent",
            "operator": "repository_contract_mutation",
            "target_file": "consumer.py",
            "target_symbol": "",
            "modified_files": ["consumer.py", "provider.py"],
            "modified_entities": ["provide", "consume"],
            "bug_patch": (
                "diff --git a/consumer.py b/consumer.py\n"
                "diff --git a/provider.py b/provider.py\n"
            ),
        }


def test_runner_preserves_repository_candidate_scope():
    runner = ProposerRunner(engine=None)
    plan = BugGenerationPlan(
        plan_id="repo-plan",
        target_repo_id="repo",
        target_base_commit="abc",
        target_file="provider.py",
        target_symbol="provide",
        strategy="repo_agent",
    )
    repo = RepoSpec(repo_id="repo", repo_dir="/repo", base_commit="abc")

    candidate = runner._coerce_candidate(RawRepoCandidate(), plan, repo)

    assert candidate.file_path == "consumer.py"
    assert candidate.symbol_name == ""
    assert candidate.modified_files == ["consumer.py", "provider.py"]
    assert candidate.modified_entities == ["provide", "consume"]


def test_task_batch_normalization_preserves_repository_candidate_scope():
    candidate = TaskBatchBuilder()._normalize_candidate(
        RawRepoCandidate(),
        plans_by_id={
            "repo-plan": {
                "plan_id": "repo-plan",
                "strategy": "repo_agent",
                "target_file": "provider.py",
                "target_symbol": "provide",
            }
        },
        repo_specs=[{"repo_id": "repo", "base_commit": "abc"}],
    )

    assert candidate.file_path == "consumer.py"
    assert candidate.symbol_name == ""
    assert candidate.modified_files == ["consumer.py", "provider.py"]
    assert candidate.modified_entities == ["provide", "consume"]


class RawProposerResult:
    node_id = "node"
    accepted_candidates = []
    rejected_candidates = []
    completed = False

    def __init__(self):
        self.pending_candidates = [RawSwesmithCandidate()]
        self.plans = [
            {
                "plan_id": "plan",
                "target_repo_id": "repo",
                "target_base_commit": "abc",
                "target_file": "module.py",
                "target_symbol": "func",
                "strategy": "procedural",
                "operator": "change_operator",
            }
        ]


class RawRunner:
    def __init__(self):
        self.request = None

    def generate_batch(self, request):
        self.request = request
        return RawProposerResult()


class RecordingValidator:
    def __init__(self):
        self.calls = []

    def validate(self, **kwargs):
        self.calls.append(kwargs)
        return CandidateValidationReport(
            candidate_id=kwargs["candidate_id"],
            passed=False,
            rejection_reasons=["expected_rejection"],
        )


class SingleRepoPool:
    pool_dir = "/repo_pool"

    def all_repos(self):
        class Spec:
            repo_id = "repo"
            base_commit = "abc"
            path = "/repo"
            test_command = "pytest -q"
            install_command = "pip install -e ."
            timeout_sec = 120

        return [Spec()]


def test_task_batch_builder_normalizes_raw_swesmith_candidate():
    validator = RecordingValidator()
    runner = RawRunner()
    result = TaskBatchBuilder(batch_size=1, max_candidates=1).build_for_node(
        node_id="node",
        repo_pool=SingleRepoPool(),
        validator=validator,
        proposer_runner=runner,
    )

    assert result.candidates_validated == 1
    assert validator.calls[0]["candidate_patch"] == "diff --git a/module.py b/module.py\n+bug"
    assert validator.calls[0]["repo_id"] == "repo"
    assert validator.calls[0]["target_file"] == "module.py"
    assert validator.calls[0]["target_symbol"] == "func"
    assert runner.request.repo_specs[0].repo_id == "repo"


def test_task_batch_builder_preserves_multifile_candidate_metadata():
    builder = TaskBatchBuilder(batch_size=1, max_candidates=1)
    candidate = {
        "candidate_id": "multi",
        "plan_id": "plan",
        "repo_id": "repo",
        "strategy": "repo_agent",
        "patch": (
            "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-c\n+d\n"
        ),
        "modified_files": ["a.py", "b.py"],
        "modified_entities": ["producer", "consumer"],
    }

    normalized = builder._normalize_candidate(
        candidate,
        plans_by_id={"plan": {"target_repo_id": "repo"}},
        repo_specs=[{"repo_id": "repo"}],
    )

    assert normalized.modified_files == ["a.py", "b.py"]
    assert normalized.modified_entities == ["producer", "consumer"]


def test_repo_chain_does_not_pair_plan_symbol_with_wrong_multifile_target():
    normalized = TaskBatchBuilder()._normalize_candidate(
        {
            "candidate_id": "chain",
            "plan_id": "plan",
            "strategy": "repo_chain",
            "target_file": "consumer.py",
            "target_symbol": "",
            "modified_files": ["consumer.py", "producer.py"],
            "patch": "diff --git a/consumer.py b/consumer.py\n",
        },
        plans_by_id={
            "plan": {
                "target_repo_id": "repo",
                "target_file": "producer.py",
                "target_symbol": "producer_symbol",
            }
        },
        repo_specs=[{"repo_id": "repo"}],
    )

    assert normalized.file_path == "consumer.py"
    assert normalized.symbol_name == ""


def test_generated_validation_command_is_rebuilt_from_trusted_repo_config():
    builder = TaskBatchBuilder()
    setup_patch = (
        "diff --git a/test/units/test_contract.py b/test/units/test_contract.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/test/units/test_contract.py\n"
        "@@ -0,0 +1 @@\n"
        "+def test_contract(): assert True\n"
    )

    command = builder._trusted_test_command(
        {"test_command": "python -m pytest -q"},
        {
            "generated_test_command": "touch /tmp/should-never-run",
            "generated_test_files": ["test/units/test_contract.py"],
        },
        setup_patch,
    )

    assert command == "python -m pytest -q test/units/test_contract.py"
