"""Invariants for the joint Solver-Proposer evolution mainline."""

from __future__ import annotations

from types import SimpleNamespace

from godel0.config import load_config
from godel0.controller.budget import Budget
from godel0.schemas.evaluation import CandidateValidationReport
from godel0.tasks.batch import TaskBatchBuilder
from initial_agent.src.proposer.planner import ProposerPlanner
from initial_agent.src.proposer.request import CandidateArtifact, ProposerResult


def test_formal_run_targets_ten_valid_tasks_per_node():
    config = load_config("configs/evolve20_ansible_formal.yaml")
    assert config.run.max_nodes == 20
    assert config.run.max_expansions > config.run.max_nodes
    assert config.tasks.batch_size == 10
    assert config.tasks.max_generation_candidates >= 50


def test_failed_attempts_do_not_count_as_successful_epochs():
    budget = Budget(max_nodes=20, max_expansions=200)
    for _ in range(20):
        budget.record_expansion()
    assert budget.nodes_created == 0
    assert not budget.exhausted()
    assert budget.remaining() == 20

    for _ in range(20):
        budget.record_node()
    assert budget.exhausted()
    assert budget.remaining() == 0


class _RepoPool:
    pool_dir = "/repo_pool"

    def all_repos(self):
        return [
            SimpleNamespace(
                repo_id="repo",
                base_commit="abc",
                path="/repo",
                test_command="pytest -q",
                install_command="pip install -e .",
                timeout_sec=120,
            )
        ]


class _RetryingRunner:
    def __init__(self):
        self.requests = []

    def generate_batch(self, request):
        self.requests.append(request)
        index = len(self.requests)
        candidate = CandidateArtifact(
            candidate_id=f"cand-{index}",
            plan_id=f"plan-{index}",
            repo_id="repo",
            base_commit="abc",
            file_path="module.py",
            symbol_name="f",
            strategy="procedural",
            patch=(
                "diff --git a/module.py b/module.py\n"
                "--- a/module.py\n+++ b/module.py\n"
                "@@ -1 +1 @@\n-return 1\n+return 2\n"
            ),
            modified_files=["module.py"],
            modified_entities=["f"],
        )
        result = ProposerResult(run_id=request.run_id, node_id=request.node_id)
        result.pending_candidates.append(candidate)
        result.plans.append(
            {
                "plan_id": candidate.plan_id,
                "target_repo_id": "repo",
                "target_base_commit": "abc",
                "target_file": "module.py",
                "target_symbol": "f",
            }
        )
        return result


class _RejectOnceValidator:
    def __init__(self):
        self.calls = 0

    def validate(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return CandidateValidationReport(
                candidate_id=kwargs["candidate_id"],
                passed=False,
                rejection_reasons=["no_f2p"],
            )
        return CandidateValidationReport(
            candidate_id=kwargs["candidate_id"],
            passed=True,
            patch_applied=True,
            source_only=True,
            syntax_valid=True,
            import_valid=True,
            timeout_valid=True,
            reverse_restored=True,
            safety_valid=True,
            duplicate_valid=True,
            relevance_valid=True,
            f2p_tests=["test_f"],
        )


class _Committer:
    def __init__(self):
        self.calls = []

    def commit_task(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(task_id=f"task-{kwargs['validation_report']['candidate_id']}")


def test_batch_retries_rejections_until_k_valid_tasks(tmp_path):
    runner = _RetryingRunner()
    validator = _RejectOnceValidator()
    committer = _Committer()
    result = TaskBatchBuilder(batch_size=2, max_candidates=5).build_for_node(
        node_id="node",
        repo_pool=_RepoPool(),
        validator=validator,
        task_committer=committer,
        proposer_runner=runner,
        output_dir=tmp_path / "proposer",
    )

    assert result.complete
    assert len(result.tasks) == 2
    assert result.candidates_generated == 3
    assert result.rejection_reasons == {"no_f2p": 1}
    assert [request.generation_attempt for request in runner.requests] == [0, 1, 2]
    assert [request.target_batch_size for request in runner.requests] == [2, 2, 1]
    assert all(request.feedback_dir for request in runner.requests)
    feedback_files = list((tmp_path / "proposer" / "trusted_feedback").glob("*.json"))
    assert len(feedback_files) == 3
    assert {call["solver_test_command"] for call in committer.calls} == {"pytest -q"}


def test_proposer_result_round_trip_preserves_pending_candidates():
    original = ProposerResult(run_id="run", node_id="node")
    original.pending_candidates.append(
        CandidateArtifact(
            candidate_id="cand",
            plan_id="plan",
            repo_id="repo",
            base_commit="abc",
            file_path="a.py",
            symbol_name="f",
            strategy="procedural",
        )
    )
    restored = ProposerResult.from_dict(original.to_dict())
    assert restored.pending_candidates[0].candidate_id == "cand"


def test_configured_proposer_strategy_policy_is_honored():
    planner = ProposerPlanner()
    planner.configure_strategy_policy({"procedural": 1.0})

    strategy = planner._choose_strategy(SimpleNamespace(behavior_pattern={}))

    assert strategy == "procedural"


def test_engine_zero_yield_plan_uses_remaining_generation_budget(tmp_path):
    class ZeroThenCandidate(_RetryingRunner):
        def generate_batch(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                result = ProposerResult(run_id=request.run_id, node_id=request.node_id)
                result.plans.append({"plan_id": "failed-plan"})
                return result
            self.requests.pop()
            return super().generate_batch(request)

    class AlwaysAccept(_RejectOnceValidator):
        def validate(self, **kwargs):
            self.calls += 1
            return CandidateValidationReport(
                candidate_id=kwargs["candidate_id"],
                passed=True,
                patch_applied=True,
                source_only=True,
                syntax_valid=True,
                import_valid=True,
                timeout_valid=True,
                reverse_restored=True,
                safety_valid=True,
                duplicate_valid=True,
                relevance_valid=True,
                f2p_tests=["test_f"],
            )

    runner = ZeroThenCandidate()
    result = TaskBatchBuilder(batch_size=1, max_candidates=2).build_for_node(
        node_id="node",
        repo_pool=_RepoPool(),
        validator=AlwaysAccept(),
        task_committer=_Committer(),
        proposer_runner=runner,
        output_dir=tmp_path / "proposer",
    )

    assert result.complete
    assert result.candidates_generated == 2
    assert len(runner.requests) == 2
