"""Zero-yield bootstrap chunks must say why they produced nothing.

Job 217937 failed 20 of 29 root chunks and the controller recorded none of it:
``trusted_feedback/`` held zero ``engine-*`` entries and ``repo_chain_stats``
stayed empty, because only ``_generate_candidates`` stamped ``last_rejection``
and bootstrap does not route through it. The reasons had to be reconstructed by
hand from per-attempt rejection text files.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from godel0.tasks.batch import TaskBatchBuilder


class _FakeRepoPool:
    def __init__(self, root: Path):
        self.pool_dir = root
        self._spec = SimpleNamespace(
            repo_id="ansible",
            base_commit="HEAD",
            path=str(root),
            test_command="pytest -q",
            install_command="pip install -e .",
            timeout_sec=120,
        )

    def all_repos(self):
        return [self._spec]


class _RejectingRunner:
    """A chunk that yields nothing and reports the engine's reason."""

    def __init__(
        self,
        reason: str,
        stage: str = "",
        *,
        target_file: str = "",
        target_symbol: str = "",
    ):
        self.reason = reason
        self.stage = stage
        self.target_file = target_file
        self.target_symbol = target_symbol

    def generate_batch(self, request):
        blueprint = {"last_rejection": self.reason}
        if self.stage:
            blueprint["last_rejection_stage"] = self.stage
        return SimpleNamespace(
            accepted_candidates=[],
            rejected_candidates=[],
            pending_candidates=[],
            plans=[
                {
                    "plan_id": "bootstrap-cross_file-0",
                    "target_repo_id": "ansible",
                    "target_file": self.target_file,
                    "target_symbol": self.target_symbol,
                    "task_blueprint": blueprint,
                }
            ],
            completed=True,
            error="",
        )


def _run(tmp_path: Path, runner):
    builder = TaskBatchBuilder(
        batch_size=1,
        max_candidates=1,
        workflow_config={"plans_per_call": 1},
    )
    return builder.build_for_node(
        node_id="root",
        repo_pool=_FakeRepoPool(tmp_path),
        validator=SimpleNamespace(),
        task_committer=None,
        proposer_runner=runner,
        output_dir=tmp_path / "proposer",
        bootstrap=True,
    )


def test_engine_rejection_reaches_the_controller(tmp_path: Path):
    result = _run(
        tmp_path,
        _RejectingRunner(
            "invalid_chain_plan:mutation sites must be unique file/symbol pairs",
            stage="contract_generation_failure",
        ),
    )

    assert len(result.engine_rejections) == 1
    record = result.engine_rejections[0]
    assert "unique file/symbol pairs" in record["reason"]
    assert record["stage"] == "contract_generation_failure"
    assert result.repo_chain_stats["contract_generation_failure_count"] == 1


def test_engine_rejection_is_written_to_trusted_feedback(tmp_path: Path):
    _run(
        tmp_path,
        _RejectingRunner("mutation_patch_apply_failed:edits count must be between 3 and 8"),
    )

    feedback = sorted((tmp_path / "proposer" / "trusted_feedback").glob("engine-*.json"))
    assert feedback, "no engine rejection was persisted"
    payload = json.loads(feedback[0].read_text())
    assert payload["accepted"] is False
    assert "edits count" in payload["reason"]


def test_engine_feedback_carries_scoped_repair_identity(tmp_path: Path):
    _run(
        tmp_path,
        _RejectingRunner(
            "invalid_chain_plan:unknown symbol",
            target_file="lib/ansible/cli/adhoc.py",
            target_symbol="AdHocCLI.name",
        ),
    )

    feedback = sorted((tmp_path / "proposer" / "trusted_feedback").glob("engine-*.json"))
    payload = json.loads(feedback[0].read_text())
    assert payload["notes"] == {
        "attempt": 0,
        "plan_id": "bootstrap-cross_file-0",
        "reason": "invalid_chain_plan:unknown symbol",
        "stage": "contract_generation_failure",
        "reason_code": "invalid_chain_plan",
        "repo": "ansible",
        "base_commit": "HEAD",
        "file": "lib/ansible/cli/adhoc.py",
        "symbol": "AdHocCLI.name",
    }


def test_stage_is_inferred_when_the_engine_omits_it(tmp_path: Path):
    result = _run(tmp_path, _RejectingRunner("clean_contract:tests failed on the unmodified repository"))

    assert result.repo_chain_stats["clean_contract_failure_count"] == 1


class _StubBacking:
    def __init__(self, reason: str, stage: str = ""):
        self.last_rejection = reason
        self.last_rejection_stage = stage


def _stamp(reason: str, stage: str = ""):
    from proposer.workflows.repo_chain.workflow import RepoChainWorkflow

    plan = SimpleNamespace(task_blueprint={})
    RepoChainWorkflow._stamp_plan_rejection(plan, _StubBacking(reason, stage))
    return plan.task_blueprint


def test_bootstrap_stamps_the_backing_generator_reason():
    blueprint = _stamp(
        "invalid_chain_plan:missing chain_plan or tests",
        "contract_generation_failure",
    )

    assert blueprint["last_rejection"] == "invalid_chain_plan:missing chain_plan or tests"
    assert blueprint["last_rejection_stage"] == "contract_generation_failure"


def test_bootstrap_classifies_a_stage_when_the_generator_left_it_blank():
    blueprint = _stamp("invalid_chain_plan:mutation sites must be unique file/symbol pairs")

    assert blueprint["last_rejection_stage"] == "contract_generation_failure"


def test_bootstrap_always_records_something():
    blueprint = _stamp("")

    assert blueprint["last_rejection"] == "engine_returned_no_candidates"
    assert blueprint["last_rejection_stage"]
