"""failing_test_output must be captured from the bugged run and committed."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from godel0.proposer_trusted.candidate_validator import CandidateValidator
from godel0.proposer_trusted.task_committer import TaskCommitter
from godel0.schemas.evaluation import CandidateValidationReport
from godel0.tasks.batch import TaskBatchBuilder
from godel0.tasks.store import TaskStore


def test_format_failing_test_output_keeps_stdout_stderr_and_truncates():
    text = CandidateValidator._format_failing_test_output(
        {
            "stdout": "FAILED test_a.py::test_one - boom\n",
            "stderr": "E   AssertionError\n",
            "returncode": 1,
        }
    )
    assert "FAILED test_a.py::test_one" in text
    assert "stderr" in text
    assert "[returncode=1]" in text

    huge = "x" * 60000
    truncated = CandidateValidator._format_failing_test_output(
        {"stdout": huge, "stderr": "", "returncode": 1},
        max_chars=1000,
    )
    assert "truncated" in truncated
    assert len(truncated) < 2000


def test_commit_task_writes_failing_test_output(tmp_path: Path):
    store = TaskStore(tmp_path / "store")
    committer = TaskCommitter(store)
    task = committer.commit_task(
        batch_id="batch_x",
        proposer_node_id="root",
        repo_id="ansible",
        base_commit="abc",
        bug_strategy="repo_chain",
        bug_patch="diff --git a/a.py b/a.py\n+x\n",
        problem_statement="bug",
        f2p_tests=["t1"],
        baseline_test_command="pytest",
        failing_test_output="FAILED t1 - boom\n[returncode=1]\n",
    )
    path = store.store_dir / task.task_id / "failing_test_output.txt"
    assert path.read_text() == "FAILED t1 - boom\n[returncode=1]\n"


def test_batch_builder_passes_failing_test_output_from_report(tmp_path: Path):
    store = TaskStore(tmp_path / "store")
    committer = TaskCommitter(store)

    class FakeRepo:
        def all_repos(self):
            return [
                SimpleNamespace(
                    repo_id="toy",
                    base_commit="deadbeef",
                    path=str(tmp_path / "repo"),
                    test_command="pytest",
                    install_command="true",
                    timeout_sec=30,
                )
            ]

        @property
        def pool_dir(self):
            return tmp_path / "pool"

    class FakeValidator:
        def validate(self, **kwargs):
            return CandidateValidationReport(
                candidate_id=kwargs.get("candidate_id", "c1"),
                passed=True,
                patch_applied=True,
                source_only=True,
                clean_passed_tests=["t1", "t2"],
                bugged_failed_tests=["t1"],
                bugged_passed_tests=["t2"],
                f2p_tests=["t1"],
                p2p_tests=["t2"],
                reverse_restored=True,
                syntax_valid=True,
                import_valid=True,
                timeout_valid=True,
                safety_valid=True,
                duplicate_valid=True,
                relevance_valid=True,
                failing_test_output="FAILED t1 - expected X\n[returncode=1]\n",
            )

    class FakeProposer:
        def generate_batch(self, request):
            return SimpleNamespace(
                completed=True,
                error="",
                accepted_candidates=[],
                rejected_candidates=[],
                pending_candidates=[
                    SimpleNamespace(
                        candidate_id="cand1",
                        plan_id="plan1",
                        repo_id="toy",
                        patch="diff --git a/a.py b/a.py\n+++ b/a.py\n@@\n+broken\n",
                        issue_draft="something broke",
                        strategy="repo_chain",
                        operator="x",
                        file_path="a.py",
                        symbol_name="",
                        modified_files=["a.py"],
                        modified_entities=["foo"],
                        generation_metadata={},
                    )
                ],
                plans=[{"plan_id": "plan1", "task_blueprint": {"source_type": "bootstrap"}}],
            )

    (tmp_path / "repo").mkdir()
    (tmp_path / "pool").mkdir()
    builder = TaskBatchBuilder(batch_size=1, max_candidates=1)
    result = builder.build_for_node(
        node_id="root",
        repo_pool=FakeRepo(),
        validator=FakeValidator(),
        task_committer=committer,
        proposer_runner=FakeProposer(),
        output_dir=tmp_path / "proposer",
        task_store_dir=str(store.store_dir),
        bootstrap=True,
    )
    assert result.complete
    assert len(result.tasks) == 1
    out = store.store_dir / result.tasks[0].task_id / "failing_test_output.txt"
    assert "FAILED t1" in out.read_text()
    assert "[returncode=1]" in out.read_text()
