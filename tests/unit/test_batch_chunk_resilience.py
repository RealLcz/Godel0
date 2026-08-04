"""One crashed/timed-out proposer chunk must not abort the whole batch.

Job 211728: attempt_005 hit the NodeProposerRunner timeout, generate_batch
raised RuntimeError, and root bootstrap died even though earlier chunks had
produced candidates. The builder now consumes that attempt's budget and
continues with the next chunk.
"""

from __future__ import annotations

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


class _CrashThenEmptyRunner:
    """First chunk raises (timeout); later chunks return empty completed results."""

    def __init__(self):
        self.calls = 0

    def generate_batch(self, request):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Node proposer produced no result: exit=-15; stderr=")
        return SimpleNamespace(
            accepted_candidates=[],
            rejected_candidates=[],
            pending_candidates=[],
            plans=[],
            completed=True,
            error="",
        )


def test_proposer_chunk_crash_consumes_budget_and_continues(tmp_path: Path):
    builder = TaskBatchBuilder(batch_size=2, max_candidates=4)
    runner = _CrashThenEmptyRunner()

    result = builder.build_for_node(
        node_id="root",
        repo_pool=_FakeRepoPool(tmp_path),
        validator=SimpleNamespace(),  # non-None so the loop keeps validating
        task_committer=None,
        proposer_runner=runner,
        output_dir=tmp_path / "proposer",
        bootstrap=True,
    )

    # The crash was recorded, budget was consumed, and the builder moved on
    # to the next chunk instead of raising.
    assert "RuntimeError" in (result.proposer_error or "")
    assert runner.calls >= 2
    assert result.plans_attempted >= 2
    assert result.candidates_generated == 0
    assert result.candidates_emitted == 0
    assert result.complete is False
