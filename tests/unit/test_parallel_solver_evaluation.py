"""evaluation.max_workers must actually parallelise Level1/Level2 solving.

The knob was declared and validated but never read, so every task was solved
serially at ~25 minutes each and a 20-epoch run could not finish inside the
job wall clock.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from godel0.controller.orchestrator import EvolutionOrchestrator


class _CountingSolver:
    """Records concurrency and returns the call index as its outcome."""

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0

    def run_task(self, **kwargs):
        with self._lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            return SimpleNamespace(
                task_id=kwargs["task"],
                level=kwargs["level"],
                seed=kwargs.get("seed"),
            )
        finally:
            with self._lock:
                self.active -= 1


def _orchestrator(solver, max_workers: int) -> EvolutionOrchestrator:
    orchestrator = EvolutionOrchestrator.__new__(EvolutionOrchestrator)
    orchestrator.solver_runner = solver
    orchestrator.config = SimpleNamespace(
        evaluation=SimpleNamespace(max_workers=max_workers)
    )
    return orchestrator


def _calls(count: int) -> list[dict]:
    return [
        {"node": "n", "task": f"task_{i}", "level": 2, "seed": i}
        for i in range(count)
    ]


def test_no_calls_short_circuits():
    solver = _CountingSolver()
    assert _orchestrator(solver, 4)._run_solver_tasks([]) == []
    assert solver.calls == 0


def test_single_worker_stays_serial():
    solver = _CountingSolver()
    outcomes = _orchestrator(solver, 1)._run_solver_tasks(_calls(4))

    assert len(outcomes) == 4
    assert solver.peak == 1


def test_multiple_workers_run_concurrently():
    solver = _CountingSolver()
    outcomes = _orchestrator(solver, 4)._run_solver_tasks(_calls(8))

    assert len(outcomes) == 8
    assert solver.peak > 1
    assert solver.peak <= 4


def test_outcomes_keep_submission_order():
    solver = _CountingSolver()
    outcomes = _orchestrator(solver, 4)._run_solver_tasks(_calls(8))

    assert [o.task_id for o in outcomes] == [f"task_{i}" for i in range(8)]
    assert [o.seed for o in outcomes] == list(range(8))


def test_worker_count_never_exceeds_the_call_count():
    solver = _CountingSolver()
    _orchestrator(solver, 16)._run_solver_tasks(_calls(2))

    assert solver.peak <= 2
