"""Solver evaluation runner: runs the solver on a task and evaluates the result."""

from __future__ import annotations

import subprocess
import time
import shutil
from pathlib import Path
from typing import Literal, Optional

from ..errors import EvaluationError
from ..schemas.evaluation import EvaluationOutcome
from ..schemas.node import NodeRecord
from ..schemas.task import TaskRecord
from ..git.repository import diff_vs_commit, apply_patch, reset_to_commit, run_git, commit
from ..storage.atomic import atomic_write_json
from ..tasks.workspace import TaskWorkspace


class SolverEvaluationRunner:
    """Runs the solver on a task and evaluates the result.

    Steps:
    1. Read immutable task from TaskStore.
    2. Create solver workspace.
    3. Checkout clean repo at base_commit.
    4. Apply bug patch to get bugged repo.
    5. Mount only public inputs for the Solver.
    6. Materialize node code commit.
    7. Invoke agent_main.py --role solver.
    8. Collect solver patch.
    9. Run trusted test evaluator.
    10. Write EvaluationOutcome and trajectory.
    """

    def __init__(
        self,
        task_store,
        workspace_manager,
        execution_backend=None,
        test_runner=None,
        repo_pool=None,
        agent_repo: Optional[Path] = None,
        agent_adapter=None,
        model: str = "deepseek/deepseek-chat",
        solver_timeout_sec: int = 3600,
        test_timeout_sec: int = 120,
    ):
        self.task_store = task_store
        self.workspace_manager = workspace_manager
        self.execution_backend = execution_backend
        self.test_runner = test_runner or SimpleTestRunner()
        self.repo_pool = repo_pool
        self.agent_repo = Path(agent_repo) if agent_repo else None
        self.agent_adapter = agent_adapter
        self.model = model
        self.solver_timeout_sec = int(solver_timeout_sec)
        self.test_timeout_sec = int(test_timeout_sec)

    def run_task(
        self,
        node: NodeRecord,
        task: TaskRecord,
        level: Literal[1, 2],
        seed: int,
        run_id: str = "",
        solver_result_patch: Optional[str] = None,
    ) -> EvaluationOutcome:
        """Run solver evaluation on a single task.

        If solver_result_patch is provided, skip the actual solver execution
        and just evaluate the given patch (useful for testing).
        """
        start_time = time.time()

        task_id = task.task_id
        node_id = node.node_id

        try:
            if solver_result_patch is not None:
                resolved = self._evaluate_patch(task, solver_result_patch, run_id, node_id)
                error_type = None
            else:
                solver_result_patch = self._run_solver(node, task, run_id, level)
                if solver_result_patch is None:
                    resolved = False
                    error_type = "solver_execution_not_configured"
                else:
                    resolved = self._evaluate_patch(task, solver_result_patch, run_id, node_id)
                    error_type = None

            runtime = time.time() - start_time

            outcome = EvaluationOutcome(
                node_id=node_id,
                task_id=task_id,
                level=level,
                resolved=resolved,
                patch_path="",
                trajectory_id=f"{node_id}_{task_id}_{level}",
                test_summary_path="",
                runtime_sec=runtime,
                error_type=error_type,
            )
            self._persist_outcome(outcome, solver_result_patch or "", run_id)
            return outcome
        except Exception as e:
            runtime = time.time() - start_time
            outcome = EvaluationOutcome(
                node_id=node_id,
                task_id=task_id,
                level=level,
                resolved=False,
                trajectory_id=f"{node_id}_{task_id}_{level}",
                runtime_sec=runtime,
                error_type=str(e),
            )
            self._persist_outcome(outcome, solver_result_patch or "", run_id)
            return outcome

    def _artifact_dir(
        self,
        run_id: str,
        node_id: str,
        task_id: str,
        level: int,
    ) -> Path:
        workspace_root = self._workspace_root(run_id, node_id, task_id)
        path = workspace_root / "trajectories" / f"level_{level}" / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _persist_outcome(
        self,
        outcome: EvaluationOutcome,
        solver_patch: str,
        run_id: str,
    ) -> None:
        artifact_dir = self._artifact_dir(
            run_id,
            outcome.node_id,
            outcome.task_id,
            outcome.level,
        )
        # BUG-07: persist the solver patch as a real file and record its path on
        # the outcome so downstream stats (empty_patch / test_only) can read it.
        # Previously the patch only lived inside trajectory_eval.json and
        # outcome.patch_path was always "", so every task looked empty.
        if solver_patch.strip():
            patch_file = artifact_dir / "model_patch.diff"
            patch_file.write_text(solver_patch, encoding="utf-8")
            outcome.patch_path = str(patch_file)
        data = outcome.model_dump(mode="json")
        data.update(
            {
                "success": outcome.resolved,
                "model_patch": solver_patch,
                "empty_patch": not solver_patch.strip(),
                "error_stage": outcome.error_type or "",
            }
        )
        atomic_write_json(artifact_dir / "trajectory_eval.json", data)

    def _evaluate_patch(
        self,
        task: TaskRecord,
        solver_patch: str,
        run_id: str = "",
        node_id: str = "",
    ) -> bool:
        """Evaluate whether the solver patch resolves the task.

        A task is resolved if:
        - The patch is non-empty
        - Source code is modified (not just tests)
        - F2P tests all pass after applying the patch
        """
        if not solver_patch.strip():
            return False

        from ..git.patch import is_source_only, count_patch_lines

        if not is_source_only(solver_patch):
            return False

        added, deleted = count_patch_lines(solver_patch)
        if added == 0 and deleted == 0:
            return False

        f2p_tests = self.task_store.get_f2p_tests(task.task_id)
        if not f2p_tests:
            return False

        source_repo = self._source_repo_for_task(task)
        if source_repo is None:
            return False

        workspace_root = self._workspace_root(run_id, node_id, task.task_id)
        task_workspace = TaskWorkspace(workspace_root)
        bug_patch = self.task_store.get_bug_patch(task.task_id)
        repo = task_workspace.setup_bugged_repo(
            source_repo=source_repo,
            base_commit=task.base_commit,
            bug_patch=bug_patch,
            task_id=task.task_id,
        )

        try:
            if not apply_patch(repo, solver_patch):
                return False
            setup_patch = self.task_store.get_setup_patch(task.task_id)
            if setup_patch and not apply_patch(repo, setup_patch):
                return False
            command = self._f2p_test_command(task.baseline_test_command, f2p_tests)
            result = self.test_runner.run_tests(
                repo, command, timeout_sec=self.test_timeout_sec
            )
            return bool(result.get("passed"))
        finally:
            task_workspace.cleanup(task.task_id)

    def _source_repo_for_task(self, task: TaskRecord) -> Optional[Path]:
        if self.repo_pool is None:
            return None
        spec = self.repo_pool.get(task.repo_id)
        if spec is None:
            return None
        return Path(spec.path)

    def _workspace_root(self, run_id: str, node_id: str, task_id: str) -> Path:
        if hasattr(self.workspace_manager, "create_solver_workspace"):
            root = self.workspace_manager.create_solver_workspace(
                run_id or "run",
                node_id or "node",
                task_id,
            )
            # The solver adapter runs with the bugged repository as its cwd.
            # Keep artifact paths absolute so trajectories are not accidentally
            # written below (and then deleted with) that temporary repository.
            return root.parent.resolve()
        root = Path(self.workspace_manager) if self.workspace_manager else Path("./scratch")
        return (root / (run_id or "run") / "solver" / (node_id or "node")).resolve()

    def _f2p_test_command(self, baseline_command: str, f2p_tests: list[str]) -> str:
        if not f2p_tests:
            return baseline_command
        parts = baseline_command.split()
        if not any("pytest" in p for p in parts):
            return baseline_command
        quoted_tests = " ".join(dict.fromkeys(f2p_tests))
        return f"{baseline_command} {quoted_tests}"

    def _run_solver(
        self,
        node: NodeRecord,
        task: TaskRecord,
        run_id: str,
        level: int,
    ) -> Optional[str]:
        """Run the configured solver adapter and return its patch, if configured."""
        if self.agent_adapter is None or self.agent_repo is None or self.repo_pool is None:
            return None

        from ..git.worktree import NodeWorktree
        from experiment_adapters.common_agent_adapter import CommonAgentRequest

        source_repo = self._source_repo_for_task(task)
        if source_repo is None:
            return None

        workspace_root = self._workspace_root(run_id, node.node_id, task.task_id)
        task_workspace = TaskWorkspace(workspace_root)
        bug_patch = self.task_store.get_bug_patch(task.task_id)
        repo = task_workspace.setup_bugged_repo(
            source_repo=source_repo,
            base_commit=task.base_commit,
            bug_patch=bug_patch,
            task_id=task.task_id,
        )
        bugged_base_commit = task_workspace.seal_bugged_snapshot(repo)
        outdir = self._artifact_dir(run_id, node.node_id, task.task_id, level)
        outdir.mkdir(parents=True, exist_ok=True)

        try:
            with NodeWorktree(self.agent_repo, workspace_root, f"eval_{node.node_id}", node.code_commit) as agent_src:
                request = CommonAgentRequest(
                    problem_statement=self.task_store.get_problem_statement(task.task_id),
                    git_dir=repo,
                    base_commit=bugged_base_commit,
                    chat_history_file=outdir / "trajectory.jsonl",
                    outdir=outdir,
                    test_description=(task.solver_test_command or task.baseline_test_command),
                    self_improve=False,
                    instance_id=task.task_id,
                    model=self.model,
                    timeout_sec=self.solver_timeout_sec,
                )
                result = self.agent_adapter.run(agent_src, request)
            if not result.success or result.patch_path is None:
                return None
            return result.patch_path.read_text()
        finally:
            task_workspace.cleanup(task.task_id)


class SimpleTestRunner:
    """Simple test runner that executes pytest commands."""

    def run_tests(
        self,
        repo_path: Path,
        test_command: str,
        timeout_sec: int = 120,
    ) -> dict:
        """Run a test command and return parsed results."""
        try:
            result = subprocess.run(
                test_command,
                shell=True,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "passed": result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": "Timeout",
                "passed": False,
            }
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "passed": False,
            }
