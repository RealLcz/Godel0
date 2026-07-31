"""Self-edit runner: invokes the coding agent to modify the agent codebase."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..git.repository import diff_vs_commit, restore_runtime_artifacts, run_git
from ..schemas.diagnosis import CycleDiagnosis
from .patch_guard import validate_changed_python_syntax

EDIT_PROTOCOL = """
Implement the improvement task in the current agent repository.

Inspect the existing implementation before editing. Make the simplest coherent
change that implements the requested capability improvement in the live runtime
path.

Do not hard-code identifiers or behavior from the observed failure instance.
Do not modify trusted evaluation code, frozen transport protocols, generated
artifacts, documentation, backup files, or unrelated tests.

Prefer modifying an existing workflow, prompt, or tool over introducing a new
subsystem when both would address the issue.

Multiple production files may be modified only when they are directly required
by the same improvement. Breadth is not evidence of quality.

Before finishing:

- inspect the final diff;
- confirm the changed code is reachable from the live runtime path;
- run the most relevant available import or test command;
- leave the repository with a non-empty source-code diff.
"""


@dataclass
class SelfEditResult:
    success: bool
    patch: str = ""
    trajectory_path: Optional[Path] = None
    error: Optional[str] = None
    wall_time_sec: float = 0.0
    attempts: int = 0
    attempt_errors: List[str] = field(default_factory=list)


class SelfEditRunner:
    """Runs the coding agent in self-improve mode to modify the agent codebase.

    The agent is given the CycleDiagnosis.problem_statement and the agent
    code worktree as its workspace. It can modify any file in the worktree.

    An attempt that ends without a usable diff (the agent died on context
    exhaustion, or left a half-written file) is retried in a fresh context
    with the previous failure quoted back, instead of costing the caller a
    whole expansion.
    """

    def __init__(self, agent_adapter=None, timeout_sec: int = 3600, max_attempts: int = 3):
        self.agent_adapter = agent_adapter
        self.timeout_sec = timeout_sec
        self.max_attempts = max(1, int(max_attempts))

    def run(
        self,
        diagnosis: CycleDiagnosis,
        worktree: Path,
        output_dir: Path,
        agent_src: Path = None,
        model: str = "deepseek/deepseek-chat",
        base_commit: str = "HEAD",
    ) -> SelfEditResult:
        """Run self-edit on the worktree, retrying unusable attempts."""
        start = time.time()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.agent_adapter is None:
            return SelfEditResult(
                success=False,
                error="No agent adapter configured for self-edit",
                wall_time_sec=time.time() - start,
            )

        attempt_errors: List[str] = []
        last_result: Optional[SelfEditResult] = None

        for attempt in range(1, self.max_attempts + 1):
            if attempt > 1:
                self._reset_worktree(worktree, base_commit)
            attempt_dir = (
                output_dir if attempt == 1 else output_dir / f"retry_{attempt:03d}"
            )
            attempt_dir.mkdir(parents=True, exist_ok=True)

            last_result = self._run_once(
                diagnosis=diagnosis,
                worktree=worktree,
                output_dir=attempt_dir,
                agent_src=agent_src,
                model=model,
                previous_errors=attempt_errors,
            )
            last_result.attempts = attempt
            last_result.attempt_errors = list(attempt_errors)

            problem = self._patch_problem(worktree, base_commit)
            if problem is None:
                last_result.success = True
                last_result.error = None
                last_result.wall_time_sec = time.time() - start
                return last_result

            attempt_errors.append(problem)

        result = last_result or SelfEditResult(success=False)
        result.success = False
        result.error = "; ".join(attempt_errors) or result.error or "self-edit produced no usable patch"
        result.attempts = self.max_attempts
        result.attempt_errors = list(attempt_errors)
        result.wall_time_sec = time.time() - start
        return result

    def _run_once(
        self,
        diagnosis: CycleDiagnosis,
        worktree: Path,
        output_dir: Path,
        agent_src: Optional[Path],
        model: str,
        previous_errors: List[str],
    ) -> SelfEditResult:
        from experiment_adapters.common_agent_adapter import CommonAgentRequest

        chat_history = output_dir / "trajectory.jsonl"
        request = CommonAgentRequest(
            problem_statement=self._build_instruction(diagnosis, previous_errors),
            git_dir=worktree,
            base_commit="HEAD",
            chat_history_file=chat_history,
            outdir=output_dir,
            self_improve=True,
            model=model,
            timeout_sec=self.timeout_sec,
        )

        result = self.agent_adapter.run(agent_src or worktree, request)

        patch = ""
        if result.patch_path and result.patch_path.exists():
            patch = result.patch_path.read_text()

        return SelfEditResult(
            success=result.success,
            patch=patch,
            trajectory_path=chat_history if chat_history.exists() else None,
            error=result.error,
        )

    def _build_instruction(
        self,
        diagnosis: CycleDiagnosis,
        previous_errors: List[str],
    ) -> str:
        parts = [diagnosis.problem_statement.rstrip(), EDIT_PROTOCOL.strip()]
        if previous_errors:
            history = "\n".join(f"- {error}" for error in previous_errors)
            parts.append(
                "The previous attempt was discarded because:\n"
                f"{history}\n"
                "Start from the clean phase base and implement the same "
                "improvement without changing frozen protocols or generated "
                "artifacts."
            )
        return "\n\n".join(parts)

    def _patch_problem(self, worktree: Path, base_commit: str) -> Optional[str]:
        """Return why the worktree diff is unusable, or None when it is fine."""
        worktree = Path(worktree)
        try:
            restore_runtime_artifacts(worktree, base_commit)
            patch = diff_vs_commit(worktree, base_commit)
        except Exception as exc:  # pragma: no cover - git failure is fatal anyway
            return f"could not diff worktree: {exc}"
        if not patch.strip():
            return "empty patch (agent finished without changing any file)"
        syntax_errors = validate_changed_python_syntax(worktree, patch)
        if syntax_errors:
            return "; ".join(syntax_errors)
        return None

    def _reset_worktree(self, worktree: Path, base_commit: str = "HEAD") -> None:
        """Discard a failed attempt so the retry starts from the phase base."""
        worktree = Path(worktree)
        run_git(worktree, "reset", "--hard", base_commit)
        run_git(worktree, "clean", "-fd")
