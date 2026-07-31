"""Self-edit runner: invokes the coding agent to modify the agent codebase."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..git.repository import diff_vs_commit, run_git
from ..schemas.diagnosis import CycleDiagnosis
from .patch_guard import validate_changed_python_syntax

# HGM-aligned self-improve protocol: improve the agent for a *class* of
# failures (not a one-line hotfix). Keep regional reads to protect context,
# but do not force minimal/single-file edits.
EDIT_PROTOCOL = """
Editing protocol (HGM-style self-improve):
- Implement the diagnosis fully enough to address this *class* of failures.
  Multiple related files are allowed when needed; do not stop at a cosmetic
  one-line change if the diagnosis calls for a real mechanism.
- Prefer extending existing tools / workflows and wiring them into the live
  path (`forward()`, proposer planners, swesmith helpers) over dead helpers.
- Locate code with `grep -n` / `sed -n 'A,Bp'`. Prefer reading only the regions
  you need; avoid dumping entire huge files into context.
- Do NOT hard-code task-specific repo/file/module/instance names as constants.
  Concrete names in the issue are illustrative examples only.
- Do not edit frozen transport schemas (`proposer/request.py`,
  `proposer/schemas.py`) or unrelated documentation.
- After editing, re-read the changed regions to confirm they are syntactically
  intact and actually invoked from the live path.
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
                self._reset_worktree(worktree)
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
                "A previous attempt on this same problem was discarded:\n"
                f"{history}\n"
                "Start from the unmodified repository and make the edit early "
                "so it survives."
            )
        return "\n\n".join(parts)

    def _patch_problem(self, worktree: Path, base_commit: str) -> Optional[str]:
        """Return why the worktree diff is unusable, or None when it is fine."""
        try:
            patch = diff_vs_commit(Path(worktree), base_commit)
        except Exception as exc:  # pragma: no cover - git failure is fatal anyway
            return f"could not diff worktree: {exc}"
        if not patch.strip():
            return "empty patch (agent finished without changing any file)"
        syntax_errors = validate_changed_python_syntax(Path(worktree), patch)
        if syntax_errors:
            return "; ".join(syntax_errors)
        return None

    def _reset_worktree(self, worktree: Path) -> None:
        """Discard a failed attempt so the retry starts from the base commit."""
        worktree = Path(worktree)
        run_git(worktree, "reset", "--hard", "HEAD")
        run_git(worktree, "clean", "-fd")
