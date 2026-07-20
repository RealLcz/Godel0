"""Self-edit runner: invokes the coding agent to modify the agent codebase."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..errors import AgentExecutionError
from ..schemas.diagnosis import CycleDiagnosis


@dataclass
class SelfEditResult:
    success: bool
    patch: str = ""
    trajectory_path: Optional[Path] = None
    error: Optional[str] = None
    wall_time_sec: float = 0.0


class SelfEditRunner:
    """Runs the coding agent in self-improve mode to modify the agent codebase.

    The agent is given the CycleDiagnosis.problem_statement and the agent
    code worktree as its workspace. It can modify any file in the worktree.
    """

    def __init__(self, agent_adapter=None, timeout_sec: int = 3600):
        self.agent_adapter = agent_adapter
        self.timeout_sec = timeout_sec

    def run(
        self,
        diagnosis: CycleDiagnosis,
        worktree: Path,
        output_dir: Path,
        agent_src: Path = None,
        model: str = "deepseek/deepseek-chat",
    ) -> SelfEditResult:
        """Run self-edit on the worktree."""
        start = time.time()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.agent_adapter is None:
            return SelfEditResult(
                success=False,
                patch="",
                trajectory_path=None,
                error="No agent adapter configured for self-edit",
                wall_time_sec=time.time() - start,
            )

        from experiment_adapters.common_agent_adapter import CommonAgentRequest

        chat_history = output_dir / "trajectory.jsonl"
        request = CommonAgentRequest(
            problem_statement=diagnosis.problem_statement,
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
            wall_time_sec=time.time() - start,
        )
