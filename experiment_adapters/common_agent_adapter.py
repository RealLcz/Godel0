"""Common Agent Adapter: unified interface for calling the coding agent."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from godel0.errors import AgentExecutionError
from godel0.execution.subprocess_runner import ExecutionBackend, ProcessResult, SubprocessRunner
from godel0.execution.workspace_manager import WorkspaceManager


@dataclass
class CommonAgentRequest:
    problem_statement: str
    git_dir: Path
    base_commit: str
    chat_history_file: Path
    outdir: Path
    test_description: Optional[str] = None
    self_improve: bool = False
    instance_id: Optional[str] = None
    model: str = "deepseek/deepseek-chat"
    timeout_sec: int = 3600
    extra_env: Dict[str, str] = field(default_factory=dict)


@dataclass
class CommonAgentResult:
    success: bool
    patch_path: Optional[Path] = None
    chat_history_path: Optional[Path] = None
    process_result: Optional[ProcessResult] = None
    error: Optional[str] = None


class CommonAgentAdapter:
    """Adapter for invoking coding_agent.py via subprocess or Apptainer.

    All LLM-based phases (Solver, Proposer planning, LM Modify, LM Rewrite,
    Issue Generation, Self-evolve) go through this adapter.
    """

    def __init__(
        self,
        execution_backend: Optional[ExecutionBackend] = None,
        workspace_manager: Optional[WorkspaceManager] = None,
    ):
        self.execution_backend = execution_backend or SubprocessRunner()
        self.workspace_manager = workspace_manager

    def run(
        self,
        agent_src: Path,
        request: CommonAgentRequest,
    ) -> CommonAgentResult:
        """Run coding_agent.py with the given request."""
        agent_src = Path(agent_src)
        coding_agent = agent_src / "coding_agent.py"

        if not coding_agent.exists():
            return CommonAgentResult(
                success=False,
                error=f"coding_agent.py not found at {coding_agent}",
            )

        request.outdir.mkdir(parents=True, exist_ok=True)
        request.chat_history_file.parent.mkdir(parents=True, exist_ok=True)

        # BUG-14/15: build the command with container paths when running under
        # Apptainer, host paths when running under SubprocessRunner. The
        # ``isinstance`` branch is gone; the backend's ``run()`` is polymorphic.
        from godel0.execution.apptainer import ApptainerRunner

        is_apptainer = isinstance(self.execution_backend, ApptainerRunner)
        if is_apptainer:
            # BUG-14: only reference container paths inside the container.
            git_dir_arg = "/workspace"
            outdir_arg = "/outputs"
            chat_history_arg = "/logs/" + Path(request.chat_history_file).name
            coding_agent_arg = "/agent/coding_agent.py"
            # Under apptainer we must use the container's python interpreter.
            command = [
                "python",
                coding_agent_arg,
                "--problem_statement", request.problem_statement,
                "--git_dir", git_dir_arg,
                "--base_commit", request.base_commit,
                "--chat_history_file", chat_history_arg,
                "--outdir", outdir_arg,
                "--model", request.model,
                "--timeout", str(request.timeout_sec),
            ]
            cwd = agent_src
        else:
            command = [
                sys.executable,
                str(coding_agent),
                "--problem_statement", request.problem_statement,
                "--git_dir", str(request.git_dir),
                "--base_commit", request.base_commit,
                "--chat_history_file", str(request.chat_history_file),
                "--outdir", str(request.outdir),
                "--model", request.model,
                "--timeout", str(request.timeout_sec),
            ]
            cwd = Path(request.git_dir)

        if request.test_description:
            command.extend(["--test_description", request.test_description])
        if request.self_improve:
            command.append("--self_improve")
        if request.instance_id:
            command.extend(["--instance_id", request.instance_id])

        env: Dict[str, str] = {}
        env.update(self._get_llm_env())
        env.update(
            {
                "PYTHONPATH": "",
                "GODEL0_ROOT": "",
                "JINHE_ROOT": "",
                "SLURM_SUBMIT_DIR": "",
                "OLDPWD": "",
            }
        )
        env.update(request.extra_env)

        if is_apptainer:
            # BUG-14: standard mount layout. Host paths are bound to their
            # container mount points; the runner rewrites cwd automatically.
            binds = {
                agent_src: "/agent:ro",
                request.git_dir: "/workspace",
                request.outdir: "/outputs",
            }
            if request.chat_history_file.parent.exists():
                binds[request.chat_history_file.parent] = "/logs"
            result = self.execution_backend.run(
                command=command,
                cwd=cwd,
                env=env,
                timeout_sec=request.timeout_sec + 60,
                binds=binds,
            )
        else:
            result = self.execution_backend.run(
                command=command,
                cwd=cwd,
                env=env,
                timeout_sec=request.timeout_sec + 60,
            )

        patch_path = request.outdir / "model_patch.diff"
        success = result.returncode == 0 and patch_path.exists()

        return CommonAgentResult(
            success=success,
            patch_path=patch_path if patch_path.exists() else None,
            chat_history_path=request.chat_history_file if request.chat_history_file.exists() else None,
            process_result=result,
            error=None if success else f"Agent exit code {result.returncode}: {result.stderr[:500]}",
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0,
        max_tokens: int = 8192,
    ) -> str:
        """Provide the direct chat interface used by repo-level generators."""
        from llm import create_client

        model = os.getenv("GODEL0_MODEL", "Qwen/Qwen3.6-35B-A3B")
        client, client_model = create_client(model)
        request = {
            "model": client_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if client_model.startswith(("Qwen/", "qwen/")):
            request["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        response = client.chat.completions.create(**request)
        return str(response.choices[0].message.content or "")

    def _get_llm_env(self) -> Dict[str, str]:
        """Get environment variables for LLM API access."""
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "initial_agent" / "src"))
            from llm import llm_container_env
            return llm_container_env()
        except ImportError:
            keys = [
                "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OpenRouter_API_KEY",
                "DEEPSEEK_API_KEY", "DEEPSEEK_API_BASE_URL",
                "DEEPSEEK_MAX_OUTPUT_TOKENS", "DEEPSEEK_API_TIMEOUT",
                "MINIMAX_API_KEY", "MINIMAX_API_BASE_URL",
                "QWEN_API_KEY", "QWEN_API_BASE_URL",
                "VLLM_HOST", "VLLM_PORT",
            ]
            return {k: os.getenv(k, "") for k in keys}
