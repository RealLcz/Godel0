"""Isolated adapter for running the proposer from a node's exact Git commit."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Optional

from ..git.worktree import NodeWorktree


class NodeProposerRunner:
    """Run ``proposer_main`` from the selected node, never from root imports.

    The trusted controller owns this adapter and the worktree.  The child
    process imports proposer/SWE-smith from that worktree, so changes to either
    component become effective immediately and remain isolated by commit.

    Phase 9: accepts an optional ``execution_backend`` (SubprocessRunner or
    ApptainerRunner) so the proposer can run in an Apptainer container for
    HPC deployments. When no backend is supplied, falls back to direct
    subprocess (backward compatible).
    """

    def __init__(
        self,
        agent_repo: Path,
        scratch_root: Path,
        timeout_sec: int = 3600,
        execution_backend=None,
    ):
        self.agent_repo = Path(agent_repo).resolve()
        self.scratch_root = Path(scratch_root).resolve()
        self.timeout_sec = timeout_sec
        self.execution_backend = execution_backend
        self.node = None

    def for_node(self, node):
        runner = NodeProposerRunner(
            self.agent_repo,
            self.scratch_root,
            timeout_sec=self.timeout_sec,
            execution_backend=self.execution_backend,
        )
        runner.node = node
        return runner

    def generate_batch(self, request):
        if self.node is None:
            raise RuntimeError("NodeProposerRunner must be bound with for_node(node)")

        invocation_id = f"proposer_{self.node.node_id}_{uuid.uuid4().hex[:8]}"
        with NodeWorktree(
            self.agent_repo,
            self.scratch_root,
            invocation_id,
            self.node.code_commit,
        ) as node_code:
            isolated_request = replace(request, agent_code_dir=str(node_code))
            request_path = Path(request.output_dir) / "proposer_request.json"
            isolated_request.save(str(request_path))

            env = os.environ.copy()
            project_root = Path(__file__).resolve().parents[3]
            env["PYTHONPATH"] = os.pathsep.join(
                [str(node_code), str(project_root), str(project_root / "src")]
            )
            command = [
                sys.executable,
                "-m",
                "proposer.proposer_main",
                "--request",
                str(request_path),
                "--output_dir",
                str(request.output_dir),
            ]
            stdout_path = Path(request.output_dir) / "proposer.stdout.log"
            stderr_path = Path(request.output_dir) / "proposer.stderr.log"
            stdout_path.parent.mkdir(parents=True, exist_ok=True)

            if self.execution_backend is not None:
                # Phase 9: run via the unified ExecutionBackend (subprocess or
                # apptainer). The backend handles cwd/env/timeout uniformly.
                result = self.execution_backend.run(
                    command=command,
                    cwd=Path(node_code),
                    env=env,
                    timeout_sec=self.timeout_sec,
                )
                stdout_path.write_text(result.stdout, encoding="utf-8")
                stderr_path.write_text(result.stderr, encoding="utf-8")
            else:
                # Backward-compatible direct subprocess path.
                with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
                    "w", encoding="utf-8"
                ) as stderr_file:
                    completed = subprocess.run(
                        command,
                        cwd=node_code,
                        env=env,
                        text=True,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        timeout=self.timeout_sec,
                    )
                # Mimic the ProcessResult shape for the result-path check below.
                from ..execution.subprocess_runner import ProcessResult

                result = ProcessResult(
                    returncode=completed.returncode,
                    stdout="",
                    stderr="",
                    wall_time_sec=0.0,
                )

            result_path = Path(request.output_dir) / "proposer_result.json"
            if not result_path.is_file():
                stderr_tail = stderr_path.read_text(
                    encoding="utf-8", errors="replace"
                )[-2000:]
                raise RuntimeError(
                    "Node proposer produced no result: "
                    f"exit={result.returncode}; stderr={stderr_tail}"
                )

            # Importing this schema in the controller is safe: it is only a
            # transport type. All intelligent policy ran in the child process.
            from initial_agent.src.proposer.request import ProposerResult

            data = json.loads(result_path.read_text(encoding="utf-8"))
            result_obj = ProposerResult.from_dict(data)
            if result.returncode not in (0, 1) and not result_obj.error:
                result_obj.error = stderr_path.read_text(
                    encoding="utf-8", errors="replace"
                )[-2000:]
            return result_obj


__all__ = ["NodeProposerRunner"]
