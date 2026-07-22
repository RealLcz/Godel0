"""Isolated adapter for running the proposer from a node's exact Git commit."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..git.worktree import NodeWorktree


class NodeProposerRunner:
    """Run ``proposer_main`` from the selected node, never from root imports.

    The trusted controller owns this adapter and the worktree.  The child
    process imports proposer/SWE-smith from that worktree, so changes to either
    component become effective immediately and remain isolated by commit.

    Phase 9 / P0-8.4/8.5: accepts an optional ``execution_backend``
    (SubprocessRunner or ApptainerRunner). Under Apptainer the runner:
      - binds node_code → /agent, output → /outputs, project_root → /godel0:ro
      - binds each ``repo_specs[*].path`` → /repos/<repo_id>
      - writes a container-specific ProposerRequest with rewritten paths
      - sets PYTHONPATH=/agent:/godel0:/godel0/src
    """

    def __init__(
        self,
        agent_repo: Path,
        scratch_root: Path,
        timeout_sec: int = 3600,
        execution_backend=None,
        project_root: Optional[Path] = None,
    ):
        self.agent_repo = Path(agent_repo).resolve()
        self.scratch_root = Path(scratch_root).resolve()
        self.timeout_sec = timeout_sec
        self.execution_backend = execution_backend
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[3]
        )
        self.node = None

    def for_node(self, node):
        runner = NodeProposerRunner(
            self.agent_repo,
            self.scratch_root,
            timeout_sec=self.timeout_sec,
            execution_backend=self.execution_backend,
            project_root=self.project_root,
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
            Path(request.output_dir).mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            project_root = self.project_root
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
                from ..execution.apptainer import ApptainerRunner

                if isinstance(self.execution_backend, ApptainerRunner):
                    container_request, binds = self._build_container_request(
                        isolated_request,
                        node_code=Path(node_code),
                        project_root=project_root,
                    )
                    container_request.save(str(request_path))
                    # Container-visible PYTHONPATH (host paths do not exist
                    # under --containall).
                    env["PYTHONPATH"] = "/agent:/godel0:/godel0/src"
                    container_command = [
                        "python",
                        "-m",
                        "proposer.proposer_main",
                        "--request",
                        "/outputs/proposer_request.json",
                        "--output_dir",
                        "/outputs",
                    ]
                    result = self.execution_backend.run(
                        command=container_command,
                        cwd=Path(node_code),
                        env=env,
                        timeout_sec=self.timeout_sec,
                        binds=binds,
                    )
                else:
                    isolated_request.save(str(request_path))
                    result = self.execution_backend.run(
                        command=command,
                        cwd=Path(node_code),
                        env=env,
                        timeout_sec=self.timeout_sec,
                    )
                stdout_path.write_text(result.stdout, encoding="utf-8")
                stderr_path.write_text(result.stderr, encoding="utf-8")
            else:
                isolated_request.save(str(request_path))
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

            from ..schemas.proposer_transport import ProposerResultV1

            data = json.loads(result_path.read_text(encoding="utf-8"))
            result_obj = ProposerResultV1.from_dict(data)
            if result.returncode not in (0, 1) and not result_obj.error:
                result_obj.error = stderr_path.read_text(
                    encoding="utf-8", errors="replace"
                )[-2000:]
            return result_obj

    def _build_container_request(
        self,
        request,
        *,
        node_code: Path,
        project_root: Path,
    ) -> Tuple[object, Dict[Path, str]]:
        """P0-8.4/8.5: rewrite host paths and assemble Apptainer binds."""
        binds: Dict[Path, str] = {
            Path(node_code).resolve(): "/agent",
            Path(request.output_dir).resolve(): "/outputs",
            Path(project_root).resolve(): "/godel0:ro",
        }

        container_specs = []
        for spec in list(getattr(request, "repo_specs", None) or []):
            host_path = Path(getattr(spec, "path", "") or "").resolve()
            repo_id = str(getattr(spec, "repo_id", "") or "repo")
            container_path = f"/repos/{repo_id}"
            if host_path.exists():
                binds[host_path] = f"{container_path}:ro"
            # Rebuild a same-type spec with container path.
            if hasattr(spec, "__dataclass_fields__"):
                container_specs.append(replace(spec, path=container_path))
            else:
                container_specs.append(spec)

        feedback_dir = getattr(request, "feedback_dir", None)
        container_feedback = "/outputs/trusted_feedback"
        if feedback_dir:
            feedback_path = Path(feedback_dir)
            out_path = Path(request.output_dir).resolve()
            try:
                if feedback_path.resolve().is_relative_to(out_path):
                    rel = feedback_path.resolve().relative_to(out_path)
                    container_feedback = f"/outputs/{rel.as_posix()}"
            except (ValueError, AttributeError):
                # Python <3.9 has no is_relative_to — fall back.
                try:
                    rel = feedback_path.resolve().relative_to(out_path)
                    container_feedback = f"/outputs/{rel.as_posix()}"
                except ValueError:
                    container_feedback = "/outputs/trusted_feedback"

        # Bind trajectory parents so failure signatures remain readable.
        traj_fields = (
            "solver_trajectories",
            "parent_failure_trajectories",
            "current_child_level1_trajectories",
        )
        traj_rewrites: Dict[str, List[str]] = {}
        for field_name in traj_fields:
            host_list = list(getattr(request, field_name, None) or [])
            rewritten: List[str] = []
            for host in host_list:
                host_path = Path(host).resolve()
                if not host_path.exists():
                    rewritten.append(host)
                    continue
                parent = host_path.parent
                # Mount each unique parent under /traj/<stable-id>
                mount_key = parent
                target = None
                for existing_host, existing_target in binds.items():
                    if Path(existing_host).resolve() == mount_key:
                        target = existing_target.split(":", 1)[0]
                        break
                if target is None:
                    import hashlib

                    digest = hashlib.sha1(str(mount_key).encode("utf-8")).hexdigest()[:12]
                    target = f"/traj/{digest}"
                    binds[mount_key] = f"{target}:ro"
                rewritten.append(f"{target}/{host_path.name}")
            traj_rewrites[field_name] = rewritten

        container_request = replace(
            request,
            agent_code_dir="/agent",
            output_dir="/outputs",
            repo_pool_dir="/repos",
            repo_specs=container_specs,
            feedback_dir=container_feedback,
            solver_trajectories=traj_rewrites.get(
                "solver_trajectories", list(getattr(request, "solver_trajectories", []) or [])
            ),
            parent_failure_trajectories=traj_rewrites.get(
                "parent_failure_trajectories",
                list(getattr(request, "parent_failure_trajectories", []) or []),
            ),
            current_child_level1_trajectories=traj_rewrites.get(
                "current_child_level1_trajectories",
                list(getattr(request, "current_child_level1_trajectories", []) or []),
            ),
        )
        return container_request, binds


__all__ = ["NodeProposerRunner"]
