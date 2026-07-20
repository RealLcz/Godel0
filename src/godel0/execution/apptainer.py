"""Apptainer execution backend."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..errors import AgentExecutionError
from .subprocess_runner import ExecutionBackend, ProcessResult


class ApptainerRunner(ExecutionBackend):
    """Run agent commands inside Apptainer containers."""

    def __init__(self, apptainer_bin: str = "apptainer"):
        self.apptainer_bin = apptainer_bin

    def run(
        self,
        *,
        command: List[str],
        cwd: Path,
        env: Dict[str, str],
        timeout_sec: int,
        image: Path,
        binds: Optional[Dict[Path, str]] = None,
        clean_env: bool = True,
        network_disabled: bool = True,
    ) -> ProcessResult:
        cmd = [self.apptainer_bin, "exec"]

        if clean_env:
            cmd.append("--cleanenv")
        cmd.append("--containall")

        if network_disabled:
            cmd.append("--network=none")

        if binds:
            for host_path, container_path in binds.items():
                mount = f"{host_path}:{container_path}"
                cmd.extend(["--bind", mount])

        cmd.append(str(image))
        cmd.extend(command)

        full_env = os.environ.copy()
        full_env.update(env)

        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=full_env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            elapsed = time.time() - start
            return ProcessResult(
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                wall_time_sec=elapsed,
            )
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start
            return ProcessResult(
                returncode=-15,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
                timed_out=True,
                wall_time_sec=elapsed,
            )
        except FileNotFoundError:
            raise AgentExecutionError(
                f"Apptainer binary '{self.apptainer_bin}' not found. "
                "Install apptainer or use backend='subprocess'."
            )
        except Exception as e:
            raise AgentExecutionError(f"Apptainer execution failed: {e}") from e
