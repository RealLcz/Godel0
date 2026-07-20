"""Execution backend protocol and subprocess runner."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..errors import AgentExecutionError, WorkspaceError


@dataclass
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    wall_time_sec: float = 0.0


class ExecutionBackend:
    """Protocol for execution backends."""

    def run(
        self,
        *,
        command: List[str],
        cwd: Path,
        env: Dict[str, str],
        timeout_sec: int,
        binds: Optional[Dict[Path, str]] = None,
    ) -> ProcessResult:
        raise NotImplementedError


class SubprocessRunner(ExecutionBackend):
    """Run agent commands directly as subprocesses (no container isolation)."""

    def run(
        self,
        *,
        command: List[str],
        cwd: Path,
        env: Dict[str, str],
        timeout_sec: int,
        binds: Optional[Dict[Path, str]] = None,
    ) -> ProcessResult:
        cwd = Path(cwd)
        cwd.mkdir(parents=True, exist_ok=True)

        full_env = os.environ.copy()
        full_env.update(env)

        start = time.time()
        try:
            proc = subprocess.run(
                command,
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
        except Exception as e:
            raise AgentExecutionError(f"Subprocess failed: {e}") from e
