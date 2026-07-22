"""Apptainer execution backend.

P0-13~22 Apptainer full-chain fixes:

- P0-13: all ``exec`` options (``--pwd``, ``--bind``, ``--cleanenv``, ...)
  MUST appear before ``image.sif``. Putting ``--pwd`` after the image is an
  argument-order bug that Apptainer silently ignores or mis-parses.
- P0-14: bind targets and ``:ro`` mode are separated. Callers may pass
  ``"/agent:ro"``; we strip the mode for path mapping and keep it only in
  the ``--bind`` string.
- P0-15: the host subprocess that launches Apptainer must retain PATH /
  HOME / TMPDIR / SLURM_* so the launcher itself can find the binary and
  temporary directories. ``--cleanenv`` already cleans the *container* env.
- P0-16/17: ``ExecutionBackendFactory`` is the single source of truth;
  ``repo_backend(repo_id)`` resolves per-repo images.
- P0-20/21: agent-facing containers keep network; trusted tests disable
  it. LLM credentials are forwarded via an explicit allowlist.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..errors import AgentExecutionError
from .subprocess_runner import ExecutionBackend, ProcessResult


# P0-21: allowlist of env vars forwarded into the container under --cleanenv.
_LLM_ENV_ALLOWLIST = (
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_API_BASE_URL",
    "VLLM_HOST",
    "VLLM_PORT",
    "VLLM_API_KEY",
    "ANTHROPIC_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "PYTHONPATH",
)

# P0-15: host launcher must keep these so Apptainer / SLURM / temp dirs work.
_HOST_ENV_KEEP = (
    "PATH",
    "HOME",
    "USER",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LC_ALL",
    "APPTAINER_CACHEDIR",
    "APPTAINER_TMPDIR",
    "SINGULARITY_CACHEDIR",
    "SINGULARITY_TMPDIR",
)


@dataclass(frozen=True)
class ContainerMount:
    """P0-14: host path, container target, and read-only flag are separate."""

    host: Path
    target: str
    read_only: bool = False

    def bind_spec(self) -> str:
        target = self.target.rstrip("/")
        if self.read_only:
            return f"{self.host}:{target}:ro"
        return f"{self.host}:{target}"

    @property
    def container_path(self) -> str:
        """Container path without any ``:ro`` / ``:rw`` mode suffix."""
        return self.target.split(":", 1)[0]


def _parse_bind_target(container_path: str) -> Tuple[str, bool]:
    """Split ``/agent:ro`` into (``/agent``, True)."""
    if ":" in container_path:
        target, mode = container_path.rsplit(":", 1)
        if mode in ("ro", "rw"):
            return target, mode == "ro"
    return container_path, False


def _host_launcher_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """P0-15: minimal host env for the Apptainer launcher process."""
    env: Dict[str, str] = {}
    for key in _HOST_ENV_KEEP:
        if key in os.environ and os.environ[key]:
            env[key] = os.environ[key]
    for key, value in os.environ.items():
        if key.startswith("SLURM_") and value:
            env[key] = value
    if extra:
        # Never blank out PATH/HOME via caller env; merge carefully.
        for key, value in extra.items():
            if key in _HOST_ENV_KEEP or key.startswith("SLURM_"):
                if value:
                    env[key] = value
    # Always ensure PATH exists so `apptainer` can be found.
    env.setdefault("PATH", os.environ.get("PATH", "/usr/bin:/bin"))
    return env


class ApptainerRunner(ExecutionBackend):
    """Run agent commands inside Apptainer containers."""

    def __init__(
        self,
        image: Path,
        apptainer_bin: str = "apptainer",
        clean_env: bool = True,
        network_disabled: bool = False,
        env_allowlist: Optional[tuple[str, ...]] = None,
    ) -> None:
        self.image = Path(image)
        self.apptainer_bin = apptainer_bin
        self.clean_env = clean_env
        self.network_disabled = network_disabled
        self.env_allowlist = tuple(env_allowlist or _LLM_ENV_ALLOWLIST)

    def run(
        self,
        *,
        command: List[str],
        cwd: Path,
        env: Dict[str, str],
        timeout_sec: int,
        binds: Optional[Dict[Path, str]] = None,
    ) -> ProcessResult:
        # P0-13: build options FIRST, then image, then the payload command.
        cmd = [self.apptainer_bin, "exec"]

        if self.clean_env:
            cmd.append("--cleanenv")
        cmd.append("--containall")

        if self.network_disabled:
            cmd.append("--network=none")

        # P0-14: parse binds into ContainerMount so path mapping never sees
        # the ``:ro`` suffix as part of the container cwd/path.
        mounts: List[ContainerMount] = []
        container_cwd = str(cwd)
        if binds:
            for host_path, raw_target in binds.items():
                target, read_only = _parse_bind_target(raw_target)
                mount = ContainerMount(
                    host=Path(host_path),
                    target=target,
                    read_only=read_only,
                )
                mounts.append(mount)
                cmd.extend(["--bind", mount.bind_spec()])

                host_str = str(Path(host_path).resolve()) if Path(host_path).exists() else str(host_path)
                cwd_str = str(cwd)
                if cwd_str == host_str or cwd_str == str(host_path):
                    container_cwd = mount.container_path
                elif cwd_str.startswith(host_str + os.sep):
                    container_cwd = mount.container_path + cwd_str[len(host_str):]
                elif cwd_str.startswith(str(host_path) + os.sep):
                    container_cwd = mount.container_path + cwd_str[len(str(host_path)):]

        # P0-13: --pwd MUST be before the image.
        cmd.extend(["--pwd", container_cwd])

        # P0-21: under --cleanenv, forward only allowlisted LLM env vars
        # into the container via --env.
        if self.clean_env:
            forwarded: Dict[str, str] = {}
            source_env = dict(os.environ)
            source_env.update(env or {})
            for key in self.env_allowlist:
                if key in source_env and source_env[key]:
                    forwarded[key] = source_env[key]
            for key, value in forwarded.items():
                cmd.extend(["--env", f"{key}={value}"])
            # P0-15: host launcher keeps PATH/HOME/TMPDIR/SLURM_*; do NOT
            # pass an empty env to subprocess.run.
            full_env = _host_launcher_env()
        else:
            full_env = os.environ.copy()
            full_env.update(env or {})

        # P0-13: image comes AFTER all exec options.
        cmd.append(str(self.image))
        cmd.extend(command)

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


class ExecutionBackendFactory:
    """P0-16/17: single source of truth for execution backends.

    - ``agent_backend()``: network-enabled image for Solver / Proposer /
      Diagnoser / Self-Improve.
    - ``repo_backend(repo_id)``: network-disabled, per-repo image for
      trusted repository tests and RepoChain local validation.
    """

    def __init__(
        self,
        agent_image: Optional[Path] = None,
        repo_image: Optional[Path] = None,
        repo_image_dir: Optional[Path] = None,
        apptainer_bin: str = "apptainer",
        use_apptainer: bool = False,
    ) -> None:
        self.agent_image = Path(agent_image) if agent_image else None
        self.repo_image = Path(repo_image) if repo_image else None
        self.repo_image_dir = Path(repo_image_dir) if repo_image_dir else None
        self.apptainer_bin = apptainer_bin
        self.use_apptainer = use_apptainer

    def _resolve_repo_image(self, repo_id: str) -> Optional[Path]:
        if self.repo_image:
            return self.repo_image
        if self.repo_image_dir and repo_id:
            # Accept both ``ansible.sif`` and ``ansible_repochain.sif``.
            for name in (f"{repo_id}.sif", f"{repo_id}_repochain.sif"):
                candidate = self.repo_image_dir / name
                if candidate.is_file():
                    return candidate
        return None

    def agent_backend(self) -> ExecutionBackend:
        if self.use_apptainer and self.agent_image:
            return ApptainerRunner(
                image=self.agent_image,
                apptainer_bin=self.apptainer_bin,
                clean_env=True,
                network_disabled=False,
            )
        from .subprocess_runner import SubprocessRunner

        return SubprocessRunner()

    def repo_backend(
        self,
        repo_id: str = "",
        image: Optional[Path] = None,
    ) -> ExecutionBackend:
        if self.use_apptainer:
            resolved = image if image is not None else self._resolve_repo_image(repo_id)
            if resolved is not None:
                return ApptainerRunner(
                    image=resolved,
                    apptainer_bin=self.apptainer_bin,
                    clean_env=True,
                    network_disabled=True,
                )
        from .subprocess_runner import SubprocessRunner

        return SubprocessRunner()


__all__ = [
    "ApptainerRunner",
    "ContainerMount",
    "ExecutionBackendFactory",
]
