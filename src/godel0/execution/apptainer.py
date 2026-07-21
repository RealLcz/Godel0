"""Apptainer execution backend.

BUG-13~17: full Apptainer chain fixes.

- BUG-13: ``image``, ``clean_env``, ``network_disabled`` move to the
  constructor so ``run()`` is signature-compatible with ``SubprocessRunner``.
  Callers no longer need ``isinstance`` branches; the same
  ``backend.run(command=, cwd=, env=, timeout_sec=, binds=)`` call works for
  both backends.
- BUG-14: container commands only reference container paths
  (``/agent``, ``/workspace``, ``/outputs``, ``/logs``, ``/control``). The
  caller still passes host paths in ``binds``; the runner maps them to their
  container mount points and rewrites ``cwd`` to the container path.
- BUG-16: network policy is phased. Agent-facing phases (solver, proposer,
  diagnoser, self-improve) need online LLM access, so ``network_disabled``
  defaults to ``False``. Trusted repository tests construct a separate
  ``ApptainerRunner`` with ``network_disabled=True``.
- BUG-17: under ``--cleanenv`` only an allowlist of LLM env vars is forwarded
  explicitly via ``--env``; no more ``full_env.update(env)``.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..errors import AgentExecutionError
from .subprocess_runner import ExecutionBackend, ProcessResult


# BUG-17: allowlist of env vars forwarded into the container under --cleanenv.
# Anything not in this list is dropped, so the evolvable agent cannot inject
# arbitrary env vars and the LLM credentials are guaranteed to be present.
_LLM_ENV_ALLOWLIST = (
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
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
    "PYTHONPATH",
)


class ApptainerRunner(ExecutionBackend):
    """Run agent commands inside Apptainer containers.

    BUG-13: the image, clean-env, and network policy are fixed at
    construction so ``run()`` is polymorphic with ``SubprocessRunner.run``.
    """

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
        # BUG-16: default to network-enabled so online LLM API calls work.
        # Trusted repository tests should construct a separate runner with
        # network_disabled=True.
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
        cmd = [self.apptainer_bin, "exec"]

        if self.clean_env:
            cmd.append("--cleanenv")
        cmd.append("--containall")

        # BUG-16: phased network policy. Agent-facing phases default to
        # network-enabled; trusted repository tests opt into --network=none.
        if self.network_disabled:
            cmd.append("--network=none")

        # BUG-14: standard mount layout. ``binds`` maps host paths to their
        # container mount points; we pass the bind through and rewrite ``cwd``
        # to the matching container path so the in-container command never
        # references a host absolute path.
        container_cwd = str(cwd)
        if binds:
            for host_path, container_path in binds.items():
                mount = f"{host_path}:{container_path}"
                cmd.extend(["--bind", mount])
                host_str = str(host_path)
                # If the requested cwd is under a bound host path, rewrite it
                # to the container path so the in-container command is correct.
                if str(cwd) == host_str:
                    container_cwd = container_path
                elif str(cwd).startswith(host_str + os.sep):
                    container_cwd = container_path + str(cwd)[len(host_str):]

        # BUG-17: under --cleanenv, only forward the allowlisted LLM env vars
        # explicitly via --env. Do NOT do full_env.update(env).
        if self.clean_env:
            forwarded = {}
            source_env = dict(os.environ)
            source_env.update(env)
            for key in self.env_allowlist:
                if key in source_env and source_env[key]:
                    forwarded[key] = source_env[key]
            for key, value in forwarded.items():
                cmd.extend(["--env", f"{key}={value}"])
            # The subprocess itself does not need the full env; apptainer
            # injects the allowlisted vars into the container.
            full_env = {}
        else:
            full_env = os.environ.copy()
            full_env.update(env)

        cmd.append(str(self.image))
        # BUG-14: run from the container path, not the host path.
        cmd.extend(["--pwd", container_cwd])
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
    """BUG-15/26: single source of truth for execution backends.

    Solver, Proposer, Validator, and Self-edit all ask this factory for a
    backend so the image / network / env policy is configured in one place.

    - ``agent_backend()`` returns the agent-facing image (network enabled)
      used for Solver, Proposer, Diagnoser, Self-Improve.
    - ``repo_backend(repo_id)`` returns a repo-specific image (network
      disabled) used for trusted repository tests and RepoChain generation.
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
        # BUG-26: repo images live in a per-repo directory, one .sif per repo.
        # When ``repo_image`` is not set explicitly, we resolve
        # ``<repo_image_dir>/<repo_id>.sif`` lazily.
        self.repo_image_dir = Path(repo_image_dir) if repo_image_dir else None
        self.apptainer_bin = apptainer_bin
        self.use_apptainer = use_apptainer

    def _resolve_repo_image(self, repo_id: str) -> Optional[Path]:
        """BUG-26: resolve the repo-specific image for a repo id."""
        if self.repo_image:
            return self.repo_image
        if self.repo_image_dir and repo_id:
            candidate = self.repo_image_dir / f"{repo_id}.sif"
            if candidate.is_file():
                return candidate
        return None

    def agent_backend(self) -> ExecutionBackend:
        """Agent-facing backend (Solver / Proposer / Diagnoser / Self-Improve).

        BUG-16: network is enabled so online LLM API calls work.
        """
        if self.use_apptainer and self.agent_image:
            return ApptainerRunner(
                image=self.agent_image,
                apptainer_bin=self.apptainer_bin,
                clean_env=True,
                network_disabled=False,
            )
        from .subprocess_runner import SubprocessRunner

        return SubprocessRunner()

    def repo_backend(self, repo_id: str = "", image: Optional[Path] = None) -> ExecutionBackend:
        """Repo-specific backend for trusted repository tests.

        BUG-16: trusted repository tests disable network access.
        BUG-26: resolve the repo-specific image from ``repo_image_dir`` or
            from the explicit ``image`` argument (e.g. from ``RepoSpec.image``).
        """
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


__all__ = ["ApptainerRunner", "ExecutionBackendFactory"]
