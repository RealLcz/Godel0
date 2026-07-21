"""Unit tests for the Apptainer runner and backend factory (BUG-13~17)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from godel0.execution.apptainer import ApptainerRunner, ExecutionBackendFactory
from godel0.execution.subprocess_runner import SubprocessRunner


class TestApptainerRunnerConstructor:
    def test_image_is_required_at_construction(self, tmp_path):
        image = tmp_path / "agent.sif"
        image.write_text("dummy")
        runner = ApptainerRunner(image=image)
        assert runner.image == image

    def test_network_defaults_to_enabled(self, tmp_path):
        # BUG-16: agent-facing backend keeps network enabled by default.
        image = tmp_path / "agent.sif"
        image.write_text("dummy")
        runner = ApptainerRunner(image=image)
        assert runner.network_disabled is False

    def test_clean_env_defaults_to_true(self, tmp_path):
        image = tmp_path / "agent.sif"
        image.write_text("dummy")
        runner = ApptainerRunner(image=image)
        assert runner.clean_env is True


class TestApptainerRunnerRunSignature:
    def test_run_accepts_same_signature_as_subprocess(self, tmp_path):
        """BUG-13: run() must be signature-compatible with SubprocessRunner.run()."""
        import inspect

        sub_sig = inspect.signature(SubprocessRunner.run)
        app_sig = inspect.signature(ApptainerRunner.run)
        sub_params = set(sub_sig.parameters.keys())
        app_params = set(app_sig.parameters.keys())
        # ApptainerRunner.run must accept at least the same parameters.
        assert sub_params.issubset(app_params)


class TestExecutionBackendFactory:
    def test_agent_backend_returns_subprocess_when_apptainer_disabled(self):
        factory = ExecutionBackendFactory(use_apptainer=False)
        backend = factory.agent_backend()
        assert isinstance(backend, SubprocessRunner)

    def test_agent_backend_returns_apptainer_when_enabled(self, tmp_path):
        image = tmp_path / "agent.sif"
        image.write_text("dummy")
        factory = ExecutionBackendFactory(
            agent_image=image,
            use_apptainer=True,
        )
        backend = factory.agent_backend()
        assert isinstance(backend, ApptainerRunner)
        # BUG-16: agent backend has network enabled.
        assert backend.network_disabled is False

    def test_repo_backend_disables_network(self, tmp_path):
        image = tmp_path / "repo.sif"
        image.write_text("dummy")
        factory = ExecutionBackendFactory(
            repo_image=image,
            use_apptainer=True,
        )
        backend = factory.repo_backend("repo1")
        assert isinstance(backend, ApptainerRunner)
        # BUG-16: repo backend disables network.
        assert backend.network_disabled is True

    def test_repo_backend_resolves_image_from_dir(self, tmp_path):
        # BUG-26: repo images resolved from repo_image_dir/<repo_id>.sif
        repo_dir = tmp_path / "images"
        repo_dir.mkdir()
        (repo_dir / "ansible.sif").write_text("dummy")
        factory = ExecutionBackendFactory(
            repo_image_dir=repo_dir,
            use_apptainer=True,
        )
        backend = factory.repo_backend("ansible")
        assert isinstance(backend, ApptainerRunner)
        assert backend.image.name == "ansible.sif"

    def test_repo_backend_falls_back_to_subprocess_when_no_image(self, tmp_path):
        factory = ExecutionBackendFactory(
            repo_image_dir=tmp_path,
            use_apptainer=True,
        )
        backend = factory.repo_backend("nonexistent")
        assert isinstance(backend, SubprocessRunner)
