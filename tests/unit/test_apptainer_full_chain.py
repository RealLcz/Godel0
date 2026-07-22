"""P0-8: Apptainer full-chain orchestration wiring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from godel0.execution.apptainer import ApptainerRunner, ExecutionBackendFactory
from godel0.execution.subprocess_runner import ProcessResult, SubprocessRunner


class TestOrchestratorAgentBackendInjection:
    def test_build_agent_adapter_receives_execution_backend(self, monkeypatch):
        from godel0.controller import orchestrator as orch_mod

        captured = {}

        class FakeAdapter:
            def __init__(self, execution_backend=None):
                captured["backend"] = execution_backend

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setattr(
            "experiment_adapters.common_agent_adapter.CommonAgentAdapter",
            FakeAdapter,
            raising=False,
        )
        # Import path used inside _build_agent_adapter
        import experiment_adapters.common_agent_adapter as caa

        monkeypatch.setattr(caa, "CommonAgentAdapter", FakeAdapter)

        backend = SubprocessRunner()
        adapter = orch_mod.EvolutionOrchestrator._build_agent_adapter(
            execution_backend=backend
        )
        assert adapter is not None
        assert captured["backend"] is backend


class TestValidatorRepoBackendResolve:
    def test_resolve_repo_backend_uses_repo_id(self, tmp_path):
        from godel0.proposer_trusted.candidate_validator import CandidateValidator

        images = tmp_path / "images"
        images.mkdir()
        sif = images / "ansible.sif"
        sif.write_text("dummy", encoding="utf-8")
        factory = ExecutionBackendFactory(
            repo_image_dir=images,
            use_apptainer=True,
        )
        validator = CandidateValidator(
            workspace_root=tmp_path / "ws",
            backend_factory=factory,
        )
        validator._active_repo_id = "ansible"
        backend = validator._resolve_repo_backend()
        assert isinstance(backend, ApptainerRunner)
        assert backend.image == sif
        assert backend.network_disabled is True

    def test_empty_repo_id_without_dir_falls_back_to_subprocess(self, tmp_path):
        from godel0.proposer_trusted.candidate_validator import CandidateValidator

        factory = ExecutionBackendFactory(
            repo_image_dir=tmp_path / "missing",
            use_apptainer=True,
        )
        validator = CandidateValidator(
            workspace_root=tmp_path / "ws",
            backend_factory=factory,
        )
        validator._active_repo_id = "ansible"
        backend = validator._resolve_repo_backend()
        assert isinstance(backend, SubprocessRunner)


class TestSolverBackendAwareTestRunner:
    def test_evaluate_uses_repo_backend_for_repo_id(self, tmp_path):
        from godel0.evaluation.runner import BackendAwareTestRunner

        images = tmp_path / "images"
        images.mkdir()
        (images / "toy.sif").write_text("dummy", encoding="utf-8")
        factory = ExecutionBackendFactory(
            repo_image_dir=images,
            use_apptainer=True,
        )
        runner = BackendAwareTestRunner(factory)
        calls = []

        class FakeBackend:
            def run(self, *, command, cwd, env, timeout_sec, binds=None):
                calls.append({"command": command, "binds": binds, "cwd": cwd})
                return ProcessResult(returncode=0, stdout="ok", stderr="")

        with patch.object(factory, "repo_backend", return_value=FakeBackend()) as mocked:
            # Force Apptainer path by returning a real ApptainerRunner wrapper...
            # Actually FakeBackend is not ApptainerRunner so binds may be None.
            # Patch isinstance path: return ApptainerRunner and stub its run.
            app = ApptainerRunner(image=images / "toy.sif")
            with patch.object(app, "run", side_effect=FakeBackend().run):
                mocked.return_value = app
                result = runner.run_tests(
                    tmp_path / "repo",
                    "pytest -q",
                    timeout_sec=30,
                    repo_id="toy",
                )
        assert result["passed"] is True
        assert calls
        assert calls[0]["binds"] == {tmp_path / "repo": "/workspace"}
        mocked.assert_called_with("toy")


class TestNodeProposerContainerPaths:
    def test_build_container_request_rewrites_repos_and_pythonpath_binds(self, tmp_path):
        from godel0.tasks.node_proposer import NodeProposerRunner

        repo = tmp_path / "repo_pool" / "ansible"
        repo.mkdir(parents=True)
        (repo / "README").write_text("x", encoding="utf-8")
        out = tmp_path / "outputs"
        out.mkdir()
        node_code = tmp_path / "node_code"
        node_code.mkdir()
        project_root = tmp_path / "godel0_project"
        project_root.mkdir()
        (project_root / "src").mkdir()

        @dataclass
        class Spec:
            repo_id: str
            path: str
            base_commit: str = "abc"
            test_command: str = "pytest -q"

        @dataclass
        class Req:
            output_dir: str
            agent_code_dir: str = ""
            repo_pool_dir: str = ""
            repo_specs: list = field(default_factory=list)
            feedback_dir: str = ""
            solver_trajectories: list = field(default_factory=list)
            parent_failure_trajectories: list = field(default_factory=list)
            current_child_level1_trajectories: list = field(default_factory=list)

        runner = NodeProposerRunner(
            agent_repo=tmp_path / "agent",
            scratch_root=tmp_path / "scratch",
            project_root=project_root,
        )
        request = Req(
            output_dir=str(out),
            agent_code_dir=str(node_code),
            repo_pool_dir=str(tmp_path / "repo_pool"),
            repo_specs=[Spec(repo_id="ansible", path=str(repo))],
            feedback_dir=str(out / "trusted_feedback"),
        )
        container_req, binds = runner._build_container_request(
            request, node_code=node_code, project_root=project_root
        )
        assert container_req.agent_code_dir == "/agent"
        assert container_req.output_dir == "/outputs"
        assert container_req.repo_pool_dir == "/repos"
        assert container_req.repo_specs[0].path == "/repos/ansible"
        assert binds[repo.resolve()] == "/repos/ansible:ro"
        assert binds[project_root.resolve()] == "/godel0:ro"
        assert binds[node_code.resolve()] == "/agent"
        assert binds[out.resolve()] == "/outputs"
        assert container_req.feedback_dir.startswith("/outputs/")

    def test_apptainer_generate_saves_container_request(self, tmp_path, monkeypatch):
        from godel0.tasks.node_proposer import NodeProposerRunner

        # Minimal stubs: skip real worktree/git by patching NodeWorktree.
        class FakeWorktree:
            def __init__(self, *a, **k):
                self.path = tmp_path / "node_code"
                self.path.mkdir(exist_ok=True)

            def __enter__(self):
                return self.path

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(
            "godel0.tasks.node_proposer.NodeWorktree", FakeWorktree
        )

        repo = tmp_path / "ansible"
        repo.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        project_root = tmp_path / "proj"
        project_root.mkdir()
        (project_root / "src").mkdir()

        image = tmp_path / "agent.sif"
        image.write_text("x", encoding="utf-8")
        backend = ApptainerRunner(image=image)
        captured = {}

        def fake_run(*, command, cwd, env, timeout_sec, binds=None):
            captured["command"] = command
            captured["env"] = env
            captured["binds"] = binds
            # Pretend proposer wrote a result.
            (out / "proposer_result.json").write_text(
                '{"run_id":"r","node_id":"n","completed":true}',
                encoding="utf-8",
            )
            return ProcessResult(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(backend, "run", fake_run)

        @dataclass
        class Spec:
            repo_id: str
            path: str
            base_commit: str = "abc"
            test_command: str = "pytest -q"

        @dataclass
        class Req:
            output_dir: str
            agent_code_dir: str = ""
            repo_pool_dir: str = ""
            repo_specs: list = field(default_factory=list)
            feedback_dir: str | None = None
            solver_trajectories: list = field(default_factory=list)
            parent_failure_trajectories: list = field(default_factory=list)
            current_child_level1_trajectories: list = field(default_factory=list)

            def save(self, path: str) -> None:
                import json
                from dataclasses import asdict

                Path(path).write_text(json.dumps(asdict(self)), encoding="utf-8")

        runner = NodeProposerRunner(
            agent_repo=tmp_path / "agent_repo",
            scratch_root=tmp_path / "scratch",
            execution_backend=backend,
            project_root=project_root,
        )
        runner.node = SimpleNamespace(node_id="n1", code_commit="deadbeef")
        request = Req(
            output_dir=str(out),
            repo_specs=[Spec(repo_id="ansible", path=str(repo))],
        )
        runner.generate_batch(request)

        assert captured["env"]["PYTHONPATH"] == "/agent:/godel0:/godel0/src"
        assert "/repos/ansible:ro" in captured["binds"].values() or any(
            v.startswith("/repos/ansible") for v in captured["binds"].values()
        )
        saved = (out / "proposer_request.json").read_text(encoding="utf-8")
        assert '"path": "/repos/ansible"' in saved or '"/repos/ansible"' in saved
        assert '"agent_code_dir": "/agent"' in saved
        assert '"output_dir": "/outputs"' in saved


class TestUseApptainerWithRepoImageDirOnly:
    def test_factory_repo_backend_works_without_agent_image(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        (images / "ansible.sif").write_text("x", encoding="utf-8")
        factory = ExecutionBackendFactory(
            agent_image=None,
            repo_image_dir=images,
            use_apptainer=True,
        )
        assert isinstance(factory.agent_backend(), SubprocessRunner)
        repo_backend = factory.repo_backend("ansible")
        assert isinstance(repo_backend, ApptainerRunner)
