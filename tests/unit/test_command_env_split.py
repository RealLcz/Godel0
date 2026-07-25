"""Env-prefix test commands must run correctly in trusted argv runners.

Job 211728 root cause: ``PYTHONPATH=lib:test/lib python3.11 -m pytest ...``
was shlex-split and executed as argv, so the env assignment became argv[0],
every clean run failed, and all candidates were rejected as
``clean_tests_unusable``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from godel0.execution.command_env import split_env_assignments
from godel0.execution.subprocess_runner import SubprocessRunner
from godel0.proposer_trusted.candidate_validator import CandidateValidator
from godel0.evaluation.runner import BackendAwareTestRunner


class TestSplitEnvAssignments:
    def test_extracts_leading_env_prefix(self):
        env, argv = split_env_assignments(
            "PYTHONPATH=lib:test/lib python3.11 -m pytest -p no:cacheprovider --rootdir=. test/units/foo.py"
        )
        assert env == {"PYTHONPATH": "lib:test/lib"}
        assert argv[0] == "python3.11"
        assert "-m" in argv and "pytest" in argv

    def test_multiple_env_assignments(self):
        env, argv = split_env_assignments("A=1 B=two pytest -q")
        assert env == {"A": "1", "B": "two"}
        assert argv == ["pytest", "-q"]

    def test_no_env_prefix_is_unchanged(self):
        env, argv = split_env_assignments("pytest -q test/units")
        assert env == {}
        assert argv == ["pytest", "-q", "test/units"]

    def test_non_leading_assignment_stays_in_argv(self):
        env, argv = split_env_assignments("pytest -q --rootdir=. X=1")
        assert env == {}
        assert argv == ["pytest", "-q", "--rootdir=.", "X=1"]

    def test_list_input(self):
        env, argv = split_env_assignments(["PYTHONPATH=lib", "pytest", "-q"])
        assert env == {"PYTHONPATH": "lib"}
        assert argv == ["pytest", "-q"]


class TestValidatorRunTestsEnvPrefix:
    def test_env_prefix_reaches_subprocess_backend(self, tmp_path: Path):
        validator = CandidateValidator(
            workspace_root=tmp_path / "validator",
            execution_backend=SubprocessRunner(),
        )
        command = (
            f"PYTHONPATH=/godel0/prefix {sys.executable} -c "
            "\"import os; print('PP=' + os.environ.get('PYTHONPATH', 'MISSING'))\""
        )
        result = validator._run_tests(tmp_path, command)
        assert result["returncode"] == 0, result
        assert "PP=/godel0/prefix" in result["stdout"]

    def test_backend_receives_split_env_and_argv(self, tmp_path: Path):
        calls = {}

        class _Backend:
            def run(self, *, command, cwd, env, timeout_sec, binds=None):
                calls["command"] = list(command)
                calls["env"] = dict(env)
                return SimpleNamespace(
                    returncode=0, stdout="", stderr="", timed_out=False
                )

        validator = CandidateValidator(
            workspace_root=tmp_path / "validator",
            execution_backend=_Backend(),
        )
        result = validator._run_tests(
            tmp_path, "PYTHONPATH=lib:test/lib python3.11 -m pytest -v"
        )
        assert result["returncode"] == 0
        assert calls["env"] == {"PYTHONPATH": "lib:test/lib"}
        assert calls["command"][0] == "python3.11"
        assert "PYTHONPATH=lib:test/lib" not in calls["command"]


class TestBackendAwareTestRunnerEnvPrefix:
    def test_env_prefix_split_before_backend_run(self, tmp_path: Path):
        calls = {}

        class _Backend:
            def run(self, *, command, cwd, env, timeout_sec, binds=None):
                calls["command"] = list(command)
                calls["env"] = dict(env)
                return SimpleNamespace(
                    returncode=0, stdout="ok", stderr="", timed_out=False
                )

        factory = SimpleNamespace(repo_backend=lambda repo_id="": _Backend())
        runner = BackendAwareTestRunner(backend_factory=factory)
        out = runner.run_tests(
            tmp_path,
            "PYTHONPATH=lib:test/lib python3.11 -m pytest -q test/units/x.py",
            timeout_sec=30,
            repo_id="ansible",
        )
        assert out["passed"] is True
        assert calls["env"] == {"PYTHONPATH": "lib:test/lib"}
        assert calls["command"][0] == "python3.11"
