from pathlib import Path
import sys
from types import SimpleNamespace

from experiment_adapters.common_agent_adapter import (
    CommonAgentAdapter,
    CommonAgentRequest,
)
from godel0.execution.subprocess_runner import ProcessResult, SubprocessRunner


class RecordingRunner(SubprocessRunner):
    def __init__(self) -> None:
        self.cwd: Path | None = None
        self.env = None

    def run(self, *, command, cwd, env, timeout_sec, binds=None):
        self.cwd = Path(cwd)
        self.env = env
        return ProcessResult(returncode=0, stdout="", stderr="")


def test_common_agent_starts_in_target_repository(tmp_path: Path):
    agent_src = tmp_path / "agent"
    repo = tmp_path / "repo"
    outdir = tmp_path / "output"
    agent_src.mkdir()
    repo.mkdir()
    (agent_src / "coding_agent.py").write_text("# test stub\n")
    runner = RecordingRunner()
    adapter = CommonAgentAdapter(execution_backend=runner)

    adapter.run(
        agent_src,
        CommonAgentRequest(
            problem_statement="Fix the repository.",
            git_dir=repo,
            base_commit="HEAD",
            chat_history_file=outdir / "trajectory.log",
            outdir=outdir,
        ),
    )

    assert runner.cwd == repo
    assert runner.env["PYTHONPATH"] == ""
    assert runner.env["GODEL0_ROOT"] == ""
    assert runner.env["SLURM_SUBMIT_DIR"] == ""


def test_common_agent_chat_supports_repo_chain(monkeypatch):
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    fake_llm = SimpleNamespace(
        create_client=lambda model: (fake_client, model)
    )
    monkeypatch.setitem(sys.modules, "llm", fake_llm)
    monkeypatch.setenv("GODEL0_MODEL", "Qwen/Qwen3.6-35B-A3B")

    result = CommonAgentAdapter().chat("system", "contract", max_tokens=1234)

    assert result == '{"ok": true}'
    assert captured["max_tokens"] == 1234
    assert captured["model"] == "Qwen/Qwen3.6-35B-A3B"
    assert captured["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_common_agent_chat_prefers_explicit_model(monkeypatch):
    """P1-1: explicit model= must win over GODEL0_MODEL / default_model."""
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setitem(
        sys.modules,
        "llm",
        SimpleNamespace(create_client=lambda model: (fake_client, model)),
    )
    monkeypatch.setenv("GODEL0_MODEL", "env-model")

    adapter = CommonAgentAdapter(default_model="default-model")
    adapter.chat("s", "u", model="proposer-model-X")
    assert captured["model"] == "proposer-model-X"


def test_common_agent_chat_uses_default_model(monkeypatch):
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setitem(
        sys.modules,
        "llm",
        SimpleNamespace(create_client=lambda model: (fake_client, model)),
    )
    monkeypatch.delenv("GODEL0_MODEL", raising=False)

    CommonAgentAdapter(default_model="proposer-from-request").chat("s", "u")
    assert captured["model"] == "proposer-from-request"
