"""Tests for the bounded repository bug-construction agent."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from experiment_adapters.repo_bug_agent_adapter import RepoBugAgentAdapter


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "provider.py").write_text('TOKEN = "current"\n', encoding="utf-8")
    (repo / "consumer.py").write_text(
        'from provider import TOKEN\nEXPECTED = "current"\n',
        encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial")
    return repo


def _response(*, content: str = "", tool_name: str = "", arguments=None):
    tool_calls = []
    if tool_name:
        tool_calls = [
            SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(
                    name=tool_name,
                    arguments=json.dumps(arguments or {}),
                ),
            )
        ]
    message = SimpleNamespace(content=content, tool_calls=tool_calls, role="assistant")
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=message,
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        ),
    )


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def _plan():
    return SimpleNamespace(
        model="Qwen/Test",
        constraints=SimpleNamespace(generation_timeout_sec=30),
    )


def _contract_test_command(repo: Path) -> str:
    (repo / "test_contract.py").write_text(
        "from consumer import EXPECTED\n"
        "from provider import TOKEN\n\n"
        "def test_provider_contract():\n"
        "    assert TOKEN == 'current'\n\n"
        "def test_consumer_contract():\n"
        "    assert EXPECTED == 'current'\n\n"
        "def test_unaffected_behavior():\n"
        "    assert 2 + 2 == 4\n",
        encoding="utf-8",
    )
    _git(repo, "add", "test_contract.py")
    _git(repo, "commit", "-m", "add contract tests")
    return (
        f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. {sys.executable} "
        "-m pytest -p no:cacheprovider test_contract.py -v"
    )


def _plan_with_validation(command: str):
    plan = _plan()
    plan.task_blueprint = {"validation_command": command}
    return plan


def test_repo_bug_agent_applies_multifile_patch_and_records_trace(tmp_path: Path):
    repo = _repo(tmp_path)
    patch = """\
diff --git a/provider.py b/provider.py
--- a/provider.py
+++ b/provider.py
@@ -1 +1 @@
-TOKEN = "current"
+TOKEN = "legacy"
diff --git a/consumer.py b/consumer.py
--- a/consumer.py
+++ b/consumer.py
@@ -1,2 +1,2 @@
 from provider import TOKEN
-EXPECTED = "current"
+EXPECTED = "legacy"
"""
    fake = _FakeClient(
        [
            _response(tool_name="apply_patch", arguments={"patch": patch}),
            _response(content="Mutation and targeted check complete."),
        ]
    )
    adapter = RepoBugAgentAdapter(
        client_factory=lambda _model: (fake, "Qwen/Test"),
        max_llm_calls=3,
    )
    output_dir = tmp_path / "agent_output"

    result = adapter.generate_repo_bug(
        str(repo),
        "Mutate the shared token contract.",
        _plan(),
        output_dir=str(output_dir),
    )

    assert "provider.py" in result
    assert "consumer.py" in result
    assert (output_dir / "model_patch.diff").read_text() == result
    assert 'TOKEN = "legacy"' in (repo / "provider.py").read_text()
    events = [
        json.loads(line)
        for line in (output_dir / "trajectory.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == [
        "start",
        "assistant",
        "tool",
        "assistant",
        "finish",
    ]
    assert events[-1]["changed_files"] == ["consumer.py", "provider.py"]
    first_request = fake.completions.requests[0]
    assert first_request["max_tokens"] == 1536
    assert first_request["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_repo_bug_agent_stops_after_two_prose_only_answers(tmp_path: Path):
    repo = _repo(tmp_path)
    fake = _FakeClient(
        [
            _response(content="I would inspect the repository."),
            _response(content="I still have not edited it."),
        ]
    )
    adapter = RepoBugAgentAdapter(
        client_factory=lambda _model: (fake, "fake-model"),
        max_llm_calls=10,
    )

    result = adapter.generate_repo_bug(str(repo), "Mutate it.", _plan())

    assert result == ""
    assert len(fake.completions.requests) == 2


def test_repo_bug_agent_denies_read_only_tools_after_edit_deadline(tmp_path: Path):
    repo = _repo(tmp_path)
    fake = _FakeClient(
        [
            _response(tool_name="bash", arguments={"command": "touch forbidden"}),
            _response(
                tool_name="edit_files",
                arguments={
                    "edits": [
                        {
                            "path": "provider.py",
                            "old_text": 'TOKEN = "current"',
                            "new_text": 'TOKEN = "legacy"',
                        },
                        {
                            "path": "consumer.py",
                            "old_text": 'EXPECTED = "current"',
                            "new_text": 'EXPECTED = "legacy"',
                        },
                    ]
                },
            ),
            _response(content="Done."),
        ]
    )
    adapter = RepoBugAgentAdapter(
        client_factory=lambda _model: (fake, "fake-model"),
        max_llm_calls=3,
        edit_deadline_call=1,
    )

    result = adapter.generate_repo_bug(str(repo), "Mutate it.", _plan())

    assert result
    assert not (repo / "forbidden").exists()
    second_request_messages = fake.completions.requests[1]["messages"]
    assert any(
        "Tool denied" in message.get("content", "")
        for message in second_request_messages
        if isinstance(message, dict)
    )
    first_request = fake.completions.requests[0]
    assert [tool["function"]["name"] for tool in first_request["tools"]] == [
        "edit_files"
    ]
    assert first_request["tool_choice"]["function"]["name"] == "edit_files"


def test_repo_bug_agent_rejects_test_file_patch(tmp_path: Path):
    repo = _repo(tmp_path)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_contract.py").write_text("assert True\n")
    adapter = RepoBugAgentAdapter(client_factory=lambda _model: (None, "fake"))
    patch = """\
diff --git a/tests/test_contract.py b/tests/test_contract.py
--- a/tests/test_contract.py
+++ b/tests/test_contract.py
@@ -1 +1 @@
-assert True
+assert False
"""

    result = adapter._apply_patch(patch, repo)

    assert result == "Error: patch path is not allowed: tests/test_contract.py"
    assert (repo / "tests" / "test_contract.py").read_text() == "assert True\n"


def test_edit_files_is_transactional_and_returns_real_context(tmp_path: Path):
    repo = _repo(tmp_path)
    adapter = RepoBugAgentAdapter(client_factory=lambda _model: (None, "fake"))

    result = adapter._edit_files(
        [
            {
                "path": "provider.py",
                "old_text": 'TOKEN = "current"',
                "new_text": 'TOKEN = "legacy"',
            },
            {
                "path": "consumer.py",
                "old_text": 'EXPECTED = "missing"',
                "new_text": 'EXPECTED = "legacy"',
            },
        ],
        repo,
    )

    assert "transaction rejected" in result
    assert 'EXPECTED = "current"' in result
    assert (repo / "provider.py").read_text() == 'TOKEN = "current"\n'


def test_edit_files_rejects_revision_that_leaves_single_file_diff(tmp_path: Path):
    repo = _repo(tmp_path)
    adapter = RepoBugAgentAdapter(client_factory=lambda _model: (None, "fake"))
    first = adapter._edit_files(
        [
            {
                "path": "provider.py",
                "old_text": 'TOKEN = "current"',
                "new_text": 'TOKEN = "legacy"',
            },
            {
                "path": "consumer.py",
                "old_text": 'EXPECTED = "current"',
                "new_text": 'EXPECTED = "legacy"',
            },
        ],
        repo,
    )
    assert first.startswith("Files edited transactionally.")

    revision = adapter._edit_files(
        [
            {
                "path": "provider.py",
                "old_text": 'TOKEN = "legacy"',
                "new_text": 'TOKEN = "current"',
            },
            {
                "path": "consumer.py",
                "old_text": 'EXPECTED = "legacy"',
                "new_text": 'EXPECTED = "older"',
            },
        ],
        repo,
    )

    assert "fewer than two changed files" in revision
    assert (repo / "provider.py").read_text() == 'TOKEN = "legacy"\n'
    assert 'EXPECTED = "legacy"' in (repo / "consumer.py").read_text()
    assert adapter._changed_files(repo) == ["consumer.py", "provider.py"]


def test_edit_files_ignores_noop_entry_during_multifile_revision(tmp_path: Path):
    repo = _repo(tmp_path)
    adapter = RepoBugAgentAdapter(client_factory=lambda _model: (None, "fake"))
    first = adapter._edit_files(
        [
            {
                "path": "provider.py",
                "old_text": 'TOKEN = "current"',
                "new_text": 'TOKEN = "legacy"',
            },
            {
                "path": "consumer.py",
                "old_text": 'EXPECTED = "current"',
                "new_text": 'EXPECTED = "legacy"',
            },
        ],
        repo,
    )
    assert first.startswith("Files edited transactionally.")

    revision = adapter._edit_files(
        [
            {
                "path": "provider.py",
                "old_text": 'TOKEN = "legacy"',
                "new_text": 'TOKEN = "legacy"',
            },
            {
                "path": "consumer.py",
                "old_text": 'EXPECTED = "legacy"',
                "new_text": 'EXPECTED = "older"',
            },
        ],
        repo,
    )

    assert "Ignored unchanged entries: provider.py" in revision
    assert 'TOKEN = "legacy"' in (repo / "provider.py").read_text()
    assert 'EXPECTED = "older"' in (repo / "consumer.py").read_text()
    assert adapter._changed_files(repo) == ["consumer.py", "provider.py"]


def test_edit_files_supports_sequential_snippets_in_the_same_file(tmp_path: Path):
    repo = _repo(tmp_path)
    adapter = RepoBugAgentAdapter(client_factory=lambda _model: (None, "fake"))

    result = adapter._edit_files(
        [
            {
                "path": "consumer.py",
                "old_text": "from provider import TOKEN",
                "new_text": "from provider import TOKEN as SOURCE_TOKEN",
            },
            {
                "path": "consumer.py",
                "old_text": 'EXPECTED = "current"',
                "new_text": "EXPECTED = SOURCE_TOKEN",
            },
            {
                "path": "provider.py",
                "old_text": 'TOKEN = "current"',
                "new_text": 'TOKEN = "legacy"',
            },
        ],
        repo,
    )

    assert result.startswith("Files edited transactionally.")
    assert (repo / "consumer.py").read_text() == (
        "from provider import TOKEN as SOURCE_TOKEN\nEXPECTED = SOURCE_TOKEN\n"
    )
    assert (repo / "provider.py").read_text() == 'TOKEN = "legacy"\n'


def test_bash_cannot_discard_existing_candidate_patch(tmp_path: Path):
    repo = _repo(tmp_path)
    adapter = RepoBugAgentAdapter(client_factory=lambda _model: (None, "fake"))
    edited = adapter._edit_files(
        [
            {
                "path": "provider.py",
                "old_text": 'TOKEN = "current"',
                "new_text": 'TOKEN = "legacy"',
            },
            {
                "path": "consumer.py",
                "old_text": 'EXPECTED = "current"',
                "new_text": 'EXPECTED = "legacy"',
            },
        ],
        repo,
    )
    assert edited.startswith("Files edited transactionally.")
    patch_before = adapter._repository_diff(repo)

    result = adapter._run_bash("git checkout -- .", repo)

    assert "BASH_WRITE_REVERTED" in result
    assert adapter._repository_diff(repo) == patch_before
    assert 'TOKEN = "legacy"' in (repo / "provider.py").read_text()


def test_revision_preserves_files_already_proven_to_cause_f2p(tmp_path: Path):
    adapter = RepoBugAgentAdapter(client_factory=lambda _model: (None, "fake"))
    probe = {
        "standalone_inert_files": ["consumer.py"],
        "only_file_f2p": {
            "provider.py": ["test_contract.py::test_provider_contract"],
            "consumer.py": [],
        },
    }
    repairing_input = {
        "edits": [
            {
                "path": "provider.py",
                "old_text": 'TOKEN = "legacy"',
                "new_text": 'TOKEN = "current"',
            },
            {
                "path": "consumer.py",
                "old_text": 'EXPECTED = "current"',
                "new_text": 'EXPECTED = "legacy"',
            },
        ]
    }

    rejected = adapter._revision_edit_restriction(
        "edit_files",
        repairing_input,
        probe,
    )

    assert "repair active bug files" in rejected
    preserving_input = dict(repairing_input)
    preserving_input["edits"] = [dict(edit) for edit in repairing_input["edits"]]
    preserving_input["edits"][0]["new_text"] = 'TOKEN = "legacy"'
    assert adapter._revision_edit_restriction(
        "edit_files",
        preserving_input,
        probe,
    ) == ""


def test_failed_edit_gets_one_correction_read_before_edit_is_forced_again(tmp_path: Path):
    repo = _repo(tmp_path)
    failed_edits = {
        "edits": [
            {
                "path": "provider.py",
                "old_text": 'TOKEN = "missing"',
                "new_text": 'TOKEN = "legacy"',
            },
            {
                "path": "consumer.py",
                "old_text": 'EXPECTED = "missing"',
                "new_text": 'EXPECTED = "legacy"',
            },
        ]
    }
    valid_edits = {
        "edits": [
            {
                "path": "provider.py",
                "old_text": 'TOKEN = "current"',
                "new_text": 'TOKEN = "legacy"',
            },
            {
                "path": "consumer.py",
                "old_text": 'EXPECTED = "current"',
                "new_text": 'EXPECTED = "legacy"',
            },
        ]
    }
    fake = _FakeClient(
        [
            _response(tool_name="edit_files", arguments=failed_edits),
            _response(tool_name="bash", arguments={"command": "sed -n '1,4p' provider.py"}),
            _response(tool_name="edit_files", arguments=valid_edits),
        ]
    )
    adapter = RepoBugAgentAdapter(
        client_factory=lambda _model: (fake, "fake-model"),
        max_llm_calls=3,
        edit_deadline_call=1,
    )

    result = adapter.generate_repo_bug(str(repo), "Mutate it.", _plan())

    assert result
    assert fake.completions.requests[0]["tool_choice"]["function"]["name"] == "edit_files"
    assert fake.completions.requests[1]["tool_choice"] == "auto"
    assert fake.completions.requests[2]["tool_choice"]["function"]["name"] == "edit_files"


def test_quality_probe_accepts_coupled_failures_and_identifies_inert_file(
    tmp_path: Path,
):
    repo = _repo(tmp_path)
    command = _contract_test_command(repo)
    adapter = RepoBugAgentAdapter(client_factory=lambda _model: (None, "fake"))
    clean_result = adapter._run_validation_command(command, repo, timeout=30)
    clean_passed, _ = adapter._pytest_statuses(clean_result)
    assert len(clean_passed) == 3

    result = adapter._edit_files(
        [
            {
                "path": "provider.py",
                "old_text": 'TOKEN = "current"',
                "new_text": 'TOKEN = "legacy"',
            },
            {
                "path": "consumer.py",
                "old_text": 'EXPECTED = "current"',
                "new_text": 'EXPECTED = "legacy"',
            },
        ],
        repo,
    )
    assert result.startswith("Files edited transactionally.")
    strict_probe = adapter._probe_candidate(repo, command, clean_passed)

    assert strict_probe["strict_ready"] is True
    assert strict_probe["standalone_inert_files"] == []
    assert strict_probe["single_repair_sufficient_files"] == []
    assert len(strict_probe["full_f2p"]) == 2
    assert strict_probe["full_p2p_count"] == 1
    assert adapter._probe_matches_patch(strict_probe, adapter._repository_diff(repo))

    assert adapter._discard_repository_diff(repo)
    result = adapter._edit_files(
        [
            {
                "path": "provider.py",
                "old_text": 'TOKEN = "current"',
                "new_text": 'TOKEN = "legacy"',
            },
            {
                "path": "consumer.py",
                "old_text": "from provider import TOKEN",
                "new_text": "from provider import TOKEN as TOKEN",
            },
        ],
        repo,
    )
    assert result.startswith("Files edited transactionally.")
    inert_probe = adapter._probe_candidate(repo, command, clean_passed)

    assert inert_probe["strict_ready"] is False
    assert inert_probe["standalone_inert_files"] == ["consumer.py"]
    assert inert_probe["single_repair_sufficient_files"] == ["provider.py"]


def test_non_strict_probe_is_rejected_and_worktree_is_cleaned(tmp_path: Path):
    repo = _repo(tmp_path)
    command = _contract_test_command(repo)
    edits = {
        "edits": [
            {
                "path": "provider.py",
                "old_text": 'TOKEN = "current"',
                "new_text": 'TOKEN = "legacy"',
            },
            {
                "path": "consumer.py",
                "old_text": "from provider import TOKEN",
                "new_text": "from provider import TOKEN as TOKEN",
            },
        ]
    }
    fake = _FakeClient(
        [
            _response(tool_name="edit_files", arguments=edits),
            _response(content="Done."),
            _response(content="Done."),
        ]
    )
    adapter = RepoBugAgentAdapter(
        client_factory=lambda _model: (fake, "fake-model"),
        max_llm_calls=3,
        edit_deadline_call=1,
        shell_timeout_sec=30,
    )
    output_dir = tmp_path / "rejected_output"

    result = adapter.generate_repo_bug(
        str(repo),
        "Mutate it.",
        _plan_with_validation(command),
        output_dir=str(output_dir),
    )

    assert result == ""
    assert adapter._repository_diff(repo) == ""
    assert (output_dir / "model_patch.diff").read_text() == ""
    events = [
        json.loads(line)
        for line in (output_dir / "trajectory.jsonl").read_text().splitlines()
    ]
    assert any(event["event"] == "quality_probe" for event in events)
    assert any(event["event"] == "quality_gate_rejected" for event in events)
