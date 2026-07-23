"""Existing-passing-test contract admission for RepoChain."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from swesmith.repo_chain import RepoChainGenerator


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_chain_plan_rejection_allows_empty_tests_when_not_required():
    gen = RepoChainGenerator(agent_adapter=SimpleNamespace())
    payload = {
        "chain_plan": {
            "root_invariant": "hosts keep unique names",
            "entrypoint": "Host.__init__",
            "endpoint": "inventory lookup",
            "mutation_sites": [
                {
                    "file": "lib/ansible/inventory/host.py",
                    "symbol": "Host",
                    "role": "identity",
                    "change": "drop name normalization",
                },
                {
                    "file": "lib/ansible/inventory/group.py",
                    "symbol": "Group",
                    "role": "carrier",
                    "change": "skip host registration",
                },
            ],
        },
        "tests": [],
        "contract_source": "existing_tests",
    }
    context = [
        "lib/ansible/inventory/host.py",
        "lib/ansible/inventory/group.py",
    ]
    assert (
        gen._chain_plan_rejection(
            payload,
            context_files=context,
            min_files=2,
            max_files=6,
            min_sites=2,
            max_sites=8,
            require_generated_tests=False,
        )
        == ""
    )
    assert gen._chain_plan_rejection(
        payload,
        context_files=context,
        min_files=2,
        max_files=6,
        min_sites=2,
        max_sites=8,
        require_generated_tests=True,
    )


def test_taxonomy_for_existing_tests_uses_file_nodeids(tmp_path: Path):
    gen = RepoChainGenerator(agent_adapter=SimpleNamespace())
    content = (
        "def test_host_name():\n"
        "    assert True\n"
        "\n"
        "class TestGroup:\n"
        "    def test_add(self):\n"
        "        assert True\n"
    )
    payload = {
        "contract_source": "existing_tests",
        "tests": [{"path": "test/units/inventory/test_host.py", "content": content}],
    }
    tax = gen._contract_test_taxonomy(
        payload, ["test/units/inventory/test_host.py"]
    )
    assert "test/units/inventory/test_host.py::test_host_name" in tax["FAIL_TO_PASS"]
    assert "test/units/inventory/test_host.py::TestGroup::test_add" in tax["FAIL_TO_PASS"]
    assert tax["PASS_TO_PASS"] == []


def test_select_passing_existing_tests_filters_failures(tmp_path: Path, monkeypatch):
    _write(
        tmp_path / "lib/ansible/inventory/host.py",
        "class Host:\n    pass\n",
    )
    _write(
        tmp_path / "test/units/inventory/test_host.py",
        "def test_host():\n    assert True\n",
    )
    _write(
        tmp_path / "test/units/inventory/test_broken.py",
        "def test_broken():\n    assert False\n",
    )

    gen = RepoChainGenerator(agent_adapter=SimpleNamespace())
    gen._current_repo_id = "ansible"

    def fake_run(workspace, command, timeout):
        code = 0 if "test_host.py" in command and "test_broken" not in command else 1
        return SimpleNamespace(returncode=code, stdout="", stderr="")

    monkeypatch.setattr(gen, "_run_command", fake_run)
    plan = SimpleNamespace(task_blueprint={})
    repo_spec = SimpleNamespace(test_command="pytest")
    selected, err, result = gen._select_passing_existing_tests(
        plan=plan,
        repo_spec=repo_spec,
        workspace=str(tmp_path),
        production_files=["lib/ansible/inventory/host.py"],
        timeout_sec=60,
        budget=2,
    )
    assert err == ""
    assert selected == ["test/units/inventory/test_host.py"]
    assert result is not None
    assert result.returncode == 0


def test_format_existing_contract_tests_includes_paths(tmp_path: Path):
    _write(
        tmp_path / "test/units/inventory/test_host.py",
        "from ansible.inventory.host import Host\n\ndef test_host():\n    assert Host\n",
    )
    gen = RepoChainGenerator(agent_adapter=SimpleNamespace())
    text = gen._format_existing_contract_tests(
        tmp_path, ["test/units/inventory/test_host.py"]
    )
    assert "EXISTING CONTRACT TEST" in text
    assert "test/units/inventory/test_host.py" in text
    assert "Host" in text
