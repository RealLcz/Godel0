"""Tests for existing-test grounding used by RepoChain contract generation."""

from __future__ import annotations

from pathlib import Path

from swesmith.test_grounding import (
    build_existing_test_grounding,
    excerpt_existing_test,
    production_to_unit_subdir,
    retrieve_nearby_existing_tests,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_production_to_unit_subdir_ansible_layout():
    assert (
        production_to_unit_subdir("lib/ansible/inventory/host.py") == "inventory"
    )
    assert (
        production_to_unit_subdir("lib/ansible/executor/task_executor.py")
        == "executor"
    )


def test_retrieve_nearby_existing_tests_prefers_mirrored_units(tmp_path: Path):
    _write(
        tmp_path / "lib/ansible/inventory/host.py",
        "class Host:\n    pass\n",
    )
    _write(
        tmp_path / "test/units/inventory/test_host.py",
        "from ansible.inventory.host import Host\n\ndef test_host():\n    assert Host\n",
    )
    _write(
        tmp_path / "test/units/cli/test_cli.py",
        "def test_cli():\n    assert True\n",
    )
    found = retrieve_nearby_existing_tests(
        tmp_path,
        ["lib/ansible/inventory/host.py"],
        budget=2,
    )
    assert found
    assert found[0] == "test/units/inventory/test_host.py"
    assert "test/units/cli/test_cli.py" not in found


def test_build_existing_test_grounding_marks_grounding_only(tmp_path: Path):
    _write(
        tmp_path / "lib/ansible/vars/manager.py",
        "class VariableManager:\n    pass\n",
    )
    _write(
        tmp_path / "test/units/vars/test_variable_manager.py",
        '''
from ansible.vars.manager import VariableManager

def test_variable_manager_construct():
    vm = VariableManager()
    assert vm is not None
'''.lstrip(),
    )
    text, paths = build_existing_test_grounding(
        tmp_path,
        ["lib/ansible/vars/manager.py"],
        budget=2,
    )
    assert paths == ["test/units/vars/test_variable_manager.py"]
    assert "grounding" in text.lower()
    assert "EXISTING TEST" in text
    assert "VariableManager" in text
    assert "Do NOT copy" in text or "newly named" in text


def test_excerpt_keeps_imports_and_tests():
    source = '''
# copyright header
from ansible.inventory.host import Host

class TestHost:
    def setUp(self):
        self.host = Host("a")

    def test_equality(self):
        assert self.host.name == "a"
'''
    excerpt = excerpt_existing_test(source)
    assert "from ansible.inventory.host import Host" in excerpt
    assert "def setUp" in excerpt or "def test_equality" in excerpt
