"""Tests for RepoProfile and the registry."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "initial_agent" / "src"))

from proposer.repo_profiles import (
    AnsibleProfile,
    RepoProfile,
    RepoProfileRegistry,
    get_profile,
)


class TestRepoProfile:
    def test_default_profile_for_unknown_repo(self):
        profile = get_profile("unknown-repo")
        assert profile.name == "default"
        assert profile.source_roots == ["."]

    def test_ansible_profile_matches(self):
        profile = get_profile("ansible")
        assert profile.name == "ansible"
        assert profile.source_roots == ["lib", "test/lib"]
        assert profile.contract_renderer == "ansible_playbook_cli"

    def test_ansible_profile_matches_case_insensitive(self):
        profile = get_profile("Ansible-core")
        assert profile.name == "ansible"

    def test_ansible_module_path(self):
        profile = AnsibleProfile()
        assert profile.module_path("ansible.plugins.loader") == "lib/ansible/plugins/loader"

    def test_default_module_path(self):
        profile = RepoProfile()
        assert profile.module_path("mypackage.mymodule") == "mypackage/mymodule.py"

    def test_registry_custom_profile(self):
        registry = RepoProfileRegistry()

        @dataclass
        class CustomProfile(RepoProfile):
            name: str = "custom"

            def matches(self, repo_id: str) -> bool:
                return repo_id == "custom-repo"

        registry.register(CustomProfile())
        profile = registry.get("custom-repo")
        assert profile.name == "custom"


from dataclasses import dataclass
