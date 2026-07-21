"""AnsibleRepoProfile: profile for the Ansible repository.

Encapsulates all Ansible-specific knowledge that was previously hardcoded:
  - source_roots: ["lib", "test/lib"]
  - contract_renderer: "ansible_playbook_cli"
  - public_entrypoints: ["ansible-playbook CLI"]
  - module_prefix: "ansible"
  - contract_scenario and contract_test_style for CLI-based contracts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .base import RepoProfile


@dataclass
class AnsibleProfile(RepoProfile):
    name: str = "ansible"
    source_roots: List[str] = field(default_factory=lambda: ["lib", "test/lib"])
    test_roots: List[str] = field(default_factory=lambda: ["test/lib"])
    contract_renderer: str = "ansible_playbook_cli"
    public_entrypoints: List[str] = field(default_factory=lambda: ["ansible-playbook CLI"])
    environment: str = "python"
    test_command: str = "python -m pytest test/lib/"
    contract_scenario: str = (
        "Use the repository's public ansible-playbook CLI with local "
        "connection and temporary YAML. Generate a playbook that exercises "
        "one target case and one nearby compatibility control."
    )
    contract_test_style: str = (
        "Use the repository's public ansible-playbook CLI with local "
        "connection and temporary YAML. Do not instantiate or mock "
        "Ansible internal classes."
    )
    require_expected_counts: bool = True
    module_prefix: str = "ansible"

    def matches(self, repo_id: str) -> bool:
        return "ansible" in (repo_id or "").lower()

    def module_path(self, module: str) -> str:
        """Map an ansible.* module to its file path under lib/."""
        if not module or not module.startswith("ansible"):
            return ""
        return "lib/" + module.replace(".", "/")


__all__ = ["AnsibleProfile"]
