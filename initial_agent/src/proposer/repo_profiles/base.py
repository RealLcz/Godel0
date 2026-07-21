"""RepoProfile: repository-specific abstractions for the RepoChain workflow.

A RepoProfile encapsulates everything RepoChain needs to know about a specific
repository family: source roots, test roots, contract renderer, public
entrypoints, environment, test command, contract scenario/style, module path
mapping, and test templates. This removes hardcoded ``if repo_id == "ansible"``
checks scattered across the proposer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RepoProfile:
    """Base repository profile.

    Subclasses override fields to customize behavior for a repo family.
    The registry selects a profile by repo_id (fuzzy match).
    """
    name: str = "default"
    source_roots: List[str] = field(default_factory=lambda: ["."])
    test_roots: List[str] = field(default_factory=lambda: ["test", "tests"])
    contract_renderer: str = "existing_test"
    public_entrypoints: List[str] = field(default_factory=list)
    environment: str = "python"
    test_command: str = "pytest -q"
    contract_scenario: str = ""
    contract_test_style: str = ""
    test_template: str = ""
    require_expected_counts: bool = False
    module_prefix: str = ""

    def matches(self, repo_id: str) -> bool:
        """Return True if this profile applies to the given repo_id."""
        return False

    def module_path(self, module: str) -> str:
        """Map a Python module name to a file path relative to repo root."""
        if not module:
            return ""
        return module.replace(".", "/") + ".py"

    def test_command_for_files(self, test_files: List[str]) -> str:
        """Build a test command scoped to specific test files."""
        if not test_files:
            return self.test_command
        return self.test_command + " " + " ".join(test_files)


__all__ = ["RepoProfile"]
