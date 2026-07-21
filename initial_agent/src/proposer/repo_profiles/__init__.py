"""RepoProfile: repository-specific abstractions for RepoChain.

Removes hardcoded ``if repo_id == "ansible"`` checks by delegating to a
profile selected via RepoProfileRegistry.get(repo_id).
"""

from .base import RepoProfile
from .ansible import AnsibleProfile
from .registry import RepoProfileRegistry, get_profile

__all__ = ["RepoProfile", "AnsibleProfile", "RepoProfileRegistry", "get_profile"]
