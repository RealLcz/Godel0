"""RepoProfileRegistry: selects the right RepoProfile for a repo_id.

The registry tries each registered profile's ``matches`` method and falls back
to the base ``RepoProfile`` for unknown repos.
"""

from __future__ import annotations

from typing import List

from .ansible import AnsibleProfile
from .base import RepoProfile


class RepoProfileRegistry:
    """Registry of repository profiles."""

    def __init__(self) -> None:
        self._profiles: List[RepoProfile] = [
            AnsibleProfile(),
        ]
        self._default = RepoProfile()

    def register(self, profile: RepoProfile) -> None:
        self._profiles.append(profile)

    def get(self, repo_id: str) -> RepoProfile:
        """Return the profile matching repo_id, or the default profile."""
        for profile in self._profiles:
            if profile.matches(repo_id):
                return profile
        return self._default


# Module-level singleton for convenience.
_REGISTRY = RepoProfileRegistry()


def get_profile(repo_id: str) -> RepoProfile:
    """Get the RepoProfile for a repo_id (module-level convenience)."""
    return _REGISTRY.get(repo_id)


__all__ = ["RepoProfileRegistry", "get_profile"]
