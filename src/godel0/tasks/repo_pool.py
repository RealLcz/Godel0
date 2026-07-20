"""Repository pool management.

A RepoPool holds specifications for all base repositories that the Proposer
can target for bug generation. Each RepoSpec points to a checked-out git
repository at a specific base_commit, with a known test command.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class RepoSpec(BaseModel):
    """Specification for a repository in the pool.

    This is the canonical repo spec used across the entire control layer
    and passed to the Proposer and SWESmithEngine.
    """

    repo_id: str
    base_commit: str
    path: str
    test_command: str = "pytest -q"
    install_command: str = "pip install -e ."
    timeout_sec: int = 120
    test_parser: str = "pytest"
    image: Optional[str] = None
    source_dirs: List[str] = Field(default_factory=lambda: ["src", "."])

    @property
    def repo_path(self) -> Path:
        """Alias for path, used by the SWESmithEngine."""
        return Path(self.path)

    @property
    def repo_dir(self) -> Path:
        """Alias for path, used by the CodeLocator."""
        return Path(self.path)

    def to_engine_dict(self) -> Dict[str, Any]:
        """Convert to dict for the SWESmithEngine's RepoSpec."""
        return {
            "repo_id": self.repo_id,
            "repo_path": str(self.path),
            "base_commit": self.base_commit,
            "test_command": self.test_command,
            "source_dirs": self.source_dirs,
        }

    def to_proposer_dict(self) -> Dict[str, Any]:
        """Convert to dict for the proposer's RepoSpec."""
        return {
            "repo_id": self.repo_id,
            "repo_dir": str(self.path),
            "base_commit": self.base_commit,
        }


class RepoPool:
    """Manages a pool of repositories for task generation.

    The pool directory has this layout::

        repo_pool/
        ├── repos.jsonl          # RepoSpec entries, one per line
        ├── toy_repo/            # checked-out repository
        │   ├── toy_module.py
        │   ├── test_toy.py
        │   └── .git/
        └── another_repo/
            └── ...

    To add a repository to the pool:
        1. Clone/checkout the repo into repo_pool/<repo_id>/
        2. Call RepoPool.add(RepoSpec(repo_id=..., base_commit=..., path=...))
    """

    def __init__(self, pool_dir: Path):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self._repos: Dict[str, RepoSpec] = {}
        self._load()

    def _load(self) -> None:
        """Load repo specs from repos.jsonl."""
        repos_file = self.pool_dir / "repos.jsonl"
        if not repos_file.exists():
            return
        with open(repos_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                spec = RepoSpec(**data)
                self._repos[spec.repo_id] = spec

    def get(self, repo_id: str) -> Optional[RepoSpec]:
        """Get a repo spec by ID."""
        return self._repos.get(repo_id)

    def all_repos(self) -> List[RepoSpec]:
        """Get all repo specs."""
        return list(self._repos.values())

    def add(self, spec: RepoSpec) -> None:
        """Add a repo to the pool and persist."""
        self._repos[spec.repo_id] = spec
        self._save()

    def remove(self, repo_id: str) -> None:
        """Remove a repo from the pool."""
        if repo_id in self._repos:
            del self._repos[repo_id]
            self._save()

    def exists(self, repo_id: str) -> bool:
        """Check if a repo exists in the pool."""
        return repo_id in self._repos

    def _save(self) -> None:
        """Save repo specs to repos.jsonl."""
        repos_file = self.pool_dir / "repos.jsonl"
        with open(repos_file, "w") as f:
            for spec in self._repos.values():
                f.write(spec.model_dump_json() + "\n")

    def get_for_proposer(self) -> List[Dict[str, Any]]:
        """Return all repo specs as dicts suitable for the proposer."""
        return [spec.to_proposer_dict() for spec in self._repos.values()]

    def materialize_repo(self, repo_id: str, destination: Path) -> Path:
        """Copy a repo from the pool to a destination for isolated use."""
        import shutil
        spec = self.get(repo_id)
        if spec is None:
            raise ValueError(f"Repo '{repo_id}' not found in pool")
        dest = Path(destination)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(
            spec.repo_path,
            dest,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".git", "*.egg-info", ".pytest_cache"
            ),
        )
        return dest
