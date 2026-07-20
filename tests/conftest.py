"""Shared test fixtures."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tests.fixtures.toy_repo import create_toy_repo, get_toy_repo_commit


@pytest.fixture
def tmp_workspace(tmp_path):
    """Provide a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def toy_repo(tmp_path):
    """Create a toy repository for testing."""
    repo_dir = create_toy_repo(tmp_path)
    commit = get_toy_repo_commit(repo_dir)
    return {
        "path": repo_dir,
        "commit": commit,
        "source_file": repo_dir / "toy_module.py",
        "test_file": repo_dir / "test_toy.py",
    }


@pytest.fixture
def task_store(tmp_path):
    """Create a TaskStore in a temp directory."""
    from godel0.tasks.store import TaskStore
    return TaskStore(tmp_path / "task_store")


@pytest.fixture
def node_archive(tmp_path):
    """Create a NodeArchive in a temp directory."""
    from godel0.tree.archive import NodeArchive
    return NodeArchive(tmp_path / "archive.jsonl")


@pytest.fixture
def fake_llm():
    """Create a FakeLLMClient."""
    from tests.fixtures.fake_llm import FakeLLMClient
    return FakeLLMClient(responses=[
        '{"primary_root_cause": "test cause", "problem_statement": "test problem"}'
    ])
