"""Toy repository fixtures for testing.

Creates a minimal repository with a clamp() function and tests.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


TOY_SOURCE = '''"""Toy module for testing."""


def clamp(x, low, high):
    """Clamp x to [low, high] range."""
    return max(low, min(x, high))


def divide(a, b):
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
'''


TOY_TESTS = '''"""Tests for toy module."""
from toy_module import clamp, divide


def test_clamp_normal():
    assert clamp(5, 0, 10) == 5


def test_clamp_lower_boundary():
    assert clamp(-5, 0, 10) == 0


def test_clamp_upper_boundary():
    assert clamp(15, 0, 10) == 10


def test_divide_normal():
    assert divide(10, 2) == 5.0


def test_divide_by_zero():
    try:
        divide(10, 0)
        assert False, "Should have raised"
    except ValueError:
        pass
'''


def create_toy_repo(parent_dir: Path) -> Path:
    """Create a toy repository with a clamp function and tests.

    Returns the path to the repo. The repo is a git repo with one commit.
    """
    repo_dir = parent_dir / "toy_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    (repo_dir / "toy_module.py").write_text(TOY_SOURCE)
    (repo_dir / "test_toy.py").write_text(TOY_TESTS)
    (repo_dir / "conftest.py").write_text("import sys\nsys.path.insert(0, '.')\n")

    subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(repo_dir), capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(repo_dir), capture_output=True,
    )
    subprocess.run(["git", "add", "-A"], cwd=str(repo_dir), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial toy repo"],
        cwd=str(repo_dir), capture_output=True,
    )

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_dir), capture_output=True, text=True,
    )
    return repo_dir


def get_toy_repo_commit(repo_dir: Path) -> str:
    """Get the HEAD commit of the toy repo."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_dir), capture_output=True, text=True,
    )
    return result.stdout.strip()


def run_toy_tests(repo_dir: Path) -> dict:
    """Run tests in the toy repo and return results."""
    result = subprocess.run(
        ["python", "-m", "pytest", "-q", "--tb=short"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
    }
