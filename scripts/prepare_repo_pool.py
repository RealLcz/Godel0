#!/usr/bin/env python3
"""Prepare the repository pool for task generation.

This script sets up the repo_pool/ directory with base repositories that
the Proposer can target for bug generation.

Usage:
    # Toy repo only (for quick testing)
    python scripts/prepare_repo_pool.py --pool-dir ./repo_pool --toy

    # Register Ansible (requires existing clone at repo_pool/ansible)
    python scripts/prepare_repo_pool.py --pool-dir ./repo_pool --ansible

    # Both
    python scripts/prepare_repo_pool.py --pool-dir ./repo_pool --toy --ansible

    # Clone an external repo
    python scripts/prepare_repo_pool.py --pool-dir ./repo_pool --clone https://github.com/psf/requests.git
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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


def fibonacci(n):
    """Compute the n-th Fibonacci number."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
'''

TOY_TESTS = '''"""Tests for toy module."""
from toy_module import clamp, divide, fibonacci


def test_clamp_normal():
    assert clamp(5, 0, 10) == 5


def test_clamp_lower():
    assert clamp(-5, 0, 10) == 0


def test_clamp_upper():
    assert clamp(15, 0, 10) == 10


def test_divide_normal():
    assert divide(10, 2) == 5.0


def test_divide_by_zero():
    try:
        divide(10, 0)
        assert False, "Should have raised"
    except ValueError:
        pass


def test_fibonacci_base():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1


def test_fibonacci_recursive():
    assert fibonacci(10) == 55
'''


def setup_toy_repo(pool_dir: Path) -> dict:
    """Create a toy repository in the pool directory."""
    repo_id = "toy_repo"
    repo_path = pool_dir / repo_id
    repo_path.mkdir(parents=True, exist_ok=True)

    (repo_path / "toy_module.py").write_text(TOY_SOURCE)
    (repo_path / "test_toy.py").write_text(TOY_TESTS)
    (repo_path / "conftest.py").write_text("import sys\nsys.path.insert(0, '.')\n")

    if not (repo_path / ".git").exists():
        subprocess.run(["git", "init"], cwd=str(repo_path), capture_output=True)
        subprocess.run(["git", "config", "user.email", "godel0@test.com"], cwd=str(repo_path), capture_output=True)
        subprocess.run(["git", "config", "user.name", "Godel0"], cwd=str(repo_path), capture_output=True)
        subprocess.run(["git", "add", "-A"], cwd=str(repo_path), capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial toy repo for Godel0"], cwd=str(repo_path), capture_output=True)

    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_path), capture_output=True, text=True)
    base_commit = result.stdout.strip()

    spec = {
        "repo_id": repo_id,
        "base_commit": base_commit,
        "path": str(repo_path),
        "test_command": "python -m pytest test_toy.py -v",
        "install_command": "pip install -e .",
        "timeout_sec": 60,
        "test_parser": "pytest",
        "image": None,
        "source_dirs": ["."],
    }

    print(f"Toy repo created at {repo_path}")
    print(f"  base_commit: {base_commit}")

    tests_result = subprocess.run(spec["test_command"], shell=True, cwd=str(repo_path), capture_output=True, text=True, timeout=30)
    print(f"  Tests: {'PASS' if tests_result.returncode == 0 else 'FAIL'}")

    return spec


def register_ansible(pool_dir: Path) -> dict:
    """Register an existing Ansible clone in the repo pool.

    The Ansible repo should already be cloned at pool_dir/ansible.
    This function:
    1. Verifies the repo exists and is a git repo.
    2. Gets the HEAD commit SHA.
    3. Creates a RepoSpec with the correct test command.
    4. Returns the spec for registration.
    """
    repo_id = "ansible"
    repo_path = pool_dir / repo_id

    if not repo_path.exists():
        print(f"ERROR: Ansible repo not found at {repo_path}")
        print("Please clone it first:")
        print(f"  cd {pool_dir} && git clone --depth 1 -b stable-2.18 https://github.com/ansible/ansible.git")
        sys.exit(1)

    if not (repo_path / ".git").exists():
        print(f"ERROR: {repo_path} is not a git repository")
        sys.exit(1)

    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_path), capture_output=True, text=True)
    base_commit = result.stdout.strip()

    # Check which Python version works
    python_bin = "python3.11"
    try:
        subprocess.run([python_bin, "--version"], capture_output=True, check=True)
    except Exception:
        python_bin = "python3"

    # Ansible uses PYTHONPATH=lib:test/lib for imports
    # --rootdir=. prevents pytest from picking up parent pytest.ini
    # No -q flag: it conflicts with -v and suppresses test names needed for F2P parsing
    test_command = f"PYTHONPATH=lib:test/lib {python_bin} -m pytest -p no:cacheprovider --rootdir=."

    spec = {
        "repo_id": repo_id,
        "base_commit": base_commit,
        "path": str(repo_path),
        "test_command": test_command,
        "install_command": f"pip install -e .",
        "timeout_sec": 120,
        "test_parser": "pytest",
        "image": None,
        "source_dirs": ["lib", "test/lib"],
    }

    print(f"Ansible repo registered at {repo_path}")
    print(f"  base_commit: {base_commit}")
    print(f"  test_command: {test_command}")
    print(f"  source_dirs: {spec['source_dirs']}")

    # Verify tests work
    test_file = "test/units/module_utils/common/test_dict_transformations.py"
    verify_cmd = f"PYTHONPATH=lib:test/lib {python_bin} -m pytest {test_file} -q"
    verify_result = subprocess.run(verify_cmd, shell=True, cwd=str(repo_path), capture_output=True, text=True, timeout=30)
    print(f"  Verify test ({test_file}): {'PASS' if verify_result.returncode == 0 else 'FAIL'}")

    return spec


def add_external_repo(pool_dir: Path, repo_url: str, repo_id: str = "") -> dict:
    """Clone an external repository into the pool."""
    if not repo_id:
        repo_id = repo_url.rstrip("/").split("/")[-1]
        if repo_id.endswith(".git"):
            repo_id = repo_id[:-4]

    repo_path = pool_dir / repo_id

    if not repo_path.exists():
        subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True)
    else:
        print(f"Repo {repo_id} already exists at {repo_path}")

    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_path), capture_output=True, text=True)
    base_commit = result.stdout.strip()

    spec = {
        "repo_id": repo_id,
        "base_commit": base_commit,
        "path": str(repo_path),
        "test_command": "pytest -q",
        "install_command": "pip install -e .",
        "timeout_sec": 120,
        "test_parser": "pytest",
        "image": None,
        "source_dirs": ["src", "."],
    }

    print(f"External repo {repo_id} added at {repo_path}")
    print(f"  base_commit: {base_commit}")

    return spec


def save_specs(pool_dir: Path, specs: list[dict]) -> None:
    """Save repo specs to repos.jsonl, merging with any existing specs."""
    repos_file = pool_dir / "repos.jsonl"

    # Load existing specs
    existing = {}
    if repos_file.exists():
        with open(repos_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    existing[data["repo_id"]] = data

    # Merge new specs
    for spec in specs:
        existing[spec["repo_id"]] = spec

    # Save all
    with open(repos_file, "w") as f:
        for spec in existing.values():
            f.write(json.dumps(spec) + "\n")

    print(f"\nSaved {len(specs)} repo specs to {repos_file}")
    print(f"Total repos in pool: {len(existing)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare the repository pool for task generation")
    parser.add_argument("--pool-dir", default="./repo_pool", help="Path to the repository pool directory")
    parser.add_argument("--toy", action="store_true", help="Create a toy repository")
    parser.add_argument("--ansible", action="store_true", help="Register existing Ansible clone in the pool")
    parser.add_argument("--clone", action="append", default=[], help="Clone an external repo (URL)")
    parser.add_argument("--repo-id", action="append", default=[], help="Repo ID for the corresponding --clone")
    args = parser.parse_args()

    pool_dir = Path(args.pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)

    specs = []

    if args.toy:
        print("=== Setting up toy repository ===")
        specs.append(setup_toy_repo(pool_dir))

    if args.ansible:
        print("\n=== Registering Ansible ===")
        specs.append(register_ansible(pool_dir))

    for i, url in enumerate(args.clone):
        repo_id = args.repo_id[i] if i < len(args.repo_id) else ""
        print(f"\n=== Cloning {url} ===")
        specs.append(add_external_repo(pool_dir, url, repo_id))

    if not specs:
        print("No repos to add. Use --toy, --ansible, or --clone <url>.")
        return

    save_specs(pool_dir, specs)
    print(f"\nRepo pool ready at {pool_dir}")
    print(f"  Repos: {[s['repo_id'] for s in specs]}")


if __name__ == "__main__":
    main()
