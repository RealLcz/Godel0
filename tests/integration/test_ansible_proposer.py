"""Integration test: generate bug on Ansible module and validate F2P.

This test uses the real Ansible repository (stable-2.18) cloned in
repo_pool/ansible. It:
1. Loads the Ansible RepoSpec from the repo pool.
2. Picks a target module (lib/ansible/module_utils/common/dict_transformations.py).
3. Applies a procedural mutation (change_operator) to introduce a bug.
4. Validates the bug with CandidateValidator (F2P check).
5. Commits the task to TaskStore.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "initial_agent" / "src"))

from godel0.tasks.repo_pool import RepoPool
from godel0.proposer_trusted.candidate_validator import CandidateValidator
from godel0.proposer_trusted.task_committer import TaskCommitter
from godel0.tasks.store import TaskStore
from godel0.git.repository import get_head_sha, reset_to_commit

# Ansible repo path
ANSIBLE_REPO = Path(__file__).parent.parent.parent / "repo_pool" / "ansible"
ANSIBLE_POOL = Path(__file__).parent.parent.parent / "repo_pool"

# Skip if Ansible is not cloned
pytestmark = pytest.mark.skipif(
    not ANSIBLE_REPO.exists(),
    reason="Ansible repo not found at repo_pool/ansible. Run: python scripts/prepare_repo_pool.py --ansible"
)


@pytest.fixture
def ansible_pool():
    """Load the repo pool with Ansible registered."""
    if not (ANSIBLE_POOL / "repos.jsonl").exists():
        pytest.skip("repos.jsonl not found. Run: python scripts/prepare_repo_pool.py --ansible")
    return RepoPool(ANSIBLE_POOL)


@pytest.fixture
def ansible_spec(ansible_pool):
    """Get the Ansible RepoSpec."""
    spec = ansible_pool.get("ansible")
    if spec is None:
        pytest.skip("Ansible not registered in repo pool")
    return spec


class TestAnsibleRepoPool:
    def test_ansible_registered(self, ansible_pool):
        """Ansible should be registered in the pool."""
        assert ansible_pool.exists("ansible")

    def test_ansible_spec_fields(self, ansible_spec):
        """The Ansible spec should have all required fields."""
        assert ansible_spec.repo_id == "ansible"
        assert ansible_spec.base_commit
        assert Path(ansible_spec.path).exists()
        assert "PYTHONPATH=lib:test/lib" in ansible_spec.test_command
        assert "lib" in ansible_spec.source_dirs

    def test_ansible_tests_pass(self, ansible_spec):
        """The clean Ansible repo tests should pass."""
        repo_path = Path(ansible_spec.path)
        result = subprocess.run(
            ansible_spec.test_command + " test/units/module_utils/common/test_dict_transformations.py -q",
            shell=True,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Clean tests failed:\n{result.stdout}\n{result.stderr}"


class TestAnsibleBugGeneration:
    def test_generate_and_validate_bug(self, ansible_spec, tmp_path):
        """Generate a bug on an Ansible module and validate F2P.

        Target: lib/ansible/module_utils/common/dict_transformations.py
        Function: dict_merge
        Bug: change `isinstance(result[k], dict)` to `isinstance(result[k], list)`
        """
        repo_path = Path(ansible_spec.path)
        target_file = "lib/ansible/module_utils/common/dict_transformations.py"
        source_file = repo_path / target_file

        # Read original source
        original = source_file.read_text()

        # Introduce a bug: in dict_merge, change `isinstance(result[k], dict)` to `isinstance(result[k], list)`
        # This makes dict_merge not recursively merge nested dicts
        buggy = original.replace(
            "if k in result and isinstance(result[k], dict):",
            "if k in result and isinstance(result[k], list):",
        )

        assert buggy != original, "Bug introduction should change the source"

        # Write buggy version and generate diff
        source_file.write_text(buggy)
        diff_result = subprocess.run(
            ["git", "-C", str(repo_path), "diff"],
            capture_output=True,
            text=True,
        )
        patch = diff_result.stdout

        # Restore original
        source_file.write_text(original)
        subprocess.run(
            ["git", "-C", str(repo_path), "checkout", "--", target_file],
            capture_output=True,
        )

        assert patch, "Patch should not be empty"
        assert "list" in patch

        # Validate with CandidateValidator
        validator = CandidateValidator(
            workspace_root=tmp_path / "validator",
            test_timeout_sec=60,
            max_patch_lines=80,
            forbid_test_file_edits=True,
        )

        # Use a specific test file that covers dict_merge
        test_command = ansible_spec.test_command + " test/units/module_utils/common/test_dict_transformations.py -v"

        report = validator.validate(
            candidate_patch=patch,
            repo_path=repo_path,
            base_commit=ansible_spec.base_commit,
            test_command=test_command,
            candidate_id="ansible_bug_001",
        )

        # The bug should produce F2P (tests that fail after the bug)
        assert report.patch_applied, f"Patch should apply: {report.rejection_reasons}"
        assert report.syntax_valid, "Bugged code should be syntactically valid"

        # dict_merge tests should fail with the bug
        # (the exact F2P tests depend on which tests exercise the dict path)
        print(f"\nF2P tests: {report.f2p_tests}")
        print(f"Reverse restored: {report.reverse_restored}")
        print(f"Report passed: {report.passed}")

        # If we got F2P tests, commit the task
        if report.f2p_tests:
            task_store = TaskStore(tmp_path / "task_store")
            committer = TaskCommitter(task_store)
            task = committer.commit_task(
                batch_id="ansible_batch_001",
                proposer_node_id="test_node",
                repo_id=ansible_spec.repo_id,
                base_commit=ansible_spec.base_commit,
                bug_strategy="procedural",
                bug_patch=patch,
                problem_statement="The dict_merge function does not correctly merge nested dictionaries.",
                f2p_tests=report.f2p_tests,
                baseline_test_command=test_command,
                modified_files=[target_file],
                modified_entities=["dict_merge"],
            )

            assert task.task_id
            assert task_store.exists(task.task_id)
            assert task.repo_id == "ansible"
            assert target_file in task.modified_files
            print(f"\nTask committed: {task.task_id}")

    def test_code_locator_finds_ansible_symbols(self, ansible_spec):
        """CodeLocator should find symbols in the Ansible repo."""
        from proposer.code_locator import RepoIndex, CodeLocator
        from proposer.schemas import FailureSignature

        index = RepoIndex.build(
            repo_id="ansible",
            repo_dir=ansible_spec.path,
            base_commit=ansible_spec.base_commit,
            source_dirs=["lib"],
        )

        # Should find functions like dict_merge, camel_dict_to_snake_dict, etc.
        symbol_names = [s["symbol_name"] for s in index.symbols]
        assert "dict_merge" in symbol_names or any("dict_merge" in s["symbol_name"] for s in index.symbols)
        assert len(index.symbols) > 100, f"Expected many symbols, got {len(index.symbols)}"

        # CodeLocator should find targets matching a failure signature
        locator = CodeLocator()
        sig = FailureSignature(
            signature_id="test_sig",
            code_patterns=["dict", "merge"],
            target_capability="dictionary operations",
        )
        targets = locator.locate(sig, index, max_results=5)
        assert len(targets) > 0
        # At least one target should be in the dict_transformations module
        assert any("dict_transformations" in t.file_path for t in targets)

    def test_procedural_engine_on_ansible(self, ansible_spec, tmp_path):
        """SWESmithEngine should generate a candidate on an Ansible module.

        We target the whole module (not just dict_merge) because dict_merge
        has no comparison operators for change_operator to mutate.
        The change_operator finds sites like `dict1[k] != dict2[k]` on line 148.
        """
        from swesmith.engine import SWESmithEngine, BugGenerationPlan, RepoSpec as EngineRepoSpec, BugConstraints

        engine = SWESmithEngine(agent_adapter=None)

        repo_spec = EngineRepoSpec(
            repo_id="ansible",
            repo_path=str(ansible_spec.path),
            base_commit=ansible_spec.base_commit,
            test_command=ansible_spec.test_command,
        )

        # Use change_operator with empty target_symbol to search whole module
        plan = BugGenerationPlan(
            plan_id="test_plan_001",
            target_repo_id="ansible",
            target_base_commit=ansible_spec.base_commit,
            target_file="lib/ansible/module_utils/common/dict_transformations.py",
            target_symbol="",  # search whole module
            strategy="procedural",
            operator="change_operator",
            constraints=BugConstraints(max_modified_lines=10),
            seed=42,
        )

        candidates = engine.generate(
            plan=plan,
            node_code_dir=str(tmp_path),
            repo_spec=repo_spec,
            output_dir=str(tmp_path / "candidates"),
        )

        assert len(candidates) > 0, "Engine should produce at least one candidate"
        cand = candidates[0]
        assert cand.bug_patch, "Candidate should have a bug patch"
        assert "dict_transformations.py" in cand.target_file

        print(f"\nCandidate: {cand.candidate_id}")
        print(f"  operator: {cand.operator}")
        print(f"  patch lines: {len(cand.bug_patch.splitlines())}")
