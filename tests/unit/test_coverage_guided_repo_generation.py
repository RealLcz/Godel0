"""Tests for the coverage-guided repository-level fallback generator."""

from __future__ import annotations

import ast
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "initial_agent" / "src"))

from swesmith.patch_utils import count_modified_lines, make_git_diff
from swesmith.repo_level import (
    RepositoryWorkspace,
    apply_repository_patch,
    repository_diff,
)


SCRIPT_PATH = ROOT / "scripts" / "run_repo_level_closed_loop.py"
SPEC = importlib.util.spec_from_file_location("repo_closed_loop_helpers", SCRIPT_PATH)
assert SPEC and SPEC.loader
closed_loop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(closed_loop)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_minimal_mutations_are_small_and_syntax_valid():
    source = (
        "def choose(flag, left, right):\n"
        "    if flag:\n"
        "        return left == right\n"
        "    return True\n"
    )

    candidates = closed_loop._minimal_mutation_candidates(
        source,
        "lib/choice.py",
        seed=42,
    )

    assert {candidate["operator"] for candidate in candidates} >= {
        "change_operator",
        "invert_boolean",
        "invert_condition",
    }
    for candidate in candidates:
        assert count_modified_lines(candidate["patch"]) <= 12
        ast.parse(candidate["mutated_source"])


def test_coverage_guided_generator_requires_coupled_producer_consumer_files(
    tmp_path: Path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    (repo / "lib").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "lib" / "alpha.py").write_text(
        "def alpha(value):\n"
        "    if value == 1:\n"
        "        return True\n"
        "    return False\n",
        encoding="utf-8",
    )
    (repo / "lib" / "beta.py").write_text(
        "from lib.alpha import alpha\n\n"
        "def beta(value):\n"
        "    if alpha(value):\n"
        "        return 'ok'\n"
        "    return 'bad'\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_contract.py").write_text(
        "from lib.beta import beta\n\n"
        "def test_contract():\n"
        "    assert beta(1) == 'ok'\n\n"
        "def test_unaffected():\n"
        "    assert 'stable'.upper() == 'STABLE'\n",
        encoding="utf-8",
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base_commit = _git(repo, "rev-parse", "HEAD")

    monkeypatch.setitem(
        closed_loop.DOMAINS,
        "toy",
        {
            "description": "toy producer contract",
            "anchors": ["lib/alpha.py", "lib/beta.py"],
            "tests": ["tests/test_contract.py"],
            "contract": "Both modules preserve their public predicates.",
        },
    )

    def accept_candidate(**kwargs):
        candidate = kwargs["candidate"]
        assert candidate.modified_files == ["lib/alpha.py", "lib/beta.py"]
        return {
            "task_id": candidate.candidate_id,
            "strict_repo_level": True,
            "modified_files": candidate.modified_files,
        }

    monkeypatch.setattr(closed_loop, "_validate_and_package", accept_candidate)
    command = (
        f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. {sys.executable} "
        "-m pytest -p no:cacheprovider"
    )

    task, row = closed_loop._generate_coverage_guided_task(
        phase="bootstrap",
        attempt_index=0,
        domain_id="toy",
        blueprint={"domain": "toy"},
        source_trajectory_ids=[],
        source_repo=repo,
        base_commit=base_commit,
        test_prefix=command,
        validator=object(),
        output_dir=tmp_path / "output",
        test_timeout=30,
        seed=123,
    )

    assert task and task["strict_repo_level"] is True
    assert row["result"] == "accepted"
    assert [component["path"] for component in row["active_components"]] == [
        "lib/alpha.py",
        "lib/beta.py",
    ]
    assert all(component["f2p_tests"] for component in row["active_components"])
    assert row["semantic_coupling"]["valid"] is True
    assert row["semantic_coupling"]["tier"] == "strong"
    assert row["semantic_coupling"]["shared_f2p_tests"] == [
        "tests/test_contract.py::test_contract"
    ]


def test_component_selector_rejects_independent_multibug_pair():
    components = [
        {
            "path": "lib/alpha.py",
            "f2p_tests": ["tests/test_contract.py::test_alpha"],
            "p2p_count": 2,
        },
        {
            "path": "lib/beta.py",
            "f2p_tests": ["tests/test_contract.py::test_beta"],
            "p2p_count": 2,
        },
    ]
    selected, evidence = closed_loop._select_semantically_coupled_components(
        components,
        {
            "lib/alpha.py": "def alpha():\n    return True\n",
            "lib/beta.py": "def beta():\n    return False\n",
        },
    )

    assert selected == []
    assert evidence["valid"] is False


def test_component_selector_supports_three_file_adaptive_topology():
    shared_test = "tests/test_contract.py::test_end_to_end"
    components = [
        {"path": "lib/a.py", "f2p_tests": [shared_test], "p2p_count": 3},
        {"path": "lib/b.py", "f2p_tests": [shared_test], "p2p_count": 3},
        {"path": "lib/c.py", "f2p_tests": [shared_test], "p2p_count": 3},
    ]
    selected, evidence = closed_loop._select_semantically_coupled_components(
        components,
        {
            "lib/a.py": "def produce():\n    return True\n",
            "lib/b.py": "from lib.a import produce\n",
            "lib/c.py": "from lib.b import consume\n",
        },
        required_components=3,
    )

    assert len(selected) == 3
    assert evidence["required_components"] == 3
    assert len(evidence["static_dependencies"]) == 2
    assert closed_loop._adaptive_min_modified_files(
        {"capability_gap_code": "incomplete_cross_file_repair"}
    ) == 3


def test_component_selector_uses_runtime_edges_and_requires_full_execution():
    shared_test = "tests/test_contract.py::test_end_to_end"
    paths = ["lib/a.py", "lib/b.py", "lib/c.py"]
    components = [
        {
            "path": path,
            "f2p_tests": [shared_test],
            "p2p_count": 1,
            "runtime_execution_files": paths,
        }
        for path in paths
    ]
    selected, evidence = closed_loop._select_semantically_coupled_components(
        components,
        {path: "" for path in paths},
        required_components=3,
        dynamic_edges=[
            {"caller": "lib/a.py", "callee": "lib/b.py", "call_count": 1},
            {"caller": "lib/b.py", "callee": "lib/c.py", "call_count": 1},
        ],
        required_shared_test=shared_test,
    )

    assert [row["path"] for row in selected] == paths
    assert len(evidence["runtime_dependencies"]) == 2

    components[0]["runtime_execution_files"] = paths[:2]
    selected, _ = closed_loop._select_semantically_coupled_components(
        components,
        {path: "" for path in paths},
        required_components=3,
        dynamic_edges=[
            {"caller": "lib/a.py", "callee": "lib/b.py", "call_count": 1},
            {"caller": "lib/b.py", "callee": "lib/c.py", "call_count": 1},
        ],
        required_shared_test=shared_test,
    )
    assert selected == []


def test_trace_ranges_limit_mutations_to_executed_function():
    source = (
        "def executed(value):\n"
        "    if value == 1:\n"
        "        return True\n"
        "    return False\n\n"
        "def ignored(value):\n"
        "    if value == 2:\n"
        "        return True\n"
        "    return False\n"
    )
    ranges = closed_loop._traced_symbol_ranges(
        source,
        [{"symbol": "executed", "first_line": 1, "call_count": 1}],
    )
    candidates = closed_loop._minimal_mutation_candidates(
        source,
        "lib/module.py",
        seed=7,
        allowed_line_ranges=ranges,
    )

    assert ranges == [(1, 4)]
    assert candidates
    assert all(
        "def ignored(value):\n    if value == 2:\n        return True"
        in candidate["mutated_source"]
        for candidate in candidates
    )


def test_join_patch_blocks_preserves_trailing_blank_context(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    first = "def first():\n    value = 1\n    return value\n\n"
    second = "def second():\n    value = 2\n    return value\n\n"
    (repo / "first.py").write_text(first, encoding="utf-8")
    (repo / "second.py").write_text(second, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    blocks = [
        make_git_diff(first, first.replace("value = 1", "value = 10"), "first.py"),
        make_git_diff(second, second.replace("value = 2", "value = 20"), "second.py"),
    ]

    patch = closed_loop._join_patch_blocks(blocks)
    result = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=repo,
        input=patch,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_file_ablation_reuses_complete_matching_checkpoint(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    first = "value = 1\n\n"
    second = "value = 2\n\n"
    (repo / "first.py").write_text(first, encoding="utf-8")
    (repo / "second.py").write_text(second, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base_commit = _git(repo, "rev-parse", "HEAD")
    patch = closed_loop._join_patch_blocks(
        [
            make_git_diff(
                first,
                first.replace("value = 1", "value = 10"),
                "first.py",
            ),
            make_git_diff(
                second,
                second.replace("value = 2", "value = 20"),
                "second.py",
            ),
        ]
    )
    checkpoint = tmp_path / "ablation.json"
    arguments = {
        "base_commit": base_commit,
        "bug_patch": patch,
        "f2p_tests": ["command::regression"],
        "test_prefix": "ignored",
        "test_command": "false",
        "validation_mode": "exit_code",
        "timeout": 10,
        "checkpoint_path": checkpoint,
    }

    first_result = closed_loop._file_ablation(source_repo=repo, **arguments)
    cached_result = closed_loop._file_ablation(
        source_repo=tmp_path / "missing-repo",
        **arguments,
    )

    assert first_result["ablation_valid"] is True
    assert cached_result == first_result


def test_adversarial_solver_patch_gate_rejects_known_shortcuts(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    fixed = "value = 1\n"
    bugged = "value = 2\n"
    shortcut = "value = 3\n"
    (repo / "state.py").write_text(fixed, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base_commit = _git(repo, "rev-parse", "HEAD")
    bug_patch = make_git_diff(fixed, bugged, "state.py")
    shortcut_patch = make_git_diff(bugged, shortcut, "state.py")
    solving_patch = make_git_diff(bugged, fixed, "state.py")
    arguments = {
        "source_repo": repo,
        "base_commit": base_commit,
        "bug_patch": bug_patch,
        "test_command": "grep -q 'value = 1' state.py",
        "timeout": 10,
    }

    rejected = closed_loop._adversarial_solver_patch_results(
        solver_patches=[{"id": "known_shortcut", "patch": shortcut_patch}],
        **arguments,
    )
    escaped = closed_loop._adversarial_solver_patch_results(
        solver_patches=[{"id": "actual_fix", "patch": solving_patch}],
        **arguments,
    )

    assert rejected["valid"] is True
    assert rejected["all_rejected"] is True
    assert rejected["results"][0]["test_returncode"] == 1
    assert escaped["valid"] is True
    assert escaped["all_rejected"] is False
    assert escaped["results"][0]["test_returncode"] == 0


def test_failure_fingerprint_uses_terminal_test_assertion():
    output = """
tests/test_contract.py::TestPipeline::test_end_to_end FAILED [100%]

________________ TestPipeline.test_end_to_end _________________

    def test_end_to_end(self):
>       self.assertEqual(result, 'expected')
E       AssertionError: 'actual' != 'expected'

tests/test_contract.py:42: AssertionError
=========================== short test summary info ============================
FAILED tests/test_contract.py::TestPipeline::test_end_to_end
"""

    fingerprints = closed_loop._pytest_failure_fingerprints(output)

    assert fingerprints[
        "tests/test_contract.py::TestPipeline::test_end_to_end"
    ] == {
        "nodeid": "tests/test_contract.py::TestPipeline::test_end_to_end",
        "test_location": "tests/test_contract.py:42",
        "error_type": "AssertionError",
        "assertion": "self.assertEqual(result, 'expected')",
        "message": "AssertionError: 'actual' != 'expected'",
        "fingerprint": "tests/test_contract.py:42:AssertionError",
    }


def test_command_failure_fingerprint_uses_failed_contract_marker():
    output = """
GODEL0_CONTRACT role.no_dupes.inroles expected=1 actual=1 command_rc=0
GODEL0_CONTRACT role.inheritance.linear expected=3 actual=1 command_rc=0
"""

    fingerprints = closed_loop._command_failure_fingerprints(
        output,
        "integration::role_instance_cache",
    )

    fingerprint = fingerprints["integration::role_instance_cache"]
    assert fingerprint["kind"] == "command_contract"
    assert fingerprint["failed_checks"] == [
        {
            "check": "role.inheritance.linear",
            "expected": "3",
            "actual": "1",
            "command_rc": 0,
        }
    ]
    assert fingerprint["fingerprint"].startswith(
        "integration::role_instance_cache:"
    )


def test_command_failure_fingerprint_rejects_warning_only_output():
    output = """
[WARNING]: You are running a development version.
run this version only while modifying the engine.
"""

    assert closed_loop._command_failure_fingerprints(
        output,
        "integration::role_instance_cache",
    ) == {}


def test_command_failure_fingerprint_supports_native_integration_errors():
    output = "fatal: [testhost]: FAILED! => undefined variable"

    fingerprints = closed_loop._command_failure_fingerprints(
        output,
        "integration::delegate_to_evaluation",
    )

    fingerprint = fingerprints["integration::delegate_to_evaluation"]
    assert fingerprint["kind"] == "command_error"
    assert fingerprint["evidence"] == [output]


def test_ansible_runtime_packages_role_contract_fixture(tmp_path: Path):
    runtime = closed_loop._prepare_ansible_runtime(tmp_path)
    playbook = Path(runtime["role_instance_contract_playbook"])
    command = closed_loop._integration_test_command(
        runtime,
        "test/integration/targets/roles",
        "true",
    )

    assert playbook.is_file()
    assert (playbook.parent / "parent_reset_shortcut.patch").is_file()
    assert str(playbook) in command
    assert "GODEL0_ROLE_INSTANCE_CONTRACT_PLAYBOOK=" in command

    public_runtime_dir = tmp_path / "public_runtime"
    public_runtime = closed_loop._prepare_ansible_runtime(
        tmp_path,
        runtime_dir=public_runtime_dir,
        include_contracts=False,
    )
    public_command = closed_loop._integration_test_command(
        public_runtime,
        "test/integration/targets/roles",
        "true",
        include_contract_env=False,
    )
    assert str(playbook) not in public_command
    assert "GODEL0_ROLE_INSTANCE_CONTRACT_PLAYBOOK=" not in public_command
    assert not (public_runtime_dir / "contracts").exists()

    public_template = closed_loop._integration_test_command(
        closed_loop._solver_public_ansible_runtime(),
        "test/integration/targets/roles",
        "true",
        include_contract_env=False,
    )
    assert closed_loop.SOLVER_PUBLIC_RUNTIME_TOKEN in public_template


def test_reference_test_patch_isolated_from_source_changes(tmp_path: Path):
    repo = tmp_path / "reference_repo"
    (repo / "lib").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "lib" / "feature.py").write_text("VALUE = 'old'\n")
    (repo / "tests" / "test_feature.py").write_text("assert True\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "buggy")
    parent = _git(repo, "rev-parse", "HEAD")

    (repo / "lib" / "feature.py").write_text("VALUE = 'fixed'\n")
    (repo / "tests" / "test_feature.py").write_text(
        "from lib.feature import VALUE\nassert VALUE == 'fixed'\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "fix")
    commit = _git(repo, "rev-parse", "HEAD")

    patch = closed_loop._reference_test_patch(
        source_repo=repo,
        reference_parent=parent,
        reference_commit=commit,
        test_files=["tests/test_feature.py"],
    )

    assert "tests/test_feature.py" in patch
    assert "lib/feature.py" not in patch


def test_solver_hides_reference_tests_then_injects_for_validation(
    tmp_path: Path,
    monkeypatch,
):
    repo = tmp_path / "solver_repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "feature.py").write_text("VALUE = 'old'\n")
    (repo / "tests" / "test_feature.py").write_text(
        "from pkg.feature import VALUE\n\ndef test_value():\n    assert VALUE == 'old'\n"
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "buggy")
    parent = _git(repo, "rev-parse", "HEAD")

    (repo / "pkg" / "feature.py").write_text("VALUE = 'fixed'\n")
    (repo / "tests" / "test_feature.py").write_text(
        "from pkg.feature import VALUE\n\ndef test_value():\n    assert VALUE == 'fixed'\n"
    )
    (repo / "changelog").mkdir()
    (repo / "changelog" / "fix.md").write_text(
        "The fix creates independent feature instances.\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "fix")
    commit = _git(repo, "rev-parse", "HEAD")

    source_fix = _git(repo, "diff", parent, commit, "--", "pkg/feature.py") + "\n"
    with RepositoryWorkspace(str(repo), commit) as workspace:
        assert apply_repository_patch(workspace, source_fix, reverse=True)
        bug_patch = repository_diff(workspace, "HEAD")

    class InspectingAdapter:
        def run(self, _agent_src, request):
            visible_test = request.git_dir / "tests" / "test_feature.py"
            public_runtime = request.git_dir / ".godel0_solver_runtime"
            assert "'old'" in visible_test.read_text()
            assert not (request.git_dir / "changelog" / "fix.md").exists()
            assert (public_runtime / "ansible.cfg").is_file()
            assert not (public_runtime / "contracts").exists()
            assert closed_loop.SOLVER_PUBLIC_RUNTIME_TOKEN not in request.test_description
            assert str(public_runtime) in request.test_description
            assert "test_feature.py" not in request.test_description
            assert request.extra_env["HOME"] == str(public_runtime / "home")
            assert "isolated_solver_scratch" in request.extra_env["TMPDIR"]
            return SimpleNamespace(error=None, success=True)

    task = {
        "task_id": "hidden_reference_test",
        "domain": "unit",
        "base_commit": commit,
        "bug_patch": bug_patch,
        "problem_statement": "Restore the feature contract.",
        "test_command": "PYTHONPATH=. python -m pytest -q tests/test_feature.py",
        "solver_test_command": (
            f"test -f {closed_loop.SOLVER_PUBLIC_RUNTIME_TOKEN}/ansible.cfg"
        ),
        "solver_validation_command": (
            "test -f changelog/fix.md && "
            "PYTHONPATH=. python -m pytest -q tests/test_feature.py"
        ),
        "control_test_command": "",
        "validation_mode": "exit_code",
        "f2p_tests": [],
        "p2p_tests": [],
        "reference_parent": parent,
        "reference_commit": commit,
        "solver_hidden_test_files": ["tests/test_feature.py"],
        "solver_hidden_reference_files": [
            "tests/test_feature.py",
            "changelog/fix.md",
        ],
    }

    scratch_root = tmp_path / "isolated_solver_scratch"
    monkeypatch.setenv("GODEL0_SOLVER_SCRATCH_ROOT", str(scratch_root))
    result = closed_loop._run_solver(
        task=task,
        phase="hidden_test_smoke",
        source_repo=repo,
        agent_src=ROOT / "initial_agent" / "src",
        adapter=InspectingAdapter(),
        model="unused",
        output_dir=tmp_path / "output",
        agent_timeout=30,
        test_timeout=30,
    )

    assert result["hidden_tests_injected"] is True
    assert result["hidden_reference_changes_injected"] is True
    assert result["solver_workspace_isolated"] is True
    assert result["public_runtime_contracts_absent"] is True
    assert result["hidden_test_files"] == ["tests/test_feature.py"]
    assert result["hidden_reference_files"] == [
        "tests/test_feature.py",
        "changelog/fix.md",
    ]
    assert str(scratch_root) in result["solver_test_command"]
    assert str(tmp_path / "output") not in result["solver_test_command"]
    assert result["test_returncode"] != 0
    assert result["resolved"] is False


def test_causal_patch_audit_rejects_mechanical_multibug_shortcuts():
    patch = """diff --git a/lib/a.py b/lib/a.py
--- a/lib/a.py
+++ b/lib/a.py
@@ -1,3 +1,3 @@
-for item in values:
+for item in list(values)[:1]:
     consume(item)
diff --git a/lib/b.py b/lib/b.py
--- a/lib/b.py
+++ b/lib/b.py
@@ -1,2 +1,2 @@
-self.enabled = False
+self.enabled = True
"""

    assert closed_loop._causal_patch_shortcut_reasons(patch) == [
        "arbitrary_iteration_truncation",
        "mechanical_boolean_flip",
    ]


def test_causal_patch_audit_accepts_protocol_handoff_change():
    patch = """diff --git a/lib/producer.py b/lib/producer.py
--- a/lib/producer.py
+++ b/lib/producer.py
@@ -1,2 +1,2 @@
-return Result(value=value)
+return Result(payload=value)
"""

    assert closed_loop._causal_patch_shortcut_reasons(patch) == []


def test_historical_pr_gate_requires_no_single_oracle_file_fix():
    shared = {
        "runtime_contract_valid": True,
        "endpoint_failure_valid": True,
        "causal_patch_quality_valid": True,
        "coupling_required": False,
        "semantic_coupling": {},
        "historical_provenance_valid": True,
    }
    ablation = {
        "ablation_valid": True,
        "all_files_oracle_necessary": False,
        "any_single_file_oracle_fix_passes": False,
    }

    assert closed_loop._strict_repo_level_valid(
        strictness_policy="historical_pr",
        ablation=ablation,
        **shared,
    )
    ablation["any_single_file_oracle_fix_passes"] = True
    assert not closed_loop._strict_repo_level_valid(
        strictness_policy="historical_pr",
        ablation=ablation,
        **shared,
    )
    ablation["any_single_file_oracle_fix_passes"] = False
    assert not closed_loop._strict_repo_level_valid(
        strictness_policy="historical_pr",
        ablation=ablation,
        adversarial_resistance_valid=False,
        **shared,
    )
    assert not closed_loop._strict_repo_level_valid(
        strictness_policy="historical_pr",
        ablation=ablation,
        solver_test_visibility_valid=False,
        **shared,
    )
    assert not closed_loop._strict_repo_level_valid(
        strictness_policy="historical_pr",
        ablation=ablation,
        solver_runtime_isolation_valid=False,
        **shared,
    )


def test_synthetic_gate_still_requires_each_injected_file_to_be_active():
    assert not closed_loop._strict_repo_level_valid(
        strictness_policy="synthetic_causal",
        ablation={
            "ablation_valid": True,
            "all_files_oracle_necessary": False,
            "any_single_file_oracle_fix_passes": False,
        },
        runtime_contract_valid=True,
        endpoint_failure_valid=True,
        causal_patch_quality_valid=True,
        coupling_required=False,
        semantic_coupling={},
        historical_provenance_valid=False,
    )


def test_solver_evaluation_command_uses_only_clean_baseline_tests():
    parameterized = "tests/test_contract.py::test_control[param with space]"
    command = closed_loop._solver_evaluation_command(
        test_prefix="python -m pytest -p no:cacheprovider",
        original_test_command="python -m pytest tests -v",
        validation_mode="pytest",
        f2p_tests=["tests/test_contract.py::test_regression"],
        p2p_tests=[parameterized],
    )

    assert "tests/test_contract.py::test_regression" in command
    assert "tests/test_contract.py::test_control" in command
    assert parameterized not in command
    assert "python -m pytest tests -v" not in command
    spaced_target = "tests/path with space/test_contract.py::test_control"
    assert shlex.quote(spaced_target) in closed_loop._test_command(
        "python -m pytest",
        [spaced_target],
    )
    assert closed_loop._solver_evaluation_command(
        test_prefix="ignored",
        original_test_command="bash integration.sh",
        validation_mode="exit_code",
        f2p_tests=["command::primary"],
        p2p_tests=["command::control"],
    ) == "bash integration.sh"


def test_pytest_status_parser_preserves_parameterized_node_ids():
    nodeid = "tests/test_contract.py::test_control[param with space]"
    passed, failed = closed_loop._pytest_status_sets(
        {
            "stdout": (
                f"{nodeid} PASSED [ 50%]\n"
                "FAILED tests/test_contract.py::test_failure[other value] - boom\n"
            ),
            "stderr": "",
        }
    )

    assert passed == {nodeid}
    assert failed == {"tests/test_contract.py::test_failure[other value]"}


def test_solver_outcome_ignores_non_baseline_parameter_failure():
    expected = "tests/test_contract.py::test_control[clean parameter]"
    passed, missing, failed = closed_loop._evaluate_expected_pytest_tests(
        expected_tests={expected},
        result={
            "returncode": 1,
            "stdout": (
                f"{expected} PASSED [ 50%]\n"
                "tests/test_contract.py::test_control[baseline failure] FAILED [100%]\n"
            ),
            "stderr": "",
        },
    )

    assert passed is True
    assert missing == []
    assert failed == []


def test_trajectory_diagnosis_is_grounded_in_patch_scope():
    bug_patch = """diff --git a/manager.py b/manager.py
--- a/manager.py
+++ b/manager.py
@@ -1 +1 @@
-parsed = True
+parsed = False
diff --git a/data.py b/data.py
--- a/data.py
+++ b/data.py
@@ -1 +1 @@
-if group in groups:
+if group not in groups:
"""
    solver_patch = """diff --git a/data.py b/data.py
--- a/data.py
+++ b/data.py
@@ -1 +1 @@
-if group not in groups:
+if group in groups:
"""
    evidence = closed_loop._build_trajectory_evidence(
        result={
            "task_id": "inventory-task",
            "domain": "inventory_model",
            "resolved": False,
            "modified_files": ["data.py"],
            "solver_patch": solver_patch,
            "tool_calls": 5,
            "test_returncode": 1,
        },
        task={
            "modified_files": ["manager.py", "data.py"],
            "bug_patch": bug_patch,
            "f2p_tests": ["tests/test_inventory.py::test_inventory"],
        },
        trajectory="read manager.py\nread data.py\n",
        test_output="tests/test_inventory.py::test_inventory FAILED\n",
    )
    diagnosis = closed_loop._ground_trajectory_diagnosis(
        evidence,
        {"domain_id": "yaml_loading", "capability_gap": "invented claim"},
    )

    assert evidence["facts"]["F3_EXACT_ORACLE_REVERTS"] == ["data.py"]
    assert evidence["facts"]["F4_MISSING_SOLVER_FILES"] == ["manager.py"]
    assert diagnosis["capability_gap_code"] == "incomplete_cross_file_repair"
    assert diagnosis["failure_stage"] == "patch_generation"
    assert diagnosis["domain_id"] == "yaml_loading"
    assert "invented claim" not in diagnosis["evidence"]
    assert diagnosis["grounding_valid"] is True
