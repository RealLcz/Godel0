"""Tests for repository-level bug generators."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "initial_agent" / "src"))

from proposer.code_locator import RepoIndex
from proposer.planner import ProposerPlanner
from proposer.schemas import FailureSignature
from swesmith.engine import BugConstraints, BugGenerationPlan, RepoSpec, SWESmithEngine
from swesmith.patch_utils import extract_changed_files
from swesmith.repo_level import (
    apply_repository_patch,
    filter_patch,
    RepositoryWorkspace,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _create_fixed_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "contract_repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "provider.py").write_text(
        'TOKEN = "legacy"\n\n\ndef token():\n    return TOKEN\n'
    )
    (repo / "pkg" / "consumer.py").write_text(
        'from pkg.provider import token\n\n\ndef accepted():\n    return token() == "legacy"\n'
    )
    (repo / "tests" / "test_contract.py").write_text(
        "from pkg.consumer import accepted\n\n\ndef test_contract():\n    assert accepted()\n"
    )

    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "buggy contract")
    buggy_commit = _git(repo, "rev-parse", "HEAD")

    (repo / "pkg" / "provider.py").write_text(
        'TOKEN = "current"\n\n\ndef token():\n    return TOKEN\n'
    )
    (repo / "pkg" / "consumer.py").write_text(
        'from pkg.provider import token\n\n\ndef accepted():\n    return token() == "current"\n'
    )
    (repo / "tests" / "test_contract.py").write_text(
        "from pkg.consumer import accepted\n"
        "from pkg.provider import token\n\n\n"
        "def test_contract():\n    assert accepted()\n    assert token() == \"current\"\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "fix cross-file token contract")
    fixed_commit = _git(repo, "rev-parse", "HEAD")
    return repo, buggy_commit, fixed_commit


def _repo_spec(repo: Path, commit: str) -> RepoSpec:
    return RepoSpec(
        repo_id="contract",
        repo_path=str(repo),
        base_commit=commit,
        test_command=f"{sys.executable} -m pytest -q",
    )


def _repo_constraints() -> BugConstraints:
    return BugConstraints(
        min_modified_files=2,
        max_modified_files=4,
        max_modified_lines=20,
        allow_test_edits=False,
    )


def _create_chain_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "chain_repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "provider.py").write_text(
        'def token():\n    return "current"\n'
    )
    (repo / "pkg" / "transform.py").write_text(
        'from pkg.provider import token\n\n\ndef decorated():\n    return token() + "!"\n'
    )
    (repo / "pkg" / "consumer.py").write_text(
        'from pkg.transform import decorated\n\n\ndef accepted():\n    return decorated() == "current!"\n'
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "clean chain")
    return repo, _git(repo, "rev-parse", "HEAD")


class _RepoChainAdapter:
    def generate_repo_chain_contract(self, workspace: str, request: str, plan):
        return {
            "chain_plan": {
                "root_invariant": "the current token survives provider to consumer",
                "entrypoint": "pkg.provider.token",
                "endpoint": "pkg.consumer.accepted",
                "capability_gap": "cross-file state propagation",
                "context_files": list(plan.target_files),
                "mutation_sites": [
                    {"file": "pkg/provider.py", "symbol": "token", "role": "producer", "change": "return stale token"},
                    {"file": "pkg/transform.py", "symbol": "decorated", "role": "carrier", "change": "use wrong marker"},
                    {"file": "pkg/consumer.py", "symbol": "accepted", "role": "consumer", "change": "compare stale contract"},
                ],
                "contract_tests": ["each layer", "end to end"],
                "rationale": "all sites propagate one token contract",
            },
            "tests": [
                {
                    "path": "test/units/test_generated_chain_contract.py",
                    "content": (
                        "from pkg.consumer import accepted\n"
                        "from pkg.provider import token\n"
                        "from pkg.transform import decorated\n\n"
                        "def test_chain_contract():\n"
                        "    assert token() == 'current'\n"
                        "    assert decorated() == 'current!'\n"
                        "    assert accepted()\n"
                    ),
                }
            ],
        }

    def generate_repo_chain_bug(self, workspace: str, request: str, plan):
        return {
            "edits": [
                {
                    "file": "pkg/provider.py",
                    "symbol": "token",
                    "before": 'return "current"',
                    "after": 'return "legacy"',
                    "intent": "produce the stale token",
                },
                {
                    "file": "pkg/transform.py",
                    "symbol": "decorated",
                    "before": 'return token() + "!"',
                    "after": 'return token() + "?"',
                    "intent": "corrupt the carried marker",
                },
                {
                    "file": "pkg/consumer.py",
                    "symbol": "accepted",
                    "before": 'return decorated() == "current!"',
                    "after": 'return decorated() == "legacy!"',
                    "intent": "accept a mismatched endpoint contract",
                },
            ]
        }


class _PartialCouplingAdapter(_RepoChainAdapter):
    def generate_repo_chain_contract(self, workspace: str, request: str, plan):
        payload = super().generate_repo_chain_contract(workspace, request, plan)
        payload["tests"][0]["content"] = (
            "from pkg.provider import token\n\n\n"
            "def test_provider_contract():\n"
            "    assert token() == 'current'\n"
        )
        return payload


def test_repo_chain_generates_contract_first_multifile_candidate(tmp_path: Path):
    repo, commit = _create_chain_repo(tmp_path)
    engine = SWESmithEngine(agent_adapter=_RepoChainAdapter())
    plan = BugGenerationPlan(
        plan_id="chain",
        target_repo_id="contract",
        target_base_commit=commit,
        target_file="pkg/provider.py",
        target_files=[
            "pkg/provider.py",
            "pkg/transform.py",
            "pkg/consumer.py",
        ],
        strategy="repo_chain",
        constraints=BugConstraints(
            min_modified_files=3,
            max_modified_files=3,
            max_modified_lines=20,
            context_file_budget=3,
            min_mutation_sites=3,
            max_mutation_sites=3,
            require_generated_tests=True,
            generation_timeout_sec=30,
        ),
    )

    candidates = engine.generate(
        plan,
        node_code_dir="",
        repo_spec=RepoSpec(
            repo_id="contract",
            repo_path=str(repo),
            base_commit=commit,
            test_command=f"{sys.executable} -m pytest -q",
        ),
        output_dir=str(tmp_path / "out_chain"),
    )

    assert len(candidates) == 1, engine.repo_chain.last_rejection
    candidate = candidates[0]
    assert candidate.strategy == "repo_chain"
    assert set(candidate.modified_files) == {
        "pkg/provider.py",
        "pkg/transform.py",
        "pkg/consumer.py",
    }
    metadata = candidate.generation_metadata
    assert metadata["clean_contract_returncode"] == 0
    assert metadata["bugged_contract_returncode"] != 0
    assert metadata["restored_contract_returncode"] == 0
    assert metadata["causal_ablation"]["repair_only_one_file_all_fail"]
    assert set(metadata["generated_test_files"]) == {
        "test/units/test_generated_chain_contract.py"
    }
    candidate_dir = tmp_path / "out_chain" / candidate.candidate_id
    assert (candidate_dir / "contract.patch").is_file()
    assert (candidate_dir / "chain_plan.json").is_file()
    assert (candidate_dir / "oracle.patch").is_file()
    assert (candidate_dir / "problem_statement.md").is_file()
    assert (candidate_dir / "task.json").is_file()
    assert (tmp_path / "out_chain" / "mutation_edits_attempt_1.json").is_file()
    assert (tmp_path / "out_chain" / "mutation_attempt_1.diff").is_file()
    task = __import__("json").loads((candidate_dir / "task.json").read_text())
    assert task["schema_version"] == "godel0.swebench_like.v1"
    assert task["setup_patch"] == candidate.bug_patch
    assert task["patch"] == (candidate_dir / "oracle.patch").read_text()
    with RepositoryWorkspace(str(repo), commit) as workspace:
        assert apply_repository_patch(workspace, task["setup_patch"])
        assert apply_repository_patch(workspace, task["patch"])
        assert (Path(workspace) / "pkg/provider.py").read_text() == (
            repo / "pkg/provider.py"
        ).read_text()


def test_repo_chain_records_partial_coupling_without_overriding_f2p_validity(
    tmp_path: Path,
):
    repo, commit = _create_chain_repo(tmp_path)
    engine = SWESmithEngine(agent_adapter=_PartialCouplingAdapter())
    plan = BugGenerationPlan(
        plan_id="partial-coupling",
        target_repo_id="contract",
        target_base_commit=commit,
        target_file="pkg/provider.py",
        target_files=["pkg/provider.py", "pkg/transform.py", "pkg/consumer.py"],
        strategy="repo_chain",
        constraints=BugConstraints(
            min_modified_files=3,
            max_modified_files=3,
            max_modified_lines=20,
            context_file_budget=3,
            min_mutation_sites=3,
            max_mutation_sites=3,
            require_generated_tests=True,
            generation_timeout_sec=30,
        ),
    )

    candidates = engine.generate(
        plan,
        node_code_dir="",
        repo_spec=RepoSpec(
            repo_id="contract",
            repo_path=str(repo),
            base_commit=commit,
            test_command=f"{sys.executable} -m pytest -q",
        ),
        output_dir=str(tmp_path / "out_partial"),
    )

    assert len(candidates) == 1, engine.repo_chain.last_rejection
    coupling = candidates[0].generation_metadata["semantic_coupling"]
    assert coupling["valid"] is False
    assert coupling["tier"] == "generated_contract_with_partial_coupling"
    assert candidates[0].generation_metadata["causal_ablation"][
        "repair_only_one_file_passed"
    ]["pkg/provider.py"]


def test_repo_chain_renders_structured_ansible_contract_cases():
    generator = SWESmithEngine().repo_chain
    plan = BugGenerationPlan(
        plan_id="render-cases",
        task_blueprint={"contract_test_renderer": "ansible_playbook_cli"},
    )
    payload = {
        "tests": [
            {
                "path": "test/units/playbook/test_generated_contract.py",
                "content": "renderer placeholder",
            }
        ],
        "contract_cases": [
            {
                "name": "custom_loop_var",
                "playbook": (
                    "---\n- hosts: localhost\n  tasks:\n"
                    "    - name: target output\n      debug:\n        msg: target output\n"
                ),
                "files": {"included.yml": "---\n[]\n"},
                "expected_output": ["target output"],
                "expected_counts": {"target output": 1},
                "compatibility_control": False,
            },
            {
                "name": "default_loop_var",
                "playbook": "---\n- hosts: localhost\n  tasks: []\n",
                "files": {"included.yml": "---\n[]\n"},
                "expected_output": ["control output"],
                "compatibility_control": True,
            },
        ],
    }

    tests = generator._materialize_contract_tests(plan, payload)

    assert len(tests) == 1
    assert tests[0]["path"] == "test/units/playbook/test_generated_contract.py"
    assert "ansible-playbook" in tests[0]["content"]
    assert "custom_loop_var" in tests[0]["content"]
    assert "control output" in tests[0]["content"]
    assert "TASK [target output]" in tests[0]["content"]
    assert "def test_generated_repo_cli_control" in tests[0]["content"]
    assert "'--version'" in tests[0]["content"]
    namespace = {}
    exec(tests[0]["content"], namespace)
    trusted_controls = [
        case
        for case in namespace["CASES"]
        if case["name"] == "godel0_empty_play_control"
    ]
    assert len(trusted_controls) == 1
    assert trusted_controls[0]["compatibility_control"] is True
    assert trusted_controls[0]["expected_output"] == ["PLAY [localhost]"]
    taxonomy = generator._contract_test_taxonomy(
        payload, ["test/units/playbook/test_generated_contract.py"]
    )
    assert (
        "test/units/playbook/test_generated_contract.py::"
        "test_generated_repo_contract[godel0_empty_play_control]"
        in taxonomy["PASS_TO_PASS"]
    )
    assert (
        "test/units/playbook/test_generated_contract.py::"
        "test_generated_repo_cli_control"
        in taxonomy["PASS_TO_PASS"]
    )


def test_repo_chain_counts_debug_message_not_task_name_substring():
    generator = SWESmithEngine().repo_chain
    plan = BugGenerationPlan(
        plan_id="render-message-count",
        task_blueprint={"contract_test_renderer": "ansible_playbook_cli"},
    )
    payload = {
        "tests": [
            {
                "path": "test/units/playbook/test_generated_message_count.py",
                "content": "renderer placeholder",
            }
        ],
        "contract_cases": [
            {
                "name": "target",
                "playbook": (
                    "---\n- hosts: localhost\n  tasks:\n"
                    "    - name: Task in explicit block\n"
                    "      debug:\n        msg: explicit\n"
                ),
                "files": {},
                "expected_output": ["explicit"],
                "expected_counts": {"explicit": 1},
                "compatibility_control": False,
            },
            {
                "name": "control",
                "playbook": "---\n- hosts: localhost\n  tasks: []\n",
                "files": {},
                "expected_output": ["control"],
                "compatibility_control": True,
            },
        ],
    }

    tests = generator._materialize_contract_tests(plan, payload)

    assert len(tests) == 1
    namespace = {}
    exec(tests[0]["content"], namespace)
    assert namespace["CASES"][0]["expected_counts"] == {
        '"msg": "explicit"': 1
    }


def test_repo_chain_drops_unobservable_block_container_count():
    generator = SWESmithEngine().repo_chain
    plan = BugGenerationPlan(
        plan_id="render-block-count",
        task_blueprint={
            "contract_test_renderer": "ansible_playbook_cli",
            "require_expected_counts": True,
        },
    )
    payload = {
        "tests": [
            {
                "path": "test/units/playbook/test_generated_block_count.py",
                "content": "renderer placeholder",
            }
        ],
        "contract_cases": [
            {
                "name": "target",
                "playbook": (
                    "---\n- hosts: localhost\n  tasks:\n"
                    "    - name: Explicit block\n      block:\n"
                    "        - name: Child task\n"
                    "          debug:\n            msg: child_marker\n"
                ),
                "files": {},
                "expected_output": ["child_marker"],
                "expected_counts": {
                    "TASK [Explicit block]": 1,
                    "child_marker": 1,
                },
                "compatibility_control": False,
            },
            {
                "name": "control",
                "playbook": "---\n- hosts: localhost\n  tasks: []\n",
                "files": {},
                "expected_output": ["control"],
                "compatibility_control": True,
            },
        ],
    }

    tests = generator._materialize_contract_tests(plan, payload)

    namespace = {}
    exec(tests[0]["content"], namespace)
    assert namespace["CASES"][0]["expected_counts"] == {
        '"msg": "child_marker"': 1
    }


def test_repo_chain_unescapes_literal_brackets_in_counts():
    generator = SWESmithEngine().repo_chain
    plan = BugGenerationPlan(
        plan_id="render-bracket-count",
        task_blueprint={"contract_test_renderer": "ansible_playbook_cli"},
    )
    payload = {
        "tests": [
            {
                "path": "test/units/playbook/test_generated_bracket_count.py",
                "content": "renderer placeholder",
            }
        ],
        "contract_cases": [
            {
                "name": "target",
                "playbook": (
                    "---\n- hosts: localhost\n  gather_facts: false\n"
                    "  tasks:\n    - debug: {msg: marker}\n"
                ),
                "files": {},
                "expected_output": ["marker"],
                "expected_counts": {r"ok: \[localhost\]": 1},
                "compatibility_control": False,
            },
            {
                "name": "control",
                "playbook": "---\n- hosts: localhost\n  tasks: []\n",
                "files": {},
                "expected_output": ["PLAY [localhost]"],
                "compatibility_control": True,
            },
        ],
    }

    tests = generator._materialize_contract_tests(plan, payload)

    namespace = {}
    exec(tests[0]["content"], namespace)
    assert namespace["CASES"][0]["expected_counts"] == {
        "ok: [localhost]": 1
    }


def test_repo_chain_rejects_duplicate_planned_mutation_sites():
    generator = SWESmithEngine().repo_chain
    payload = {
        "chain_plan": {
            "root_invariant": "one invariant",
            "entrypoint": "a.f",
            "endpoint": "b.g",
            "mutation_sites": [
                {"file": "a.py", "symbol": "f"},
                {"file": "a.py", "symbol": "f"},
                {"file": "b.py", "symbol": "g"},
            ],
        },
        "tests": [{"path": "test/units/test_generated.py", "content": "assert True"}],
    }

    rejection = generator._chain_plan_rejection(
        payload,
        context_files=["a.py", "b.py"],
        min_files=2,
        max_files=4,
        min_sites=3,
        max_sites=4,
    )

    assert rejection == "mutation sites must be unique file/symbol pairs"


def test_repo_chain_requires_edits_for_every_planned_site(tmp_path: Path):
    generator = SWESmithEngine().repo_chain
    chain = {
        "mutation_sites": [
            {"file": "a.py", "symbol": "f"},
            {"file": "b.py", "symbol": "g"},
            {"file": "c.py", "symbol": "h"},
        ]
    }
    payload = {
        "edits": [
            {"file": "a.py", "symbol": "f"},
            {"file": "b.py", "symbol": "g"},
        ]
    }

    edits, rejection = generator._materialize_symbol_edits(
        tmp_path,
        payload,
        chain,
        min_files=2,
        max_files=4,
        min_sites=2,
        max_sites=4,
    )

    assert edits == []
    assert rejection == "edits must cover all 3 planned mutation sites exactly once"


def test_repo_chain_rejects_new_comments_that_reveal_mutation(tmp_path: Path):
    generator = SWESmithEngine().repo_chain
    (tmp_path / "a.py").write_text("def f():\n    return 1\n")
    (tmp_path / "b.py").write_text("def g():\n    return 1\n")
    chain = {
        "mutation_sites": [
            {"file": "a.py", "symbol": "f"},
            {"file": "b.py", "symbol": "g"},
        ]
    }
    payload = {
        "edits": [
            {
                "file": "a.py",
                "symbol": "f",
                "before": "    return 1\n",
                "after": "    # Bug: return the wrong value\n    return 0\n",
            },
            {
                "file": "b.py",
                "symbol": "g",
                "before": "    return 1\n",
                "after": "    return 0\n",
            },
        ]
    }

    edits, rejection = generator._materialize_symbol_edits(
        tmp_path,
        payload,
        chain,
        min_files=2,
        max_files=4,
        min_sites=2,
        max_sites=4,
    )

    assert edits == []
    assert rejection == "edit 0 adds comments that may reveal the generated regression"


def test_repo_chain_normalizes_only_unique_uniform_indentation_offsets():
    generator = SWESmithEngine().repo_chain
    symbol = "def f():\n        if ready:\n            return 1\n"

    aligned = generator._align_edit_indentation(
        symbol,
        "if ready:\n    return 1",
        "if ready:\n    return 2",
    )

    assert aligned == (
        "        if ready:\n            return 1",
        "        if ready:\n            return 2",
    )


def test_pr_replay_reverses_complete_multifile_fix(tmp_path: Path):
    repo, _buggy_commit, fixed_commit = _create_fixed_repo(tmp_path)
    engine = SWESmithEngine()
    plan = BugGenerationPlan(
        plan_id="replay",
        target_repo_id="contract",
        target_base_commit=fixed_commit,
        strategy="pr_replay",
        reference_commit=fixed_commit,
        constraints=_repo_constraints(),
    )

    candidates = engine.generate(
        plan,
        node_code_dir="",
        repo_spec=_repo_spec(repo, fixed_commit),
        output_dir=str(tmp_path / "out_replay"),
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.modified_files == ["pkg/consumer.py", "pkg/provider.py"]
    assert extract_changed_files(candidate.bug_patch) == candidate.modified_files
    assert "tests/test_contract.py" not in candidate.bug_patch
    assert candidate.mutation_site["reference_parent"] == f"{fixed_commit}^1"
    assert candidate.generation_metadata["reference_test_files"] == [
        "tests/test_contract.py"
    ]
    assert candidate.generation_metadata["reference_hidden_files"] == [
        "tests/test_contract.py"
    ]
    assert 'TOKEN = "current"' in (repo / "pkg" / "provider.py").read_text()

    with RepositoryWorkspace(str(repo), fixed_commit) as workspace:
        assert apply_repository_patch(workspace, candidate.bug_patch)
        workspace_path = Path(workspace)
        assert 'TOKEN = "legacy"' in (workspace_path / "pkg" / "provider.py").read_text()
        assert '== "legacy"' in (workspace_path / "pkg" / "consumer.py").read_text()
        test_result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        assert test_result.returncode != 0
        assert apply_repository_patch(workspace, candidate.bug_patch, reverse=True)
        restored = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        assert restored.returncode == 0, restored.stdout + restored.stderr


def test_filter_patch_preserves_blank_trailing_context(tmp_path: Path):
    repo = tmp_path / "blank_context_repo"
    repo.mkdir()
    (repo / "first.py").write_text("value = 'old'\n\n")
    (repo / "second.py").write_text("value = 'old'\n\n")
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "old")
    old_commit = _git(repo, "rev-parse", "HEAD")
    (repo / "first.py").write_text("value = 'new'\n\n")
    (repo / "second.py").write_text("value = 'new'\n\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "new")
    new_commit = _git(repo, "rev-parse", "HEAD")
    result = subprocess.run(
        ["git", "-C", str(repo), "diff", old_commit, new_commit, "--"],
        capture_output=True,
        text=True,
        check=True,
    )

    filtered = filter_patch(
        result.stdout,
        include_files=["first.py", "second.py"],
    )

    with RepositoryWorkspace(str(repo), new_commit) as workspace:
        assert apply_repository_patch(workspace, filtered, reverse=True)
        assert (Path(workspace) / "first.py").read_text() == "value = 'old'\n\n"
        assert (Path(workspace) / "second.py").read_text() == "value = 'old'\n\n"


class _EditingRepoAgent:
    def __init__(self, edit_consumer: bool = True):
        self.edit_consumer = edit_consumer
        self.request = ""
        self.output_dir = None

    def generate_repo_bug(
        self,
        workspace: str,
        request: str,
        plan,
        *,
        output_dir: str,
    ) -> str:
        self.request = request
        self.output_dir = output_dir
        root = Path(workspace)
        provider = root / "pkg" / "provider.py"
        provider.write_text(provider.read_text().replace('"current"', '"legacy"'))
        if self.edit_consumer:
            consumer = root / "pkg" / "consumer.py"
            consumer.write_text(consumer.read_text().replace('"current"', '"legacy"'))
        return ""


class _CommonStyleRepoAgent:
    def __init__(self):
        self.request = None

    def run(self, agent_src: Path, request):
        self.request = request
        root = request.git_dir
        provider = root / "pkg" / "provider.py"
        consumer = root / "pkg" / "consumer.py"
        provider.write_text(provider.read_text().replace('"current"', '"legacy"'))
        consumer.write_text(consumer.read_text().replace('"current"', '"legacy"'))
        return SimpleNamespace(patch_path=None)


def test_repo_agent_captures_full_worktree_diff(tmp_path: Path):
    repo, _buggy_commit, fixed_commit = _create_fixed_repo(tmp_path)
    adapter = _EditingRepoAgent()
    engine = SWESmithEngine(agent_adapter=adapter)
    plan = BugGenerationPlan(
        plan_id="agent",
        target_repo_id="contract",
        target_base_commit=fixed_commit,
        target_file="pkg/provider.py",
        target_symbol="token",
        target_files=["pkg/provider.py"],
        target_symbols=["token"],
        strategy="repo_agent",
        constraints=_repo_constraints(),
        task_blueprint={"contract": "provider-consumer token agreement"},
    )

    candidates = engine.generate(
        plan,
        node_code_dir=str(ROOT / "initial_agent" / "src"),
        repo_spec=_repo_spec(repo, fixed_commit),
        output_dir=str(tmp_path / "out_agent"),
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert set(candidate.modified_files) == {"pkg/provider.py", "pkg/consumer.py"}
    assert candidate.target_symbol == ""
    assert candidate.generation_metadata["task_blueprint"]["contract"]
    assert "Inspect the whole repository" in adapter.request
    assert adapter.output_dir.endswith("repo_agent_run")
    assert 'TOKEN = "current"' in (repo / "pkg" / "provider.py").read_text()


def test_repo_agent_rejects_single_file_candidate(tmp_path: Path):
    repo, _buggy_commit, fixed_commit = _create_fixed_repo(tmp_path)
    engine = SWESmithEngine(agent_adapter=_EditingRepoAgent(edit_consumer=False))
    plan = BugGenerationPlan(
        plan_id="single-file",
        target_base_commit=fixed_commit,
        strategy="repo_agent",
        constraints=_repo_constraints(),
    )

    candidates = engine.generate(
        plan,
        node_code_dir=str(ROOT / "initial_agent" / "src"),
        repo_spec=_repo_spec(repo, fixed_commit),
        output_dir=str(tmp_path / "out_single"),
    )

    assert candidates == []


def test_repo_agent_supports_common_agent_adapter_contract(tmp_path: Path):
    repo, _buggy_commit, fixed_commit = _create_fixed_repo(tmp_path)
    adapter = _CommonStyleRepoAgent()
    engine = SWESmithEngine(agent_adapter=adapter)
    plan = BugGenerationPlan(
        plan_id="common-adapter",
        target_base_commit=fixed_commit,
        strategy="repo_agent",
        constraints=_repo_constraints(),
        model="test-model",
    )

    candidates = engine.generate(
        plan,
        node_code_dir=str(ROOT / "initial_agent" / "src"),
        repo_spec=_repo_spec(repo, fixed_commit),
        output_dir=str(tmp_path / "out_common"),
    )

    assert len(candidates) == 1
    assert len(candidates[0].modified_files) == 2
    assert adapter.request.git_dir.name == "repo"
    assert adapter.request.base_commit == "HEAD"
    assert adapter.request.model == "test-model"


def test_planner_routes_repository_failures_to_repo_generators(tmp_path: Path):
    repo, _buggy_commit, fixed_commit = _create_fixed_repo(tmp_path)
    index = RepoIndex.build("contract", str(repo), fixed_commit, source_dirs=["pkg"])
    planner = ProposerPlanner()

    repo_signature = FailureSignature(
        signature_id="incomplete",
        source_trajectory_id="trajectory",
        failure_stage="patch_generation",
        root_cause="solver updated only one side of a contract",
        target_capability="multi-file patch completeness",
        code_patterns=["token"],
    )
    repo_plan = planner.create_plan(repo_signature, index)
    assert repo_plan is not None
    assert repo_plan.strategy == "repo_chain"
    assert repo_plan.constraints.min_modified_files == 2
    assert repo_plan.constraints.max_modified_files == 6

    replay_signature = repo_signature.model_copy(
        update={
            "signature_id": "replay",
            "behavior_pattern": {"reference_commit": fixed_commit},
        }
    )
    replay_plan = planner.create_plan(replay_signature, index)
    assert replay_plan is not None
    assert replay_plan.strategy == "pr_replay"
    assert replay_plan.reference_commit == fixed_commit


def test_engine_registers_repository_strategies():
    strategies = SWESmithEngine().list_strategies()
    assert "pr_replay" in strategies
    assert "repo_agent" in strategies
    assert "repo_chain" in strategies
