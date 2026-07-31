"""Tests for HGM-style dual diagnose + self-edit plumbing."""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from godel0.evolution.agent_code_dump import dump_agent_code, role_code_summary
from godel0.evolution.entry_selector import (
    choose_proposer_failure,
    choose_solver_failure,
    list_proposer_failures,
)
from godel0.evolution.hgm_diagnose import (
    HgmDiagnoseClips,
    HgmEntryDiagnoser,
    build_proposer_diagnose_messages,
    build_solver_diagnose_messages,
    clip_text,
    parse_diagnose_json,
    wrap_problem_statement,
)
from godel0.evolution.self_edit import EDIT_PROTOCOL, SelfEditRunner
from godel0.evolution.child_builder import ChildBuilder
from godel0.evolution.entry_selector import ProposerFailureEntry, SolverFailureEntry
from godel0.schemas.diagnosis import CycleDiagnosis
from godel0.schemas.node import NodeRecord, NodeStatus


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_edit_protocol_prefers_focused_live_path_changes():
    assert "smallest change" not in EDIT_PROTOCOL.lower()
    assert "at most 2 files" not in EDIT_PROTOCOL.lower()
    assert "do not stop at a cosmetic" not in EDIT_PROTOCOL.lower()
    assert "class of failures" not in EDIT_PROTOCOL.lower()
    assert "simplest coherent" in EDIT_PROTOCOL.lower()
    assert "breadth is not evidence of quality" in EDIT_PROTOCOL.lower()


def test_clip_text_keeps_head_and_tail():
    text = "A" * 1000 + "MID" + "B" * 1000
    out = clip_text(text, 200)
    assert len(out) <= 200
    assert out.startswith("A")
    assert out.endswith("B")
    assert "truncated" in out


def test_choose_proposer_failure_prefers_rejected(tmp_path: Path):
    proposer = tmp_path / "proposer"
    feedback = proposer / "trusted_feedback"
    feedback.mkdir(parents=True)
    (feedback / "cand_ok.json").write_text(
        json.dumps({"candidate_id": "cand_ok", "accepted": True, "reason": ""}),
        encoding="utf-8",
    )
    (feedback / "cand_bad.json").write_text(
        json.dumps(
            {
                "candidate_id": "cand_bad",
                "accepted": False,
                "reason": "no_f2p",
                "notes": {"candidate_id": "cand_bad", "passed": False},
            }
        ),
        encoding="utf-8",
    )
    attempt = proposer / "attempt_000"
    attempt.mkdir()
    (attempt / "proposer.stdout.log").write_text("planning...\n", encoding="utf-8")
    cand_dir = attempt / "proposer_candidates" / "plan-x" / "cand_bad"
    cand_dir.mkdir(parents=True)
    (cand_dir / "bug.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    (cand_dir / "problem_statement.md").write_text("broken task\n", encoding="utf-8")

    failures = list_proposer_failures(proposer)
    assert len(failures) == 1
    assert failures[0].candidate_id == "cand_bad"
    assert failures[0].bug_patch.startswith("diff")
    assert "broken task" in failures[0].problem_statement

    chosen = choose_proposer_failure(proposer, rng=random.Random(0))
    assert chosen is not None
    assert chosen.candidate_id == "cand_bad"


def test_choose_solver_failure_prefers_level2(tmp_path: Path):
    scratch = tmp_path / "solver" / "root"
    traj_dir = scratch / "trajectories" / "level_2" / "task_fail" / "rollout_0"
    traj_dir.mkdir(parents=True)
    (traj_dir / "trajectory.jsonl").write_text('{"step":1}\n', encoding="utf-8")
    (traj_dir / "model_patch.diff").write_text("+x\n", encoding="utf-8")
    task_store = tmp_path / "task_store" / "task_fail"
    task_store.mkdir(parents=True)
    (task_store / "problem_statement.md").write_text("fix the bug\n", encoding="utf-8")
    (task_store / "failing_test_output.txt").write_text("FAILED\n", encoding="utf-8")

    level2 = tmp_path / "level2.json"
    level2.write_text(
        json.dumps(
            {
                "node_id": "root",
                "task_batch_id": "b",
                "evaluated_task_ids": ["task_fail"],
                "solved_task_ids": [],
                "failed_task_ids": ["task_fail"],
                "accuracy": 0.0,
            }
        ),
        encoding="utf-8",
    )

    entry = choose_solver_failure(
        level2_result_path=level2,
        level1_result_path=None,
        scratch_solver_root=scratch,
        task_store_root=tmp_path / "task_store",
        rng=random.Random(1),
    )
    assert entry is not None
    assert entry.task_id == "task_fail"
    assert entry.level == 2
    assert "fix the bug" in entry.problem_statement
    assert entry.predicted_patch.strip() == "+x"


def test_diagnose_prompt_contains_code_logs_and_failure():
    entry = ProposerFailureEntry(
        candidate_id="cand_x",
        reason="causal_ablation_failure",
        stdout_log="generated plan",
        stderr_log="warn",
        bug_patch="diff --git a/a b/a\n",
        problem_statement="Make X fail",
        failing_test_output="3 failed",
        validation_report={"passed": False, "rejection_reasons": ["no_f2p"]},
    )
    system, user = build_proposer_diagnose_messages(
        entry, code_dump="def plan():\n    pass\n", clips=HgmDiagnoseClips()
    )
    assert "def plan" in system
    assert "cand_x" in user
    assert "generated plan" in user
    assert "diff --git" in user
    assert "3 failed" in user

    solver = SolverFailureEntry(
        task_id="task_1",
        level=2,
        problem_statement="Issue text",
        trajectory_text="tool: editor",
        predicted_patch="+fix",
        eval_log="FAIL_TO_PASS: 1",
    )
    system2, user2 = build_solver_diagnose_messages(
        solver, code_dump="class AgenticSystem: pass\n", clips=HgmDiagnoseClips()
    )
    assert "AgenticSystem" in system2
    assert "Issue text" in user2
    assert "tool: editor" in user2
    assert "+fix" in user2


def test_wrap_and_parse_diagnose_json():
    wrapped = wrap_problem_statement(
        role="solver",
        implementation_suggestion="Add better localization.",
        problem_description="Agent stops too early.",
    )
    assert "Add better localization" in wrapped
    assert "GENERALIZATION" in role_code_summary("solver") or "coding_agent" in wrapped.lower() or "To Implement" in wrapped

    raw = """```json
{
  "failure_summary": "agent stopped early",
  "primary_root_cause": "weak localization loop",
  "generalization": "affects many localization failures",
  "single_improvement": "retry localization with failing tests",
  "edit_scope": ["coding_agent.py", "tools/"],
  "implementation_suggestion": "extend tools",
  "expected_behavior_change": "agent re-reads failing tests before editing",
  "problem_description": "do X"
}
```"""
    data = parse_diagnose_json(raw)
    assert data["implementation_suggestion"] == "extend tools"
    assert data["primary_root_cause"] == "weak localization loop"


def test_hgm_diagnoser_returns_none_without_adapter(tmp_path: Path):
    repo = tmp_path / "agent"
    (repo / "proposer").mkdir(parents=True)
    (repo / "proposer" / "planner.py").write_text("x=1\n", encoding="utf-8")
    diagnoser = HgmEntryDiagnoser(chat_adapter=None, model="")
    entry = ProposerFailureEntry(candidate_id="c1", reason="no_f2p")
    diagnosis = diagnoser.diagnose_proposer(
        node_id="root", entry=entry, agent_repo=repo
    )
    assert diagnosis is None


def test_parse_diagnose_json_rejects_dangerous_bypass():
    raw = """```json
{
  "failure_summary": "validation exception",
  "primary_root_cause": "validator too strict",
  "generalization": "all candidates",
  "single_improvement": "mark passed=True after exception",
  "edit_scope": ["proposer/"],
  "implementation_suggestion": "accept on exception in validator adapter",
  "expected_behavior_change": "more candidates pass",
  "problem_description": "bypass validation on exception"
}
```"""
    with pytest.raises(Exception):
        parse_diagnose_json(raw)


def test_parse_diagnose_json_rejects_large_edit_scope():
    raw = """```json
{
  "failure_summary": "x",
  "primary_root_cause": "y",
  "generalization": "z",
  "single_improvement": "a",
  "edit_scope": ["a", "b", "c", "d", "e"],
  "implementation_suggestion": "extend tools",
  "expected_behavior_change": "better",
  "problem_description": "do X"
}
```"""
    with pytest.raises(Exception):
        parse_diagnose_json(raw)


def test_dump_agent_code_role_scoped(tmp_path: Path):
    repo = tmp_path / "agent"
    (repo / "proposer").mkdir(parents=True)
    (repo / "proposer" / "a.py").write_text("proposer=1\n", encoding="utf-8")
    (repo / "coding_agent.py").write_text("solver=1\n", encoding="utf-8")
    (repo / "tools").mkdir()
    (repo / "tools" / "edit.py").write_text("tool=1\n", encoding="utf-8")
    prop = dump_agent_code(repo, "proposer")
    assert "proposer=1" in prop
    assert "solver=1" not in prop
    sol = dump_agent_code(repo, "solver")
    assert "solver=1" in sol
    assert "tool=1" in sol


class _PhaseAdapter:
    def __init__(self):
        self.calls = []

    def run(self, agent_src, request):
        self.calls.append(request.problem_statement)
        git_dir = Path(request.git_dir)
        # Distinguish phases by a marker in the problem statement.
        if "PROPOSER_MARK" in request.problem_statement:
            target = git_dir / "proposer" / "planner.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("def plan():\n    return 'p'\n", encoding="utf-8")
        elif "SOLVER_MARK" in request.problem_statement:
            target = git_dir / "coding_agent.py"
            target.write_text("def forward():\n    return 's'\n", encoding="utf-8")
        return SimpleNamespace(success=True, patch_path=None, error=None)


@pytest.fixture()
def agent_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "agent_repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "proposer").mkdir()
    (repo / "proposer" / "planner.py").write_text("def plan():\n    return 0\n", encoding="utf-8")
    (repo / "coding_agent.py").write_text("def forward():\n    return 0\n", encoding="utf-8")
    (repo / "tools").mkdir()
    (repo / "tools" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def test_child_builder_dual_runs_both_phases(agent_repo: Path, tmp_path: Path):
    adapter = _PhaseAdapter()
    builder = ChildBuilder(
        agent_repo=agent_repo,
        scratch_root=tmp_path / "scratch",
        self_edit_runner=SelfEditRunner(agent_adapter=adapter, max_attempts=1),
        output_root=tmp_path / "nodes",
    )
    # Bypass heavy child/phase gates for unit test.
    builder._run_child_gates = lambda worktree, gates_dir: []  # type: ignore[method-assign]
    builder._run_phase_gates = lambda role, worktree, gates_dir: []  # type: ignore[method-assign]
    # Patch guard may reject unknown paths depending on allowlist; stub final check.
    builder.patch_guard.check = lambda patch: SimpleNamespace(passed=True, reasons=[])  # type: ignore

    parent = NodeRecord(
        node_id="root",
        parent_node_id=None,
        code_commit=subprocess.check_output(
            ["git", "-C", str(agent_repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        code_ref="refs/godel0/nodes/root",
        status=NodeStatus.COMPLETE,
    )
    proposer_diag = CycleDiagnosis(
        node_id="root",
        primary_root_cause="proposer",
        source_stages=["proposer"],
        problem_statement="PROPOSER_MARK improve proposer",
    )
    solver_diag = CycleDiagnosis(
        node_id="root",
        primary_root_cause="solver",
        source_stages=["solver"],
        problem_statement="SOLVER_MARK improve solver",
    )
    result = builder.build_dual(
        parent,
        model="test-model",
        proposer_diagnosis=proposer_diag,
        solver_diagnosis=solver_diag,
        proposer_entry={"candidate_id": "c1"},
        solver_entry={"task_id": "t1"},
    )
    assert result.passed, result.errors
    assert result.node is not None
    assert len(adapter.calls) == 2
    child_diag = tmp_path / "nodes" / result.node.node_id / "diagnosis"
    assert (child_diag / "proposer_problem_statement.md").is_file()
    assert (child_diag / "solver_problem_statement.md").is_file()
    assert (child_diag / "problem_statement.md").is_file()
    final_patch = (tmp_path / "nodes" / result.node.node_id / "self_evolve" / "final.patch").read_text()
    assert "planner.py" in final_patch or "coding_agent.py" in final_patch


def test_child_builder_dual_skips_missing_phase(agent_repo: Path, tmp_path: Path):
    adapter = _PhaseAdapter()
    builder = ChildBuilder(
        agent_repo=agent_repo,
        scratch_root=tmp_path / "scratch",
        self_edit_runner=SelfEditRunner(agent_adapter=adapter, max_attempts=1),
        output_root=tmp_path / "nodes",
    )
    builder._run_child_gates = lambda worktree, gates_dir: []  # type: ignore[method-assign]
    builder._run_phase_gates = lambda role, worktree, gates_dir: []  # type: ignore[method-assign]
    builder.patch_guard.check = lambda patch: SimpleNamespace(passed=True, reasons=[])  # type: ignore
    parent = NodeRecord(
        node_id="root",
        parent_node_id=None,
        code_commit=subprocess.check_output(
            ["git", "-C", str(agent_repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        code_ref="refs/godel0/nodes/root",
        status=NodeStatus.COMPLETE,
    )
    result = builder.build_dual(
        parent,
        model="test-model",
        proposer_diagnosis=None,
        solver_diagnosis=CycleDiagnosis(
            node_id="root",
            primary_root_cause="solver",
            source_stages=["solver"],
            problem_statement="SOLVER_MARK only",
        ),
    )
    assert result.passed, result.errors
    assert len(adapter.calls) == 1
