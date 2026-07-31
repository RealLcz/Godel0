"""Child builder: creates a child node from parent + diagnosis."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from ..constants import (
    PROPOSER_ALLOWED_PATCH_PREFIXES,
    SOLVER_ALLOWED_PATCH_PREFIXES,
)
from ..git.repository import commit as git_commit
from ..git.repository import diff_vs_commit, restore_runtime_artifacts
from ..git.worktree import NodeWorktree, commit_child
from ..schemas.diagnosis import CycleDiagnosis
from ..schemas.mutation import MutationManifest
from ..schemas.node import NodeRecord, NodeStatus
from ..storage.atomic import atomic_write_json, atomic_write_text
from .gates import validate_proposer_schema_compatibility
from .mutation_manifest import build_mutation_manifest
from .patch_guard import PatchGuard, validate_changed_python_syntax
from .self_edit import SelfEditResult, SelfEditRunner

__all__ = [
    "ChildBuildResult",
    "ChildBuilder",
    "validate_changed_python_syntax",
]


@dataclass
class ChildBuildResult:
    passed: bool
    node: Optional[NodeRecord] = None
    manifest: Optional[MutationManifest] = None
    self_edit_result: Optional[SelfEditResult] = None
    proposer_self_edit_result: Optional[SelfEditResult] = None
    solver_self_edit_result: Optional[SelfEditResult] = None
    errors: list[str] = None
    failure_stage: Optional[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ChildBuilder:
    """Builds a child node from a parent and diagnosis.

    Steps (single-diagnosis path):
    1. Create child_id.
    2. Create worktree from parent commit.
    3. Save diagnosis.
    4. Run self-edit (coding agent).
    5. Build mutation manifest.
    6. Check patch allowlist.
    7. Run agent unit tests (if available).
    8. Commit child.
    9. Create NodeRecord(status=CANDIDATE).
    """

    def __init__(
        self,
        agent_repo: Path,
        scratch_root: Path,
        patch_guard: Optional[PatchGuard] = None,
        self_edit_runner: Optional[SelfEditRunner] = None,
        output_root: Optional[Path] = None,
    ):
        self.agent_repo = Path(agent_repo).resolve()
        self.scratch_root = Path(scratch_root).resolve()
        self.patch_guard = patch_guard or PatchGuard()
        self.self_edit_runner = self_edit_runner or SelfEditRunner()
        self.output_root = (
            Path(output_root).resolve() if output_root else self.scratch_root
        )

    def build(
        self,
        parent: NodeRecord,
        diagnosis: CycleDiagnosis,
        model: str = "deepseek/deepseek-chat",
    ) -> ChildBuildResult:
        """Build a child from parent and a single diagnosis (legacy joint path)."""
        child_id = f"node_{uuid.uuid4().hex[:12]}"
        parent_commit = parent.code_commit

        output_dir = self.output_root / child_id / "self_evolve"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with NodeWorktree(self.agent_repo, self.scratch_root, child_id, parent_commit) as worktree:
                self_edit_result = self._run_self_edit(
                    diagnosis=diagnosis,
                    worktree=worktree,
                    output_dir=output_dir,
                    model=model,
                    base_commit=parent_commit,
                )

                if not self_edit_result.success and self_edit_result.error:
                    return ChildBuildResult(
                        passed=False,
                        self_edit_result=self_edit_result,
                        errors=[f"Self-edit failed: {self_edit_result.error}"],
                        failure_stage="self_edit",
                    )

                restore_runtime_artifacts(worktree, parent_commit)
                patch = diff_vs_commit(worktree, parent_commit)
                guard_report = self.patch_guard.check(patch)
                if not guard_report.passed:
                    return ChildBuildResult(
                        passed=False,
                        self_edit_result=self_edit_result,
                        errors=[f"Patch guard: {r}" for r in guard_report.reasons],
                        failure_stage="patch_guard",
                    )

                syntax_errors = validate_changed_python_syntax(worktree, patch)
                if syntax_errors:
                    return ChildBuildResult(
                        passed=False,
                        self_edit_result=self_edit_result,
                        errors=[f"Syntax guard: {error}" for error in syntax_errors],
                        failure_stage="syntax_guard",
                    )

                manifest = build_mutation_manifest(
                    parent_node_id=parent.node_id,
                    child_node_id=child_id,
                    worktree_path=worktree,
                    base_commit=parent_commit,
                    diagnosis_problem_statement=diagnosis.problem_statement,
                )

                gate_errors = self._run_child_gates(worktree, output_dir.parent / "gates")
                if gate_errors:
                    return ChildBuildResult(
                        passed=False,
                        manifest=manifest,
                        self_edit_result=self_edit_result,
                        errors=[f"Child gate: {error}" for error in gate_errors],
                        failure_stage="child_gates",
                    )

                restore_runtime_artifacts(worktree, parent_commit)
                patch = diff_vs_commit(worktree, parent_commit)

                diagnosis_dir = output_dir.parent / "diagnosis"
                diagnosis_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_json(
                    diagnosis_dir / "diagnosis.json",
                    diagnosis.model_dump(mode="json"),
                )
                atomic_write_text(
                    diagnosis_dir / "problem_statement.md",
                    diagnosis.problem_statement.rstrip() + "\n",
                )
                atomic_write_json(
                    output_dir.parent / "mutation_manifest.json",
                    manifest.model_dump(mode="json"),
                )
                atomic_write_text(output_dir / "final.patch", patch)

                child_sha = commit_child(
                    self.agent_repo,
                    worktree,
                    child_id,
                    f"Child node {child_id} from {parent.node_id}",
                )

                child_record = NodeRecord(
                    node_id=child_id,
                    parent_node_id=parent.node_id,
                    code_commit=child_sha,
                    code_ref=f"refs/godel0/nodes/{child_id}",
                    status=NodeStatus.CANDIDATE,
                    mutation_manifest_path=str(output_dir.parent / "mutation_manifest.json"),
                )

                return ChildBuildResult(
                    passed=True,
                    node=child_record,
                    manifest=manifest,
                    self_edit_result=self_edit_result,
                )

        except Exception as e:
            return ChildBuildResult(
                passed=False,
                errors=[f"Child build error: {str(e)}"],
                failure_stage="exception",
            )

    def build_dual(
        self,
        parent: NodeRecord,
        model: str = "deepseek/deepseek-chat",
        *,
        proposer_diagnosis: Optional[CycleDiagnosis] = None,
        solver_diagnosis: Optional[CycleDiagnosis] = None,
        proposer_entry: Optional[dict] = None,
        solver_entry: Optional[dict] = None,
        allow_empty_phase: bool = False,
    ) -> ChildBuildResult:
        """Build one child via sequential proposer then solver self-edits.

        Each provided diagnosis triggers one self-edit on the same worktree.
        After a successful proposer phase, an intermediate commit is created so
        solver-phase retries cannot wipe proposer edits. The final patch is
        always the cumulative diff versus the parent commit.
        """
        if proposer_diagnosis is None and solver_diagnosis is None:
            return ChildBuildResult(
                passed=False,
                errors=["No proposer or solver diagnosis provided for dual self-edit"],
                failure_stage="no_diagnosis",
            )

        child_id = f"node_{uuid.uuid4().hex[:12]}"
        parent_commit = parent.code_commit
        node_dir = self.output_root / child_id
        output_dir = node_dir / "self_evolve"
        diagnosis_dir = node_dir / "diagnosis"
        output_dir.mkdir(parents=True, exist_ok=True)
        diagnosis_dir.mkdir(parents=True, exist_ok=True)

        proposer_result: Optional[SelfEditResult] = None
        solver_result: Optional[SelfEditResult] = None
        last_result: Optional[SelfEditResult] = None
        combined_statement_parts: list[str] = []

        try:
            with NodeWorktree(
                self.agent_repo, self.scratch_root, child_id, parent_commit
            ) as worktree:
                phase_base = parent_commit

                if proposer_diagnosis is not None:
                    if proposer_entry is not None:
                        atomic_write_json(
                            diagnosis_dir / "proposer_entry.json", proposer_entry
                        )
                    atomic_write_json(
                        diagnosis_dir / "proposer_diagnosis.json",
                        proposer_diagnosis.model_dump(mode="json"),
                    )
                    atomic_write_text(
                        diagnosis_dir / "proposer_problem_statement.md",
                        proposer_diagnosis.problem_statement.rstrip() + "\n",
                    )
                    combined_statement_parts.append(
                        "## Proposer phase\n\n" + proposer_diagnosis.problem_statement.strip()
                    )
                    proposer_out = output_dir / "proposer"
                    proposer_out.mkdir(parents=True, exist_ok=True)
                    proposer_result = self._run_self_edit(
                        diagnosis=proposer_diagnosis,
                        worktree=worktree,
                        output_dir=proposer_out,
                        model=model,
                        base_commit=phase_base,
                    )
                    last_result = proposer_result

                    restore_runtime_artifacts(worktree, phase_base)
                    phase_patch = diff_vs_commit(worktree, phase_base)
                    if not phase_patch.strip():
                        if not allow_empty_phase:
                            return ChildBuildResult(
                                passed=False,
                                self_edit_result=proposer_result,
                                proposer_self_edit_result=proposer_result,
                                errors=[
                                    "Proposer self-edit produced no usable patch: "
                                    + (proposer_result.error or "empty")
                                ],
                                failure_stage="proposer_empty_patch",
                            )
                    else:
                        phase_errors = self._validate_phase_patch(
                            role="proposer",
                            worktree=worktree,
                            patch=phase_patch,
                            base_commit=phase_base,
                            gates_dir=node_dir / "gates" / "proposer",
                        )
                        if phase_errors:
                            return ChildBuildResult(
                                passed=False,
                                self_edit_result=proposer_result,
                                proposer_self_edit_result=proposer_result,
                                errors=phase_errors,
                                failure_stage="proposer_phase_gate",
                            )
                        atomic_write_text(proposer_out / "phase.patch", phase_patch)
                        phase_base = git_commit(
                            worktree,
                            f"godel0: proposer self-edit for {child_id}",
                        )

                if solver_diagnosis is not None:
                    if solver_entry is not None:
                        atomic_write_json(
                            diagnosis_dir / "solver_entry.json", solver_entry
                        )
                    atomic_write_json(
                        diagnosis_dir / "solver_diagnosis.json",
                        solver_diagnosis.model_dump(mode="json"),
                    )
                    atomic_write_text(
                        diagnosis_dir / "solver_problem_statement.md",
                        solver_diagnosis.problem_statement.rstrip() + "\n",
                    )
                    combined_statement_parts.append(
                        "## Solver phase\n\n" + solver_diagnosis.problem_statement.strip()
                    )
                    solver_out = output_dir / "solver"
                    solver_out.mkdir(parents=True, exist_ok=True)
                    solver_result = self._run_self_edit(
                        diagnosis=solver_diagnosis,
                        worktree=worktree,
                        output_dir=solver_out,
                        model=model,
                        base_commit=phase_base,
                    )
                    last_result = solver_result

                    restore_runtime_artifacts(worktree, phase_base)
                    phase_patch = diff_vs_commit(worktree, phase_base)
                    if not phase_patch.strip():
                        if not allow_empty_phase:
                            return ChildBuildResult(
                                passed=False,
                                self_edit_result=solver_result,
                                proposer_self_edit_result=proposer_result,
                                solver_self_edit_result=solver_result,
                                errors=[
                                    "Solver self-edit produced no usable patch: "
                                    + (solver_result.error or "empty")
                                ],
                                failure_stage="solver_empty_patch",
                            )
                    else:
                        phase_errors = self._validate_phase_patch(
                            role="solver",
                            worktree=worktree,
                            patch=phase_patch,
                            base_commit=phase_base,
                            gates_dir=node_dir / "gates" / "solver",
                        )
                        if phase_errors:
                            return ChildBuildResult(
                                passed=False,
                                self_edit_result=solver_result,
                                proposer_self_edit_result=proposer_result,
                                solver_self_edit_result=solver_result,
                                errors=phase_errors,
                                failure_stage="solver_phase_gate",
                            )
                        atomic_write_text(solver_out / "phase.patch", phase_patch)

                restore_runtime_artifacts(worktree, parent_commit)
                patch = diff_vs_commit(worktree, parent_commit)
                if not patch.strip():
                    return ChildBuildResult(
                        passed=False,
                        self_edit_result=last_result,
                        proposer_self_edit_result=proposer_result,
                        solver_self_edit_result=solver_result,
                        errors=["Dual self-edit produced no cumulative patch vs parent"],
                        failure_stage="empty_cumulative_patch",
                    )

                guard_report = self.patch_guard.check(patch)
                if not guard_report.passed:
                    return ChildBuildResult(
                        passed=False,
                        self_edit_result=last_result,
                        proposer_self_edit_result=proposer_result,
                        solver_self_edit_result=solver_result,
                        errors=[f"Patch guard: {r}" for r in guard_report.reasons],
                        failure_stage="final_patch_guard",
                    )

                syntax_errors = validate_changed_python_syntax(worktree, patch)
                if syntax_errors:
                    return ChildBuildResult(
                        passed=False,
                        self_edit_result=last_result,
                        proposer_self_edit_result=proposer_result,
                        solver_self_edit_result=solver_result,
                        errors=[f"Syntax guard: {error}" for error in syntax_errors],
                        failure_stage="final_syntax_guard",
                    )

                primary_diagnosis = solver_diagnosis or proposer_diagnosis
                assert primary_diagnosis is not None
                combined_statement = "\n\n".join(combined_statement_parts).strip()
                if combined_statement:
                    primary_diagnosis = primary_diagnosis.model_copy(
                        update={"problem_statement": combined_statement}
                    )

                manifest = build_mutation_manifest(
                    parent_node_id=parent.node_id,
                    child_node_id=child_id,
                    worktree_path=worktree,
                    base_commit=parent_commit,
                    diagnosis_problem_statement=primary_diagnosis.problem_statement,
                )

                gate_errors = self._run_child_gates(worktree, node_dir / "gates")
                if gate_errors:
                    return ChildBuildResult(
                        passed=False,
                        manifest=manifest,
                        self_edit_result=last_result,
                        proposer_self_edit_result=proposer_result,
                        solver_self_edit_result=solver_result,
                        errors=[f"Child gate: {error}" for error in gate_errors],
                        failure_stage="child_gates",
                    )

                restore_runtime_artifacts(worktree, parent_commit)
                patch = diff_vs_commit(worktree, parent_commit)

                atomic_write_json(
                    diagnosis_dir / "diagnosis.json",
                    primary_diagnosis.model_dump(mode="json"),
                )
                atomic_write_text(
                    diagnosis_dir / "problem_statement.md",
                    primary_diagnosis.problem_statement.rstrip() + "\n",
                )
                atomic_write_json(
                    node_dir / "mutation_manifest.json",
                    manifest.model_dump(mode="json"),
                )
                atomic_write_text(output_dir / "final.patch", patch)

                child_sha = commit_child(
                    self.agent_repo,
                    worktree,
                    child_id,
                    f"Child node {child_id} from {parent.node_id}",
                )

                child_record = NodeRecord(
                    node_id=child_id,
                    parent_node_id=parent.node_id,
                    code_commit=child_sha,
                    code_ref=f"refs/godel0/nodes/{child_id}",
                    status=NodeStatus.CANDIDATE,
                    mutation_manifest_path=str(node_dir / "mutation_manifest.json"),
                )

                return ChildBuildResult(
                    passed=True,
                    node=child_record,
                    manifest=manifest,
                    self_edit_result=last_result,
                    proposer_self_edit_result=proposer_result,
                    solver_self_edit_result=solver_result,
                )

        except Exception as e:
            return ChildBuildResult(
                passed=False,
                proposer_self_edit_result=proposer_result,
                solver_self_edit_result=solver_result,
                errors=[f"Child build error: {str(e)}"],
                failure_stage="exception",
            )

    def _validate_phase_patch(
        self,
        *,
        role: str,
        worktree: Path,
        patch: str,
        base_commit: str,
        gates_dir: Path,
    ) -> list[str]:
        """Role-specific PatchGuard + syntax + import/schema checks for one phase."""
        prefixes = (
            PROPOSER_ALLOWED_PATCH_PREFIXES
            if role == "proposer"
            else SOLVER_ALLOWED_PATCH_PREFIXES
        )
        guard = PatchGuard(allowed_prefixes=prefixes)
        report = guard.check(patch)
        errors: list[str] = []
        if not report.passed:
            errors.extend([f"{role} phase patch guard: {r}" for r in report.reasons])

        syntax_errors = validate_changed_python_syntax(worktree, patch)
        if syntax_errors:
            errors.extend([f"{role} phase syntax: {e}" for e in syntax_errors])

        phase_gate_errors = self._run_phase_gates(role, worktree, gates_dir)
        errors.extend(phase_gate_errors)
        return errors

    def _run_phase_gates(
        self, role: str, worktree: Path, gates_dir: Path
    ) -> list[str]:
        """Mechanical import/schema gates for a single evolution phase."""
        gates_dir.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []
        env = self._gate_env(worktree)

        if role == "proposer":
            schema_errors = validate_proposer_schema_compatibility(worktree)
            if schema_errors:
                errors.extend(
                    [f"proposer schema compatibility: {e}" for e in schema_errors]
                )
            commands = [
                (
                    "proposer_help",
                    [sys.executable, "-B", "-m", "proposer.proposer_main", "--help"],
                ),
            ]
        else:
            commands = [
                (
                    "coding_agent_help",
                    [sys.executable, "-B", "coding_agent.py", "--help"],
                ),
                (
                    "import_coding_agent",
                    [sys.executable, "-B", "-c", "import coding_agent"],
                ),
                (
                    "import_llm_withtools",
                    [sys.executable, "-B", "-c", "import llm_withtools"],
                ),
            ]

        for name, command in commands:
            completed = subprocess.run(
                command,
                cwd=worktree,
                env=env,
                text=True,
                capture_output=True,
                timeout=600,
            )
            log = (
                f"command: {' '.join(command)}\n"
                f"exit_code: {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}\n"
            )
            atomic_write_text(gates_dir / f"{name}.txt", log)
            if completed.returncode != 0:
                errors.append(f"{role} phase {name} failed with exit {completed.returncode}")
        return errors

    def _run_self_edit(
        self,
        diagnosis: CycleDiagnosis,
        worktree: Path,
        output_dir: Path,
        model: str,
        base_commit: str,
    ) -> SelfEditResult:
        """Invoke the self-edit runner, tolerating runners without base_commit."""
        try:
            return self.self_edit_runner.run(
                diagnosis=diagnosis,
                worktree=worktree,
                output_dir=output_dir,
                model=model,
                base_commit=base_commit,
            )
        except TypeError:
            return self.self_edit_runner.run(
                diagnosis=diagnosis,
                worktree=worktree,
                output_dir=output_dir,
                model=model,
            )

    def _gate_env(self, worktree: Path) -> dict:
        project_root = Path(__file__).resolve().parents[3]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(worktree), str(project_root)])
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def _run_child_gates(self, worktree: Path, gates_dir: Path) -> list[str]:
        """Validate the whole joint Agent commit in an isolated process."""
        gates_dir.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]
        env = self._gate_env(worktree)
        commands = [
            (
                "agent_codebase",
                [
                    sys.executable,
                    "-B",
                    str(project_root / "scripts" / "validate_agent_codebase.py"),
                    "--code-dir",
                    str(worktree),
                ],
            ),
            (
                "proposer_import",
                [sys.executable, "-B", "-m", "proposer.proposer_main", "--help"],
            ),
            (
                "solver_import",
                [sys.executable, "-B", "-c", "import coding_agent"],
            ),
        ]

        errors: list[str] = []
        for name, command in commands:
            completed = subprocess.run(
                command,
                cwd=worktree,
                env=env,
                text=True,
                capture_output=True,
                timeout=600,
            )
            log = (
                f"command: {' '.join(command)}\n"
                f"exit_code: {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}\n"
            )
            atomic_write_text(gates_dir / f"{name}.txt", log)
            if completed.returncode != 0:
                errors.append(f"{name} failed with exit {completed.returncode}")

        # Final schema compatibility check on the joint child.
        schema_errors = validate_proposer_schema_compatibility(worktree)
        if schema_errors:
            errors.extend([f"schema compatibility: {e}" for e in schema_errors])
        return errors
