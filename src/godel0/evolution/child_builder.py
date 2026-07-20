"""Child builder: creates a child node from parent + diagnosis."""

from __future__ import annotations

import uuid
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..errors import PatchGuardError
from ..schemas.diagnosis import CycleDiagnosis
from ..schemas.mutation import MutationManifest
from ..schemas.node import NodeRecord, NodeStatus
from ..git.repository import get_head_sha, diff_vs_commit
from ..git.worktree import NodeWorktree, commit_child
from .patch_guard import PatchGuard
from .mutation_manifest import build_mutation_manifest
from .self_edit import SelfEditRunner, SelfEditResult
from ..storage.atomic import atomic_write_json, atomic_write_text


def validate_changed_python_syntax(worktree: Path, patch: str) -> list[str]:
    """Compile changed Python sources without importing or writing bytecode."""
    from ..git.patch import extract_changed_files

    errors: list[str] = []
    for relative_path in extract_changed_files(patch):
        if not relative_path.endswith(".py"):
            continue
        source_path = worktree / relative_path
        if not source_path.is_file():
            continue
        try:
            compile(source_path.read_bytes(), str(source_path), "exec")
        except (SyntaxError, ValueError) as exc:
            line = getattr(exc, "lineno", None)
            location = f":{line}" if line else ""
            errors.append(f"Invalid Python syntax in {relative_path}{location}: {exc.msg if isinstance(exc, SyntaxError) else exc}")
    return errors


@dataclass
class ChildBuildResult:
    passed: bool
    node: Optional[NodeRecord] = None
    manifest: Optional[MutationManifest] = None
    self_edit_result: Optional[SelfEditResult] = None
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ChildBuilder:
    """Builds a child node from a parent and diagnosis.

    Steps:
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
        """Build a child from parent and diagnosis."""
        child_id = f"node_{uuid.uuid4().hex[:12]}"
        parent_commit = parent.code_commit

        output_dir = self.output_root / child_id / "self_evolve"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with NodeWorktree(self.agent_repo, self.scratch_root, child_id, parent_commit) as worktree:
                self_edit_result = self.self_edit_runner.run(
                    diagnosis=diagnosis,
                    worktree=worktree,
                    output_dir=output_dir,
                    model=model,
                )

                if not self_edit_result.success and self_edit_result.error:
                    return ChildBuildResult(
                        passed=False,
                        self_edit_result=self_edit_result,
                        errors=[f"Self-edit failed: {self_edit_result.error}"],
                    )

                patch = diff_vs_commit(worktree, parent_commit)
                guard_report = self.patch_guard.check(patch)
                if not guard_report.passed:
                    return ChildBuildResult(
                        passed=False,
                        self_edit_result=self_edit_result,
                        errors=[f"Patch guard: {r}" for r in guard_report.reasons],
                    )

                syntax_errors = validate_changed_python_syntax(worktree, patch)
                if syntax_errors:
                    return ChildBuildResult(
                        passed=False,
                        self_edit_result=self_edit_result,
                        errors=[f"Syntax guard: {error}" for error in syntax_errors],
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
                    )

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
            )

    def _run_child_gates(self, worktree: Path, gates_dir: Path) -> list[str]:
        """Validate the whole joint Agent commit in an isolated process."""
        gates_dir.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(worktree), str(project_root)])
        commands = [
            (
                "agent_codebase",
                [
                    sys.executable,
                    str(project_root / "scripts" / "validate_agent_codebase.py"),
                    "--code-dir",
                    str(worktree),
                ],
            ),
            (
                "proposer_import",
                [sys.executable, "-m", "proposer.proposer_main", "--help"],
            ),
        ]
        if (worktree / "tests").is_dir():
            commands.append(
                ("agent_tests", [sys.executable, "-m", "pytest", "-q", "tests"])
            )

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
        return errors
