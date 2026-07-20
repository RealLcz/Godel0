from __future__ import annotations

import hashlib
import os
from pathlib import PurePosixPath
from typing import List

from .candidate import CandidateArtifact
from .engine import BugGenerationPlan, RepoSpec
from .repo_level import (
    RepositoryWorkspace,
    RepositoryWorkspaceError,
    apply_repository_patch,
    declared_target_symbols,
    filter_patch,
    repository_diff,
    repository_path,
    run_git,
    validate_repository_patch,
)
from .patch_utils import extract_changed_files


class PRReplayGenerator:
    """Create a repository-level bug by reversing a real multi-file fix.

    ``reference_patch`` and ``reference_commit`` describe a forward fix
    (buggy -> fixed). The task base must contain the fixed state. The generator
    reverses the source hunks in an isolated clone and returns the resulting
    fixed -> buggy patch.
    """

    def generate(
        self,
        plan: BugGenerationPlan,
        node_code_dir: str,
        repo_spec: RepoSpec,
        output_dir: str,
    ) -> List[CandidateArtifact]:
        source_repo = repository_path(repo_spec, node_code_dir)
        if not source_repo or not os.path.isdir(source_repo):
            return []

        reference_commit = str(getattr(plan, "reference_commit", "") or "")
        base_commit = str(
            getattr(plan, "target_base_commit", "")
            or getattr(repo_spec, "base_commit", "")
            or reference_commit
            or "HEAD"
        )
        full_fix_patch = self._load_fix_patch(plan, source_repo, reference_commit)
        if not full_fix_patch:
            return []

        reference_parent = str(
            getattr(plan, "reference_parent", "")
            or (f"{reference_commit}^1" if reference_commit else "")
        )
        reference_changed_files = extract_changed_files(full_fix_patch)
        reference_test_files = [
            path
            for path in reference_changed_files
            if self._is_test_path(path)
        ]

        explicit_files = list(getattr(plan, "target_files", None) or [])
        fix_patch = filter_patch(
            full_fix_patch,
            include_files=explicit_files or None,
            allow_test_edits=bool(plan.constraints.allow_test_edits),
        )
        if not fix_patch:
            return []

        workspace_parent = output_dir or None
        try:
            with RepositoryWorkspace(
                source_repo,
                base_commit=base_commit,
                parent_dir=workspace_parent,
                prefix=f"pr_replay_{plan.plan_id}_",
            ) as workspace:
                if not apply_repository_patch(workspace, fix_patch, reverse=True):
                    return []
                bug_patch = repository_diff(workspace, "HEAD")
        except RepositoryWorkspaceError:
            return []

        summary = validate_repository_patch(
            bug_patch,
            plan.constraints,
            require_multiple_files=True,
        )
        if not summary.valid:
            return []

        candidate_files = set(summary.changed_files)
        reference_hidden_files = [
            path for path in reference_changed_files if path not in candidate_files
        ]

        candidate_id = self._make_candidate_id(plan, bug_patch)
        symbols = declared_target_symbols(plan)
        artifact = CandidateArtifact(
            candidate_id=candidate_id,
            plan_id=plan.plan_id,
            strategy="pr_replay",
            operator="reverse_real_fix",
            target_file=summary.changed_files[0],
            target_symbol="",
            bug_patch=bug_patch,
            mutation_site={
                "reference_commit": reference_commit,
                "reference_parent": reference_parent,
                "changed_files": summary.changed_files,
            },
            seed=getattr(plan, "seed", 0),
            before_snippet="",
            after_snippet="",
            generation_metadata={
                "reference_kind": "commit" if reference_commit else "patch",
                "reference_patch_sha256": hashlib.sha256(fix_patch.encode()).hexdigest(),
                "reference_changed_files": reference_changed_files,
                "reference_test_files": reference_test_files,
                "reference_hidden_files": reference_hidden_files,
                "modified_lines": summary.modified_lines,
                "task_blueprint": dict(getattr(plan, "task_blueprint", None) or {}),
            },
            modified_files=summary.changed_files,
            modified_entities=symbols,
        )
        if output_dir:
            artifact.save(os.path.join(output_dir, candidate_id))
        return [artifact]

    @staticmethod
    def _is_test_path(path: str) -> bool:
        normalized = PurePosixPath(path)
        parts = set(normalized.parts)
        name = normalized.name.lower()
        return bool(
            parts.intersection({"test", "tests"})
            or name.startswith("test_")
            or name.endswith(("_test.py", ".spec.js", ".test.js"))
        )

    def _load_fix_patch(
        self,
        plan: BugGenerationPlan,
        source_repo: str,
        reference_commit: str,
    ) -> str:
        inline = str(getattr(plan, "reference_patch", "") or "")
        if inline.strip():
            return inline

        patch_path = str(getattr(plan, "reference_patch_path", "") or "")
        if patch_path:
            candidates = [patch_path]
            if not os.path.isabs(patch_path):
                candidates.append(os.path.join(source_repo, patch_path))
            for candidate in candidates:
                if os.path.isfile(candidate):
                    with open(candidate, "r", encoding="utf-8") as handle:
                        return handle.read()

        commit = reference_commit or str(getattr(plan, "target_base_commit", "") or "")
        if not commit:
            commit = str(getattr(plan, "target_base_commit", "") or "HEAD")
        parent = str(getattr(plan, "reference_parent", "") or f"{commit}^1")
        result = run_git(
            source_repo,
            "diff",
            "--binary",
            "--full-index",
            parent,
            commit,
            "--",
        )
        return result.stdout if result.returncode == 0 else ""

    def _make_candidate_id(self, plan: BugGenerationPlan, patch: str) -> str:
        digest = hashlib.sha256(
            f"{plan.plan_id}:pr_replay:{patch}".encode("utf-8")
        ).hexdigest()[:12]
        return f"cand_{digest}"


__all__ = ["PRReplayGenerator"]
