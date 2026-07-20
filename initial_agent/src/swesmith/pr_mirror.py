from __future__ import annotations

import hashlib
import json
import os
from typing import Any, List, Optional

from .candidate import CandidateArtifact
from .engine import BugGenerationPlan, RepoSpec
from .patch_utils import make_git_diff, count_modified_lines, extract_changed_files
from .workspace import WorkspaceManager, WorkspaceSpec


PR_MIRROR_REQUEST_TEMPLATE = """\
# PR Mirror Task

A real PR diff was applied to this repository. We want to create an approximate
reverse modification (bug introduction) that mimics the kind of bug this PR fixed.

## PR Metadata

{pr_metadata}

## PR Diff

```diff
{pr_diff}
```

## Target File

{target_file}

## Current Source (at base commit)

```python
{source}
```

## Instructions

Understand what the PR fixed, then introduce a bug that the PR would fix.
Apply the edit to the file in the workspace.
"""


class PRMirror:
    def __init__(self, agent_adapter: Any = None) -> None:
        self.agent_adapter = agent_adapter
        self.workspace_manager = WorkspaceManager()

    def generate(
        self,
        plan: BugGenerationPlan,
        node_code_dir: str,
        repo_spec: RepoSpec,
        output_dir: str,
        pr_diff_dir: Optional[str] = None,
    ) -> List[CandidateArtifact]:
        target_file = plan.target_file
        if not target_file:
            return []

        file_path = target_file
        if not os.path.isabs(file_path):
            file_path = os.path.join(repo_spec.repo_path or node_code_dir, target_file)

        if not os.path.exists(file_path):
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            original_source = f.read()

        pr_diff = ""
        pr_metadata: dict = {}
        if pr_diff_dir:
            pr_diff, pr_metadata = self._load_pr_data(pr_diff_dir, plan.target_repo_id)
        elif hasattr(plan, "pr_diff") and plan.pr_diff:
            pr_diff = plan.pr_diff

        if not pr_diff:
            return []

        workspace_dir = os.path.join(
            output_dir or "/tmp/swesmith_pr_mirror",
            f"ws_{plan.plan_id}",
        )
        spec = WorkspaceSpec(
            workspace_dir=workspace_dir,
            target_file=target_file,
            target_symbol=plan.target_symbol,
            base_commit=plan.target_base_commit,
            repo_id=plan.target_repo_id,
            expose_full_source=True,
            restricted=False,
        )
        self.workspace_manager.create_workspace(spec, source_content=original_source)

        request_text = PR_MIRROR_REQUEST_TEMPLATE.format(
            pr_metadata=json.dumps(pr_metadata, indent=2),
            pr_diff=pr_diff,
            target_file=target_file,
            source=original_source,
        )
        self.workspace_manager.write_request(workspace_dir, request_text)
        self.workspace_manager.write_file(workspace_dir, target_file, original_source)

        if self.agent_adapter is None:
            return []

        try:
            modified_source = self._call_agent(workspace_dir, target_file, request_text)
        except Exception:
            return []

        if not modified_source or modified_source == original_source:
            return []

        try:
            import ast as _ast
            _ast.parse(modified_source)
        except SyntaxError:
            return []

        patch = make_git_diff(original_source, modified_source, filename=target_file)
        if not patch:
            return []

        if count_modified_lines(patch) > plan.constraints.max_modified_lines:
            return []

        candidate_id = self._make_candidate_id(plan)
        artifact = CandidateArtifact(
            candidate_id=candidate_id,
            plan_id=plan.plan_id,
            strategy="pr_mirror",
            operator="pr_mirror",
            target_file=target_file,
            target_symbol=plan.target_symbol,
            bug_patch=patch,
            mutation_site={"pr_source": pr_metadata.get("pr_id", "unknown")},
            seed=plan.seed,
            before_snippet=original_source[:500],
            after_snippet=modified_source[:500],
            generation_metadata={
                "agent": str(type(self.agent_adapter).__name__),
                "pr_metadata": pr_metadata,
            },
        )

        if output_dir:
            candidate_dir = os.path.join(output_dir, candidate_id)
            artifact.save(candidate_dir)

        return [artifact]

    def _load_pr_data(self, pr_diff_dir: str, repo_id: str) -> tuple:
        repo_pr_dir = os.path.join(pr_diff_dir, repo_id) if repo_id else pr_diff_dir
        if not os.path.isdir(repo_pr_dir):
            return "", {}

        diff_path = os.path.join(repo_pr_dir, "pr.diff")
        meta_path = os.path.join(repo_pr_dir, "metadata.json")

        pr_diff = ""
        if os.path.exists(diff_path):
            with open(diff_path, "r", encoding="utf-8") as f:
                pr_diff = f.read()

        metadata = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}

        return pr_diff, metadata

    def _call_agent(
        self,
        workspace_dir: str,
        target_file: str,
        request_text: str,
    ) -> str:
        if hasattr(self.agent_adapter, "mirror_pr"):
            return self.agent_adapter.mirror_pr(workspace_dir, target_file, request_text)
        if hasattr(self.agent_adapter, "run"):
            result = self.agent_adapter.run(workspace_dir, request_text)
            if isinstance(result, str):
                target_path = os.path.join(workspace_dir, target_file)
                if os.path.exists(target_path):
                    with open(target_path, "r", encoding="utf-8") as f:
                        return f.read()
                return result
        target_path = os.path.join(workspace_dir, target_file)
        if os.path.exists(target_path):
            with open(target_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def _make_candidate_id(self, plan: BugGenerationPlan) -> str:
        raw = f"{plan.plan_id}:pr_mirror:{plan.target_file}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        return f"cand_{digest}"
