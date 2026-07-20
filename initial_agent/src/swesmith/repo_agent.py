from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, List

from .candidate import CandidateArtifact
from .engine import BugGenerationPlan, RepoSpec
from .repo_level import (
    RepositoryWorkspace,
    RepositoryWorkspaceError,
    apply_repository_patch,
    declared_target_files,
    declared_target_symbols,
    extract_patch_from_response,
    repository_diff,
    repository_path,
    validate_repository_patch,
)


REPO_AGENT_SYSTEM_PROMPT = """\
You are constructing a repository-level software repair task. Work directly in
the supplied repository. Introduce one coherent behavioral regression whose
correct repair requires coordinated reasoning across multiple production files.
Do not merely concatenate unrelated mutations. Do not edit tests, generated
artifacts, dependency locks, or git metadata. Do not commit your changes.
"""


REPO_AGENT_REQUEST_TEMPLATE = """\
# Repository-level bug construction

Create a bug candidate in the current repository according to this blueprint:

```json
{blueprint}
```

Capability being probed: {desired_behavior}
Anchor files (hints, not the complete allowed set): {target_files}
Anchor symbols: {target_symbols}

Constraints:
- Modify between {min_files} and {max_files} related production files.
- Modify at most {max_lines} added/deleted lines in total.
- The edits must express one shared contract or causal chain.
- A repair of only one touched file should remain incomplete.
- Deliberately remove, invert, or corrupt existing behavior already covered by
  tests; do not add fallback/error handling or compatibility behavior.
- At least two touched files must each independently trigger a baseline failure
  for the shared contract, while the complete patch retains passing tests.
- Preserve syntax and avoid broad disabling, hard-coded test answers, and test edits.
- Inspect the whole repository and its tests before editing.

Apply the bug-introducing edits directly to the repository worktree. Do not
return an explanatory-only answer and do not commit the changes.
"""


class RepoAgentGenerator:
    """Run a coding agent over a full repository to introduce a coupled bug."""

    def __init__(self, agent_adapter: Any = None) -> None:
        self.agent_adapter = agent_adapter

    def generate(
        self,
        plan: BugGenerationPlan,
        node_code_dir: str,
        repo_spec: RepoSpec,
        output_dir: str,
    ) -> List[CandidateArtifact]:
        if self.agent_adapter is None:
            return []
        source_repo = repository_path(repo_spec, node_code_dir)
        if not source_repo or not os.path.isdir(source_repo):
            return []

        base_commit = str(
            getattr(plan, "target_base_commit", "")
            or getattr(repo_spec, "base_commit", "")
            or "HEAD"
        )
        request_text = self._build_request(plan)
        agent_output = os.path.join(output_dir or source_repo, "repo_agent_run")

        try:
            with RepositoryWorkspace(
                source_repo,
                base_commit=base_commit,
                parent_dir=output_dir or None,
                prefix=f"repo_agent_{plan.plan_id}_",
            ) as workspace:
                try:
                    response_patch = self._invoke_agent(
                        plan=plan,
                        node_code_dir=node_code_dir,
                        workspace=workspace,
                        output_dir=agent_output,
                        request_text=request_text,
                        test_command=str(getattr(repo_spec, "test_command", "") or ""),
                    )
                except Exception:
                    return []
                bug_patch = repository_diff(workspace, "HEAD")
                if not bug_patch and response_patch:
                    extracted = extract_patch_from_response(response_patch)
                    if not extracted and response_patch.lstrip().startswith("diff --git "):
                        extracted = response_patch
                    if extracted and apply_repository_patch(workspace, extracted):
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

        candidate_id = self._make_candidate_id(plan, bug_patch)
        entities = declared_target_symbols(plan)
        artifact = CandidateArtifact(
            candidate_id=candidate_id,
            plan_id=plan.plan_id,
            strategy="repo_agent",
            operator="repository_contract_mutation",
            target_file=summary.changed_files[0],
            target_symbol="",
            bug_patch=bug_patch,
            mutation_site={
                "anchor_files": declared_target_files(plan),
                "anchor_symbols": entities,
                "changed_files": summary.changed_files,
            },
            seed=getattr(plan, "seed", 0),
            before_snippet="",
            after_snippet="",
            generation_metadata={
                "agent": type(self.agent_adapter).__name__,
                "modified_lines": summary.modified_lines,
                "task_blueprint": dict(getattr(plan, "task_blueprint", None) or {}),
                "request_sha256": hashlib.sha256(request_text.encode()).hexdigest(),
            },
            modified_files=summary.changed_files,
            modified_entities=entities,
        )
        if output_dir:
            artifact.save(os.path.join(output_dir, candidate_id))
        return [artifact]

    def _build_request(self, plan: BugGenerationPlan) -> str:
        constraints = plan.constraints
        min_files = max(2, int(getattr(constraints, "min_modified_files", 2) or 2))
        max_files = int(getattr(constraints, "max_modified_files", min_files) or min_files)
        blueprint = dict(getattr(plan, "task_blueprint", None) or {})
        failure_signature = getattr(plan, "failure_signature", None)
        if failure_signature is not None and "failure_signature" not in blueprint:
            blueprint["failure_signature"] = self._to_plain_data(failure_signature)
        if getattr(plan, "rationale", ""):
            blueprint.setdefault("rationale", plan.rationale)
        return REPO_AGENT_REQUEST_TEMPLATE.format(
            blueprint=json.dumps(blueprint, indent=2, ensure_ascii=False, default=str),
            desired_behavior=getattr(constraints, "desired_behavior", "") or "repository reasoning",
            target_files=json.dumps(declared_target_files(plan)),
            target_symbols=json.dumps(declared_target_symbols(plan)),
            min_files=min_files,
            max_files=max_files,
            max_lines=int(getattr(constraints, "max_modified_lines", 20) or 20),
        )

    def _invoke_agent(
        self,
        *,
        plan: BugGenerationPlan,
        node_code_dir: str,
        workspace: str,
        output_dir: str,
        request_text: str,
        test_command: str,
    ) -> str:
        adapter = self.agent_adapter
        os.makedirs(output_dir, exist_ok=True)
        stale_patch = Path(output_dir) / "model_patch.diff"
        if stale_patch.exists():
            stale_patch.unlink()

        if hasattr(adapter, "generate_repo_bug"):
            generate_repo_bug = adapter.generate_repo_bug
            try:
                parameters = inspect.signature(generate_repo_bug).parameters.values()
                supports_output_dir = any(
                    parameter.name == "output_dir"
                    or parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                )
            except (TypeError, ValueError):
                supports_output_dir = False
            if supports_output_dir:
                result = generate_repo_bug(
                    workspace,
                    request_text,
                    plan,
                    output_dir=output_dir,
                )
            else:
                result = generate_repo_bug(workspace, request_text, plan)
            return self._result_patch(result)
        if hasattr(adapter, "run_repo_task"):
            result = adapter.run_repo_task(request_text, workspace, plan)
            return self._result_patch(result)
        if hasattr(adapter, "run_task"):
            result = adapter.run_task(
                request_text,
                REPO_AGENT_SYSTEM_PROMPT,
                str(getattr(plan, "model", "") or ""),
                workspace,
            )
            return self._result_patch(result)
        if hasattr(adapter, "run"):
            try:
                from experiment_adapters.common_agent_adapter import CommonAgentRequest

                request = CommonAgentRequest(
                    problem_statement=REPO_AGENT_SYSTEM_PROMPT + "\n\n" + request_text,
                    git_dir=Path(workspace),
                    base_commit="HEAD",
                    chat_history_file=Path(output_dir) / "trajectory.jsonl",
                    outdir=Path(output_dir),
                    test_description=test_command or None,
                    model=str(getattr(plan, "model", "") or "deepseek/deepseek-chat"),
                    timeout_sec=int(
                        getattr(plan.constraints, "generation_timeout_sec", 3600) or 3600
                    ),
                )
                result = adapter.run(Path(node_code_dir), request)
                return self._result_patch(result)
            except (ImportError, TypeError, OSError):
                return ""
        return ""

    def _result_patch(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, Path):
            return result.read_text(encoding="utf-8") if result.is_file() else ""
        patch_path = getattr(result, "patch_path", None)
        if patch_path:
            path = Path(patch_path)
            if path.is_file():
                return path.read_text(encoding="utf-8")
        patch = getattr(result, "patch", "")
        return str(patch or "")

    def _to_plain_data(self, value: Any) -> Any:
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return value
        return str(value)

    def _make_candidate_id(self, plan: BugGenerationPlan, patch: str) -> str:
        digest = hashlib.sha256(
            f"{plan.plan_id}:repo_agent:{patch}".encode("utf-8")
        ).hexdigest()[:12]
        return f"cand_{digest}"


__all__ = ["RepoAgentGenerator"]
