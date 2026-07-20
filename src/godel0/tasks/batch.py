"""Task batch builder for proposer batch generation.

Wires the RepoPool -> Proposer -> SWESmithEngine -> CandidateValidator
-> TaskCommitter pipeline.
"""

from __future__ import annotations

import json
import os
import shlex
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Any
from types import SimpleNamespace

from ..schemas.task import TaskRecord
from ..git.patch import extract_changed_files
from ..storage.atomic import atomic_write_json
from ..proposer_trusted.statement_auditor import audit_statement


@dataclass
class TaskBatchResult:
    batch_id: str
    node_id: str
    tasks: List[TaskRecord] = field(default_factory=list)
    complete: bool = False
    rejected_candidates: int = 0
    rejection_reasons: dict = field(default_factory=dict)
    candidates_generated: int = 0
    candidates_validated: int = 0
    validation_reports: List[dict] = field(default_factory=list)
    proposer_error: str = ""
    engine_rejections: List[dict] = field(default_factory=list)


class TaskBatchBuilder:
    """Builds a batch of validated tasks for a node.

    The builder orchestrates:
    1. Read repo specs from the RepoPool.
    2. Create a ProposerRequest with repo_specs.
    3. Run the ProposerRunner to generate candidates.
    4. For each candidate, run the trusted CandidateValidator.
    5. For each validated candidate, commit a task via TaskCommitter.
    6. Stop when K tasks are collected or budget is exhausted.
    """

    def __init__(
        self,
        batch_size: int = 10,
        max_candidates: int = 50,
        strategy_weights: Optional[dict[str, float]] = None,
        contract_test_renderer: str = "",
    ):
        self.batch_size = batch_size
        self.max_candidates = max_candidates
        self.strategy_weights = dict(strategy_weights or {})
        self.contract_test_renderer = str(contract_test_renderer or "")

    def build_for_node(
        self,
        node_id: str,
        repo_pool=None,
        validator=None,
        task_committer=None,
        proposer_runner=None,
        solver_trajectories: Optional[List[str]] = None,
        parent_task_ids: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
        agent_code_dir: Optional[str] = None,
        model: str = "deepseek/deepseek-chat",
        run_id: str = "run",
        task_store_dir: str = "./task_store",
    ) -> TaskBatchResult:
        """Build a task batch from the repo pool through validation.

        Args:
            node_id: The proposer node ID.
            repo_pool: RepoPool instance with base repositories.
            validator: CandidateValidator for trusted validation.
            task_committer: TaskCommitter for committing validated tasks.
            proposer_runner: ProposerRunner for generating candidates.
            solver_trajectories: Paths to solver trajectory JSONL files.
            parent_task_ids: Parent's solved task IDs.
            output_dir: Directory for proposer output.
            agent_code_dir: Path to the agent code directory.
            model: LLM model to use.
            run_id: Run ID.
            task_store_dir: Path to the task store.

        Returns:
            TaskBatchResult with committed tasks and statistics.
        """
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        result = TaskBatchResult(batch_id=batch_id, node_id=node_id)

        if repo_pool is None or proposer_runner is None:
            # Without a repo pool or proposer, return empty batch
            result.complete = False
            return result

        # Get repo specs from the pool
        repo_specs = []
        for spec in repo_pool.all_repos():
            repo_specs.append({
                "repo_id": spec.repo_id,
                "base_commit": spec.base_commit,
                "path": str(Path(spec.path).resolve()),
                "test_command": spec.test_command,
                "install_command": spec.install_command,
                "timeout_sec": spec.timeout_sec,
            })

        if not repo_specs:
            print("No repos in pool, cannot generate tasks")
            result.complete = False
            return result

        # Create the proposer request
        output_dir = Path(output_dir or f"./outputs/{node_id}/proposer").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        from initial_agent.src.proposer.request import ProposerRequest, RepoSpecInfo
        request = ProposerRequest(
            node_id=node_id,
            run_id=run_id,
            agent_code_dir=agent_code_dir or "",
            repo_pool_dir=str(Path(repo_pool.pool_dir).resolve()),
            task_store_dir=str(Path(task_store_dir).resolve()),
            output_dir=str(output_dir),
            target_batch_size=self.batch_size,
            max_candidates=self.max_candidates,
            solver_trajectories=[
                str(Path(value).resolve()) for value in (solver_trajectories or [])
            ],
            parent_task_ids=parent_task_ids or [],
            model=model,
            strategy_weights=self.strategy_weights,
            feedback_dir=str(output_dir / "trusted_feedback"),
            contract_test_renderer=self.contract_test_renderer,
            repo_specs=[
                RepoSpecInfo(
                    repo_id=r["repo_id"],
                    base_commit=r["base_commit"],
                    path=r["path"],
                    test_command=r["test_command"],
                    install_command=r.get("install_command", "pip install -e ."),
                    timeout_sec=r.get("timeout_sec", 120),
                )
                for r in repo_specs
            ],
        )

        attempt = 0
        while (
            len(result.tasks) < self.batch_size
            and result.candidates_generated < self.max_candidates
        ):
            remaining_tasks = self.batch_size - len(result.tasks)
            remaining_candidates = self.max_candidates - result.candidates_generated
            attempt_dir = output_dir / f"attempt_{attempt:03d}"
            attempt_request = replace(
                request,
                output_dir=str(attempt_dir),
                target_batch_size=min(remaining_tasks, remaining_candidates),
                max_candidates=remaining_candidates,
                generation_attempt=attempt,
            )
            proposer_result = proposer_runner.generate_batch(attempt_request)
            proposer_error = str(getattr(proposer_result, "error", "") or "")
            if proposer_error:
                result.proposer_error = proposer_error
            pending_candidates = getattr(proposer_result, "pending_candidates", [])
            candidates_to_validate = (
                list(proposer_result.accepted_candidates) + list(pending_candidates)
            )
            generated_this_attempt = (
                len(proposer_result.accepted_candidates)
                + len(proposer_result.rejected_candidates)
                + len(pending_candidates)
            )
            # A plan consumed generation budget even when the evolvable engine
            # rejected it before emitting a candidate. Counting only emitted
            # artifacts made a zero-yield attempt terminate the whole batch
            # immediately instead of using the configured retry budget.
            generated_this_attempt = max(
                generated_this_attempt,
                len(getattr(proposer_result, "plans", []) or []),
            )
            result.candidates_generated += generated_this_attempt

            if validator is None:
                result.complete = bool(proposer_result.completed)
                return result

            plans_by_id = {
                p.get("plan_id"): p
                for p in getattr(proposer_result, "plans", [])
                if isinstance(p, dict) and p.get("plan_id")
            }
            for plan in plans_by_id.values():
                blueprint = dict(plan.get("task_blueprint") or {})
                rejection = str(blueprint.get("last_rejection") or "")
                if rejection:
                    rejection_record = {
                        "attempt": attempt,
                        "plan_id": plan.get("plan_id"),
                        "reason": rejection,
                    }
                    result.engine_rejections.append(rejection_record)
                    feedback_id = f"engine-{attempt}-{plan.get('plan_id') or 'plan'}"
                    atomic_write_json(
                        output_dir / "trusted_feedback" / f"{feedback_id}.json",
                        {
                            "candidate_id": feedback_id,
                            "accepted": False,
                            "reason": rejection,
                            "notes": rejection_record,
                        },
                    )
            for raw_cand in candidates_to_validate:
                if len(result.tasks) >= self.batch_size:
                    break
                cand = self._normalize_candidate(raw_cand, plans_by_id, repo_specs)

                if not cand.patch:
                    result.rejected_candidates += 1
                    result.rejection_reasons["empty_patch"] = result.rejection_reasons.get("empty_patch", 0) + 1
                    continue

                repo_spec = None
                for r in repo_specs:
                    if r["repo_id"] == cand.repo_id:
                        repo_spec = r
                        break
                if repo_spec is None:
                    result.rejected_candidates += 1
                    result.rejection_reasons["no_repo_spec"] = result.rejection_reasons.get("no_repo_spec", 0) + 1
                    continue

                generation_metadata = dict(getattr(cand, "generation_metadata", {}) or {})
                setup_patch = str(generation_metadata.get("generated_test_patch") or "")
                generated_test_command = self._trusted_test_command(
                    repo_spec,
                    generation_metadata,
                    setup_patch,
                )
                report = validator.validate(
                    candidate_patch=cand.patch,
                    repo_path=Path(repo_spec["path"]),
                    base_commit=repo_spec["base_commit"],
                    test_command=generated_test_command,
                    candidate_id=cand.candidate_id,
                    repo_id=repo_spec["repo_id"],
                    target_file=getattr(cand, "file_path", ""),
                    target_symbol=getattr(cand, "symbol_name", ""),
                    operator=getattr(cand, "operator", ""),
                    setup_patch=setup_patch,
                )
                result.candidates_validated += 1
                problem_statement = cand.issue_draft or "Bug found in repository."
                if report.passed:
                    statement_valid, statement_issues = audit_statement(
                        problem_statement,
                        cand.patch,
                        report.f2p_tests,
                    )
                    if not statement_valid:
                        report.passed = False
                        report.rejection_reasons.extend(
                            f"statement_audit:{issue}" for issue in statement_issues
                        )
                report_data = report.model_dump(mode="json")
                result.validation_reports.append(report_data)
                feedback_reason = "; ".join(report.rejection_reasons)
                atomic_write_json(
                    output_dir / "trusted_feedback" / f"{cand.candidate_id}.json",
                    {
                        "candidate_id": cand.candidate_id,
                        "accepted": bool(report.passed),
                        "reason": feedback_reason,
                        "notes": report_data,
                    },
                )

                if report.passed and task_committer:
                    task = task_committer.commit_task(
                        batch_id=batch_id,
                        proposer_node_id=node_id,
                        repo_id=repo_spec["repo_id"],
                        base_commit=repo_spec["base_commit"],
                        bug_strategy=cand.strategy,
                        bug_patch=cand.patch,
                        problem_statement=problem_statement,
                        f2p_tests=report.f2p_tests,
                        baseline_test_command=generated_test_command,
                        # The solver sees the repository's normal verification
                        # command, never a hard-coded unrelated Ansible test or
                        # the private generated contract path. Trusted scoring
                        # still runs the exact F2P command above.
                        solver_test_command=str(repo_spec["test_command"]),
                        modified_files=cand.modified_files,
                        modified_entities=cand.modified_entities,
                        validation_report=report_data,
                        setup_patch=setup_patch,
                    )
                    result.tasks.append(task)
                else:
                    result.rejected_candidates += 1
                    for reason in report.rejection_reasons:
                        result.rejection_reasons[reason] = result.rejection_reasons.get(reason, 0) + 1

            attempt += 1
            if generated_this_attempt == 0:
                if not result.proposer_error:
                    result.proposer_error = "proposer_generated_zero_candidates"
                break

        result.complete = len(result.tasks) >= self.batch_size
        return result

    def _normalize_candidate(
        self,
        candidate: Any,
        plans_by_id: dict,
        repo_specs: list[dict],
    ) -> SimpleNamespace:
        """Normalize proposer/SWE-smith candidate objects at the trust boundary."""
        data = self._candidate_dict(candidate)
        plan = plans_by_id.get(data.get("plan_id")) or {}
        only_repo = repo_specs[0] if len(repo_specs) == 1 else {}

        def first(*keys: str, default: str = "") -> str:
            for key in keys:
                value = data.get(key)
                if value not in (None, ""):
                    return str(value)
            return default

        patch = first("patch", "bug_patch")
        modified_files = data.get("modified_files") or data.get("changed_files") or []
        if isinstance(modified_files, str):
            modified_files = [modified_files]
        modified_files = [str(value) for value in modified_files if value]
        if not modified_files and patch:
            modified_files = extract_changed_files(patch)
        modified_entities = data.get("modified_entities") or data.get("changed_entities") or []
        if isinstance(modified_entities, str):
            modified_entities = [modified_entities]
        modified_entities = [str(value) for value in modified_entities if value]
        fallback_file = first(
            "file_path",
            "target_file",
            default=str(plan.get("target_file") or ""),
        )
        if not modified_files and fallback_file:
            modified_files = [fallback_file]
        strategy = first("strategy", default=str(plan.get("strategy") or "unknown"))
        if strategy in {"pr_replay", "repo_agent", "repo_chain"}:
            fallback_symbol = ""
        else:
            fallback_symbol = first(
                "symbol_name",
                "target_symbol",
                default=str(plan.get("target_symbol") or ""),
            )
        if not modified_entities and fallback_symbol:
            modified_entities = [fallback_symbol]

        return SimpleNamespace(
            candidate_id=first("candidate_id", default=f"cand_{uuid.uuid4().hex[:12]}"),
            plan_id=first("plan_id"),
            repo_id=first(
                "repo_id",
                default=str(plan.get("target_repo_id") or only_repo.get("repo_id") or ""),
            ),
            base_commit=first(
                "base_commit",
                default=str(plan.get("target_base_commit") or only_repo.get("base_commit") or ""),
            ),
            strategy=strategy,
            operator=first("operator", default=str(plan.get("operator") or "")),
            patch=patch,
            issue_draft=first("issue_draft"),
            file_path=fallback_file or (modified_files[0] if modified_files else ""),
            symbol_name=fallback_symbol,
            modified_files=modified_files,
            modified_entities=modified_entities,
            generation_metadata=dict(data.get("generation_metadata") or {}),
        )

    def _candidate_dict(self, candidate: Any) -> dict:
        """Convert a candidate object/dataclass/dict to a plain dict."""
        if isinstance(candidate, dict):
            return dict(candidate)
        to_dict = getattr(candidate, "to_dict", None)
        if callable(to_dict):
            data = to_dict()
            if isinstance(data, dict):
                return dict(data)
        return dict(getattr(candidate, "__dict__", {}))

    def _trusted_test_command(
        self,
        repo_spec: dict,
        generation_metadata: dict,
        setup_patch: str,
    ) -> str:
        """Construct validation commands only from trusted repo config + paths.

        The evolvable node may propose generated test files, but it must never
        be able to inject an arbitrary shell command into the trusted runner.
        """
        base = str(repo_spec.get("test_command") or "pytest -q")
        if setup_patch.strip():
            setup_files = extract_changed_files(setup_patch)
            declared = [
                str(path)
                for path in generation_metadata.get("generated_test_files") or []
            ]
            if (
                not setup_files
                or set(declared) != set(setup_files)
                or any(
                    path.startswith("/")
                    or ".." in Path(path).parts
                    or not path.startswith(("test/", "tests/"))
                    for path in setup_files
                )
            ):
                # CandidateValidator will reject an invalid setup patch; use the
                # trusted baseline here rather than honoring node-provided command.
                return base
            return base + " " + " ".join(shlex.quote(path) for path in setup_files)
        # Existing-test mode: no setup patch is written, but the proposer may
        # declare existing repository test files to scope validation to. Only
        # honor paths that are safe relative test paths; otherwise fall back to
        # the trusted baseline (which runs the repo's full test command).
        declared_existing = [
            str(path)
            for path in generation_metadata.get("generated_test_files") or []
        ]
        if not declared_existing:
            return base
        repo_path = Path(repo_spec.get("path") or "")
        safe_existing: List[str] = []
        for path in declared_existing:
            if (
                path.startswith("/")
                or ".." in Path(path).parts
                or not path.startswith(("test/", "tests/"))
            ):
                continue
            if repo_path and not (repo_path / path).is_file():
                continue
            safe_existing.append(path)
        if not safe_existing:
            return base
        return base + " " + " ".join(shlex.quote(path) for path in safe_existing)
