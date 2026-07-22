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


def compute_effective_quotas(
    batch_size: int,
    nominal_parent: int,
    nominal_child: int,
    available_parent: int,
    available_child: int,
    *,
    bootstrap: bool = False,
) -> dict:
    """P0-10/11: compute effective source quotas before generation.

    Supports donation:
      - parent has 3 sources, child many → 3+7
      - child has 0 sources → 10+0
      - parent has 0 sources → 0+10
      - both unknown (0,0) → keep nominal 5+5
    """
    if bootstrap:
        return {
            "parent_failure": 0,
            "current_child_level1": 0,
            "bootstrap": int(batch_size),
        }
    k = max(1, int(batch_size))
    avail_p = max(0, int(available_parent))
    avail_c = max(0, int(available_child))
    nominal_p = max(0, int(nominal_parent))
    # Both unknown: keep the configured nominal split.
    if avail_p == 0 and avail_c == 0:
        parent_quota = min(nominal_p, k)
        return {
            "parent_failure": parent_quota,
            "current_child_level1": k - parent_quota,
            "bootstrap": 0,
        }

    parent_quota = min(nominal_p, avail_p) if avail_p > 0 else 0
    child_quota = k - parent_quota
    if avail_c < child_quota:
        child_quota = avail_c
        parent_quota = k - child_quota
    # Do NOT re-cap parent by avail_p after donation: generation may reuse
    # the available parent trajectories to fill the donated slots (5+5→10+0).

    parent_quota = max(0, min(parent_quota, k))
    child_quota = max(0, k - parent_quota)
    return {
        "parent_failure": parent_quota,
        "current_child_level1": child_quota,
        "bootstrap": 0,
    }


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
        source_quotas: Optional[dict] = None,
        workflow_config: Optional[dict] = None,
        allow_workflow_fallback: bool = False,
        allow_human_curated_data: bool = False,
    ):
        self.batch_size = batch_size
        self.max_candidates = max_candidates
        self.strategy_weights = dict(strategy_weights or {})
        self.contract_test_renderer = str(contract_test_renderer or "")
        # Phase 6: Task Source Quota. source_quotas is a dict like
        # {"parent_failure": 5, "current_child_level1": 5}. When present, the
        # builder tags each committed task with its source type.
        self.source_quotas = dict(source_quotas or {})
        self.workflow_config = dict(workflow_config or {})
        self.allow_workflow_fallback = bool(allow_workflow_fallback)
        self.allow_human_curated_data = bool(allow_human_curated_data)

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
        bootstrap: bool = False,
        parent_failure_trajectories: Optional[List[str]] = None,
        current_child_level1_trajectories: Optional[List[str]] = None,
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
            parent_failure_trajectories: BUG-08/09 parent Level2 unresolved
                trajectories. When populated, the builder enforces the
                ``parent_failure`` quota against this bucket.
            current_child_level1_trajectories: BUG-08/09 current-child Level1
                unresolved/forgotten trajectories. When populated, the builder
                enforces the ``current_child_level1`` quota against this bucket.

        Returns:
            TaskBatchResult with committed tasks and statistics.
        """
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        result = TaskBatchResult(batch_id=batch_id, node_id=node_id)

        # BUG-08/09: 5+5 quota with dynamic fallback. Prefer the split
        # trajectory buckets when available; otherwise fall back to the flat
        # list. ``source_counts`` tracks how many accepted tasks come from
        # each source so we can stop accepting a source once it is full and
        # reallocate the remainder to the other source (5+5 -> 4+6 -> ...).
        parent_failure_trajectories = list(parent_failure_trajectories or [])
        current_child_level1_trajectories = list(current_child_level1_trajectories or [])
        if not parent_failure_trajectories and not current_child_level1_trajectories:
            parent_failure_trajectories = list(solver_trajectories or [])

        quotas = {
            "parent_failure": int(self.source_quotas.get("parent_failure", 0) or 0),
            "current_child_level1": int(self.source_quotas.get("current_child_level1", 0) or 0),
        }
        # When quotas are not configured, fall back to an even split.
        if sum(quotas.values()) == 0:
            half = max(1, self.batch_size // 2)
            quotas = {"parent_failure": half, "current_child_level1": self.batch_size - half}

        # P0-10/11: compute effective quotas BEFORE generation based on
        # available failure-source counts, then use those as the generation
        # targets. Supports 5+5 -> 3+7 -> 10+0 when one side underfills.
        # Bootstrap skips the split entirely.
        if bootstrap:
            quotas = compute_effective_quotas(
                self.batch_size,
                quotas.get("parent_failure", 0),
                quotas.get("current_child_level1", 0),
                available_parent=0,
                available_child=0,
                bootstrap=True,
            )
        else:
            quotas = compute_effective_quotas(
                self.batch_size,
                quotas["parent_failure"],
                quotas["current_child_level1"],
                available_parent=len(parent_failure_trajectories),
                available_child=len(current_child_level1_trajectories),
                bootstrap=False,
            )

        source_counts = {"parent_failure": 0, "current_child_level1": 0, "bootstrap": 0}
        quota_fallback_log: list[dict] = []


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
        # BUG-08/09: pass the split trajectory buckets so the proposer can tag
        # each plan with its source type. ``solver_trajectories`` remains the
        # union for backward compatibility with proposer code that still reads
        # only that field.
        combined_trajectories = list(solver_trajectories or [])
        for value in parent_failure_trajectories + current_child_level1_trajectories:
            if value not in combined_trajectories:
                combined_trajectories.append(value)
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
                str(Path(value).resolve()) for value in combined_trajectories
            ],
            parent_failure_trajectories=[
                str(Path(value).resolve()) for value in parent_failure_trajectories
            ],
            current_child_level1_trajectories=[
                str(Path(value).resolve()) for value in current_child_level1_trajectories
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
            bootstrap=bootstrap,
            workflow_config=dict(self.workflow_config),
            generation_quotas={
                "parent_failure": int(quotas.get("parent_failure", 0) or 0),
                "current_child_level1": int(quotas.get("current_child_level1", 0) or 0),
            },
            allow_workflow_fallback=self.allow_workflow_fallback,
            allow_human_curated_data=self.allow_human_curated_data,
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
                    # BUG-08/09: classify the candidate against the split
                    # trajectory buckets and enforce the 5+5 quota with
                    # dynamic fallback (5+5 -> 4+6 -> ...).
                    source_type, source_trajectory = self._classify_source_v2(
                        cand,
                        parent_failure_trajectories,
                        current_child_level1_trajectories,
                        parent_task_ids or [],
                        bootstrap=bootstrap,
                    )
                    if not self._accepts_source(
                        source_type, source_counts, quotas, self.batch_size
                    ):
                        # Quota for this source is full and the other source
                        # has not yet donated its fallback surplus. Skip but
                        # record the candidate so a later fallback can accept
                        # it if the other source underfills.
                        result.rejected_candidates += 1
                        result.rejection_reasons[
                            f"quota_full:{source_type}"
                        ] = result.rejection_reasons.get(
                            f"quota_full:{source_type}", 0
                        ) + 1
                        continue
                    # Record fallback whenever we accept a source past its
                    # nominal quota.
                    nominal = quotas.get(source_type, 0)
                    if nominal and source_counts[source_type] >= nominal:
                        quota_fallback_log.append({
                            "candidate_id": cand.candidate_id,
                            "source_type": source_type,
                            "nominal_quota": nominal,
                            "accepted_so_far": source_counts[source_type] + 1,
                            "reason": "other_source_underfilled",
                        })
                    # P0-12: stamp full provenance from trajectory/plan metadata.
                    meta = dict(getattr(cand, "generation_metadata", {}) or {})
                    plan_meta = plans_by_id.get(getattr(cand, "plan_id", ""), {}) or {}
                    blueprint = dict(plan_meta.get("task_blueprint") or {})
                    source_node_id = str(
                        meta.get("source_node_id")
                        or meta.get("source_node")
                        or (
                            node_id
                            if source_type in {"current_child_level1", "bootstrap"}
                            else ""
                        )
                        or node_id
                    )
                    source_task_id = str(
                        meta.get("source_task_id")
                        or blueprint.get("source_task_id")
                        or ""
                    )
                    source_failure_stage = str(
                        meta.get("source_failure_stage")
                        or blueprint.get("failure_stage")
                        or ""
                    )
                    # P0-8: f2p_tests come ONLY from trusted report, never from
                    # candidate-declared metadata.
                    task = task_committer.commit_task(
                        batch_id=batch_id,
                        proposer_node_id=node_id,
                        repo_id=repo_spec["repo_id"],
                        base_commit=repo_spec["base_commit"],
                        bug_strategy=cand.strategy,
                        bug_patch=cand.patch,
                        problem_statement=problem_statement,
                        f2p_tests=list(report.f2p_tests),
                        baseline_test_command=generated_test_command,
                        solver_test_command=str(repo_spec["test_command"]),
                        modified_files=cand.modified_files,
                        modified_entities=cand.modified_entities,
                        validation_report=report_data,
                        setup_patch=setup_patch,
                        source_node=source_node_id,
                        source_trajectory=source_trajectory,
                        source_type=source_type,
                        source_node_id=source_node_id,
                        source_trajectory_id=source_trajectory,
                        source_task_id=source_task_id,
                        source_failure_stage=source_failure_stage,
                    )
                    result.tasks.append(task)
                    source_counts[source_type] = source_counts.get(source_type, 0) + 1
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
        # BUG-08/09: persist quota fallback metadata so the run is auditable.
        try:
            atomic_write_json(
                output_dir / "quota_summary.json",
                {
                    "quotas": quotas,
                    "source_counts": source_counts,
                    "fallback_events": quota_fallback_log,
                },
            )
        except Exception:
            pass
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

    def _classify_source(
        self,
        candidate: Any,
        solver_trajectories: List[str],
        parent_task_ids: List[str],
    ) -> str:
        """Classify a candidate's task source for Phase 6 quotas.

        Returns "parent_failure" if the candidate was conditioned on a parent
        trajectory, "current_child_level1" if conditioned on a current-child
        Level 1 failure, or "bootstrap" if no trajectories were used.
        """
        cand_dict = self._candidate_dict(candidate)
        plan_id = str(cand_dict.get("plan_id") or "")
        source_traj_ids = []
        if isinstance(cand_dict.get("generation_metadata"), dict):
            source_traj_ids = list(
                cand_dict["generation_metadata"].get("source_trajectory_ids") or []
            )
        if not solver_trajectories and not parent_task_ids:
            return "bootstrap"
        if source_traj_ids:
            for traj in source_traj_ids:
                if any(traj in path for path in solver_trajectories):
                    return "parent_failure"
            return "current_child_level1"
        # Default: if we have parent trajectories, assume parent_failure.
        if solver_trajectories:
            return "parent_failure"
        return "current_child_level1"

    def _classify_source_v2(
        self,
        candidate: Any,
        parent_failure_trajectories: List[str],
        current_child_level1_trajectories: List[str],
        parent_task_ids: List[str],
        bootstrap: bool = False,
    ) -> tuple[str, str]:
        """BUG-09: classify a candidate against the split trajectory buckets.

        Returns ``(source_type, source_trajectory)`` where ``source_type`` is
        one of ``parent_failure``, ``current_child_level1``, or ``bootstrap``,
        and ``source_trajectory`` is the concrete trajectory id/path the
        candidate was conditioned on (empty only for bootstrap).
        """
        cand_dict = self._candidate_dict(candidate)
        source_traj_ids: list[str] = []
        if isinstance(cand_dict.get("generation_metadata"), dict):
            source_traj_ids = list(
                cand_dict["generation_metadata"].get("source_trajectory_ids") or []
            )

        if bootstrap and not parent_failure_trajectories and not current_child_level1_trajectories:
            return "bootstrap", ""

        def _match(traj_id: str, bucket: List[str]) -> bool:
            for path in bucket:
                if traj_id and (traj_id in path or path in traj_id or traj_id == path):
                    return True
            return False

        if source_traj_ids:
            for traj_id in source_traj_ids:
                if _match(traj_id, parent_failure_trajectories):
                    return "parent_failure", str(traj_id)
            for traj_id in source_traj_ids:
                if _match(traj_id, current_child_level1_trajectories):
                    return "current_child_level1", str(traj_id)
            # Unknown trajectory id; default by which bucket is non-empty.
            if parent_failure_trajectories:
                return "parent_failure", str(source_traj_ids[0])
            if current_child_level1_trajectories:
                return "current_child_level1", str(source_traj_ids[0])
            return "bootstrap", str(source_traj_ids[0])

        # No explicit trajectory id; infer from non-empty buckets.
        if parent_failure_trajectories:
            return "parent_failure", str(parent_failure_trajectories[0])
        if current_child_level1_trajectories:
            return "current_child_level1", str(current_child_level1_trajectories[0])
        return "bootstrap", ""

    @staticmethod
    def _accepts_source(
        source_type: str,
        source_counts: dict,
        quotas: dict,
        batch_size: int,
    ) -> bool:
        """BUG-08 / P0-11: effective-quota targets with donation fallback.

        A source is accepted when either:
        - Its count is below its (effective) quota, OR
        - The other source is at/over its quota and the batch still has room, OR
        - The other source's quota is 0 (exhausted of sources) so this side
          may fill the remainder.
        """
        if source_type == "bootstrap":
            return True
        total = sum(v for k, v in source_counts.items() if k != "bootstrap")
        if total >= batch_size:
            return False
        nominal = int(quotas.get(source_type, 0) or 0)
        if source_counts.get(source_type, 0) < nominal:
            return True
        other = "current_child_level1" if source_type == "parent_failure" else "parent_failure"
        other_nominal = int(quotas.get(other, 0) or 0)
        other_count = source_counts.get(other, 0)
        if other_nominal == 0 or other_count >= other_nominal:
            return True
        return False

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
