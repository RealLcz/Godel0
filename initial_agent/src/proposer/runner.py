from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from .candidate_feedback import CandidateFeedbackProcessor
from .code_locator import CodeLocator, RepoIndex, RepoSpec
from .planner import ProposerPlanner
from .request import CandidateArtifact, ProposerRequest, ProposerResult, new_candidate_id
from .schemas import BugGenerationPlan, FailureSignature
from .statement_generator import StatementGenerator
from .trajectory_analyzer import EvaluationOutcomeView, TrajectoryAnalyzer, TrajectoryView


class AgentAdapter(Protocol):
    """Minimal protocol for an agent adapter used by the runner.

    The real implementation (e.g. CommonAgentAdapter) lives outside this
    skeleton. The runner only relies on `run_task` for LM-driven steps
    such as bug-introduction and issue-draft generation.
    """

    def run_task(
        self,
        prompt: str,
        system_message: str,
        model: str,
        workspace_dir: str,
    ) -> str: ...


class EngineLike(Protocol):
    """Minimal protocol for an SWE-smith style engine.

    The real SWESmithEngine dispatches by `plan.strategy` to materialize
    candidate patches. This skeleton only requires a `generate` method.
    """

    def generate(
        self,
        plan: BugGenerationPlan,
        node_code_dir: str,
        repo_spec: RepoSpec,
        output_dir: str,
    ) -> List[CandidateArtifact]: ...


@dataclass
class GenerationTrace:
    plans: List[BugGenerationPlan]
    signatures: List[FailureSignature]
    candidates: List[CandidateArtifact]


class ProposerRunner:
    """Orchestrates the proposer batch generation pipeline.

    Pipeline:
      1. Analyze solver trajectories -> FailureSignature list.
      2. For each signature, create a BugGenerationPlan (WHAT/which/strategy).
      3. Generate candidates via the SWE-smith engine (HOW).
      4. Output candidates for trusted validation (external).
      5. Process validation feedback (if any) into accepted/rejected.
      6. Return ProposerResult with accepted candidates and issue drafts.

    The proposer must NOT directly write to TaskStore or read trusted
    private inputs. It only interacts with the trusted validator through
    standard request/response files.
    """

    def __init__(
        self,
        agent_adapter: Optional[AgentAdapter] = None,
        engine: Optional[EngineLike] = None,
        workflow: Any = None,
        workflow_config: Any = None,
        allow_workflow_fallback: bool = False,
        allow_human_curated_data: bool = False,
    ) -> None:
        self.agent_adapter = agent_adapter
        self.engine = engine
        # BUG-02/03: route every non-bootstrap plan through RepoChainWorkflow
        # so the runtime actually exercises the RepoChain stages instead of
        # going straight to SWESmithEngine.generate(). When ``workflow`` is
        # None we lazily build a default RepoChainWorkflow wrapping the engine.
        self._workflow = workflow
        self.workflow_config = workflow_config
        # P0-6: production forbids silent fallback to SWESmithEngine.
        self.allow_workflow_fallback = bool(allow_workflow_fallback)
        # P0-23: main experiment forbids PR-replay / human-curated data.
        self.allow_human_curated_data = bool(allow_human_curated_data)
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.code_locator = CodeLocator()
        self.planner = ProposerPlanner(code_locator=self.code_locator)
        self.feedback_processor = CandidateFeedbackProcessor()
        self.statement_generator = StatementGenerator()
        self._workflow_fallback_used = False

    @property
    def workflow(self):
        """Lazily instantiate the default RepoChainWorkflow (P0-5/P0-6)."""
        if self._workflow is None:
            try:
                from proposer.workflows.repo_chain import RepoChainWorkflow

                kwargs = {
                    "agent_adapter": self.agent_adapter,
                    "engine": self.engine,
                    "trajectory_analyzer": self.trajectory_analyzer,
                    "code_locator": self.code_locator,
                }
                cfg = self.workflow_config
                if cfg is not None:
                    kwargs["config"] = cfg
                    if hasattr(cfg, "mutation_backends"):
                        kwargs["mutation_backend_weights"] = dict(
                            cfg.mutation_backends or {}
                        )
                    if hasattr(cfg, "require_causal_ablation"):
                        kwargs["require_causal_ablation"] = bool(
                            cfg.require_causal_ablation
                        )
                self._workflow = RepoChainWorkflow(**kwargs)
            except Exception as exc:
                # P0-6: production must crash rather than silently degrade
                # to plain SWE-smith generation.
                if not self.allow_workflow_fallback:
                    raise RuntimeError(
                        "RepoChainWorkflow unavailable in production mode"
                    ) from exc
                self._workflow = None
                self._workflow_fallback_used = True
        return self._workflow

    def generate_batch(self, request: ProposerRequest) -> ProposerResult:
        result = ProposerResult.new_for(request)
        # P0-6: always record which workflow ran so silent degradation is
        # visible in batch artifacts.
        result.workflow = "repo_chain"
        result.workflow_fallback = False
        # Prefer request-carried workflow config when the runner was not
        # constructed with an explicit config object (subprocess path).
        if self.workflow_config is None and getattr(request, "workflow_config", None):
            self.workflow_config = request.workflow_config
        if getattr(request, "allow_workflow_fallback", None) is not None:
            self.allow_workflow_fallback = bool(request.allow_workflow_fallback)
        if getattr(request, "allow_human_curated_data", None) is not None:
            self.allow_human_curated_data = bool(request.allow_human_curated_data)
        try:
            self._enforce_human_data_policy(request)
            traces = self._load_trajectories(request)
            outcomes = self._load_outcomes(request, traces)
            signatures = self.trajectory_analyzer.extract_signatures(traces, outcomes)
            result.failure_signatures = [s.model_dump() for s in signatures]

            if not signatures:
                if getattr(request, "bootstrap", False):
                    # Root bootstrap: no solver trajectories yet. Generate T_0
                    # from a capability prior via RepoChainWorkflow.bootstrap.
                    candidates = self._bootstrap_candidates(request)
                    self._write_candidates(request, candidates, [])
                    for cand in candidates:
                        result.add_candidate(cand, accepted=True)
                        result.add_pending_candidate(cand)
                    result.completed = len(candidates) > 0
                    result.workflow_fallback = bool(self._workflow_fallback_used)
                    return result
                result.completed = True
                return result

            repo_index = self._build_repo_index(request)
            feedbacks = self.feedback_processor.load_feedback(request.feedback_dir)
            self.planner.configure_strategy_policy(
                request.strategy_weights,
                offset=request.generation_attempt * max(1, request.target_batch_size),
            )
            plans = self.planner.create_plans(
                signatures,
                repo_index,
                base_commit=repo_index.base_commit,
                max_plans=request.target_batch_size,
            )
            for index, plan in enumerate(plans):
                plan.seed = request.generation_attempt * 1000 + index + 1
                # P0-5: stamp RepoChain difficulty constraints onto every plan.
                self._stamp_repo_chain_constraints(plan)
                rejected_feedback = [
                    {
                        "candidate_id": feedback.candidate_id,
                        "reason": feedback.reason,
                    }
                    for feedback in feedbacks
                    if not feedback.accepted
                ][-20:]
                if rejected_feedback:
                    # This is a standard trusted response file, not validator
                    # private state. Engines can use it to avoid repeating the
                    # same invalid construction on the next generation attempt.
                    plan.task_blueprint["trusted_validation_feedback"] = rejected_feedback
            candidates = self._generate_candidates(request, plans, repo_index)
            result.workflow_fallback = bool(self._workflow_fallback_used)
            # Generation may attach an engine rejection reason to the plan.
            result.plans = [p.model_dump() for p in plans]
            self._write_candidates(request, candidates, plans)

            partitioned = self.feedback_processor.partition(candidates, feedbacks)

            for cand in partitioned["accepted"]:
                self._finalize_accepted(request, cand, plans)
                result.add_candidate(cand, accepted=True)
            for cand in partitioned["rejected"]:
                result.add_candidate(cand, accepted=False)
            partitioned_ids = {
                cand.candidate_id
                for cand in partitioned["accepted"] + partitioned["rejected"]
            }
            for cand in candidates:
                if cand.candidate_id not in partitioned_ids:
                    result.add_pending_candidate(cand)

            accepted_needed = request.target_batch_size
            if len(result.accepted_candidates) >= accepted_needed:
                result.completed = True
            elif len(candidates) >= request.max_candidates:
                result.completed = True
        except Exception as exc:  # pragma: no cover - skeleton guard
            result.error = f"{type(exc).__name__}: {exc}"
            result.completed = False
            result.workflow_fallback = bool(self._workflow_fallback_used)
        return result

    def _enforce_human_data_policy(self, request: ProposerRequest) -> None:
        """P0-23: refuse PR-replay / human-curated data in the main setting."""
        if self.allow_human_curated_data:
            return
        weights = dict(getattr(request, "strategy_weights", None) or {})
        cfg = self.workflow_config or getattr(request, "workflow_config", None) or {}
        if isinstance(cfg, dict):
            backends = dict(cfg.get("mutation_backends") or {})
        else:
            backends = dict(getattr(cfg, "mutation_backends", None) or {})
        pr_weight = max(
            float(weights.get("pr_replay", 0.0) or 0.0),
            float(weights.get("pr_mirror", 0.0) or 0.0),
            float(backends.get("pr_replay", 0.0) or 0.0),
            float(backends.get("pr_mirror", 0.0) or 0.0),
        )
        if pr_weight > 0.0:
            # Prefer zeroing request weights so a misconfigured YAML cannot
            # silently introduce human-curated PR data into the main run.
            for key in ("pr_replay", "pr_mirror"):
                if key in weights:
                    weights[key] = 0.0
            request.strategy_weights = weights
            import logging

            logging.getLogger("proposer.runner").warning(
                "P0-23: forced pr_replay/pr_mirror weights to 0 "
                "(allow_human_curated_data=False)"
            )

    def _stamp_repo_chain_constraints(self, plan: BugGenerationPlan) -> None:
        """P0-5: force plan constraints from RepoChainWorkflowConfig."""
        cfg = self.workflow_config
        if cfg is None:
            return
        constraints = plan.task_blueprint.setdefault("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
            plan.task_blueprint["constraints"] = constraints
        for src, dst in (
            ("min_files", "min_modified_files"),
            ("max_files", "max_modified_files"),
            ("min_mutation_sites", "min_mutation_sites"),
            ("max_mutation_sites", "max_mutation_sites"),
            ("context_file_budget", "context_file_budget"),
        ):
            if hasattr(cfg, src):
                constraints[dst] = getattr(cfg, src)

    def _load_trajectories(self, request: ProposerRequest) -> List[TrajectoryView]:
        views: List[TrajectoryView] = []
        for path in request.solver_trajectories:
            if not os.path.isfile(path):
                continue
            views.append(TrajectoryView.from_jsonl(path))
        return views

    def _load_outcomes(
        self,
        request: ProposerRequest,
        traces: List[TrajectoryView],
    ) -> List[EvaluationOutcomeView]:
        outcomes: List[EvaluationOutcomeView] = []
        for path in request.solver_trajectories:
            outcome_path = os.path.splitext(path)[0] + "_eval.json"
            if os.path.isfile(outcome_path):
                outcomes.append(EvaluationOutcomeView.from_json(outcome_path))
        if not outcomes and traces:
            for traj in traces:
                outcomes.append(EvaluationOutcomeView(trajectory_id=traj.trajectory_id))
        return outcomes

    def _bootstrap_candidates(self, request: ProposerRequest) -> List[CandidateArtifact]:
        """Generate bootstrap candidates from a capability prior.

        Used when ``request.bootstrap`` is True and there are no solver
        trajectories (root node). Routes through RepoChainWorkflow.bootstrap so
        the root T_0 is produced by the RepoChain workflow (stages 2-8), not by
        a raw lm_modify plan. BUG-04/05: previously this called
        ``build_bootstrap_plans([], repo_spec)`` (empty prior -> 0 plans) and
        then ran each plan through ``engine.generate`` (lm_modify backend),
        bypassing RepoChain entirely.
        """
        try:
            from proposer.workflows.repo_chain.bootstrap import (
                BOOTSTRAP_CAPABILITY_PRIOR,
            )
        except Exception as exc:
            if not self.allow_workflow_fallback:
                raise RuntimeError(
                    "RepoChain bootstrap unavailable in production mode"
                ) from exc
            self._workflow_fallback_used = True
            return []
        if not request.repo_specs:
            return []
        spec = request.repo_specs[0]
        repo_spec = RepoSpec(
            repo_id=spec.repo_id,
            repo_dir=spec.path,
            base_commit=spec.base_commit,
            test_command=spec.test_command,
        )
        # P0-5/P0-6: bootstrap must use the same RepoChainWorkflow path as
        # normal generation (no silent engine fallback).
        workflow = self.workflow
        if workflow is None:
            if not self.allow_workflow_fallback:
                raise RuntimeError(
                    "RepoChainWorkflow required for bootstrap but unavailable"
                )
            self._workflow_fallback_used = True
            return []
        cand_dir = os.path.join(request.output_dir, "proposer_candidates", "bootstrap")
        os.makedirs(cand_dir, exist_ok=True)
        candidates = workflow.bootstrap(
            repo_spec=repo_spec,
            output_dir=cand_dir,
            capability_prior=BOOTSTRAP_CAPABILITY_PRIOR,
            target_count=int(request.target_batch_size or 10),
            max_candidates=int(request.max_candidates or 50),
        )
        # Stamp each candidate with the request model / plan id metadata so the
        # downstream commit step has the provenance it expects.
        for index, cand in enumerate(candidates):
            if hasattr(cand, "plan_id") and not cand.plan_id:
                cand.plan_id = f"bootstrap-{index}"
            if hasattr(cand, "generation_metadata") and isinstance(
                cand.generation_metadata, dict
            ):
                cand.generation_metadata.setdefault("bootstrap", True)
                cand.generation_metadata.setdefault("source_type", "bootstrap")
        return candidates

    def _build_repo_index(self, request: ProposerRequest) -> RepoIndex:
        """Build a RepoIndex from the repo_specs carried in the request.

        If the request carries explicit repo_specs, use the first one
        (or the one matching a plan's target_repo_id).
        Otherwise, fall back to scanning repo_pool_dir directly.
        """
        # Prefer explicit repo_specs from the request
        if request.repo_specs:
            spec = request.repo_specs[0]
            # Use RepoProfileRegistry to get source_dirs (no hardcoded checks).
            from proposer.repo_profiles import get_profile

            profile = get_profile(spec.repo_id)
            source_dirs = profile.source_roots
            return RepoIndex.build(
                repo_id=spec.repo_id,
                repo_dir=spec.path,
                base_commit=spec.base_commit,
                source_dirs=source_dirs,
            )

        # Fallback: scan the repo_pool_dir
        repo_dir = request.repo_pool_dir
        if not repo_dir or not os.path.isdir(repo_dir):
            raise FileNotFoundError(
                f"repo_pool_dir '{repo_dir}' does not exist and no "
                f"repo_specs were provided in the request"
            )
        return RepoIndex.build(
            repo_id=os.path.basename(os.path.normpath(repo_dir)) or "repo",
            repo_dir=repo_dir,
            base_commit="",
        )

    def _get_repo_spec_for_plan(self, request: ProposerRequest, plan: BugGenerationPlan):
        """Find the RepoSpecInfo for a plan's target repo."""
        if plan.target_repo_id:
            spec = request.get_repo(plan.target_repo_id)
            if spec:
                return spec
        return request.first_repo()

    def _generate_candidates(
        self,
        request: ProposerRequest,
        plans: List[BugGenerationPlan],
        repo_index: RepoIndex,
    ) -> List[CandidateArtifact]:
        candidates: List[CandidateArtifact] = []
        if self.engine is None:
            return candidates
        repo_spec = RepoSpec.from_index(repo_index)
        workflow = self.workflow
        for plan in plans:
            if len(candidates) >= request.max_candidates:
                break

            # Use the correct repo spec for this plan
            plan_repo = self._get_repo_spec_for_plan(request, plan)
            if plan_repo:
                repo_spec = RepoSpec(
                    repo_id=plan_repo.repo_id,
                    repo_dir=plan_repo.path,
                    base_commit=plan_repo.base_commit,
                    test_command=plan_repo.test_command,
                    install_command=plan_repo.install_command,
                    timeout_sec=plan_repo.timeout_sec,
                )
                plan.target_base_commit = plan_repo.base_commit
                plan.target_repo_id = plan_repo.repo_id
            plan.model = request.model

            cand_dir = os.path.join(request.output_dir, "proposer_candidates", plan.plan_id)
            os.makedirs(cand_dir, exist_ok=True)
            self._write_plan(cand_dir, plan)
            try:
                # P0-6: production must not silently fall back to SWE-smith.
                if workflow is not None:
                    produced = workflow.generate(
                        plan=plan,
                        node_code_dir=request.agent_code_dir,
                        repo_spec=repo_spec,
                        output_dir=cand_dir,
                    )
                else:
                    if not self.allow_workflow_fallback:
                        raise RuntimeError(
                            "RepoChainWorkflow unavailable in production mode"
                        )
                    self._workflow_fallback_used = True
                    produced = self.engine.generate(
                        plan=plan,
                        node_code_dir=request.agent_code_dir,
                        repo_spec=repo_spec,
                        output_dir=cand_dir,
                    )
            except Exception as exc:
                rejection = str(
                    getattr(getattr(self.engine, "repo_chain", None), "last_rejection", "")
                    or getattr(self.engine, "last_rejection", "")
                    or f"{type(exc).__name__}: {exc}"
                )
                plan.task_blueprint["last_rejection"] = rejection
                produced = []
            if not produced:
                rejection = str(
                    getattr(getattr(self.engine, "repo_chain", None), "last_rejection", "")
                    or getattr(self.engine, "last_rejection", "")
                    or "engine_returned_no_candidates"
                )
                plan.task_blueprint["last_rejection"] = rejection
            for raw in produced:
                cand = self._coerce_candidate(raw, plan, repo_spec)
                if not cand.candidate_id:
                    cand.candidate_id = new_candidate_id()
                if not cand.plan_id:
                    cand.plan_id = plan.plan_id
                candidates.append(cand)
        return candidates

    def _coerce_candidate(
        self,
        raw: Any,
        plan: BugGenerationPlan,
        repo_spec: RepoSpec,
    ) -> CandidateArtifact:
        """Normalize engine-specific candidate objects to the proposer contract."""
        if isinstance(raw, CandidateArtifact):
            return raw

        to_dict = getattr(raw, "to_dict", None)
        data = to_dict() if callable(to_dict) else dict(getattr(raw, "__dict__", {}))
        if not isinstance(data, dict):
            data = {}

        patch = str(data.get("patch") or data.get("bug_patch") or "")
        modified_files_value = data.get("modified_files") or data.get("changed_files") or []
        if isinstance(modified_files_value, str):
            modified_files_value = [modified_files_value]
        modified_files = list(modified_files_value)
        if not modified_files and patch:
            try:
                from swesmith.patch_utils import extract_changed_files

                modified_files = extract_changed_files(patch)
            except ImportError:
                modified_files = []
        modified_entities = list(
            data.get("modified_entities") or data.get("changed_entities") or []
        )

        strategy = str(data.get("strategy") or plan.strategy)
        if strategy in {"pr_replay", "repo_agent", "repo_chain"}:
            symbol_name = str(data.get("symbol_name") or data.get("target_symbol") or "")
        else:
            symbol_name = str(
                data.get("symbol_name") or data.get("target_symbol") or plan.target_symbol
            )

        return CandidateArtifact(
            candidate_id=str(data.get("candidate_id") or new_candidate_id()),
            plan_id=str(data.get("plan_id") or plan.plan_id),
            repo_id=str(data.get("repo_id") or plan.target_repo_id or repo_spec.repo_id),
            base_commit=str(data.get("base_commit") or plan.target_base_commit or repo_spec.base_commit),
            file_path=str(data.get("file_path") or data.get("target_file") or plan.target_file),
            symbol_name=symbol_name,
            strategy=strategy,
            operator=str(data.get("operator") or plan.operator or ""),
            patch=patch,
            issue_draft=str(
                data.get("issue_draft")
                or (data.get("generation_metadata") or {}).get("problem_statement")
                or ""
            ),
            local_test_notes=dict(data.get("local_test_notes") or {}),
            generation_trajectory=list(data.get("generation_trajectory") or []),
            modified_files=modified_files,
            modified_entities=modified_entities,
            generation_metadata=self._stamp_provenance(data, plan),
            status=str(data.get("status") or "pending_validation"),
        )

    def _stamp_provenance(self, data: dict, plan: BugGenerationPlan) -> dict:
        """BUG-09: stamp source_trajectory_ids / source_type onto metadata.

        The plan already carries ``source_trajectory_ids`` (from the
        trajectory analyzer). We copy them onto the candidate's
        ``generation_metadata`` so the trusted ``TaskBatchBuilder`` can
        classify the source without re-reading the plan.
        """
        metadata = dict(data.get("generation_metadata") or {})
        ids = list(plan.source_trajectory_ids or [])
        if ids:
            metadata.setdefault("source_trajectory_ids", ids)
        metadata.setdefault("source_node", getattr(plan, "reference_parent", "") or "")
        metadata.setdefault("plan_id", plan.plan_id)
        return metadata

    def _write_plan(self, cand_dir: str, plan: BugGenerationPlan) -> None:
        path = os.path.join(cand_dir, "plan.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(plan.model_dump_json(indent=2))

    def _write_candidates(
        self,
        request: ProposerRequest,
        candidates: List[CandidateArtifact],
        plans: List[BugGenerationPlan],
    ) -> None:
        plan_by_id = {p.plan_id: p for p in plans}
        for cand in candidates:
            cand_dir = os.path.join(
                request.output_dir,
                "proposer_candidates",
                cand.plan_id or "orphan",
            )
            os.makedirs(cand_dir, exist_ok=True)
            with open(os.path.join(cand_dir, "candidate.json"), "w", encoding="utf-8") as f:
                import json

                json.dump(cand.to_dict(), f, indent=2, ensure_ascii=False)
            if cand.patch:
                with open(os.path.join(cand_dir, "bug.patch"), "w", encoding="utf-8") as f:
                    f.write(cand.patch)
            plan = plan_by_id.get(cand.plan_id)
            if cand.issue_draft:
                statement = cand.issue_draft.rstrip() + "\n"
                Path(cand_dir, "problem_statement.md").write_text(
                    statement, encoding="utf-8"
                )
                Path(cand_dir, "issue_draft.md").write_text(
                    statement, encoding="utf-8"
                )
            else:
                self.statement_generator.write_issue_draft(cand, cand_dir, plan)

    def _finalize_accepted(
        self,
        request: ProposerRequest,
        cand: CandidateArtifact,
        plans: List[BugGenerationPlan],
    ) -> None:
        plan = next((p for p in plans if p.plan_id == cand.plan_id), None)
        cand_dir = os.path.join(
            request.output_dir,
            "proposer_candidates",
            cand.plan_id or "orphan",
        )
        if cand.issue_draft:
            statement = cand.issue_draft.rstrip() + "\n"
            Path(cand_dir, "problem_statement.md").write_text(
                statement, encoding="utf-8"
            )
            Path(cand_dir, "issue_draft.md").write_text(
                statement, encoding="utf-8"
            )
        else:
            self.statement_generator.write_issue_draft(cand, cand_dir, plan)


__all__ = ["ProposerRunner", "AgentAdapter", "EngineLike", "GenerationTrace"]
