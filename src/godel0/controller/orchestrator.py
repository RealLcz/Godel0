"""Evolution orchestrator: the main loop."""

from __future__ import annotations

import random
import shutil
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from ..config import Godel0Config, assert_no_human_curated_data, config_to_dict
from ..constants import ROOT_NODE_ID
from ..errors import BudgetExhaustedError
from ..schemas.node import NodeRecord, NodeStatus
from ..storage.atomic import read_json
from ..storage.atomic import atomic_write_json
from ..storage.event_log import log_event
from ..tree.archive import NodeArchive
from ..tree.selection import ParentSelector
from .budget import Budget
from .run_context import RunContext
from .scorer import compute_scores


_TEST_PATH_PATTERNS = ("test_", "_test.py", "/tests/", "/test/", "conftest.py")


def _is_test_path(path: str) -> bool:
    """Return True if a repository path looks like a test file."""
    if not path:
        return False
    normalized = path.replace("\\", "/")
    lower = normalized.lower()
    return any(pattern in lower for pattern in _TEST_PATH_PATTERNS)


class EvolutionOrchestrator:
    """Main evolution loop orchestrator."""

    def __init__(
        self,
        config: Godel0Config,
        archive: NodeArchive,
        selector: ParentSelector,
        run_context: RunContext,
        budget: Budget,
        cycle_builder=None,
        special_detector=None,
        evidence_selector=None,
        diagnoser=None,
        child_builder=None,
        level1_evaluator=None,
        task_provider=None,
        task_batch_builder=None,
        level2_evaluator=None,
        repo_pool=None,
        validator=None,
        task_committer=None,
        proposer_runner=None,
        task_store=None,
        solver_runner=None,
    ):
        self.config = config
        self.archive = archive
        self.selector = selector
        self.run_context = run_context
        self.budget = budget
        self.cycle_builder = cycle_builder
        self.special_detector = special_detector
        self.evidence_selector = evidence_selector
        self.diagnoser = diagnoser
        self.child_builder = child_builder
        self.level1_evaluator = level1_evaluator
        # TaskProvider seam: the HGM-style loop calls get_tasks(node, context)
        # and is agnostic to BenchmarkTaskProvider vs ProposerTaskProvider.
        # For backwards compatibility, a raw TaskBatchBuilder is wrapped.
        self.task_provider = task_provider
        self.task_batch_builder = task_batch_builder
        if self.task_provider is None and self.task_batch_builder is not None:
            from ..tasks.proposer_provider import ProposerTaskProvider

            self.task_provider = ProposerTaskProvider(
                batch_builder=self.task_batch_builder,
                repo_pool=repo_pool,
                validator=validator,
                task_committer=task_committer,
                proposer_runner=proposer_runner,
                task_store_dir=config.paths.task_store,
            )
        self.level2_evaluator = level2_evaluator
        self.repo_pool = repo_pool
        self.validator = validator
        self.task_committer = task_committer
        self.proposer_runner = proposer_runner
        self.task_store = task_store
        self.solver_runner = solver_runner
        self.rng = random.Random(config.run.seed)

    @classmethod
    def from_config(cls, config: Godel0Config) -> "EvolutionOrchestrator":
        """Create an orchestrator from config."""
        from ..config import assert_no_human_curated_data

        # P0-23: zero out PR-replay weights in the main experiment.
        config = assert_no_human_curated_data(config)

        runs_dir = Path(config.paths.runs)
        run_context = RunContext.create(config, runs_dir)

        atomic_write_yaml(run_context.paths.config_path, config_to_dict(config))

        archive = NodeArchive(run_context.paths.archive_path)
        cls._initialize_root(config, archive, run_context)
        # BUG-10: the main experiment uses HGM-style Thompson Sampling, not
        # epsilon-greedy. EpsilonGreedySelector is kept for ablations only and
        # must be selected explicitly via config.selection.strategy.
        selector = cls._build_selector(config)
        budget = Budget(
            max_nodes=config.run.max_nodes,
            max_expansions=config.run.max_expansions,
        )

        components = cls._build_components(config, run_context)

        return cls(
            config=config,
            archive=archive,
            selector=selector,
            run_context=run_context,
            budget=budget,
            **components,
        )

    @classmethod
    def _build_components(cls, config: Godel0Config, run_context: RunContext) -> dict:
        """Build the default trusted components used by the evolution loop."""
        from ..evaluation.level1 import Level1Evaluator
        from ..evaluation.level2 import Level2Evaluator
        from ..evaluation.runner import SolverEvaluationRunner
        from ..evolution.child_builder import ChildBuilder
        from ..evolution.patch_guard import PatchGuard
        from ..evolution.self_edit import SelfEditRunner
        from ..execution.workspace_manager import WorkspaceManager
        from ..proposer_trusted.candidate_validator import CandidateValidator
        from ..proposer_trusted.task_committer import TaskCommitter
        from ..tasks.batch import TaskBatchBuilder
        from ..tasks.proposer_provider import ProposerTaskProvider
        from ..tasks.repo_pool import RepoPool
        from ..tasks.store import TaskStore

        repo_pool = RepoPool(Path(config.paths.repo_pool))
        task_store = TaskStore(Path(config.paths.task_store))
        workspace_manager = WorkspaceManager(Path(config.execution.scratch_root))
        validator = CandidateValidator(
            workspace_root=Path(config.execution.scratch_root) / run_context.run_id / "validator",
            test_timeout_sec=config.proposer.candidate_timeout_sec,
            max_patch_lines=config.proposer.max_patch_lines,
            forbid_test_file_edits=config.proposer.forbid_test_file_edits,
            # P0-7: authoritative trusted causal ablation gate.
            require_causal_ablation=bool(
                config.proposer.repo_chain.require_causal_ablation
            ),
            # BUG-15/10.8: trusted repository tests run through the repo
            # backend so the chain is end-to-end Apptainer. The backend is
            # built later from the ExecutionBackendFactory; we patch it in
            # after the factory is constructed below.
        )
        task_committer = TaskCommitter(task_store)

        agent_adapter = cls._build_agent_adapter()
        from ..tasks.node_proposer import NodeProposerRunner
        from ..execution.subprocess_runner import SubprocessRunner

        # Phase 9 / BUG-13~17: build the unified execution backend. Default is
        # subprocess; when config.execution.backend == "apptainer" and an
        # agent_image is configured, use ApptainerRunner for HPC container
        # isolation. The ExecutionBackendFactory exposes separate agent-facing
        # (network enabled) and repo-facing (network disabled) backends so the
        # whole chain is end-to-end Apptainer.
        from ..execution.apptainer import ExecutionBackendFactory

        backend_factory = ExecutionBackendFactory(
            agent_image=Path(config.execution.agent_image) if config.execution.agent_image else None,
            repo_image_dir=Path(config.execution.repo_image_dir) if config.execution.repo_image_dir else None,
            apptainer_bin=config.execution.apptainer_bin,
            use_apptainer=(
                config.execution.backend == "apptainer"
                and bool(config.execution.agent_image)
            ),
        )
        # BUG-16: agent-facing backend keeps network enabled so online LLM
        # API calls work. Repo-facing backend (trusted tests) disables network.
        execution_backend = backend_factory.agent_backend()
        repo_backend = backend_factory.repo_backend()
        # BUG-15/10.8: inject the repo backend into the validator so trusted
        # repository tests also run inside Apptainer.
        validator.execution_backend = repo_backend

        proposer_runner = NodeProposerRunner(
            agent_repo=Path(config.paths.agent_repo),
            scratch_root=Path(config.execution.scratch_root)
            / run_context.run_id
            / "node_proposer",
            timeout_sec=config.proposer.candidate_timeout_sec
            * config.tasks.max_generation_candidates,
            execution_backend=execution_backend,
        )

        from ..evolution.cycle_builder import NodeCycleBuilder
        from ..evolution.special_detectors import CompositeSpecialDetector
        from ..evolution.evidence_selector import CycleEvidenceSelector
        from ..evolution.diagnose import CycleDiagnoser

        return {
            "child_builder": ChildBuilder(
                agent_repo=Path(config.paths.agent_repo),
                scratch_root=Path(config.execution.scratch_root) / run_context.run_id / "children",
                patch_guard=PatchGuard(),
                self_edit_runner=SelfEditRunner(
                    agent_adapter=agent_adapter,
                    timeout_sec=config.agent.self_evolve_timeout_sec,
                ),
                output_root=run_context.paths.nodes_dir,
            ),
            "cycle_builder": NodeCycleBuilder(),
            "special_detector": CompositeSpecialDetector(),
            "evidence_selector": CycleEvidenceSelector(
                max_solver_trajectories=config.diagnosis.max_solver_trajectories,
                max_proposer_candidates=config.diagnosis.max_proposer_candidates,
                max_tool_incidents=config.diagnosis.max_tool_incidents,
                max_raw_chars_per_item=config.diagnosis.max_raw_chars_per_item,
                max_total_evidence_chars=config.diagnosis.max_total_evidence_chars,
                include_success_contrast=config.diagnosis.include_success_contrast,
            ),
            "diagnoser": CycleDiagnoser(chat_adapter=agent_adapter),
            "level1_evaluator": Level1Evaluator(
                regression_threshold=config.scoring.regression_threshold
            ),
            "task_batch_builder": TaskBatchBuilder(
                batch_size=config.tasks.batch_size,
                max_candidates=config.tasks.max_generation_candidates,
                strategy_weights=config.proposer.repo_chain.mutation_backends,
                contract_test_renderer=config.proposer.contract_test_renderer,
                source_quotas={
                    "parent_failure": config.tasks.sources.parent_failure.quota,
                    "current_child_level1": config.tasks.sources.current_child_level1.quota,
                },
                workflow_config=config_to_dict(config)["proposer"]["repo_chain"],
                allow_workflow_fallback=config.proposer.allow_workflow_fallback,
                allow_human_curated_data=config.proposer.allow_human_curated_data,
            ),
            "task_provider": ProposerTaskProvider(
                batch_builder=TaskBatchBuilder(
                    batch_size=config.tasks.batch_size,
                    max_candidates=config.tasks.max_generation_candidates,
                    strategy_weights=config.proposer.repo_chain.mutation_backends,
                    contract_test_renderer=config.proposer.contract_test_renderer,
                    source_quotas={
                        "parent_failure": config.tasks.sources.parent_failure.quota,
                        "current_child_level1": config.tasks.sources.current_child_level1.quota,
                    },
                    workflow_config=config_to_dict(config)["proposer"]["repo_chain"],
                    allow_workflow_fallback=config.proposer.allow_workflow_fallback,
                    allow_human_curated_data=config.proposer.allow_human_curated_data,
                ),
                repo_pool=repo_pool,
                validator=validator,
                task_committer=task_committer,
                proposer_runner=proposer_runner,
                task_store_dir=config.paths.task_store,
            ),
            "level2_evaluator": Level2Evaluator(),
            "repo_pool": repo_pool,
            "validator": validator,
            "task_committer": task_committer,
            "proposer_runner": proposer_runner,
            "task_store": task_store,
            "solver_runner": SolverEvaluationRunner(
                task_store=task_store,
                workspace_manager=workspace_manager,
                repo_pool=repo_pool,
                agent_repo=Path(config.paths.agent_repo),
                agent_adapter=agent_adapter,
                model=config.models.solver_model,
                solver_timeout_sec=max(
                    config.evaluation.level1_timeout_sec,
                    config.evaluation.level2_timeout_sec,
                ),
                test_timeout_sec=config.proposer.candidate_timeout_sec,
            ),
        }

    @staticmethod
    def _build_agent_adapter():
        """Create a coding-agent adapter only when LLM configuration is present."""
        env_keys = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OpenRouter_API_KEY",
            "DEEPSEEK_API_KEY",
            "QWEN_API_KEY",
            "MINIMAX_API_KEY",
            "VLLM_HOST",
        ]
        if not any(os.getenv(key) for key in env_keys):
            return None
        from experiment_adapters.common_agent_adapter import CommonAgentAdapter

        return CommonAgentAdapter()

    @staticmethod
    def _build_selector(config: "Godel0Config"):
        """Build the parent selector from config.

        BUG-10: the main experiment defaults to ThompsonSamplingSelector.
        EpsilonGreedySelector is only available as an explicit ablation.
        """
        from ..tree.selection import ThompsonSamplingSelector

        strategy = getattr(config.scoring.selection, "strategy", "thompson_sampling")
        if strategy == "epsilon_greedy":
            from ..tree.selection import EpsilonGreedySelector

            return EpsilonGreedySelector(
                epsilon=config.scoring.selection.epsilon
            )
        return ThompsonSamplingSelector(
            num_pseudo_descendant_evals=config.scoring.selection.num_pseudo_descendant_evals,
        )

    @classmethod
    def _initialize_root(
        cls,
        config: Godel0Config,
        archive: NodeArchive,
        run_context: RunContext,
    ) -> None:
        """Create the root agent git repo/ref and archive record if needed."""
        from ..controller.scorer import compute_scores
        from ..git.node_refs import create_node_ref, node_exists
        from ..git.repository import commit, get_head_sha, init_repo, run_git

        agent_repo = Path(config.paths.agent_repo)
        source = Path("initial_agent/src")

        if not agent_repo.exists():
            agent_repo.mkdir(parents=True, exist_ok=True)
        if not any(agent_repo.iterdir()) and source.exists():
            shutil.copytree(
                source,
                agent_repo,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
            )

        if not (agent_repo / ".git").exists():
            init_repo(agent_repo)
            root_sha = commit(agent_repo, "root agent")
        else:
            status = run_git(agent_repo, "status", "--porcelain", check=False)
            try:
                root_sha = get_head_sha(agent_repo)
            except Exception:
                root_sha = commit(agent_repo, "root agent")
            if status.stdout.strip():
                root_sha = commit(agent_repo, "root agent")

        from ..evolution.gates import (
            ProposerExtensionGate,
            SolverCoreParityGate,
            SolverPathIsolationGate,
        )

        parity = SolverCoreParityGate().run(
            agent_repo, Path("initial_agent/solver_core.lock.json")
        )
        isolation = SolverPathIsolationGate().run(agent_repo)
        extension = ProposerExtensionGate().run(agent_repo)
        gate_dir = run_context.paths.root_dir / "gates"
        atomic_write_json(gate_dir / "solver_core_parity.json", parity.__dict__)
        atomic_write_json(gate_dir / "solver_path_isolation.json", isolation.__dict__)
        atomic_write_json(gate_dir / "proposer_extension.json", extension.__dict__)
        if not (parity.passed and isolation.passed and extension.passed):
            raise RuntimeError(
                "Root gate failed: "
                f"parity={parity.passed}, isolation={isolation.passed}, "
                f"extension={extension.passed}"
            )

        if not node_exists(agent_repo, ROOT_NODE_ID):
            create_node_ref(agent_repo, ROOT_NODE_ID, root_sha)

        if archive.get(ROOT_NODE_ID) is None:
            root = NodeRecord(
                node_id=ROOT_NODE_ID,
                parent_node_id=None,
                code_commit=root_sha,
                code_ref=f"refs/godel0/nodes/{ROOT_NODE_ID}",
                status=NodeStatus.CANDIDATE,
            )
            archive.add(root)
            run_context.paths.ensure_node_dirs(ROOT_NODE_ID)
            archive.save_node_json(root, run_context.paths.node_json(ROOT_NODE_ID))

    def run(self) -> None:
        """Run until the requested number of *completed child nodes* exists."""
        print(f"Starting evolution run: {self.run_context.run_id}")
        print(
            "Budget: "
            f"target_successful_epochs={self.budget.max_nodes}, "
            f"max_attempts={self.budget.max_expansions}"
        )

        if not self._ensure_root_bootstrap():
            raise RuntimeError(
                "Root bootstrap failed: no complete K-task proposer/solver cycle; "
                "evolution was not started"
            )

        while not self.budget.exhausted():
            self.budget.record_expansion()

            try:
                parent = self._select_parent()
                if parent is None:
                    print("No eligible parents. Waiting for root initialization.")
                    break

                print(f"\nExpanding parent: {parent.node_id} (score={parent.node_score})")
                log_event(
                    self.run_context.paths.events_path,
                    "parent_selected",
                    run_id=self.run_context.run_id,
                    node_id=parent.node_id,
                    payload={"score": parent.node_score},
                )

                diagnosis = self._prepare_diagnosis(parent)
                child_result = self._build_child(parent, diagnosis)
                if not child_result or not child_result.passed:
                    errs = child_result.errors if child_result else ["No result"]
                    print(f"Child build failed: {errs}")
                    continue

                child = child_result.node
                if child is None:
                    print("Child node is None, skipping")
                    continue

                print(f"Child created: {child.node_id}")

                level1_result = self._evaluate_level1(parent, child)
                if level1_result is None or not level1_result.passed:
                    retention = (
                        level1_result.retention_rate if level1_result is not None else 0.0
                    )
                    print(f"Level 1 failed: retention={retention:.2f}")
                    child.status = NodeStatus.LEVEL1_FAILED
                    self.archive.update(child)
                    continue

                batch_result = self._generate_batch(
                    child,
                    parent=parent,
                    level1_result=level1_result,
                )
                batch_complete = (
                    batch_result.get("complete", False)
                    if isinstance(batch_result, dict)
                    else bool(getattr(batch_result, "complete", False))
                )
                batch_task_count = len(getattr(batch_result, "tasks", [])) if batch_result else 0
                if (
                    batch_result is None
                    or not batch_complete
                    or batch_task_count != self.config.tasks.batch_size
                ):
                    print("Proposer batch failed")
                    child.status = NodeStatus.PROPOSER_FAILED
                    self.archive.update(child)
                    continue

                level2_result = self._evaluate_level2(child, batch_result)
                if (
                    level2_result is None
                    or len(level2_result.outcomes) != self.config.tasks.batch_size
                ):
                    print("Level 2 incomplete; child rejected")
                    child.status = NodeStatus.REJECTED
                    self.archive.update(child)
                    continue

                scores = self._compute_and_apply_scores(
                    child, level1_result, level2_result
                )

                child.status = NodeStatus.COMPLETE
                child.completed_at = datetime.now(timezone.utc)
                self.archive.update(child)
                self.budget.record_node()

                print(f"Node {child.node_id} committed: score={scores.node_score:.4f}")
                log_event(
                    self.run_context.paths.events_path,
                    "node_committed",
                    run_id=self.run_context.run_id,
                    node_id=child.node_id,
                    payload={
                        "node_score": scores.node_score,
                        "solver_score": scores.solver_score,
                        "proposer_score": scores.proposer_score,
                    },
                )

            except BudgetExhaustedError:
                break
            except Exception as e:
                print(f"Expansion error: {e}")
                import traceback
                traceback.print_exc()
                log_event(
                    self.run_context.paths.events_path,
                    "expansion_error",
                    run_id=self.run_context.run_id,
                    payload={"error": str(e)},
                )

        print(
            "\nEvolution complete. "
            f"Successful epochs: {self.budget.nodes_created}; "
            f"attempts: {self.budget.expansions_attempted}"
        )

    def _select_parent(self):
        eligible = self.archive.eligible_parents(
            self.config.scoring.min_parent_solved_tasks,
            scoring_mode=self.config.scoring.mode,
        )
        if not eligible:
            return None
        return self.selector.select(
            self.archive,
            self.rng,
            self.config.scoring.min_parent_solved_tasks,
        )

    def _ensure_root_bootstrap(self) -> bool:
        """Create a real T_root and evaluate the root solver before evolution."""
        root = self.archive.get(ROOT_NODE_ID)
        if root is None:
            return False
        existing = (
            self.task_store.tasks_for_batch(root.generated_task_batch_id)
            if self.task_store is not None and root.generated_task_batch_id
            else []
        )
        if (
            root.status == NodeStatus.COMPLETE
            and len(existing) == self.config.tasks.batch_size
            and root.level2_result_path
            and Path(root.level2_result_path).is_file()
        ):
            return True

        print(
            f"Bootstrapping root with {self.config.tasks.batch_size} "
            "trusted-valid tasks before any self-edit"
        )
        batch = self._generate_batch(root, parent=None, level1_result=None)
        if not batch.complete or len(batch.tasks) != self.config.tasks.batch_size:
            root.status = NodeStatus.PROPOSER_FAILED
            self.archive.update(root)
            return False

        level2 = self._evaluate_level2(root, batch)
        if level2 is None or len(level2.outcomes) != self.config.tasks.batch_size:
            root.status = NodeStatus.REJECTED
            self.archive.update(root)
            return False

        # P0-3: Root's ONLY special-case is retention_rate=1.0 (no parent).
        # valid_yield / causal_ablation_pass / batch_complete must come from
        # the real generation statistics, never hard-coded to 1.0.
        gate = self._hgm_quality_from_batch(root, batch)
        scores = compute_scores(
            retention_rate=1.0,
            frontier_accuracy=level2.accuracy,
            regression_weight=self.config.scoring.regression_weight,
            target_accuracy=self.config.scoring.proposer_target_accuracy,
            mode=self.config.scoring.mode,
            valid_yield=gate["valid_yield"],
            causal_ablation_pass=gate["causal_ablation_pass"],
            batch_complete=gate["batch_complete"],
            hgm_valid_yield_threshold=self.config.scoring.hgm_valid_yield_threshold,
            hgm_causal_ablation_pass_threshold=self.config.scoring.hgm_causal_ablation_pass_threshold,
            hgm_difficulty_min=self.config.scoring.hgm_difficulty_min,
        )
        root.retention_rate = scores.retention_rate
        root.frontier_accuracy = scores.frontier_accuracy
        root.solver_score = scores.solver_score
        root.proposer_score = scores.proposer_score
        root.node_score = scores.node_score
        root.selection_eligible = bool(scores.eligible)
        # BUG-11: record the root's trusted Level2 utilities so the Thompson
        # Sampling selector can include the root in its Beta posterior.
        root.utility_measures = [
            1.0 if outcome.resolved else 0.0
            for outcome in level2.outcomes
        ]
        root.evaluated_task_ids = [
            outcome.task_id for outcome in level2.outcomes
        ]
        needs_evolution_parent = self.budget.max_nodes > 0
        if (
            needs_evolution_parent
            and (root.solved_task_count or 0)
            < self.config.scoring.min_parent_solved_tasks
        ):
            root.status = NodeStatus.REJECTED
            self.archive.update(root)
            self.archive.save_node_json(root, self.run_context.paths.node_json(root.node_id))
            raise RuntimeError(
                "Root bootstrap produced a complete batch but the solver solved only "
                f"{root.solved_task_count or 0} tasks; at least "
                f"{self.config.scoring.min_parent_solved_tasks} are required for an "
                "eligible parent"
            )
        if needs_evolution_parent and root.node_score <= 0:
            root.status = NodeStatus.REJECTED
            self.archive.update(root)
            self.archive.save_node_json(root, self.run_context.paths.node_json(root.node_id))
            raise RuntimeError(
                "Root bootstrap score is zero, so root cannot be selected as a parent"
            )
        root.status = NodeStatus.COMPLETE
        root.completed_at = datetime.now(timezone.utc)
        self.archive.update(root)
        self.archive.save_node_json(root, self.run_context.paths.node_json(root.node_id))
        return True

    def _hgm_quality_from_batch(self, node, batch) -> dict:
        """P0-3: compute HGM gate inputs from a real generation batch.

        Used by root bootstrap (retention special-cased to 1.0 elsewhere) and
        unit-tested so valid_yield / causal_ablation_pass are never hard-coded.
        """
        proposer_summary = self._proposer_stats(node) if node is not None else {}
        accepted = int(
            proposer_summary.get("accepted", len(getattr(batch, "tasks", []) or []))
            or 0
        )
        candidates_validated = int(
            proposer_summary.get("candidates_validated", 0)
            or getattr(batch, "candidates_validated", 0)
            or 0
        )
        # ``_proposer_stats`` historically used ``generated`` for validated.
        if candidates_validated <= 0:
            candidates_validated = int(
                proposer_summary.get("generated", 0)
                or getattr(batch, "candidates_validated", 0)
                or getattr(batch, "candidates_generated", 0)
                or 0
            )
        if candidates_validated <= 0:
            candidates_validated = int(getattr(batch, "candidates_validated", 0) or 0)
            accepted = int(len(getattr(batch, "tasks", []) or []))
        valid_yield = (
            (accepted / candidates_validated) if candidates_validated > 0 else None
        )
        extended = (
            self._extended_proposer_stats(node, proposer_summary)
            if node is not None
            else {}
        )
        ablation_failures = int(extended.get("causal_ablation_failure_count", 0) or 0)
        if ablation_failures <= 0:
            # Also derive from batch rejection reasons when summary is thin.
            rejections = dict(getattr(batch, "rejection_reasons", {}) or {})
            for reason, count in rejections.items():
                reason_lower = str(reason).lower()
                if any(
                    key in reason_lower
                    for key in (
                        "causal_ablation",
                        "single_file_repair",
                        "independently_active",
                        "not_multi_file",
                        "trusted_causal_ablation",
                    )
                ):
                    ablation_failures += int(count)
        causal_ablation_pass = None
        if candidates_validated > 0:
            causal_ablation_pass = max(
                0.0, 1.0 - (ablation_failures / max(candidates_validated, 1))
            )
        batch_complete = bool(
            getattr(batch, "complete", False)
            and len(getattr(batch, "tasks", []) or []) == self.config.tasks.batch_size
        )
        return {
            "valid_yield": valid_yield,
            "causal_ablation_pass": causal_ablation_pass,
            "batch_complete": batch_complete,
            "accepted": accepted,
            "candidates_validated": candidates_validated,
        }

    def _prepare_diagnosis(self, parent):
        """Build and persist one joint-cycle diagnosis for the whole node."""
        from ..schemas.evaluation import Level1Result, Level2Result

        level1 = None
        if parent.level1_result_path and Path(parent.level1_result_path).is_file():
            level1 = Level1Result.model_validate(read_json(Path(parent.level1_result_path)))
        level2 = None
        if parent.level2_result_path and Path(parent.level2_result_path).is_file():
            level2 = Level2Result.model_validate(read_json(Path(parent.level2_result_path)))

        proposer_stats = self._proposer_stats(parent)
        artifacts = {
            "solver_trajectories": self._trajectory_excerpts(parent.node_id),
            "proposer_candidates": proposer_stats.pop("reports", []),
            "success_contrast": self._success_contrast(parent),
            "chain_plans": self._chain_plans(parent),
            "ablation_results": self._ablation_results(parent),
            "task_quality_summary": self._task_quality_summary(parent, proposer_stats),
        }

        # Failed child attempts are evidence about the parent system. They are
        # never eligible nodes, but their rejection reasons must influence the
        # next joint diagnosis.
        failed_children = [
            child
            for child in self.archive.children_of(parent.node_id)
            if child.status in (NodeStatus.LEVEL1_FAILED, NodeStatus.PROPOSER_FAILED)
        ]
        if failed_children:
            latest = max(failed_children, key=lambda value: value.created_at)
            failed_stats = self._proposer_stats(latest)
            if failed_stats.get("generated", 0) or failed_stats.get("rejections"):
                proposer_stats = {
                    key: value for key, value in failed_stats.items() if key != "reports"
                }
                artifacts["proposer_candidates"] = failed_stats.get("reports", [])
            artifacts["solver_trajectories"].extend(
                self._trajectory_excerpts(latest.node_id)
            )
            if not artifacts.get("chain_plans"):
                artifacts["chain_plans"] = self._chain_plans(latest)
            if not artifacts.get("ablation_results"):
                artifacts["ablation_results"] = self._ablation_results(latest)

        summary = self.cycle_builder.build(
            parent,
            level1=level1,
            proposer_stats=proposer_stats,
            level2=level2,
            is_root=parent.parent_node_id is None,
        )
        special_config = config_to_dict(self.config)["special_cases"]
        special_config["regression_threshold"] = self.config.scoring.regression_threshold
        solver_stats = self._solver_stats(parent, level2)
        extended_proposer_stats = self._extended_proposer_stats(parent, proposer_stats)
        tool_events = self._tool_events(parent.node_id)
        alerts = self.special_detector.detect(
            summary,
            trajectories=None,
            candidates=None,
            tool_events=tool_events,
            solver_stats=solver_stats,
            proposer_stats=extended_proposer_stats,
            config=special_config,
        )
        evidence = self.evidence_selector.select(summary, alerts, artifacts)
        diagnosis = self.diagnoser.diagnose(
            parent.node_id,
            summary,
            evidence,
            agent_code_summary=(
                "This commit is one joint Agent node containing coding_agent.py, "
                "proposer/, swesmith/, shared tools/, prompts and runtime. Ansible "
                "is only the external task repository."
            ),
        )

        self.run_context.paths.ensure_node_dirs(parent.node_id)
        atomic_write_json(
            self.run_context.paths.cycle_summary_path(parent.node_id),
            summary.model_dump(mode="json"),
        )
        atomic_write_json(
            self.run_context.paths.special_alerts_path(parent.node_id),
            [alert.model_dump(mode="json") for alert in alerts],
        )
        atomic_write_json(
            self.run_context.paths.evidence_bundle_path(parent.node_id),
            evidence.model_dump(mode="json"),
        )
        atomic_write_json(
            self.run_context.paths.diagnosis_path(parent.node_id),
            diagnosis.model_dump(mode="json"),
        )
        self.run_context.paths.problem_statement_path(parent.node_id).write_text(
            diagnosis.problem_statement.rstrip() + "\n",
            encoding="utf-8",
        )
        return diagnosis

    def _proposer_stats(self, node) -> dict:
        path = self.run_context.paths.proposer_dir(node.node_id) / "generation_summary.json"
        if not path.is_file():
            return {
                "requested": self.config.tasks.batch_size,
                "generated": 0,
                "accepted": 0,
                "rejections": {},
                "operators": {},
                "reports": [],
            }
        data = read_json(path)
        rejections = dict(data.get("rejection_reasons") or {})
        for item in data.get("engine_rejections") or []:
            reason = str(item.get("reason") or "engine_rejection")
            rejections[reason] = rejections.get(reason, 0) + 1
        return {
            "requested": self.config.tasks.batch_size,
            "generated": int(data.get("candidates_generated", 0)),
            "accepted": len(data.get("task_ids") or []),
            "rejections": rejections,
            "operators": {},
            "reports": list(data.get("validation_reports") or []),
        }

    def _trajectory_excerpts(self, node_id: str) -> list[str]:
        scratch = Path(self.config.execution.scratch_root)
        excerpts: list[str] = []
        for path in scratch.glob(
            f"{self.run_context.run_id}/solver/**/trajectories/**/trajectory.jsonl"
        ):
            if node_id not in path.parts or not path.is_file():
                continue
            excerpts.append(path.read_text(encoding="utf-8", errors="replace")[-20000:])
        return excerpts

    def _solver_stats(self, node, level2=None) -> dict:
        """Assemble solver-side stats for special detection.

        Counts empty patches, test-only patches, timeouts, context overflows,
        stochastic tasks, repeated tool loops, and localization collapses from
        the Level 2 outcomes and trajectory files.

        BUG-21: ``stochastic_task_count`` and ``localization_collapse_count``
        are computed from real trajectory structure (multiple rollouts of the
        same task, and structural signals like no production file opened /
        repeated search without narrowing), not left as always-zero counters.
        ``solver_rollouts`` reports the max number of rollouts observed for any
        single task so the detector can suppress stochasticity alerts when
        ``solver_rollouts < solver_stochasticity_min_rollouts``.
        """
        stats = {
            "empty_patch_count": 0,
            "test_only_patch_count": 0,
            "evaluated_count": 0,
            "timeout_count": 0,
            "context_overflow_count": 0,
            "stochastic_task_count": 0,
            "repeated_tool_loop_count": 0,
            "localization_collapse_count": 0,
            # BUG-21: max rollouts observed for any single task in this node.
            "solver_rollouts": 0,
            # BUG-21: number of distinct tasks with >=2 rollouts.
            "tasks_with_multiple_rollouts": 0,
        }
        if level2 is not None:
            stats["evaluated_count"] = len(level2.outcomes)
            from ..git.patch import extract_changed_files

            for outcome in level2.outcomes:
                patch_path = getattr(outcome, "patch_path", None)
                patch = ""
                if patch_path and Path(patch_path).is_file():
                    patch = Path(patch_path).read_text(encoding="utf-8", errors="replace")
                if not patch.strip():
                    stats["empty_patch_count"] += 1
                else:
                    # BUG-07: classify test-only patches by parsing changed
                    # files instead of substring-matching "test" in the patch
                    # text (which false-positives on any patch mentioning tests).
                    changed_files = extract_changed_files(patch)
                    production_files = [
                        p for p in changed_files if not _is_test_path(p)
                    ]
                    if changed_files and not production_files:
                        stats["test_only_patch_count"] += 1
                error_type = str(getattr(outcome, "error_type", "") or "").lower()
                if "timeout" in error_type or "timed out" in error_type:
                    stats["timeout_count"] += 1
                if "context" in error_type or "token" in error_type:
                    stats["context_overflow_count"] += 1

        # BUG-21 / P1-4: collect per-task rollout outcomes so stochasticity
        # only fires when the SAME task has inconsistent resolved outcomes
        # across rollouts (not merely because it was rolled out multiple times).
        rollout_counts: dict[str, int] = {}
        rollout_outcomes: dict[str, list[bool]] = {}
        excerpts = self._trajectory_excerpts(node.node_id)
        for excerpt in excerpts:
            lower = excerpt.lower()
            if "context length" in lower or "token limit" in lower:
                stats["context_overflow_count"] += 1
            if lower.count('"tool":') > 20 and lower.count("repeated") > 0:
                stats["repeated_tool_loop_count"] += 1
            if self._trajectory_localization_collapsed(excerpt):
                stats["localization_collapse_count"] += 1
            task_id = self._trajectory_task_id(excerpt)
            if task_id:
                rollout_counts[task_id] = rollout_counts.get(task_id, 0) + 1
                resolved = self._trajectory_resolved(excerpt)
                if resolved is not None:
                    rollout_outcomes.setdefault(task_id, []).append(resolved)
        if rollout_counts:
            stats["solver_rollouts"] = max(rollout_counts.values())
            stats["tasks_with_multiple_rollouts"] = sum(
                1 for c in rollout_counts.values() if c >= 2
            )
            # P1-4: only count tasks with inconsistent outcomes across rollouts.
            stochastic = 0
            for task_id, outcomes in rollout_outcomes.items():
                if len(outcomes) >= 2 and (True in outcomes) and (False in outcomes):
                    stochastic += 1
            stats["stochastic_task_count"] = stochastic
        return stats

    @staticmethod
    def _trajectory_resolved(excerpt: str) -> Optional[bool]:
        """P1-4: best-effort resolved flag from a trajectory excerpt."""
        if not excerpt:
            return None
        import json as _json

        for line in excerpt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = _json.loads(line)
            except Exception:
                continue
            if not isinstance(entry, dict):
                continue
            if "resolved" in entry:
                return bool(entry["resolved"])
            if "success" in entry:
                return bool(entry["success"])
            status = str(entry.get("status") or "").lower()
            if status in {"resolved", "success", "passed", "pass"}:
                return True
            if status in {"failed", "unresolved", "error", "fail"}:
                return False
        return None

    @staticmethod
    def _trajectory_localization_collapsed(excerpt: str) -> bool:
        """BUG-21: structural localization-collapse heuristic.

        Returns True when the trajectory shows none of the narrowing signals
        we expect from a healthy localization (opening a production file,
        locating a symbol, or producing a patch touching production code).
        """
        if not excerpt:
            return False
        lower = excerpt.lower()
        # Heuristic positive signals of successful localization.
        opened_production = (
            '"tool": "view' in lower
            or '"tool": "open' in lower
            or '"tool": "cat' in lower
            or '"tool": "read' in lower
            or '"name": "view' in lower
            or '"name": "open' in lower
            or '"name": "cat' in lower
            or '"name": "read' in lower
        )
        edited_file = (
            '"tool": "edit' in lower
            or '"tool": "str_replace' in lower
            or '"tool": "create' in lower
            or '"name": "edit' in lower
            or '"name": "str_replace' in lower
            or '"name": "create' in lower
        )
        # Negative signal: repeated search without narrowing.
        repeated_search = lower.count('"tool": "search') >= 4 or lower.count('"tool": "grep') >= 4
        return (not opened_production) and (not edited_file) and repeated_search

    @staticmethod
    def _trajectory_task_id(excerpt: str) -> str:
        """BUG-21: best-effort extraction of task_id from a trajectory excerpt."""
        if not excerpt:
            return ""
        import json as _json

        for line in excerpt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = _json.loads(line)
            except Exception:
                continue
            for key in ("task_id", "instance_id", "id"):
                value = entry.get(key) if isinstance(entry, dict) else None
                if value:
                    return str(value)
        return ""


    def _extended_proposer_stats(self, node, base_stats: dict) -> dict:
        """Extend proposer stats with RepoChain-specific counters.

        Reads the extended generation_summary.json written by the proposer,
        which now includes causal_ablation_failure_count,
        contract_generation_failure_count, clean_contract_failure_count,
        no_f2p_count, no_p2p_count, duplicate_count, statement_leakage_count.
        Falls back to 0 for each counter that the proposer has not yet emitted.
        """
        path = self.run_context.paths.proposer_dir(node.node_id) / "generation_summary.json"
        extended = {
            "causal_ablation_failure_count": 0,
            "contract_generation_failure_count": 0,
            "clean_contract_failure_count": 0,
            "no_f2p_count": 0,
            "no_p2p_count": 0,
            "duplicate_count": 0,
            "statement_leakage_count": 0,
        }
        if path.is_file():
            data = read_json(path)
            # P1-3: prefer structured repo_chain_stats when the proposer emitted it.
            structured = data.get("repo_chain_stats") or {}
            if isinstance(structured, dict):
                for key in extended:
                    if key in structured:
                        extended[key] = int(structured.get(key, 0) or 0)
            for key in extended:
                if key in data:
                    extended[key] = int(data.get(key, extended[key]) or 0)
            # Also derive no_f2p / no_p2p from rejection_reasons if present.
            rejections = data.get("rejection_reasons") or {}
            for reason, count in rejections.items():
                reason_lower = str(reason).lower()
                if "no_f2p" in reason_lower or "f2p" in reason_lower:
                    extended["no_f2p_count"] += int(count)
                if "no_p2p" in reason_lower or "p2p" in reason_lower:
                    extended["no_p2p_count"] += int(count)
                if "duplicate" in reason_lower:
                    extended["duplicate_count"] += int(count)
                if "leakage" in reason_lower or "leak" in reason_lower:
                    extended["statement_leakage_count"] += int(count)
                if any(
                    token in reason_lower
                    for token in (
                        "causal_ablation",
                        "single_file_repair",
                        "independently_active",
                        "not_multi_file",
                        "trusted_causal_ablation",
                    )
                ):
                    extended["causal_ablation_failure_count"] += int(count)
        return extended

    def _tool_events(self, node_id: str) -> list[dict]:
        """Extract tool-call events from solver trajectories for shared detection."""
        events: list[dict] = []
        for excerpt in self._trajectory_excerpts(node_id):
            import json as _json

            for line in excerpt.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = _json.loads(line)
                except Exception:
                    continue
                if isinstance(entry, dict) and entry.get("type") == "tool_call":
                    tool_name = str(entry.get("tool") or entry.get("name") or "unknown")
                    error = str(entry.get("error") or "")
                    if error:
                        events.append({"tool": tool_name, "error": error})
        return events

    def _success_contrast(self, node) -> str:
        """Build a success-contrast excerpt for the diagnoser.

        BUG-23: the contrast must be a real failure/success comparison, not a
        one-line "SOLVED task X". We surface the solved task id, the tool
        sequence, the files inspected, the patch files, the test behavior,
        and a trajectory excerpt so the diagnoser can actually contrast.
        """
        if not (node.level2_result_path and Path(node.level2_result_path).is_file()):
            return ""
        from ..schemas.evaluation import Level2Result

        level2 = Level2Result.model_validate(read_json(Path(node.level2_result_path)))
        solved = set(level2.solved_task_ids)
        if not solved:
            return ""
        # Prefer the first solved task with a non-empty patch and trajectory.
        scratch = Path(self.config.execution.scratch_root)
        for tid in level2.solved_task_ids:
            outcome = next((o for o in level2.outcomes if o.task_id == tid), None)
            if outcome is None:
                continue
            parts: list[str] = [f"SUCCESS CONTRAST — task {tid}"]
            # Patch files.
            patch_path = getattr(outcome, "patch_path", None)
            patch_text = ""
            if patch_path and Path(patch_path).is_file():
                patch_text = Path(patch_path).read_text(encoding="utf-8", errors="replace")
            if patch_text:
                from ..git.patch import extract_changed_files

                changed = extract_changed_files(patch_text)
                parts.append(f"patch_files: {', '.join(changed) if changed else '(none)'}")
                parts.append(f"patch_excerpt:\n{patch_text[:2000]}")
            # Test behavior.
            f2p = list(getattr(outcome, "fail_to_pass", []) or [])
            p2p = list(getattr(outcome, "pass_to_pass", []) or [])
            if f2p:
                parts.append(f"fail_to_pass: {f2p[:8]}")
            if p2p:
                parts.append(f"pass_to_pass: {p2p[:8]}")
            # Trajectory excerpt (tool sequence, files inspected).
            for traj_path in scratch.glob(
                f"{self.run_context.run_id}/solver/**/trajectories/**/{tid}*/trajectory.jsonl"
            ):
                if not traj_path.is_file():
                    continue
                text = traj_path.read_text(encoding="utf-8", errors="replace")
                tool_seq = self._extract_tool_sequence(text)
                if tool_seq:
                    parts.append(f"tool_sequence: {tool_seq}")
                parts.append(f"trajectory_excerpt:\n{text[-4000:]}")
                break
            else:
                # Fallback: search by node_id + task_id in path parts.
                for traj_path in scratch.glob(
                    f"{self.run_context.run_id}/solver/**/trajectories/**/trajectory.jsonl"
                ):
                    if node.node_id not in traj_path.parts or tid not in traj_path.parts:
                        continue
                    text = traj_path.read_text(encoding="utf-8", errors="replace")
                    tool_seq = self._extract_tool_sequence(text)
                    if tool_seq:
                        parts.append(f"tool_sequence: {tool_seq}")
                    parts.append(f"trajectory_excerpt:\n{text[-4000:]}")
                    break
            return "\n\n".join(parts)
        return ""

    @staticmethod
    def _extract_tool_sequence(trajectory_text: str) -> str:
        """BUG-23: extract the ordered tool names from a trajectory JSONL."""
        import json as _json

        tools: list[str] = []
        for line in str(trajectory_text).splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = _json.loads(line)
            except Exception:
                continue
            if isinstance(entry, dict):
                tool = entry.get("tool") or (entry.get("action") or {}).get("tool")
                if not tool and isinstance(entry.get("action"), dict):
                    tool = entry["action"].get("tool") or entry["action"].get("name")
                if tool:
                    tools.append(str(tool))
        return ", ".join(tools[:20]) if tools else ""


    def _chain_plans(self, node) -> list[str]:
        """Extract RepoChain plan summaries from the proposer output."""
        proposer_dir = self.run_context.paths.proposer_dir(node.node_id)
        plans: list[str] = []
        result_path = proposer_dir / "proposer_result.json"
        if result_path.is_file():
            import json as _json

            try:
                data = _json.loads(result_path.read_text(encoding="utf-8"))
                for plan in data.get("plans") or []:
                    plans.append(_json.dumps(plan, ensure_ascii=False)[:4000])
            except Exception:
                pass
        return plans

    def _ablation_results(self, node) -> str:
        """Extract causal ablation results from the proposer output."""
        proposer_dir = self.run_context.paths.proposer_dir(node.node_id)
        summary_path = proposer_dir / "generation_summary.json"
        if summary_path.is_file():
            data = read_json(summary_path)
            parts = []
            for key in ("causal_ablation_pass", "causal_ablation_failure_count"):
                if key in data:
                    parts.append(f"{key}: {data[key]}")
            return "; ".join(parts)
        return ""

    def _task_quality_summary(self, node, proposer_stats: dict) -> str:
        """Build a short task-quality summary for empty-patch diagnosis."""
        parts = [
            f"generated={proposer_stats.get('generated', 0)}",
            f"accepted={proposer_stats.get('accepted', 0)}",
            f"rejections={proposer_stats.get('rejections', {})}",
        ]
        return "TaskQualitySummary: " + ", ".join(parts)

    def _build_child(self, parent, diagnosis=None):
        from ..evolution.child_builder import ChildBuildResult
        from ..schemas.diagnosis import CycleDiagnosis

        if self.child_builder is None:
            return ChildBuildResult(passed=True, node=None)

        if diagnosis is None:
            raise RuntimeError("A persisted joint-cycle diagnosis is required")
        return self.child_builder.build(
            parent, diagnosis, self.config.models.self_improve_model
        )

    def _evaluate_level1(self, parent, child):
        if self.level1_evaluator is None or self.solver_runner is None:
            return None
        parent_task_ids = self._parent_solved_task_ids(parent)
        tasks = [self.task_store.get(tid) for tid in parent_task_ids] if self.task_store else []
        tasks = [t for t in tasks if t is not None]
        outcomes = [
            self.solver_runner.run_task(
                node=child,
                task=task,
                level=1,
                seed=self.config.run.seed,
                run_id=self.run_context.run_id,
            )
            for task in tasks
        ]
        result = self.level1_evaluator.compute_retention(parent_task_ids, outcomes)
        result.parent_node_id = parent.node_id
        result.child_node_id = child.node_id
        path = self.run_context.paths.level1_result(child.node_id)
        atomic_write_json(path, result.model_dump())
        child.level1_result_path = str(path)
        return result

    def _generate_batch(self, child, parent=None, level1_result=None):
        if self.task_provider is None and self.task_batch_builder is None:
            return None
        self.run_context.paths.ensure_node_dirs(child.node_id)
        trajectories: list[str] = []
        scratch = Path(self.config.execution.scratch_root)
        allowed_node_ids = {child.node_id}
        if parent is not None:
            allowed_node_ids.add(parent.node_id)
        for path in scratch.glob(
            f"{self.run_context.run_id}/solver/**/trajectories/**/trajectory.jsonl"
        ):
            if not any(node_id in path.parts for node_id in allowed_node_ids):
                continue
            value = str(path.resolve())
            if value not in trajectories:
                trajectories.append(value)
        parent_task_ids = self._parent_solved_task_ids(parent) if parent is not None else []
        from ..tasks.provider import TaskGenerationContext

        # P0-9: Parent source = Parent Level2 unresolved failures only.
        # Child source = Current Child Level1 forgotten/unresolved only.
        # Do NOT treat successful parent/child trajectories as weakness sources.
        parent_failure_trajectories: list[str] = []
        current_child_level1_trajectories: list[str] = []
        parent_failed_task_ids = (
            self._parent_failed_task_ids(parent) if parent is not None else []
        )
        child_forgotten_task_ids: list[str] = []
        if level1_result is not None:
            child_forgotten_task_ids = list(
                getattr(level1_result, "child_forgotten_task_ids", None)
                or getattr(level1_result, "forgotten_task_ids", None)
                or []
            )
            # Also include tasks the child newly failed to retain from parent.
            for tid in getattr(level1_result, "evaluated_task_ids", []) or []:
                retained = set(
                    getattr(level1_result, "child_retained_task_ids", None) or []
                )
                if tid not in retained and tid not in child_forgotten_task_ids:
                    child_forgotten_task_ids.append(tid)

        if parent is not None and child is not None and parent.node_id != child.node_id:
            for value in trajectories:
                if parent.node_id in value and self._trajectory_matches_any_task(
                    value, parent_failed_task_ids
                ):
                    parent_failure_trajectories.append(value)
                elif child.node_id in value and self._trajectory_matches_any_task(
                    value, child_forgotten_task_ids
                ):
                    current_child_level1_trajectories.append(value)

        context = TaskGenerationContext(
            node=child,
            parent=parent,
            level1_result=level1_result,
            parent_failure_trajectories=parent_failure_trajectories,
            current_child_level1_trajectories=current_child_level1_trajectories,
            solver_trajectories=trajectories,
            parent_task_ids=parent_task_ids,
            parent_solved_task_ids=list(parent_task_ids),
            run_id=self.run_context.run_id,
            output_dir=self.run_context.paths.proposer_dir(child.node_id),
            model=self.config.models.proposer_model,
            task_store_dir=self.config.paths.task_store,
            bootstrap=parent is None,
        )
        if self.task_provider is not None:
            result = self.task_provider.get_tasks(child, context)
        else:
            # Legacy fallback when a raw TaskBatchBuilder was injected.
            bound_runner = (
                self.proposer_runner.for_node(child)
                if hasattr(self.proposer_runner, "for_node")
                else self.proposer_runner
            )
            legacy = self.task_batch_builder.build_for_node(
                node_id=child.node_id,
                repo_pool=self.repo_pool,
                validator=self.validator,
                task_committer=self.task_committer,
                proposer_runner=bound_runner,
                solver_trajectories=trajectories,
                parent_task_ids=parent_task_ids,
                output_dir=self.run_context.paths.proposer_dir(child.node_id),
                agent_code_dir="",
                model=self.config.models.proposer_model,
                run_id=self.run_context.run_id,
                task_store_dir=self.config.paths.task_store,
                bootstrap=parent is None,
                # BUG-08/09: forward split buckets for the legacy path too.
                parent_failure_trajectories=parent_failure_trajectories,
                current_child_level1_trajectories=current_child_level1_trajectories,
            )
            from ..tasks.provider import TaskBatch

            result = TaskBatch(
                batch_id=legacy.batch_id,
                node_id=legacy.node_id,
                tasks=list(legacy.tasks),
                complete=bool(legacy.complete),
                rejected_candidates=legacy.rejected_candidates,
                rejection_reasons=dict(legacy.rejection_reasons),
                candidates_generated=legacy.candidates_generated,
                candidates_validated=legacy.candidates_validated,
                validation_reports=list(legacy.validation_reports),
                proposer_error=legacy.proposer_error,
                engine_rejections=list(legacy.engine_rejections),
            )
        child.generated_task_batch_id = result.batch_id
        summary_path = self.run_context.paths.proposer_dir(child.node_id) / "generation_summary.json"
        # BUG-20: persist RepoChain-specific counters as structured fields so
        # the special detectors can read them directly instead of substring
        # matching on rejection_reasons.
        rejection_reasons = dict(result.rejection_reasons)
        def _count(*keys: str) -> int:
            total = 0
            for reason, count in rejection_reasons.items():
                reason_lower = str(reason).lower()
                if any(k in reason_lower for k in keys):
                    total += int(count)
            return total
        repo_chain_stats = {
            "contract_generation_failure_count": _count("contract_generation", "contract_not_restored", "contract_not_built"),
            "clean_contract_failure_count": _count("clean_contract"),
            "mutation_materialization_failure_count": _count("mutation_materialization", "mutation_failed", "engine_returned_no_candidates"),
            "causal_ablation_failure_count": _count("causal_ablation", "single_file_repair", "independently_active"),
            "no_f2p_count": _count("no_f2p", "f2p"),
            "no_p2p_count": _count("no_p2p", "p2p"),
            "duplicate_count": _count("duplicate"),
            "statement_leakage_count": _count("leakage", "leak", "statement_audit"),
        }
        atomic_write_json(
            summary_path,
            {
                "batch_id": result.batch_id,
                "node_id": result.node_id,
                "complete": result.complete,
                "task_ids": [t.task_id for t in result.tasks],
                "rejected_candidates": result.rejected_candidates,
                "rejection_reasons": result.rejection_reasons,
                "candidates_generated": result.candidates_generated,
                "candidates_validated": result.candidates_validated,
                "validation_reports": result.validation_reports,
                "proposer_error": result.proposer_error,
                "engine_rejections": result.engine_rejections,
                # BUG-20: structured RepoChain counters (also available as a
                # nested ``repo_chain_stats`` object for consumers that prefer
                # the grouped shape from the bugfix guide).
                "repo_chain_stats": repo_chain_stats,
                **repo_chain_stats,
            },
        )
        return result

    def _evaluate_level2(self, child, batch_result):
        if self.level2_evaluator is None or self.solver_runner is None or batch_result is None:
            return None
        tasks = list(getattr(batch_result, "tasks", []))
        outcomes = [
            self.solver_runner.run_task(
                node=child,
                task=task,
                level=2,
                seed=self.config.run.seed,
                run_id=self.run_context.run_id,
            )
            for task in tasks
        ]
        result = self.level2_evaluator.compute_accuracy(
            child.node_id,
            getattr(batch_result, "batch_id", ""),
            outcomes,
        )
        path = self.run_context.paths.level2_result(child.node_id)
        atomic_write_json(path, result.model_dump())
        child.level2_result_path = str(path)
        child.solved_task_count = len(result.solved_task_ids)
        return result

    def _parent_solved_task_ids(self, parent) -> list[str]:
        if parent.level2_result_path:
            path = Path(parent.level2_result_path)
            if path.exists():
                data = read_json(path)
                return list(data.get("solved_task_ids", []))
        if parent.generated_task_batch_id and self.task_store is not None:
            return [t.task_id for t in self.task_store.tasks_for_batch(parent.generated_task_batch_id)]
        return []

    def _parent_failed_task_ids(self, parent) -> list[str]:
        """P0-9: Parent Level2 unresolved task ids only."""
        if parent is None:
            return []
        if parent.level2_result_path:
            path = Path(parent.level2_result_path)
            if path.exists():
                data = read_json(path)
                failed = list(data.get("failed_task_ids", []) or [])
                if failed:
                    return failed
                # Derive from outcomes when failed_task_ids is absent.
                failed = [
                    o.get("task_id")
                    for o in (data.get("outcomes") or [])
                    if isinstance(o, dict) and not o.get("resolved", True)
                ]
                return [tid for tid in failed if tid]
        return []

    @staticmethod
    def _trajectory_matches_any_task(trajectory_path: str, task_ids: list[str]) -> bool:
        """P0-9: match a trajectory path/content against unresolved task ids.

        When ``task_ids`` is empty we return False (do not treat unknown
        trajectories as failure sources). Matching is substring-based on the
        path first, then a cheap peek into the jsonl head for ``task_id``.
        """
        if not task_ids:
            return False
        path_str = str(trajectory_path)
        for tid in task_ids:
            if tid and tid in path_str:
                return True
        try:
            p = Path(trajectory_path)
            if not p.is_file():
                return False
            # Read a small head to avoid loading huge trajectories.
            head = p.read_text(encoding="utf-8", errors="replace")[:8000]
            for tid in task_ids:
                if tid and tid in head:
                    return True
        except Exception:
            return False
        return False

    def _compute_and_apply_scores(self, child, level1_result, level2_result):
        r = level1_result.retention_rate if level1_result else 0.5
        p = level2_result.accuracy if level2_result else 0.5

        # BUG-12: feed the HGM quality-gate inputs (valid_yield, causal
        # ablation pass rate, batch completeness) into compute_scores so its
        # ``eligible`` flag reflects the full gate rather than just b > 0.
        proposer_summary = self._proposer_stats(child)
        candidates_validated = int(proposer_summary.get("generated", 0)) or 0
        accepted = int(proposer_summary.get("accepted", 0))
        valid_yield = (accepted / candidates_validated) if candidates_validated > 0 else None
        extended = self._extended_proposer_stats(child, proposer_summary)
        ablation_failures = int(extended.get("causal_ablation_failure_count", 0))
        causal_ablation_pass = None
        if candidates_validated > 0 and ablation_failures is not None:
            causal_ablation_pass = max(
                0.0, 1.0 - (ablation_failures / max(candidates_validated, 1))
            )
        batch_complete = bool(
            child.generated_task_batch_id
            and level2_result is not None
            and len(level2_result.outcomes) == self.config.tasks.batch_size
        )

        scores = compute_scores(
            retention_rate=r,
            frontier_accuracy=p,
            regression_weight=self.config.scoring.regression_weight,
            target_accuracy=self.config.scoring.proposer_target_accuracy,
            mode=self.config.scoring.mode,
            valid_yield=valid_yield,
            causal_ablation_pass=causal_ablation_pass,
            batch_complete=batch_complete,
            hgm_valid_yield_threshold=self.config.scoring.hgm_valid_yield_threshold,
            hgm_causal_ablation_pass_threshold=self.config.scoring.hgm_causal_ablation_pass_threshold,
            hgm_difficulty_min=self.config.scoring.hgm_difficulty_min,
        )

        child.retention_rate = scores.retention_rate
        child.frontier_accuracy = scores.frontier_accuracy
        child.solver_score = scores.solver_score
        child.proposer_score = scores.proposer_score
        child.node_score = scores.node_score
        # BUG-12: persist the HGM quality-gate result so eligible_parents can
        # filter on it instead of the weaker proposer_score > 0 heuristic.
        child.selection_eligible = bool(scores.eligible)

        # BUG-11: populate utility_measures from trusted Level2 outcomes so the
        # Thompson Sampling selector has a Beta posterior to sample from. Each
        # solved task contributes 1.0, each unresolved task 0.0.
        if level2_result is not None:
            child.utility_measures = [
                1.0 if outcome.resolved else 0.0
                for outcome in level2_result.outcomes
            ]
            child.evaluated_task_ids = [
                outcome.task_id for outcome in level2_result.outcomes
            ]

        atomic_write_json(
            self.run_context.paths.node_scores(child.node_id),
            {
                "retention_rate": scores.retention_rate,
                "frontier_accuracy": scores.frontier_accuracy,
                "solver_score": scores.solver_score,
                "proposer_score": scores.proposer_score,
                "node_score": scores.node_score,
                "selection_eligible": child.selection_eligible,
                "utility_measures": child.utility_measures,
            }
        )

        return scores


def atomic_write_yaml(path: Path, data: dict) -> None:
    """Write YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
