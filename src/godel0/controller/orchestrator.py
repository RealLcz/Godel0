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

from ..config import Godel0Config, config_to_dict
from ..constants import ROOT_NODE_ID
from ..errors import BudgetExhaustedError
from ..schemas.node import NodeRecord, NodeStatus
from ..storage.atomic import read_json
from ..storage.atomic import atomic_write_json
from ..storage.event_log import log_event
from ..tree.archive import NodeArchive
from ..tree.selection import EpsilonGreedySelector, ParentSelector
from .budget import Budget
from .run_context import RunContext
from .scorer import compute_scores


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
        self.task_batch_builder = task_batch_builder
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
        runs_dir = Path(config.paths.runs)
        run_context = RunContext.create(config, runs_dir)

        atomic_write_yaml(run_context.paths.config_path, config_to_dict(config))

        archive = NodeArchive(run_context.paths.archive_path)
        cls._initialize_root(config, archive, run_context)
        selector = EpsilonGreedySelector(epsilon=0.1)
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
        )
        task_committer = TaskCommitter(task_store)

        agent_adapter = cls._build_agent_adapter()
        from ..tasks.node_proposer import NodeProposerRunner

        proposer_runner = NodeProposerRunner(
            agent_repo=Path(config.paths.agent_repo),
            scratch_root=Path(config.execution.scratch_root)
            / run_context.run_id
            / "node_proposer",
            timeout_sec=config.proposer.candidate_timeout_sec
            * config.tasks.max_generation_candidates,
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
                strategy_weights=config.proposer.strategies,
                contract_test_renderer=config.proposer.contract_test_renderer,
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
                model=config.models.agent_model,
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
        eligible = self.archive.eligible_parents(self.config.scoring.min_parent_solved_tasks)
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

        scores = compute_scores(
            retention_rate=1.0,
            frontier_accuracy=level2.accuracy,
            regression_weight=self.config.scoring.regression_weight,
            target_accuracy=self.config.scoring.proposer_target_accuracy,
        )
        root.retention_rate = scores.retention_rate
        root.frontier_accuracy = scores.frontier_accuracy
        root.solver_score = scores.solver_score
        root.proposer_score = scores.proposer_score
        root.node_score = scores.node_score
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

        summary = self.cycle_builder.build(
            parent,
            level1=level1,
            proposer_stats=proposer_stats,
            level2=level2,
            is_root=parent.parent_node_id is None,
        )
        special_config = config_to_dict(self.config)["special_cases"]
        special_config["regression_threshold"] = self.config.scoring.regression_threshold
        alerts = self.special_detector.detect(summary, config=special_config)
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

    def _build_child(self, parent, diagnosis=None):
        from ..evolution.child_builder import ChildBuildResult
        from ..schemas.diagnosis import CycleDiagnosis

        if self.child_builder is None:
            return ChildBuildResult(passed=True, node=None)

        if diagnosis is None:
            raise RuntimeError("A persisted joint-cycle diagnosis is required")
        return self.child_builder.build(parent, diagnosis, self.config.models.agent_model)

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
        if self.task_batch_builder is None:
            return None
        self.run_context.paths.ensure_node_dirs(child.node_id)
        trajectories: list[str] = []
        bootstrap = os.environ.get("GODEL0_BOOTSTRAP_SOLVER_TRAJECTORY", "")
        if bootstrap and Path(bootstrap).is_file():
            trajectories.append(bootstrap)
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
        bound_runner = (
            self.proposer_runner.for_node(child)
            if hasattr(self.proposer_runner, "for_node")
            else self.proposer_runner
        )
        result = self.task_batch_builder.build_for_node(
            node_id=child.node_id,
            repo_pool=self.repo_pool,
            validator=self.validator,
            task_committer=self.task_committer,
            proposer_runner=bound_runner,
            solver_trajectories=trajectories,
            parent_task_ids=parent_task_ids,
            output_dir=self.run_context.paths.proposer_dir(child.node_id),
            agent_code_dir="",
            model=self.config.models.agent_model,
            run_id=self.run_context.run_id,
            task_store_dir=self.config.paths.task_store,
        )
        child.generated_task_batch_id = result.batch_id
        summary_path = self.run_context.paths.proposer_dir(child.node_id) / "generation_summary.json"
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

    def _compute_and_apply_scores(self, child, level1_result, level2_result):
        r = level1_result.retention_rate if level1_result else 0.5
        p = level2_result.accuracy if level2_result else 0.5

        scores = compute_scores(
            retention_rate=r,
            frontier_accuracy=p,
            regression_weight=self.config.scoring.regression_weight,
            target_accuracy=self.config.scoring.proposer_target_accuracy,
        )

        child.retention_rate = scores.retention_rate
        child.frontier_accuracy = scores.frontier_accuracy
        child.solver_score = scores.solver_score
        child.proposer_score = scores.proposer_score
        child.node_score = scores.node_score

        atomic_write_json(
            self.run_context.paths.node_scores(child.node_id),
            {
                "retention_rate": scores.retention_rate,
                "frontier_accuracy": scores.frontier_accuracy,
                "solver_score": scores.solver_score,
                "proposer_score": scores.proposer_score,
                "node_score": scores.node_score,
            }
        )

        return scores


def atomic_write_yaml(path: Path, data: dict) -> None:
    """Write YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
