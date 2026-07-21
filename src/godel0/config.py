"""Configuration system: frozen dataclasses loaded from YAML with validation."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .errors import ConfigError


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    run_name: Optional[str] = None
    max_nodes: int = 200
    max_expansions: int = 200
    resume_from: Optional[str] = None


@dataclass(frozen=True)
class ModelConfig:
    # BUG-24: every chat() call must use an explicit model. Even when all
    # roles share the same model, each is recorded explicitly so experiments
    # are reproducible. ``agent_model`` remains as an alias of ``solver_model``
    # for backward compatibility with existing config files.
    solver_model: str = "deepseek/deepseek-chat"
    proposer_model: str = "deepseek/deepseek-chat"
    diagnose_model: str = "deepseek/deepseek-chat"
    self_improve_model: str = "deepseek/deepseek-chat"
    temperature: float = 0.0
    max_tokens: int = 32768

    @property
    def agent_model(self) -> str:
        """Backward-compatible alias for ``solver_model``."""
        return self.solver_model

    @property
    def agent_model_set(self) -> bool:
        return True


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 100
    max_tool_errors: int = 5
    trajectory_format: str = "jsonl"
    self_evolve_timeout_sec: int = 3600


@dataclass(frozen=True)
class TaskConfig:
    batch_size: int = 10
    max_generation_candidates: int = 50
    candidates_per_signature: int = 5
    allow_same_repo_transfer: bool = True
    allow_cross_repo_transfer: bool = True
    sources: "TaskSourceConfig" = field(default_factory=lambda: TaskSourceConfig())


@dataclass(frozen=True)
class TaskSourceConfig:
    """Quotas for task sources.

    parent_failure: tasks generated from the parent node's failure trajectories.
    current_child_level1: tasks generated from the current child's Level 1
        failures (forgotten tasks). The sum of quotas should equal batch_size;
        if it doesn't, the provider falls back dynamically.
    """
    parent_failure: "TaskSourceQuota" = field(
        default_factory=lambda: TaskSourceQuota(name="parent_failure", quota=5)
    )
    current_child_level1: "TaskSourceQuota" = field(
        default_factory=lambda: TaskSourceQuota(name="current_child_level1", quota=5)
    )


@dataclass(frozen=True)
class TaskSourceQuota:
    name: str
    quota: int = 5


@dataclass(frozen=True)
class SelectionConfig:
    """Parent selection strategy configuration.

    strategy: "thompson_sampling" (default, main experiment) or
        "epsilon_greedy" (ablation only).
    num_pseudo_descendant_evals: HGM-style pseudo-descendant count used to
        stabilize the Beta posterior when a node has many measurements.
    """
    strategy: str = "thompson_sampling"
    num_pseudo_descendant_evals: int = 10
    epsilon: float = 0.1


@dataclass(frozen=True)
class ScoringConfig:
    regression_threshold: float = 0.8
    regression_weight: float = 0.5
    proposer_target_accuracy: float = 0.5
    min_parent_solved_tasks: int = 3
    # Phase 8: Scoring Ablation.
    # "joint" (default): node_score = a * b (Godel0 original).
    # "hgm": solver_score = a, node_score = a; b is an eligibility gate only.
    mode: str = "joint"
    # HGM gate thresholds (only used in "hgm" mode).
    hgm_valid_yield_threshold: float = 0.20
    hgm_causal_ablation_pass_threshold: float = 0.50
    hgm_difficulty_min: float = 0.30
    selection: SelectionConfig = field(default_factory=SelectionConfig)


@dataclass(frozen=True)
class DiagnosisConfig:
    one_primary_root_cause: bool = True
    max_solver_trajectories: int = 4
    max_proposer_candidates: int = 4
    max_tool_incidents: int = 2
    max_raw_chars_per_item: int = 20000
    max_total_evidence_chars: int = 120000
    include_success_contrast: bool = True
    prioritize_special_alerts: bool = True
    special_alert_override_requires_reason: bool = True


@dataclass(frozen=True)
class SpecialCaseConfig:
    solver_empty_patch_ratio: float = 0.10
    solver_stochasticity_min_rollouts: int = 3
    solver_context_error_count: int = 1
    proposer_empty_batch_ratio: float = 0.50
    proposer_min_valid_yield: float = 0.20
    proposer_duplicate_ratio: float = 0.25
    proposer_leakage_ratio: float = 0.10
    difficulty_low: float = 0.40
    difficulty_high: float = 0.60


@dataclass(frozen=True)
class EvaluationConfig:
    solver_rollouts: int = 1
    deterministic: bool = True
    level1_timeout_sec: int = 3600
    level2_timeout_sec: int = 3600
    max_workers: int = 8


@dataclass(frozen=True)
class RepoChainWorkflowConfig:
    """Configuration for the RepoChain proposer workflow.

    RepoChain is the default Proposer workflow (not a mutation backend).
    The mutation_backends dict selects which low-level mutation mechanisms
    RepoChain uses internally, weighted by probability.
    """
    min_files: int = 2
    max_files: int = 6
    min_mutation_sites: int = 3
    max_mutation_sites: int = 8
    context_file_budget: int = 10
    require_generated_contracts: bool = True
    require_causal_ablation: bool = True
    mutation_backends: Dict[str, float] = field(default_factory=lambda: {
        "lm_modify": 0.5,
        "procedural": 0.2,
        "pr_replay": 0.3,
    })


@dataclass(frozen=True)
class ProposerConfig:
    initial_workflow: str = "repo_chain"
    repo_chain: RepoChainWorkflowConfig = field(default_factory=RepoChainWorkflowConfig)
    candidate_timeout_sec: int = 120
    max_patch_lines: int = 80
    forbid_test_file_edits: bool = True
    require_f2p: bool = True
    contract_test_renderer: str = ""


@dataclass(frozen=True)
class ExecutionConfig:
    backend: str = "subprocess"
    apptainer_bin: str = "apptainer"
    agent_image: Optional[str] = None
    repo_image_dir: Optional[str] = None
    scratch_root: str = "./scratch"
    clean_env: bool = True
    network_disabled: bool = True


@dataclass(frozen=True)
class PathConfig:
    agent_repo: str = "./agent_repo"
    repo_pool: str = "./repo_pool"
    runs: str = "./runs"
    task_store: str = "./task_store"


@dataclass(frozen=True)
class Godel0Config:
    run: RunConfig = field(default_factory=RunConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    diagnosis: DiagnosisConfig = field(default_factory=DiagnosisConfig)
    special_cases: SpecialCaseConfig = field(default_factory=SpecialCaseConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    proposer: ProposerConfig = field(default_factory=ProposerConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    paths: PathConfig = field(default_factory=PathConfig)


_SUBCONFIG_KEYS = {
    "run": RunConfig,
    "models": ModelConfig,
    "agent": AgentConfig,
    "tasks": TaskConfig,
    "scoring": ScoringConfig,
    "diagnosis": DiagnosisConfig,
    "special_cases": SpecialCaseConfig,
    "evaluation": EvaluationConfig,
    "proposer": ProposerConfig,
    "execution": ExecutionConfig,
    "paths": PathConfig,
}


def _build_subconfig(key: str, data: Any) -> Any:
    cls = _SUBCONFIG_KEYS[key]
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"Config section '{key}' must be a mapping, got {type(data).__name__}")
    if key == "proposer":
        rc_data = data.get("repo_chain", {})
        if rc_data is not None and not isinstance(rc_data, dict):
            raise ConfigError("proposer.repo_chain must be a mapping")
        valid_rc_keys = {f.name for f in fields(RepoChainWorkflowConfig)}
        filtered_rc = {k: v for k, v in (rc_data or {}).items() if k in valid_rc_keys}
        data = {k: v for k, v in data.items() if k != "repo_chain"}
        data["repo_chain"] = RepoChainWorkflowConfig(**filtered_rc)
    if key == "tasks":
        sources_data = data.get("sources", {})
        if sources_data is not None and not isinstance(sources_data, dict):
            raise ConfigError("tasks.sources must be a mapping")
        src = TaskSourceConfig()
        if sources_data:
            pf = sources_data.get("parent_failure", {})
            cc = sources_data.get("current_child_level1", {})
            if isinstance(pf, dict):
                pf = TaskSourceQuota(**{k: v for k, v in pf.items() if k in {f2.name for f2 in fields(TaskSourceQuota)}})
            if isinstance(cc, dict):
                cc = TaskSourceQuota(**{k: v for k, v in cc.items() if k in {f2.name for f2 in fields(TaskSourceQuota)}})
            src = TaskSourceConfig(parent_failure=pf, current_child_level1=cc)
        data = {k: v for k, v in data.items() if k != "sources"}
        data["sources"] = src
    if key == "scoring":
        sel_data = data.get("selection", {})
        if sel_data is not None and not isinstance(sel_data, dict):
            raise ConfigError("scoring.selection must be a mapping")
        valid_sel_keys = {f.name for f in fields(SelectionConfig)}
        filtered_sel = {k: v for k, v in (sel_data or {}).items() if k in valid_sel_keys}
        data = {k: v for k, v in data.items() if k != "selection"}
        data["selection"] = SelectionConfig(**filtered_sel)
    if key == "models":
        # BUG-24: backward compatibility -- legacy configs may still specify
        # ``agent_model``. Map it onto ``solver_model`` (and, when the new
        # explicit fields are absent, propagate to proposer / self_improve so
        # every chat() call still has an explicit model).
        if "agent_model" in data and "solver_model" not in data:
            data["solver_model"] = data.pop("agent_model")
        else:
            data.pop("agent_model", None)
        solver = data.get("solver_model")
        if solver:
            data.setdefault("proposer_model", solver)
            data.setdefault("diagnose_model", solver)
            data.setdefault("self_improve_model", solver)
    valid_keys = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)


def _validate(config: Godel0Config) -> None:
    if config.tasks.batch_size <= 0:
        raise ConfigError("tasks.batch_size must be > 0")
    if not (0 <= config.scoring.regression_threshold <= 1):
        raise ConfigError("scoring.regression_threshold must be in [0, 1]")
    if not (0 <= config.scoring.regression_weight <= 1):
        raise ConfigError("scoring.regression_weight must be in [0, 1]")
    if config.scoring.proposer_target_accuracy != 0.5:
        raise ConfigError("scoring.proposer_target_accuracy must be 0.5 (fixed in v1)")
    if config.tasks.max_generation_candidates < config.tasks.batch_size:
        raise ConfigError("tasks.max_generation_candidates must be >= tasks.batch_size")
    if config.evaluation.max_workers < 1:
        raise ConfigError("evaluation.max_workers must be >= 1")
    total_backends = sum(config.proposer.repo_chain.mutation_backends.values())
    if config.proposer.repo_chain.mutation_backends and abs(total_backends - 1.0) > 0.001:
        raise ConfigError(
            f"proposer.repo_chain.mutation_backends must sum to 1.0, got {total_backends}"
        )


def _apply_overrides(config_dict: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(config_dict)
    for key, value in overrides.items():
        if value is None:
            continue
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def load_config(
    path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> Godel0Config:
    """Load configuration from a YAML file with optional dot-notation overrides."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw = _apply_overrides(raw, overrides)
    kwargs: Dict[str, Any] = {}
    for key, cls in _SUBCONFIG_KEYS.items():
        kwargs[key] = _build_subconfig(key, raw.get(key))
    config = Godel0Config(**kwargs)
    _validate(config)
    return config


def config_to_dict(config: Godel0Config) -> Dict[str, Any]:
    """Serialize config back to a plain dict suitable for YAML dump."""
    result: Dict[str, Any] = {}
    for key in _SUBCONFIG_KEYS:
        sub = getattr(config, key)
        d: Dict[str, Any] = {}
        for f in fields(sub):
            d[f.name] = getattr(sub, f.name)
        result[key] = d
    return result


def save_config(config: Godel0Config, path: Union[str, Path]) -> None:
    """Save config to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config_to_dict(config), f, default_flow_style=False, sort_keys=False)
