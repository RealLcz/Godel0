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
    agent_model: str = "deepseek/deepseek-chat"
    diagnose_model: str = "deepseek/deepseek-chat"
    temperature: float = 0.0
    max_tokens: int = 32768


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


@dataclass(frozen=True)
class ScoringConfig:
    regression_threshold: float = 0.8
    regression_weight: float = 0.5
    proposer_target_accuracy: float = 0.5
    min_parent_solved_tasks: int = 3


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
class ProposerConfig:
    strategies: Dict[str, float] = field(default_factory=lambda: {
        "repo_chain": 0.30,
        "lm_modify": 0.20,
        "lm_rewrite": 0.15,
        "procedural": 0.15,
        "combine": 0.10,
        "pr_mirror": 0.05,
        "pr_replay": 0.05,
    })
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
    if key == "proposer" and "strategies" in data:
        strategies = data["strategies"]
        if not isinstance(strategies, dict):
            raise ConfigError("proposer.strategies must be a mapping")
        total = sum(strategies.values())
        if abs(total - 1.0) > 0.001:
            raise ConfigError(f"proposer.strategies probabilities must sum to 1.0, got {total}")
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
    total = sum(config.proposer.strategies.values())
    if abs(total - 1.0) > 0.001:
        raise ConfigError(f"proposer.strategies must sum to 1.0, got {total}")


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
