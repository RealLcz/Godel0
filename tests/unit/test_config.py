"""Unit tests for the config system."""

from __future__ import annotations

import pytest
from pathlib import Path

from godel0.config import load_config, Godel0Config, RunConfig
from godel0.errors import ConfigError


class TestConfig:
    def test_load_default_config(self):
        """Load the default config file."""
        config = load_config("configs/default.yaml")
        assert config.run.seed == 42
        assert config.run.max_nodes == 200
        assert config.scoring.regression_weight == 0.5
        assert config.scoring.proposer_target_accuracy == 0.5
        assert config.tasks.batch_size == 10

    def test_load_smoke_config(self):
        """Load the local smoke config."""
        config = load_config("configs/local_smoke.yaml")
        assert config.run.max_nodes == 3
        assert config.tasks.batch_size == 2
        assert config.evaluation.max_workers == 1

    def test_overrides(self):
        """Test CLI overrides."""
        config = load_config("configs/default.yaml", overrides={"run.max_nodes": 5})
        assert config.run.max_nodes == 5

    def test_invalid_batch_size(self):
        """batch_size <= 0 should raise."""
        with pytest.raises(ConfigError):
            load_config("configs/default.yaml", overrides={"tasks.batch_size": 0})

    def test_invalid_regression_threshold(self):
        """regression_threshold out of [0,1] should raise."""
        with pytest.raises(ConfigError):
            load_config("configs/default.yaml", overrides={"scoring.regression_threshold": 1.5})

    def test_invalid_target_accuracy(self):
        """proposer_target_accuracy must be 0.5."""
        with pytest.raises(ConfigError):
            load_config("configs/default.yaml", overrides={"scoring.proposer_target_accuracy": 0.6})

    def test_proposer_strategies_sum(self):
        """Mutation backend probabilities must sum to 1.0."""
        config = load_config("configs/default.yaml")
        total = sum(config.proposer.repo_chain.mutation_backends.values())
        assert abs(total - 1.0) < 0.001

    def test_config_to_dict_roundtrip(self):
        """Config should survive dict conversion."""
        from godel0.config import config_to_dict
        config = load_config("configs/default.yaml")
        d = config_to_dict(config)
        assert "run" in d
        assert "models" in d
        assert d["run"]["seed"] == 42
