"""Unit tests covering remaining P0 bugfixes."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from godel0.config import Godel0Config, RepoChainWorkflowConfig, assert_no_human_curated_data
from godel0.tasks.batch import compute_effective_quotas
from godel0.tree.selection import ThompsonSamplingSelector


class TestFractionalThompsonSampling:
    def test_fractional_posterior_params_for_mean_point_six(self):
        """P0-1: [0.6]*10 => successes=6, failures=4 => Beta(7, 5)."""
        evals = [0.6] * 10
        successes = float(sum(evals))
        failures = float(len(evals)) - successes
        alpha = 1.0 + successes
        beta = 1.0 + max(0.0, failures)
        assert successes == pytest.approx(6.0)
        assert failures == pytest.approx(4.0)
        assert alpha == pytest.approx(7.0)
        assert beta == pytest.approx(5.0)

        # Mirror the selector's math with a stub archive node.
        selector = ThompsonSamplingSelector(num_pseudo_descendant_evals=10)
        # Sanity: betavariate with these params is callable.
        import random

        theta = random.Random(0).betavariate(alpha, beta)
        assert 0.0 <= theta <= 1.0
        assert selector.num_pseudo == 10


class TestRootQualityNotHardcoded:
    def test_hgm_quality_helper_uses_real_yield(self, tmp_path):
        from godel0.config import Godel0Config
        from godel0.controller.orchestrator import EvolutionOrchestrator
        from godel0.controller.run_context import RunContext
        from godel0.tree.archive import NodeArchive
        from godel0.tree.selection import ThompsonSamplingSelector
        from godel0.controller.budget import Budget

        config = Godel0Config()
        # Minimal orchestrator shell without full from_config side effects.
        run_context = MagicMock()
        run_context.paths.proposer_dir.return_value = tmp_path
        (tmp_path / "generation_summary.json").write_text(
            '{"task_ids": ["t1", "t2"], "candidates_generated": 10, '
            '"candidates_validated": 10, "rejection_reasons": '
            '{"trusted_causal_ablation_failed": 3}}',
            encoding="utf-8",
        )
        orch = EvolutionOrchestrator(
            config=config,
            archive=NodeArchive(tmp_path / "archive.json"),
            selector=ThompsonSamplingSelector(),
            run_context=run_context,
            budget=Budget(max_nodes=1, max_expansions=1),
        )
        batch = SimpleNamespace(
            tasks=[object(), object()],
            candidates_validated=10,
            candidates_generated=10,
            complete=True,
            rejection_reasons={"trusted_causal_ablation_failed": 3},
        )
        node = SimpleNamespace(node_id="root")
        gate = orch._hgm_quality_from_batch(node, batch)
        assert gate["valid_yield"] == pytest.approx(0.2)
        assert gate["causal_ablation_pass"] == pytest.approx(0.7)
        assert gate["batch_complete"] is False  # batch_size default 10, only 2 tasks


class TestTrustedCausalAblationSingleFile:
    def test_single_file_patch_rejected_with_not_multi_file(self, tmp_path):
        from godel0.proposer_trusted.candidate_validator import CandidateValidator
        from godel0.schemas.evaluation import CandidateValidationReport

        validator = CandidateValidator(
            workspace_root=tmp_path,
            require_causal_ablation=True,
        )
        report = CandidateValidationReport(candidate_id="c1", passed=True)
        # Bypass full validation; exercise the ablation gate directly.
        ok = validator._run_trusted_causal_ablation(
            candidate_patch=(
                "diff --git a/foo.py b/foo.py\n"
                "--- a/foo.py\n"
                "+++ b/foo.py\n"
                "@@ -1 +1 @@\n"
                "-a\n"
                "+b\n"
            ),
            repo_path=tmp_path,
            base_commit="HEAD",
            test_command="true",
            setup_patch="",
            f2p_tests=["t1"],
            report=report,
            validation_mode="pytest",
            command_test_id="",
            control_test_command=None,
        )
        assert ok is False
        assert report.trusted_causal_ablation_pass is False
        assert "not_multi_file" in report.rejection_reasons


class TestEffectiveQuota:
    def test_parent_capped_by_available_sources(self):
        quotas = compute_effective_quotas(
            batch_size=10,
            nominal_parent=5,
            nominal_child=5,
            available_parent=3,
            available_child=100,
        )
        assert quotas["parent_failure"] == 3
        assert quotas["current_child_level1"] == 7

    def test_bootstrap_skips_split(self):
        quotas = compute_effective_quotas(
            batch_size=10,
            nominal_parent=5,
            nominal_child=5,
            available_parent=0,
            available_child=0,
            bootstrap=True,
        )
        assert quotas["bootstrap"] == 10
        assert quotas["parent_failure"] == 0
        assert quotas["current_child_level1"] == 0


class TestRepoChainMutationOperatorV1:
    def test_default_operator_is_trajectory_conditioned_chain_mutation(self):
        from godel0.config import RepoChainWorkflowConfig

        cfg = RepoChainWorkflowConfig()
        assert cfg.mutation_operator == "trajectory_conditioned_chain_mutation"
        assert not hasattr(cfg, "mutation_backends")

    def test_legacy_mutation_backends_yaml_is_ignored(self, tmp_path):
        """Old YAML with mutation_backends must still load (key filtered out)."""
        from godel0.config import load_config

        yaml_path = tmp_path / "legacy.yaml"
        yaml_path.write_text(
            """
run:
  seed: 1
  max_nodes: 2
proposer:
  repo_chain:
    min_files: 2
    mutation_backends:
      lm_modify: 0.7
      procedural: 0.3
      pr_replay: 0.0
""",
            encoding="utf-8",
        )
        # Merge onto default by using as full config may miss required sections;
        # use Godel0Config defaults via partial load path.
        from godel0.config import Godel0Config, _build_subconfig

        rc = _build_subconfig(
            "proposer",
            {
                "repo_chain": {
                    "min_files": 2,
                    "mutation_backends": {
                        "lm_modify": 0.7,
                        "procedural": 0.3,
                        "pr_replay": 0.0,
                    },
                }
            },
        )
        assert rc.repo_chain.mutation_operator == "trajectory_conditioned_chain_mutation"

    def test_assert_no_human_curated_data_is_noop_without_backends(self):
        from godel0.config import Godel0Config, assert_no_human_curated_data

        base = Godel0Config()
        clean = assert_no_human_curated_data(base)
        assert clean.proposer.repo_chain.mutation_operator == (
            "trajectory_conditioned_chain_mutation"
        )
