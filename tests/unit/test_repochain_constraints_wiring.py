"""P0-2: RepoChain config must update plan.constraints, not only blueprint."""
from __future__ import annotations

from types import SimpleNamespace

from proposer.schemas import BugConstraints, BugGenerationPlan, FailureSignature
from proposer.workflows.repo_chain.workflow import RepoChainWorkflow


def _plan() -> BugGenerationPlan:
    return BugGenerationPlan(
        plan_id="p1",
        failure_signature=FailureSignature(signature_id="s1"),
        strategy="repo_chain",
        # Defaults that used to leak into the generator (1/1/2).
        constraints=BugConstraints(),
        task_blueprint={},
    )


class TestRepoChainConstraintsWiring:
    def test_apply_constraints_updates_plan_constraints_object(self):
        wf = RepoChainWorkflow(
            min_files=2,
            max_files=6,
            min_mutation_sites=3,
            max_mutation_sites=8,
            context_file_budget=10,
        )
        plan = _plan()
        assert plan.constraints.min_modified_files == 1
        assert plan.constraints.max_modified_files == 1

        wf._apply_constraints_to_plan(plan)

        assert plan.constraints.min_modified_files == 2
        assert plan.constraints.max_modified_files == 6
        assert plan.constraints.min_mutation_sites == 3
        assert plan.constraints.max_mutation_sites == 8
        assert plan.constraints.context_file_budget == 10
        assert plan.constraints.require_generated_tests is True
        # Blueprint metadata stays in sync, but is not the algorithm source.
        assert plan.task_blueprint["constraints"]["min_modified_files"] == 2
        assert plan.task_blueprint["constraints"]["max_modified_files"] == 6

    def test_generator_readable_attrs_match_config(self):
        """Mimic RepoChainGenerator's constraint reads after stamping."""
        wf = RepoChainWorkflow(
            config=SimpleNamespace(
                min_files=2,
                max_files=6,
                min_mutation_sites=3,
                max_mutation_sites=8,
                context_file_budget=10,
                require_causal_ablation=True,
                mutation_operator="trajectory_conditioned_chain_mutation",
            )
        )
        plan = _plan()
        wf._apply_constraints_to_plan(plan)
        assert plan.operator == "trajectory_conditioned_chain_mutation"
        assert plan.strategy == "repo_chain"
        constraints = plan.constraints
        min_files = max(2, int(getattr(constraints, "min_modified_files", 2) or 2))
        max_files = max(min_files, int(getattr(constraints, "max_modified_files", 6) or 6))
        min_sites = max(
            min_files,
            int(getattr(constraints, "min_mutation_sites", min_files) or min_files),
        )
        max_sites = max(
            min_sites,
            int(getattr(constraints, "max_mutation_sites", 8) or 8),
        )
        assert (min_files, max_files, min_sites, max_sites) == (2, 6, 3, 8)
