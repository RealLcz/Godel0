# Gödel0 HGM-style Refactor Execution Plan

Derived from godel0_repochain_hgm_style_refactor_plan_cn.md. Scope: all Phase 1-9. Clean break (no old run compat).

## Current state (verified)

- RepoChain is one of 8 SWESmith strategies, dispatched by string in initial_agent/src/swesmith/engine.py:259-299 (SWESmithEngine.generate). PlanStrategy Literal in initial_agent/src/proposer/planner.py:10-19.
- No TaskProvider Protocol, no BenchmarkTaskProvider, no ProposerTaskProvider, no RepoProfile, no AnsibleProfile exist anywhere.
- Orchestrator (src/godel0/controller/orchestrator.py) directly threads repo_pool, validator, task_committer, proposer_runner, task_store into TaskBatchBuilder.build_for_node (13-param flat signature at src/godel0/tasks/batch.py:63). No task_provider.get_tasks(node, context) seam.
- Solver-side TaskInstance is already clean: CommonAgentRequest(problem_statement, git_dir, base_commit, test_description) in experiment_adapters/common_agent_adapter.py:16. No leak there.
- GODEL0_BOOTSTRAP_SOLVER_TRAJECTORY read at orchestrator.py:660, set in scripts/slurm/godel0_evolve20_repo_chain.slurm:89.
- special_detectors.py is half-connected: orchestrator.py:548 calls detect(summary, config=special_config) only. trajectories/candidates/tool_events/solver_stats never passed. solver_empty_patches and all SharedSpecialDetector alerts are dead code.
- evidence_selector.py uses first-N logs (max_solver=4, max_proposer=4). No alert-conditioned retrieval. success_contrast never populated.
- Ansible hardcoding in 4 sites: proposer/runner.py:191, proposer/planner.py:250, swesmith/repo_chain.py:813, swesmith/repo_chain.py:1144-1151 (+ template at 958-991).
- Scorer (src/godel0/controller/scorer.py): a = lam*r + (1-lam)*p; b = max(0, 1-2|p-0.5|); score = a*b. Only Joint variant exists.
- Config (src/godel0/config.py): ProposerConfig has strategies dict (sum to 1.0). Validated at config.py:191-193.

## Phase 1: Freeze HGM-compatible Outer Loop + TaskProvider abstraction

### P1.1 Create TaskProvider Protocol

New file: src/godel0/tasks/provider.py
- class TaskGenerationContext (dataclass): node, parent, level1_result, solver_trajectories (List[str]), parent_task_ids (List[str]), run_id, output_dir, model
- class TaskBatch (dataclass): batch_id, node_id, tasks (List[TaskRecord]), complete, rejected_candidates, rejection_reasons, candidates_generated, candidates_validated, validation_reports, proposer_error, engine_rejections  (mirrors current TaskBatchResult)
- class TaskProvider(Protocol): def get_tasks(self, node, context: TaskGenerationContext) -> TaskBatch: ...

### P1.2 BenchmarkTaskProvider

New file: src/godel0/tasks/benchmark_provider.py
- Wraps existing TaskStore.tasks_for_batch for replaying a fixed batch of tasks. Used only for ablation/HGM-baseline. For now a thin adapter; full SWE-bench ingest is out of scope but the seam exists.

### P1.3 ProposerTaskProvider

New file: src/godel0/tasks/proposer_provider.py
- ProposerTaskProvider(TaskProvider): holds repo_pool, validator, task_committer, proposer_runner, task_store_dir, batch_size, max_candidates, workflow_config (RepoChainWorkflowConfig). get_tasks(node, context) calls the existing TaskBatchBuilder pipeline internally but through the new (node, context) signature. This is the default provider.
- Existing TaskBatchBuilder.build_for_node stays as the internal implementation; ProposerTaskProvider adapts the 13-param call into the (node, context) call, removing orchestrator internals leakage.

### P1.4 Wire orchestrator to TaskProvider

Edit src/godel0/controller/orchestrator.py:
- Replace task_batch_builder slot with task_provider slot in __init__ (32-72) and _build_components (102-196).
- _generate_batch (655-713) becomes: build TaskGenerationContext from (child, parent, level1_result, trajectories, parent_task_ids, run_id, output_dir, model); return self.task_provider.get_tasks(child, context). Move trajectory globbing + bootstrap-env read into the context builder (to be removed in Phase 3).
- _ensure_root_bootstrap (429-500) calls task_provider.get_tasks(root, context_with_no_trajectories).
- Acceptance: BenchmarkTaskProvider and ProposerTaskProvider can both be plugged into the same orchestrator.run() with no code change to the loop.

## Phase 2: RepoChain Workflow Refactor (P0-1)

### P2.1 New directory layout under initial_agent/src/proposer/workflows/repo_chain/

Move logic out of swesmith/repo_chain.py into a proper workflow package. Target files:
- proposer/workflows/__init__.py
- proposer/workflows/repo_chain/__init__.py
- proposer/workflows/repo_chain/workflow.py  - RepoChainWorkflow class (orchestrates 8 stages)
- proposer/workflows/repo_chain/weakness_analysis.py  - Stage 1 (from existing trajectory_analyzer + failure_signature)
- proposer/workflows/repo_chain/repository_transfer.py  - Stage 2
- proposer/workflows/repo_chain/chain_discovery.py  - Stage 3
- proposer/workflows/repo_chain/contract_generation.py  - Stage 4 (extracted from repo_chain.py contract rendering)
- proposer/workflows/repo_chain/mutation_planning.py  - Stage 5 (delegates to mutation backends)
- proposer/workflows/repo_chain/causal_ablation.py  - Stage 6 (new, currently absent)
- proposer/workflows/repo_chain/prompts/  (move CHAIN_SYSTEM_PROMPT, CONTRACT_PROMPT here)

RepoChainWorkflow.generate(plan, node_code_dir, repo_spec, output_dir) keeps the same outward signature as the current RepoChainGenerator.generate so SWESmithEngine can still call it, but internally runs stages 1-8. For Phase 2 the stage split is structural; deep behavioral changes (causal ablation enforcement) land in later phases but the stubs must exist.

### P2.2 Demote LM/Procedural/PR-replay to Mutation Backends

Reorganize initial_agent/src/swesmith/:
- swesmith/mutations/__init__.py
- swesmith/mutations/procedural.py  (move procedural.py + operators/)
- swesmith/mutations/lm_modify.py
- swesmith/mutations/lm_rewrite.py
- swesmith/mutations/pr_replay.py  (merge pr_mirror + pr_replay)
- swesmith/mutations/combine.py
- swesmith/patch_utils.py (keep)
- swesmith/repo_level.py (keep, shared workspace utilities)

Keep swesmith/engine.py as a thin MutationBackend registry/dispatcher. SWESmithEngine.generate(plan, ...) now means "run the selected mutation backend", not "run a top-level strategy". RepoChainWorkflow.mutation_planning stage calls SWESmithEngine to materialize each mutation site.

### P2.3 Remove repo_chain from PlanStrategy

Edit initial_agent/src/proposer/planner.py:10-19 and schemas.py:57-66: drop "repo_chain" and "repo_agent" from PlanStrategy Literal. Remaining Literal = mutation backends only: lm_modify, lm_rewrite, procedural, combine, pr_mirror, pr_replay. Update SWESmithEngine.list_strategies (engine.py:304-314) and generate dispatch (259-299) accordingly. Update _choose_strategy (planner.py:147-186) to return a mutation backend name, defaulting to lm_modify, weighted by proposer.repo_chain.mutation_backends config.

### P2.4 Config schema change

Edit src/godel0/config.py ProposerConfig (93-108): replace strategies dict with:
- initial_workflow: str = "repo_chain"
- repo_chain: RepoChainWorkflowConfig (sub-config with min_files, max_files, min_mutation_sites, max_mutation_sites, context_file_budget, require_generated_contracts, require_causal_ablation, mutation_backends: Dict[str,float])
Update _SUBCONFIG_KEYS, _build_subconfig (drop the sum-to-1.0 strategies check at 166-172), _validate (191-193). Update configs/*.yaml to new schema.

## Phase 3: Root Bootstrap (P0-3)

### P3.1 Bootstrap capability prior

New file: initial_agent/src/proposer/workflows/repo_chain/bootstrap.py
- BOOTSTRAP_CAPABILITY_PRIOR: list of capability categories (cross_file_localization, multi_module_state_propagation, configuration_precedence, error_handling, compatibility_preservation, api_contract_reasoning, multi_step_repository_reasoning) with default FailureSignature seeds.
- RepoChainWorkflow.bootstrap(repo_spec, output_dir) -> List[CandidateArtifact]: runs stages 2-8 against the bootstrap prior instead of solver trajectories. Produces T_0 with no trajectory conditioning.

### P3.2 Remove GODEL0_BOOTSTRAP_SOLVER_TRAJECTORY

Edit src/godel0/controller/orchestrator.py _generate_batch (660-662): delete the env-var read. When parent is None (root), build a TaskGenerationContext with solver_trajectories=[] and a bootstrap=True flag. ProposerTaskProvider detects bootstrap and calls RepoChainWorkflow.bootstrap instead of .generate. Edit scripts/slurm/godel0_evolve20_repo_chain.slurm:89: remove the export line.

## Phase 4: HGM-style Special Failure System (P0-4)

### P4.1 Expand SolverSpecialDetector

Edit src/godel0/evolution/special_detectors.py SolverSpecialDetector.detect (12-73): add alerts solver_test_only_patch, solver_stochasticity, solver_context_overflow, solver_timeout, solver_repeated_tool_loop, solver_localization_collapse. Each needs a real predicate over trajectories + solver_stats. Add solver_stats keys: empty_patch_count, test_only_patch_count, evaluated_count, timeout_count, context_overflow_count, stochastic_task_count, repeated_tool_loop_count, localization_collapse_count.

### P4.2 Expand ProposerSpecialDetector

Edit ProposerSpecialDetector.detect (76-164): add alerts contract_generation_failure, clean_contract_failure, mutation_materialization_failure, no_f2p_dominant, no_p2p, causal_ablation_failure, duplicate_collapse, context_overflow, repo_subsystem_collapse, statement_leakage. Driven by extended proposer_stats keys: contract_failure_count, clean_contract_failure_count, no_f2p_count, no_p2p_count, causal_ablation_failure_count, duplicate_count, statement_leakage_count. causal_ablation_failure is the most important: fires when repair-one-file restores all contracts.

### P4.3 Wire detectors to real data

Edit src/godel0/controller/orchestrator.py _prepare_diagnosis (502-582):
- Assemble solver_stats dict from level2 outcomes + trajectory analysis (empty_patch_count, test_only_patch_count, timeout_count, context_overflow_count, stochastic_task_count, repeated_tool_loop_count, localization_collapse_count).
- Assemble proposer_stats extended dict from generation_summary.json (contract_failure_count, clean_contract_failure_count, no_f2p_count, no_p2p_count, causal_ablation_failure_count, duplicate_count, statement_leakage_count).
- Assemble tool_events list from trajectory tool-call logs.
- Call self.special_detector.detect(summary, trajectories=trajectories, candidates=candidate_reports, tool_events=tool_events, solver_stats=solver_stats, proposer_stats=proposer_stats, config=special_config). Update CompositeSpecialDetector.detect signature to accept proposer_stats.

### P4.4 Extend generation_summary.json

Edit src/godel0/tasks/batch.py TaskBatchResult + the summary written at orchestrator.py:697-712: add the new proposer_stats counters. Edit proposer side (initial_agent/src/proposer/runner.py + proposer_main.py) to emit them.

## Phase 5: Evidence System (P0-5)

### P5.1 Alert-conditioned evidence retrieval

Rewrite src/godel0/evolution/evidence_selector.py CycleEvidenceSelector.select:
- Input: summary, alerts, artifacts (now includes solver_trajectories, proposer_candidates, tool_events, chain_plans, ablation_results, success_contrast).
- Pick primary alert (highest priority/severity). Branch on alert_type:
  - no_f2p_dominant: 2 no_f2p candidates + 1 successful F2P candidate + 1 chain plan.
  - solver_empty_patch: 3 empty-patch trajectories + 1 successful patch trajectory + task quality summary.
  - causal_ablation_failure: failed chain plan + mutation sites + repair-one-file ablation results + 1 successful chain-level task as contrast.
  - default: balanced sample.
- Always include 1 success contrast when available. Bounded by max_total_evidence_chars.

### P5.2 Populate success_contrast and chain_plans artifacts

Edit orchestrator.py _prepare_diagnosis (514-517): populate artifacts["success_contrast"], artifacts["chain_plans"], artifacts["ablation_results"], artifacts["tool_events"] from the proposer generation_summary and solver trajectories.

## Phase 6: Task Source Quota (P0-6)

### P6.1 Config

Edit src/godel0/config.py TaskConfig (41-47): add sources sub-config with parent_failure.quota and current_child_level1.quota (both default 5). batch_size default 10.

### P6.2 Quota enforcement in ProposerTaskProvider/TaskBatchBuilder

Edit src/godel0/tasks/proposer_provider.py + batch.py build_for_node: split requested batch_size into parent_failure quota and current_child_level1 quota. Allow dynamic fallback (3+7, 0+10) when one side insufficient. Tag each TaskRecord with source_node, source_trajectory, source_type (extend TaskRecord schema in src/godel0/schemas/task.py:13-48 with these fields).

## Phase 7: RepoProfile (P0-7)

### P7.1 RepoProfile base + AnsibleProfile

New files under initial_agent/src/proposer/repo_profiles/:
- base.py: class RepoProfile with source_roots(), test_roots(), contract_renderer(), public_entrypoints(), environment(), test_command(), contract_scenario(), contract_test_style().
- registry.py: RepoProfileRegistry.get(repo_id) -> RepoProfile.
- ansible.py: AnsibleProfile(RepoProfile) - source_roots=["lib","test/lib"], contract_renderer="ansible_playbook_cli", test_command="ansible-test", public_entrypoints from bin/ansible-playbook.

### P7.2 Remove Ansible hardcoding

Edit the 4 sites:
- initial_agent/src/proposer/runner.py:191 - replace if "ansible" in spec.repo_id with RepoProfileRegistry.get(spec.repo_id).source_roots().
- initial_agent/src/proposer/planner.py:250 - replace if target.repo_id=="ansible" with profile = RepoProfileRegistry.get(target.repo_id); blueprint.update(profile.contract_blueprint()).
- initial_agent/src/swesmith/repo_chain.py:813 - replace blueprint marker check with profile.contract_renderer() check.
- initial_agent/src/swesmith/repo_chain.py:1144-1151 - replace hardcoded ansible.* + lib/ with profile.module_path(root, module) and profile.source_roots().
- initial_agent/src/swesmith/repo_chain.py:958-991 - render test template from profile.test_template().

## Phase 8: Scoring Ablation

### P8.1 Two scoring modes

Edit src/godel0/config.py ScoringConfig (50-55): add mode: str = "joint" (values: "hgm", "joint"). Edit src/godel0/controller/scorer.py compute_scores: if mode=="hgm": solver_score = a; proposer_score = b; node_score = a; b acts only as eligibility gate (batch_complete and valid_yield>=threshold and causal_ablation_pass>=threshold and difficulty_score>=threshold). If mode=="joint": current a*b formula.

### P8.2 Archive eligibility gate

Edit src/godel0/tree/archive.py NodeArchive.eligible_parents: in hgm mode, require proposer quality gate (valid_yield, causal_ablation_pass, difficulty) before a node is eligible. Gate thresholds from config.

## Phase 9: Apptainer / HPC Execution

### P9.1 Unify execution backend

Edit src/godel0/execution/apptainer.py + subprocess_runner.py: ensure Solver, Proposer, RepoChain, Trusted Validation, Self-Improvement all route through the same ExecutionBackend interface. Proposer subprocess (NodeProposerRunner) gains an apptainer variant. Keep single-machine stability first; no large-scale concurrency.

## Test and validation plan

- After each phase, run: python -m pytest tests/ -x
- After Phase 1+2: smoke run with configs/local_smoke.yaml using ProposerTaskProvider + RepoChainWorkflow.
- After Phase 3: root bootstrap smoke without GODEL0_BOOTSTRAP_SOLVER_TRAJECTORY.
- After Phase 4+5: unit tests for each new special detector and evidence branch.
- After Phase 7: verify AnsibleProfile produces identical contracts to pre-refactor on one sample.
- After Phase 8: run one epoch in both hgm and joint modes, compare scores.

## Files most affected

- src/godel0/controller/orchestrator.py (Phases 1, 3, 4, 5)
- src/godel0/tasks/ (new provider.py, benchmark_provider.py, proposer_provider.py; edit batch.py, node_proposer.py)
- src/godel0/config.py (Phases 2, 6, 8)
- src/godel0/evolution/special_detectors.py, evidence_selector.py (Phases 4, 5)
- src/godel0/controller/scorer.py, tree/archive.py (Phase 8)
- initial_agent/src/proposer/planner.py, schemas.py, runner.py (Phases 2, 7)
- initial_agent/src/proposer/workflows/repo_chain/ (new, Phase 2, 3)
- initial_agent/src/proposer/repo_profiles/ (new, Phase 7)
- initial_agent/src/swesmith/ (reorg to mutations/, Phase 2)
- initial_agent/src/swesmith/repo_chain.py -> split into workflows/repo_chain/ (Phases 2, 3, 7)
- configs/*.yaml (Phase 2 schema change)
- scripts/slurm/godel0_evolve20_repo_chain.slurm (Phase 3)
