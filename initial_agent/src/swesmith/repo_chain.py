from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Sequence

from .candidate import CandidateArtifact
from .engine import BugGenerationPlan, RepoSpec
from .repo_level import (
    RepositoryWorkspace,
    RepositoryWorkspaceError,
    apply_repository_patch,
    declared_target_files,
    declared_target_symbols,
    is_safe_repo_path,
    is_test_path,
    repository_diff,
    repository_path,
    split_patch_by_file,
    validate_repository_patch,
)


CHAIN_SYSTEM_PROMPT = """\
You design difficult but repairable repository-level software tasks. Identify
one behavioral invariant that crosses several related production modules. The
task must probe repository reasoning, not syntax repair or test memorization.
All proposed mutation sites must be manifestations of the same invariant.
"""


CONTRACT_PROMPT = """\
# Plan one cross-file regression and its hidden contracts

Capability gap inferred from prior solver behavior:
{trajectory_evidence}

Blueprint:
```json
{blueprint}
```

Allowed production paths (this is an exhaustive whitelist):
```json
{allowed_production_paths}
```

Allowed AST mutation symbols by production path (also exhaustive):
```json
{allowed_symbols}
```

The following {context_count} related repository files are the complete code
context for this planning call:
{source_bundle}

Return one JSON object and no prose:
{{
  "chain_plan": {{
    "root_invariant": "one user-visible semantic invariant",
    "entrypoint": "module.symbol",
    "endpoint": "module.symbol or observable behavior",
    "capability_gap": "reasoning skill exercised",
    "context_files": ["paths actually used"],
    "mutation_sites": [
      {{
        "file": "production path",
        "symbol": "symbol",
        "role": "producer|carrier|identity|consumer|error-boundary",
        "change": "specific behavior to corrupt"
      }}
    ],
    "contract_tests": ["layer contract", "end-to-end contract"],
    "rationale": "why all sites belong to one chain"
  }},
  "tests": [
    {{"path": "test path", "content": "complete test file content"}}
  ],
  "contract_cases": [
    {{
      "name": "case id",
      "playbook": "complete playbook YAML",
      "files": {{"relative included file": "complete YAML"}},
      "expected_output": ["observable substring"],
      "expected_counts": {{"substring that must occur exactly N times": 1}},
      "compatibility_control": false
    }}
  ]
}}

Requirements:
- Plan {min_sites} to {max_sites} mutation sites in {min_files} to {max_files}
  production files selected from the exhaustive whitelist. A path not present
  in that list makes the response invalid.
- Every mutation-site `symbol` must be copied exactly from the AST symbol
  whitelist for its file. Fields, conceptual helpers, and invented methods are
  invalid mutation symbols.
- Transfer only the solver's abstract capability gap from trajectory evidence.
  Do not reuse its repository subsystem, bug story, symbols, or domain nouns.
- Obey every `forbidden_copy` value in the blueprint. Do not mention or test
  the forbidden scenario even if it appears repeatedly in the trajectory.
- Generate focused tests before any mutation. They must pass on the current
  repository and assert public behavior, not source text or implementation
  line numbers.
- Include at least one test crossing three production modules or layers.
- Include layer contracts so repairing only one mutated file remains incomplete.
- Generate at least two pytest test functions, including one nearby compatibility
  control expected to remain passing after the mutation.
- Do not use mocks that bypass the production chain.
- Return one newly named Python test file under `test/units/`; its path must end
  in `.py` and must not replace an existing repository file. Include every
  imported name and do not rely on fixtures from another test class.
- If the blueprint specifies `contract_test_renderer`, populate
  `contract_cases` with at least one target case and one compatibility control.
  The renderer owns pytest/subprocess code, so set the test content to a short
  placeholder instead of inventing a command harness.
- `files` may be empty for a self-contained playbook. For de-duplication or
  exactly-once behavior, populate `expected_counts` so the renderer checks the
  exact number of observable occurrences.
- Every `expected_counts` key is passed to Python `output.count` as a literal
  substring. Never use regular expressions, glob syntax, `.*`, anchors, or
  escaped regex metacharacters in those keys.
- Do not expose the planned mutations in test names or comments.
"""


MUTATION_PROMPT = """\
# Materialize one planned repository-level regression

Chain plan:
```json
{chain_plan}
```

Hidden contract tests, which currently pass:
```json
{tests}
```

Relevant production context:
{source_bundle}

Return one JSON object and no prose:
{{
  "edits": [
    {{
      "file": "production path from the chain plan",
      "symbol": "exact planned symbol",
      "before": "exact contiguous source text copied from the symbol context",
      "after": "replacement source text",
      "intent": "how this corrupts the root invariant"
    }}
  ]
}}

Requirements:
- Modify {min_files} to {max_files} production files and at most {max_lines}
  added/deleted lines.
- Implement {min_sites} to {max_sites} planned manifestations of the same root
  invariant. Preserve signatures and syntax.
- Return exactly one edit for every planned mutation-site file/symbol pair and
  never repeat a file/symbol pair. The edit count must equal the number of
  mutation sites in the chain plan.
- The complete hidden contract suite must fail after the patch.
- Repairing only one touched file must not make the complete suite pass.
- Do not edit tests, generated files, dependency metadata, or documentation.
- Do not add comments revealing the regression and do not hard-code test data.
- Keep `after` concise production code. Do not include analysis, alternatives,
  walkthroughs, or comments explaining the mutation inside source strings.
- Every `before` must differ from `after` and occur exactly once inside the
  declared symbol. Copy it byte-for-byte from the supplied exact source.
- Do not return a unified diff or line numbers. Git diff generation is owned by
  the trusted materializer.
- Keep each replacement syntactically complete, including indentation.
"""


TASK_SCHEMA_VERSION = "godel0.swebench_like.v1"


class RepoChainGenerator:
    """Generate a trajectory-conditioned cross-file bug with hidden contracts."""

    def __init__(self, agent_adapter: Any = None) -> None:
        self.agent_adapter = agent_adapter
        self.last_rejection = ""
        self.last_rejection_stage = ""
        self._current_repo_id = ""

    def _stage_for_rejection(self, detail: str) -> str:
        try:
            from godel0.schemas.repo_chain_stats import stage_for_engine_rejection

            return stage_for_engine_rejection(detail)
        except Exception:
            text = str(detail or "").lower()
            if "unmodified repository" in text or "clean_contract" in text:
                return "clean_contract_failure"
            if "invalid_chain_plan" in text or "contract_not_restored" in text:
                return "contract_generation_failure"
            return "mutation_failure"

    def _reject(self, stage: str, detail: str) -> None:
        """P1-3: emit structured stage + detail for orchestrator stats."""
        self.last_rejection_stage = str(stage or "")
        self.last_rejection = str(detail or "")

    def _reject_detail(self, detail: str) -> None:
        """Classify and emit a rejection from a detail string."""
        self._reject(self._stage_for_rejection(detail), detail)

    def generate(
        self,
        plan: BugGenerationPlan,
        node_code_dir: str,
        repo_spec: RepoSpec,
        output_dir: str,
    ) -> List[CandidateArtifact]:
        self.last_rejection = ""
        self.last_rejection_stage = ""
        self._current_repo_id = getattr(repo_spec, "repo_id", "") or getattr(plan, "target_repo_id", "")
        if self.agent_adapter is None:
            self._reject_detail("missing_agent_adapter")
            return []
        source_repo = repository_path(repo_spec, node_code_dir)
        if not source_repo or not os.path.isdir(source_repo):
            self._reject_detail("missing_source_repository")
            return []

        constraints = plan.constraints
        min_files = max(2, int(getattr(constraints, "min_modified_files", 2) or 2))
        max_files = max(min_files, int(getattr(constraints, "max_modified_files", 6) or 6))
        min_sites = max(min_files, int(getattr(constraints, "min_mutation_sites", min_files) or min_files))
        max_sites = max(min_sites, int(getattr(constraints, "max_mutation_sites", 8) or 8))
        context_budget = max(
            max_files,
            int(getattr(constraints, "context_file_budget", 10) or 10),
        )
        base_commit = str(
            getattr(plan, "target_base_commit", "")
            or getattr(repo_spec, "base_commit", "")
            or "HEAD"
        )
        context_files = self._related_files(
            Path(source_repo),
            declared_target_files(plan),
            context_budget,
        )
        if len(context_files) < min_files:
            self._reject_detail("insufficient_context_files")
            return []
        source_bundle = self._source_bundle(Path(source_repo), context_files)
        trajectory_evidence = self._trajectory_evidence(plan)
        blueprint = dict(getattr(plan, "task_blueprint", None) or {})

        output_root = Path(output_dir or source_repo)
        output_root.mkdir(parents=True, exist_ok=True)
        try:
            with RepositoryWorkspace(
                source_repo,
                base_commit=base_commit,
                parent_dir=str(output_root),
                prefix=f"repo_chain_{plan.plan_id}_",
            ) as workspace:
                contract_prompt = CONTRACT_PROMPT.format(
                        trajectory_evidence=trajectory_evidence,
                        blueprint=json.dumps(blueprint, indent=2, ensure_ascii=False),
                        allowed_production_paths=json.dumps(
                            [path for path in context_files if not is_test_path(path)],
                            indent=2,
                        ),
                        allowed_symbols=json.dumps(
                            self._symbol_catalog(Path(source_repo), context_files),
                            indent=2,
                        ),
                        context_count=len(context_files),
                        source_bundle=source_bundle,
                        min_sites=min_sites,
                        max_sites=max_sites,
                        min_files=min_files,
                        max_files=max_files,
                    )
                contract_payload: Dict[str, Any] = {}
                contract_error = ""
                tests: List[Dict[str, Any]] = []
                contract_patch = ""
                test_files: List[str] = []
                test_command = ""
                clean_result: subprocess.CompletedProcess[str] | None = None
                for attempt in range(3):
                    retry_prompt = contract_prompt
                    if contract_error:
                        retry_prompt += (
                            "\n\n# Previous response rejected\n"
                            f"Reason: {contract_error}\n"
                            "Regenerate the complete JSON from scratch. Fix the stated "
                            "problem while preserving the abstract capability target."
                        )
                    contract_response = self._call_contract_agent(
                        plan=plan,
                        workspace=workspace,
                        prompt=retry_prompt,
                    )
                    response_text = (
                        json.dumps(contract_response, indent=2, ensure_ascii=False)
                        if isinstance(contract_response, dict)
                        else str(contract_response or "")
                    )
                    contract_payload = self._parse_contract_payload(contract_response)
                    stored_contract = json.dumps(
                        contract_payload
                        if contract_payload
                        else {"invalid_response": response_text},
                        indent=2,
                        ensure_ascii=False,
                    )
                    (output_root / f"contract_response_attempt_{attempt + 1}.json").write_text(
                        stored_contract, encoding="utf-8"
                    )
                    (output_root / "contract_response.json").write_text(
                        stored_contract, encoding="utf-8"
                    )
                    contract_error = self._chain_plan_rejection(
                        contract_payload,
                        context_files=context_files,
                        min_files=min_files,
                        max_files=max_files,
                        min_sites=min_sites,
                        max_sites=max_sites,
                        forbidden_terms=list(blueprint.get("forbidden_terms") or []),
                    )
                    if not contract_error:
                        contract_error = self._mutation_symbols_rejection(
                            Path(workspace), contract_payload["chain_plan"]
                        )
                    if contract_error:
                        (output_root / f"contract_rejection_attempt_{attempt + 1}.txt").write_text(
                            contract_error + "\n", encoding="utf-8"
                        )
                        continue
                    tests = self._materialize_contract_tests(plan, contract_payload)
                    if not tests:
                        contract_error = (
                            "contract_cases must contain at least one target case and "
                            "one compatibility_control case with complete observable "
                            "expectations"
                        )
                        (output_root / f"contract_rejection_attempt_{attempt + 1}.txt").write_text(
                            contract_error + "\n", encoding="utf-8"
                        )
                        continue
                    if not self._write_tests(Path(workspace), tests):
                        contract_error = (
                            "generated tests must use one new .py path under test/units"
                        )
                        (output_root / f"contract_rejection_attempt_{attempt + 1}.txt").write_text(
                            contract_error + "\n", encoding="utf-8"
                        )
                        self._discard_tests(Path(workspace), tests)
                        continue
                    contract_patch = self._test_patch(repository_diff(workspace, "HEAD"))
                    if not contract_patch or not self._contract_quality_valid(contract_patch):
                        contract_error = (
                            "generated test patch lacks behavioral assertions or uses "
                            "forbidden source inspection"
                        )
                        (output_root / f"contract_rejection_attempt_{attempt + 1}.txt").write_text(
                            contract_error + "\n", encoding="utf-8"
                        )
                        self._discard_tests(Path(workspace), tests)
                        continue
                    test_files = self._changed_test_files(contract_patch)
                    test_command = self._contract_test_command(plan, repo_spec, test_files)
                    clean_result = self._run_command(
                        workspace, test_command, constraints.generation_timeout_sec
                    )
                    if clean_result.returncode == 0:
                        break
                    self._save_diagnostic(
                        output_root,
                        f"clean_contract_failure_attempt_{attempt + 1}.txt",
                        clean_result,
                    )
                    failure_tail = (clean_result.stdout + "\n" + clean_result.stderr)[-5000:]
                    contract_error = (
                        "generated tests failed on the unmodified repository; correct "
                        f"their API assumptions and collection errors:\n{failure_tail}"
                    )
                    self._discard_tests(Path(workspace), tests)
                if contract_error:
                    # P1-3: emit stage at source (clean vs contract generation).
                    if "unmodified repository" in contract_error.lower():
                        self._reject(
                            "clean_contract_failure",
                            f"clean_contract:{contract_error}",
                        )
                    else:
                        self._reject(
                            "contract_generation_failure",
                            f"invalid_chain_plan:{contract_error}",
                        )
                    return []
                assert clean_result is not None
                contract_contents = {
                    path: (Path(workspace) / path).read_text(
                        encoding="utf-8", errors="replace"
                    )
                    for path in test_files
                }
                self._remove_runtime_artifacts(Path(workspace))
                mutation_source_bundle = self._mutation_source_bundle(
                    Path(workspace), contract_payload["chain_plan"]
                )
                if not mutation_source_bundle:
                    self._reject_detail("mutation_symbols_not_found")
                    return []

                mutation_prompt = MUTATION_PROMPT.format(
                        chain_plan=json.dumps(contract_payload["chain_plan"], indent=2, ensure_ascii=False),
                        tests=json.dumps(tests, indent=2, ensure_ascii=False),
                        source_bundle=mutation_source_bundle,
                        min_files=min_files,
                        max_files=max_files,
                        max_lines=int(getattr(constraints, "max_modified_lines", 160) or 160),
                        min_sites=min_sites,
                        max_sites=max_sites,
                    )
                response_patch = ""
                accepted_edits: List[Dict[str, str]] = []
                mutation_error = ""
                for attempt in range(3):
                    retry_prompt = mutation_prompt
                    if mutation_error:
                        retry_prompt += (
                            "\n\n# Previous mutation rejected\n"
                            f"Reason: {mutation_error}\n"
                            "Regenerate the complete JSON edits object from scratch. "
                            "Copy each before value exactly from its declared symbol."
                        )
                    mutation_response = self._call_mutation_agent(
                        plan=plan,
                        workspace=workspace,
                        prompt=retry_prompt,
                    )
                    mutation_payload = self._parse_json_object(mutation_response)
                    response_text = json.dumps(
                        mutation_payload if mutation_payload else {"invalid_response": str(mutation_response or "")},
                        indent=2,
                        ensure_ascii=False,
                    )
                    edits_path = output_root / f"mutation_edits_attempt_{attempt + 1}.json"
                    edits_path.write_text(response_text, encoding="utf-8")
                    (output_root / "mutation_edits.json").write_text(
                        response_text, encoding="utf-8"
                    )
                    direct_patch = self._source_patch(repository_diff(workspace, "HEAD"))
                    if direct_patch:
                        # Compatibility for tool-using adapters that already edited the
                        # isolated workspace. Chat adapters must use structured edits.
                        response_patch = direct_patch
                        accepted_edits = []
                        mutation_error = ""
                    else:
                        accepted_edits, mutation_error = self._materialize_symbol_edits(
                            Path(workspace),
                            mutation_payload,
                            contract_payload["chain_plan"],
                            min_files=min_files,
                            max_files=max_files,
                            min_sites=min_sites,
                            max_sites=max_sites,
                        )
                        if mutation_error:
                            (output_root / f"mutation_rejection_attempt_{attempt + 1}.txt").write_text(
                                mutation_error + "\n", encoding="utf-8"
                            )
                            continue
                        response_patch = self._source_patch(repository_diff(workspace, "HEAD"))
                    if not response_patch:
                        mutation_error = "structured edits produced no production diff"
                        (output_root / f"mutation_rejection_attempt_{attempt + 1}.txt").write_text(
                            mutation_error + "\n", encoding="utf-8"
                        )
                        continue
                    (output_root / f"mutation_attempt_{attempt + 1}.diff").write_text(
                        response_patch, encoding="utf-8"
                    )
                    (output_root / "mutation.diff").write_text(
                        response_patch, encoding="utf-8"
                    )
                    break
                if mutation_error:
                    self._reject_detail(f"mutation_patch_apply_failed:{mutation_error}")
                    return []
                self._remove_runtime_artifacts(Path(workspace))
                full_patch = repository_diff(workspace, "HEAD")
                changed_contract_contents = [
                    path
                    for path, content in contract_contents.items()
                    if not (Path(workspace) / path).is_file()
                    or (Path(workspace) / path).read_text(
                        encoding="utf-8", errors="replace"
                    )
                    != content
                ]
                observed_test_files = self._changed_test_files(
                    self._test_patch(full_patch)
                )
                if changed_contract_contents or set(observed_test_files) != set(test_files):
                    self._reject_detail(
                        "mutation_modified_contract_tests:"
                        f"changed={changed_contract_contents}:"
                        f"observed={observed_test_files}:expected={test_files}"
                    )
                    return []
                bug_patch = self._source_patch(full_patch)
                summary = validate_repository_patch(
                    bug_patch,
                    constraints,
                    require_multiple_files=True,
                )
                if not summary.valid:
                    self._reject_detail(f"invalid_bug_patch:{summary.rejection_reason}")
                    return []
                if not self._plan_matches_patch(
                    contract_payload["chain_plan"], summary.changed_files, min_sites
                ):
                    self._reject_detail("mutation_patch_does_not_match_plan")
                    return []
                bugged_result = self._run_command(
                    workspace, test_command, constraints.generation_timeout_sec
                )
                if bugged_result.returncode == 0:
                    self._reject_detail("generated_contract_did_not_fail")
                    return []
                provisional_taxonomy = self._contract_test_taxonomy(
                    contract_payload, test_files
                )
                control_ids = provisional_taxonomy["PASS_TO_PASS"]
                control_passed = any(
                    self._run_command(
                        workspace,
                        self._selected_contract_command(
                            test_command, test_files, [control_id]
                        ),
                        constraints.generation_timeout_sec,
                    ).returncode
                    == 0
                    for control_id in control_ids
                )
                if control_ids and not control_passed:
                    self._reject_detail("compatibility_control_failed_after_mutation")
                    return []
                target_command = self._selected_contract_command(
                    test_command,
                    test_files,
                    provisional_taxonomy["FAIL_TO_PASS"],
                )
                if not target_command or self._run_command(
                    workspace,
                    target_command,
                    constraints.generation_timeout_sec,
                ).returncode == 0:
                    self._reject_detail("target_contract_did_not_fail_after_mutation")
                    return []
                if not apply_repository_patch(workspace, bug_patch, reverse=True):
                    self._reject_detail("bug_patch_reverse_failed")
                    return []
                self._remove_runtime_artifacts(Path(workspace))
                restored_result = self._run_command(
                    workspace, test_command, constraints.generation_timeout_sec
                )
                if restored_result.returncode != 0:
                    self._reject("contract_generation_failure", "contract_not_restored")
                    return []
                causal = self._causal_ablation(
                    workspace,
                    bug_patch,
                    test_command,
                    constraints.generation_timeout_sec,
                )
                problem_statement = self._build_problem_statement(
                    contract_payload["chain_plan"], contract_payload
                )
                test_taxonomy = self._contract_test_taxonomy(
                    contract_payload, test_files
                )
                if not apply_repository_patch(workspace, bug_patch):
                    self._reject_detail("bug_patch_reapply_failed_for_oracle")
                    return []
                oracle_patch = self._reverse_worktree_diff(
                    workspace, summary.changed_files
                )
                if not oracle_patch or not apply_repository_patch(
                    workspace, bug_patch, reverse=True
                ):
                    self._reject_detail("oracle_patch_generation_failed")
                    return []
        except (RepositoryWorkspaceError, OSError, ValueError, subprocess.SubprocessError) as exc:
            self._reject_detail(f"generation_error:{type(exc).__name__}:{exc}")
            return []

        candidate_id = self._candidate_id(plan, bug_patch, contract_patch)
        chain_plan = dict(contract_payload["chain_plan"])
        artifact = CandidateArtifact(
            candidate_id=candidate_id,
            plan_id=plan.plan_id,
            strategy="repo_chain",
            operator="trajectory_conditioned_chain_mutation",
            target_file=summary.changed_files[0],
            target_symbol="",
            bug_patch=bug_patch,
            mutation_site={
                "chain_plan": chain_plan,
                "changed_files": summary.changed_files,
            },
            seed=getattr(plan, "seed", 0),
            before_snippet="",
            after_snippet="",
            generation_metadata={
                "agent": type(self.agent_adapter).__name__,
                "modified_lines": summary.modified_lines,
                "task_blueprint": blueprint,
                "source_trajectory_ids": list(getattr(plan, "source_trajectory_ids", []) or []),
                # P0-5: promote blueprint provenance onto candidate metadata so
                # TaskBatchBuilder does not invent source_node_id at commit.
                **{
                    key: value
                    for key, value in {
                        "source_type": str(blueprint.get("source_type") or ""),
                        "source_node_id": str(blueprint.get("source_node_id") or ""),
                        "source_task_id": str(blueprint.get("source_task_id") or ""),
                        "source_trajectory_id": str(
                            blueprint.get("source_trajectory_id") or ""
                        ),
                        "source_failure_stage": str(
                            blueprint.get("source_failure_stage")
                            or blueprint.get("failure_stage")
                            or ""
                        ),
                    }.items()
                    if value
                },
                "context_files": context_files,
                "chain_plan": chain_plan,
                "structured_symbol_edits": accepted_edits,
                "generated_test_patch": contract_patch,
                "generated_test_files": test_files,
                "generated_test_command": test_command,
                "contract_test": test_files[0] if test_files else "",
                "causal_ablation": causal,
                "semantic_coupling": {
                    # Causal ablation is a quality diagnostic, not part of the
                    # trusted-valid definition. The controller-side validator
                    # remains authoritative for F2P, reverse, relevance,
                    # duplicate, and safety checks from the specification.
                    "valid": bool(causal.get("repair_only_one_file_all_fail")),
                    "tier": (
                        "generated_cross_layer_contract"
                        if causal.get("repair_only_one_file_all_fail")
                        else "generated_contract_with_partial_coupling"
                    ),
                    "root_invariant": chain_plan.get("root_invariant", ""),
                    "entrypoint": chain_plan.get("entrypoint", ""),
                    "endpoint": chain_plan.get("endpoint", ""),
                    "required_components": len(summary.changed_files),
                    "independently_active_file_count": causal.get(
                        "independently_active_file_count", 0
                    ),
                },
                "clean_contract_returncode": clean_result.returncode,
                "bugged_contract_returncode": bugged_result.returncode,
                "restored_contract_returncode": restored_result.returncode,
                "problem_statement": problem_statement,
                "oracle_patch": oracle_patch,
                "fail_to_pass": test_taxonomy["FAIL_TO_PASS"],
                "pass_to_pass": test_taxonomy["PASS_TO_PASS"],
            },
            modified_files=summary.changed_files,
            modified_entities=[
                str(site.get("symbol") or "")
                for site in chain_plan.get("mutation_sites") or []
                if site.get("symbol")
            ],
        )
        candidate_dir = output_root / candidate_id
        artifact.save(str(candidate_dir))
        (candidate_dir / "contract.patch").write_text(contract_patch, encoding="utf-8")
        (candidate_dir / "oracle.patch").write_text(oracle_patch, encoding="utf-8")
        (candidate_dir / "problem_statement.md").write_text(
            problem_statement.rstrip() + "\n", encoding="utf-8"
        )
        task = {
            "schema_version": TASK_SCHEMA_VERSION,
            "instance_id": candidate_id,
            "repo": str(getattr(repo_spec, "repo_id", "") or ""),
            "base_commit": base_commit,
            "problem_statement": problem_statement,
            "setup_patch": bug_patch,
            "patch": oracle_patch,
            "test_patch": contract_patch,
            "FAIL_TO_PASS": test_taxonomy["FAIL_TO_PASS"],
            "PASS_TO_PASS": test_taxonomy["PASS_TO_PASS"],
        }
        (candidate_dir / "task.json").write_text(
            json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (candidate_dir / "chain_plan.json").write_text(
            json.dumps(chain_plan, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (candidate_dir / "contract_test_output.txt").write_text(
            bugged_result.stdout + "\n" + bugged_result.stderr,
            encoding="utf-8",
        )
        return [artifact]

    def _call_contract_agent(self, *, plan: BugGenerationPlan, workspace: str, prompt: str) -> Any:
        method = getattr(self.agent_adapter, "generate_repo_chain_contract", None)
        if callable(method):
            return method(workspace, prompt, plan)
        blueprint = dict(getattr(plan, "task_blueprint", None) or {})
        max_tokens = 4096 if blueprint.get("contract_test_renderer") else 8192
        return self._chat(
            CHAIN_SYSTEM_PROMPT,
            prompt,
            max_tokens=max_tokens,
            model=str(getattr(plan, "model", "") or ""),
        )

    def _call_mutation_agent(self, *, plan: BugGenerationPlan, workspace: str, prompt: str) -> Any:
        method = getattr(self.agent_adapter, "generate_repo_chain_bug", None)
        if callable(method):
            result = method(workspace, prompt, plan)
            return result
        return self._chat(
            CHAIN_SYSTEM_PROMPT,
            prompt,
            max_tokens=4096,
            model=str(getattr(plan, "model", "") or ""),
        )

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        model: str = "",
    ) -> str:
        chat = getattr(self.agent_adapter, "chat", None)
        if not callable(chat):
            return ""
        resolved = (
            str(model or "").strip()
            or str(getattr(self.agent_adapter, "default_model", "") or "").strip()
        )
        # P1-1: prefer explicit model=; fall back for older fake adapters.
        try:
            if resolved:
                return str(
                    chat(
                        system_prompt,
                        user_prompt,
                        temperature=0,
                        max_tokens=max_tokens,
                        model=resolved,
                    )
                    or ""
                )
            return str(
                chat(
                    system_prompt,
                    user_prompt,
                    temperature=0,
                    max_tokens=max_tokens,
                )
                or ""
            )
        except TypeError:
            try:
                return str(chat(system_prompt, user_prompt, temperature=0) or "")
            except TypeError:
                return str(chat(system_prompt, user_prompt) or "")

    def _parse_contract_payload(self, response: Any) -> Dict[str, Any]:
        return self._parse_json_object(response)

    def _parse_json_object(self, response: Any) -> Dict[str, Any]:
        if isinstance(response, dict):
            return response
        text = str(response or "").strip()
        if not text:
            return {}
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _valid_chain_plan(
        self,
        payload: Dict[str, Any],
        *,
        context_files: Sequence[str],
        min_files: int,
        max_files: int,
        min_sites: int,
        max_sites: int,
        forbidden_terms: Sequence[str] = (),
    ) -> bool:
        return not self._chain_plan_rejection(
            payload,
            context_files=context_files,
            min_files=min_files,
            max_files=max_files,
            min_sites=min_sites,
            max_sites=max_sites,
            forbidden_terms=forbidden_terms,
        )

    def _chain_plan_rejection(
        self,
        payload: Dict[str, Any],
        *,
        context_files: Sequence[str],
        min_files: int,
        max_files: int,
        min_sites: int,
        max_sites: int,
        forbidden_terms: Sequence[str] = (),
    ) -> str:
        chain = payload.get("chain_plan")
        tests = payload.get("tests")
        if not isinstance(chain, dict) or not isinstance(tests, list) or not tests:
            return "missing chain_plan or tests"
        payload_text = json.dumps(payload, ensure_ascii=False).lower()
        leaked_terms = [
            str(term)
            for term in forbidden_terms
            if str(term).strip() and str(term).strip().lower() in payload_text
        ]
        if leaked_terms:
            return f"copied forbidden trajectory domain terms: {leaked_terms}"
        if not all(str(chain.get(key) or "").strip() for key in ("root_invariant", "entrypoint", "endpoint")):
            return "missing root_invariant, entrypoint, or endpoint"
        sites = chain.get("mutation_sites") or []
        if not isinstance(sites, list) or not min_sites <= len(sites) <= max_sites:
            return f"mutation site count must be between {min_sites} and {max_sites}"
        site_keys = [
            (str(site.get("file") or ""), str(site.get("symbol") or ""))
            for site in sites
            if isinstance(site, dict)
        ]
        if len(site_keys) != len(sites) or len(set(site_keys)) != len(site_keys):
            return "mutation sites must be unique file/symbol pairs"
        files = {str(site.get("file") or "") for site in sites if isinstance(site, dict)}
        if not min_files <= len(files) <= max_files:
            return f"production file count must be between {min_files} and {max_files}"
        allowed_files = {path for path in context_files if not is_test_path(path)}
        invalid_files = sorted(files - allowed_files)
        if invalid_files:
            return f"mutation files outside whitelist: {invalid_files}"
        tests_valid = len(tests) == 1 and all(
            isinstance(row, dict)
            and is_safe_repo_path(str(row.get("path") or ""))
            and str(row.get("path") or "").startswith("test/units/")
            and str(row.get("path") or "").endswith(".py")
            and str(row.get("content") or "").strip()
            for row in tests
        )
        if not tests_valid:
            return "tests must contain safe test paths and complete nonempty content"
        return ""

    def _write_tests(self, root: Path, tests: Sequence[Dict[str, Any]]) -> bool:
        if len(tests) != 1:
            return False
        for row in tests:
            relative = str(row.get("path") or "")
            if (
                not is_safe_repo_path(relative)
                or not relative.startswith("test/units/")
                or not relative.endswith(".py")
            ):
                return False
            path = root / relative
            if path.exists():
                return False
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(row.get("content") or ""), encoding="utf-8")
        return True

    def _materialize_contract_tests(
        self,
        plan: BugGenerationPlan,
        payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        tests = list(payload.get("tests") or [])
        blueprint = dict(getattr(plan, "task_blueprint", None) or {})
        if blueprint.get("contract_test_renderer") != "ansible_playbook_cli":
            return tests
        if len(tests) != 1:
            return []
        cases = payload.get("contract_cases") or []
        if not isinstance(cases, list) or len(cases) < 2:
            return []
        normalized: List[Dict[str, Any]] = []
        has_target = False
        has_control = False
        for index, case in enumerate(cases):
            if not isinstance(case, dict):
                return []
            playbook = str(case.get("playbook") or "").strip()
            files = case.get("files") or {}
            expected = case.get("expected_output") or []
            expected_counts = case.get("expected_counts") or {}
            if (
                not playbook
                or not isinstance(files, dict)
                or not isinstance(expected, list)
                or not expected
                or not isinstance(expected_counts, dict)
            ):
                return []
            safe_files = {
                str(path): str(content)
                for path, content in files.items()
                if is_safe_repo_path(str(path)) and str(content).strip()
            }
            if len(safe_files) != len(files):
                return []
            is_control = bool(case.get("compatibility_control", False))
            safe_counts: Dict[str, int] = {}
            task_definition_source = "\n".join([playbook, *safe_files.values()])
            for token, count in expected_counts.items():
                token = str(token)
                try:
                    count = int(count)
                except (TypeError, ValueError):
                    return []
                if not token or count < 1:
                    return []
                if r"\[" in token or r"\]" in token:
                    token = token.replace(r"\[", "[").replace(r"\]", "]")
                if token.startswith("TASK [") and token.endswith("]"):
                    display_name = token[len("TASK [") : -1]
                    block_container_pattern = re.compile(
                        r"(?m)^(?P<indent>\s*)-\s*name:\s*[\"']?"
                        + re.escape(display_name)
                        + r"[\"']?\s*\n(?P=indent)\s+block:\s*$"
                    )
                    if block_container_pattern.search(task_definition_source):
                        # A named ``block`` is a grouping construct, not an
                        # executable task. ansible-playbook prints its child
                        # task headers but never ``TASK [<block name>]``.
                        continue
                # Models often use a task's display name as an exactly-once
                # token. Ansible prints that name both in ``TASK [name]`` and
                # again when the debug message is identical. Count the public
                # execution header in that common case so clean contracts do
                # not fail for an irrelevant formatting duplicate.
                name_pattern = re.compile(
                    r"(?m)^\s*-?\s*name:\s*[\"']?"
                    + re.escape(token)
                    + r"[\"']?\s*$"
                )
                message_pattern = re.compile(
                    r"(?m)^\s*msg:\s*[\"']?"
                    + re.escape(token)
                    + r"[\"']?\s*$"
                )
                if name_pattern.search(task_definition_source):
                    count_token = f"TASK [{token}]"
                elif message_pattern.search(task_definition_source):
                    # A short message can also occur as a substring of a
                    # descriptive task name (for example ``explicit`` in
                    # ``Task in explicit block``). Count Ansible's exact JSON
                    # debug field instead of the ambiguous raw substring.
                    count_token = f'"msg": "{token}"'
                else:
                    count_token = token
                safe_counts[count_token] = count
            if (
                blueprint.get("require_expected_counts")
                and not is_control
                and not safe_counts
            ):
                return []
            has_control = has_control or is_control
            has_target = has_target or not is_control
            normalized.append(
                {
                    "name": str(case.get("name") or f"case_{index}"),
                    "playbook": playbook,
                    "files": safe_files,
                    "expected_output": [str(value) for value in expected],
                    "expected_counts": safe_counts,
                    "compatibility_control": is_control,
                }
            )
        if not has_target or not has_control:
            return []
        trusted_control_name = "godel0_empty_play_control"
        if not any(case["name"] == trusted_control_name for case in normalized):
            trusted_control = {
                "name": trusted_control_name,
                "playbook": (
                    "---\n- hosts: localhost\n"
                    "  gather_facts: false\n"
                    "  tasks: []\n"
                ),
                "files": {},
                "expected_output": ["PLAY [localhost]"],
                "expected_counts": {},
                "compatibility_control": True,
            }
            normalized.append(trusted_control)
            # The rendered test file and the taxonomy must describe the same
            # parametrized cases. Downstream F2P/P2P metadata is derived from
            # this payload after rendering.
            payload["contract_cases"] = [
                *list(payload.get("contract_cases") or []),
                dict(trusted_control),
            ]
        content = (
            "from __future__ import annotations\n\n"
            "import json\n"
            "import os\n"
            "import subprocess\n\n"
            "import sys\n\n"
            "from pathlib import Path\n\n"
            "import pytest\n\n"
            f"CASES = json.loads({json.dumps(json.dumps(normalized))})\n\n"
            "@pytest.mark.parametrize('case', CASES, ids=lambda case: case['name'])\n"
            "def test_generated_repo_contract(tmp_path, case):\n"
            "    playbook = tmp_path / 'playbook.yml'\n"
            "    playbook.write_text(case['playbook'], encoding='utf-8')\n"
            "    for relative, file_content in case['files'].items():\n"
            "        path = tmp_path / relative\n"
            "        path.parent.mkdir(parents=True, exist_ok=True)\n"
            "        path.write_text(file_content, encoding='utf-8')\n"
            "    inventory = tmp_path / 'inventory'\n"
            "    inventory.write_text('localhost ansible_connection=local\\n', encoding='utf-8')\n"
            "    env = os.environ.copy()\n"
            "    repo_root = Path(__file__).resolve().parents[3]\n"
            "    inherited_pythonpath = env.get('PYTHONPATH', '')\n"
            "    env['PYTHONPATH'] = os.pathsep.join(filter(None, [\n"
            "        str(repo_root / 'lib'), str(repo_root / 'test' / 'lib'),\n"
            "        inherited_pythonpath,\n"
            "    ]))\n"
            "    result = subprocess.run(\n"
            "        [sys.executable, str(repo_root / 'bin' / 'ansible-playbook'), "
            "str(playbook), '-i', str(inventory)],\n"
            "        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,\n"
            "    )\n"
            "    output = result.stdout + '\\n' + result.stderr\n"
            "    assert result.returncode == 0, output\n"
            "    for expected in case['expected_output']:\n"
            "        assert expected in output, output\n"
            "    for token, count in case.get('expected_counts', {}).items():\n"
            "        assert output.count(token) == count, output\n"
            "\n\n"
            "def test_generated_repo_cli_control():\n"
            "    repo_root = Path(__file__).resolve().parents[3]\n"
            "    env = os.environ.copy()\n"
            "    inherited_pythonpath = env.get('PYTHONPATH', '')\n"
            "    env['PYTHONPATH'] = os.pathsep.join(filter(None, [\n"
            "        str(repo_root / 'lib'), str(repo_root / 'test' / 'lib'),\n"
            "        inherited_pythonpath,\n"
            "    ]))\n"
            "    result = subprocess.run(\n"
            "        [sys.executable, str(repo_root / 'bin' / 'ansible-playbook'), '--version'],\n"
            "        cwd=repo_root, env=env, capture_output=True, text=True, check=False,\n"
            "    )\n"
            "    output = result.stdout + '\\n' + result.stderr\n"
            "    assert result.returncode == 0, output\n"
            "    assert 'ansible-playbook' in output, output\n"
        )
        return [{"path": str(tests[0].get("path") or ""), "content": content}]

    def _discard_tests(self, root: Path, tests: Sequence[Dict[str, Any]]) -> None:
        for row in tests:
            relative = str(row.get("path") or "")
            if not is_safe_repo_path(relative):
                continue
            path = root / relative
            if path.is_file():
                path.unlink()
        self._remove_runtime_artifacts(root)

    def _contract_quality_valid(self, patch: str) -> bool:
        lowered = patch.lower()
        forbidden = (
            "inspect.getsource",
            "read_text()",
            "git diff",
            "subprocess.run(['git'",
        )
        return "assert" in patch and not any(token in lowered for token in forbidden)

    def _contract_test_command(
        self,
        plan: BugGenerationPlan,
        repo_spec: RepoSpec,
        test_files: Sequence[str],
    ) -> str:
        blueprint = dict(getattr(plan, "task_blueprint", None) or {})
        explicit = str(blueprint.get("generated_test_command") or "")
        files = " ".join(shlex.quote(path) for path in test_files)
        if explicit:
            return explicit.replace("{test_files}", files)
        prefix = str(getattr(repo_spec, "test_command", "") or "pytest")
        return f"{prefix} {files}".strip()

    def _selected_contract_command(
        self,
        command: str,
        test_files: Sequence[str],
        test_ids: Sequence[str],
    ) -> str:
        if not test_ids:
            return ""
        base = command
        for path in test_files:
            base = base.replace(shlex.quote(path), "").replace(path, "")
        selected = " ".join(shlex.quote(test_id) for test_id in test_ids)
        return f"{base.strip()} {selected}".strip()

    def _plan_matches_patch(self, chain: Dict[str, Any], changed_files: Sequence[str], min_sites: int) -> bool:
        sites = list(chain.get("mutation_sites") or [])
        planned_files = {str(site.get("file") or "") for site in sites if isinstance(site, dict)}
        return len(sites) >= min_sites and set(changed_files).issubset(planned_files)

    def _causal_ablation(self, workspace: str, bug_patch: str, command: str, timeout: int) -> Dict[str, Any]:
        blocks = split_patch_by_file(bug_patch)
        repair_only: Dict[str, bool] = {}
        isolated: Dict[str, bool] = {}
        for repaired_path, _ in blocks:
            applied: List[str] = []
            for path, block in blocks:
                if path != repaired_path and apply_repository_patch(workspace, block):
                    applied.append(block)
            self._remove_runtime_artifacts(Path(workspace))
            repair_only[repaired_path] = bool(
                len(applied) == len(blocks) - 1
                and self._run_command(workspace, command, timeout).returncode == 0
            )
            for block in reversed(applied):
                apply_repository_patch(workspace, block, reverse=True)
        for path, block in blocks:
            applied = apply_repository_patch(workspace, block)
            self._remove_runtime_artifacts(Path(workspace))
            isolated[path] = bool(
                applied and self._run_command(workspace, command, timeout).returncode != 0
            )
            if applied:
                apply_repository_patch(workspace, block, reverse=True)
        return {
            "repair_only_one_file_passed": repair_only,
            "repair_only_one_file_all_fail": bool(repair_only and not any(repair_only.values())),
            "isolated_file_triggers_contract": isolated,
            "independently_active_file_count": sum(isolated.values()),
        }

    def _related_files(self, root: Path, requested: Sequence[str], budget: int) -> List[str]:
        selected: List[str] = []
        for value in requested:
            normalized = self._normalize(value)
            if normalized and (root / normalized).is_file() and normalized not in selected:
                selected.append(normalized)
        if len(selected) >= budget:
            return selected[:budget]
        # Follow repository-local imports before fuzzy path similarity. For an
        # anchor such as playbook/helpers.py, the actual semantic neighbours
        # are its dynamically imported Block/Handler/Task classes; filename
        # token matching previously filled the budget with unrelated helper
        # modules and made the exhaustive whitelist unusable.
        cursor = 0
        while cursor < len(selected) and len(selected) < budget:
            relative = selected[cursor]
            cursor += 1
            path = root / relative
            if not path.is_file() or path.suffix != ".py":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                module = ""
                if isinstance(node, ast.ImportFrom):
                    module = str(node.module or "")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported = self._python_module_path(root, alias.name)
                        if imported and imported not in selected:
                            selected.append(imported)
                            if len(selected) >= budget:
                                break
                    continue
                imported = self._python_module_path(root, module)
                if imported and imported not in selected:
                    selected.append(imported)
                    if len(selected) >= budget:
                        break
        anchor_tokens = {
            token
            for path in selected
            for token in re.split(r"[^A-Za-z0-9_]+", path)
            if len(token) >= 4 and token not in {"python", "main"}
        }
        candidates: List[tuple[int, str]] = []
        for path in root.rglob("*.py"):
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                continue
            if relative in selected or "/.git/" in f"/{relative}/":
                continue
            if is_test_path(relative):
                continue
            score = sum(1 for token in anchor_tokens if token.lower() in relative.lower())
            if score:
                candidates.append((-score, relative))
        for _, relative in sorted(candidates):
            if len(selected) >= budget:
                break
            selected.append(relative)
        return selected

    def _python_module_path(self, root: Path, module: str) -> str:
        # Use RepoProfile to determine module prefix and path mapping
        # (no hardcoded "ansible" check).
        try:
            from proposer.repo_profiles import get_profile

            profile = get_profile(str(getattr(self, "_current_repo_id", "") or ""))
            prefix = profile.module_prefix
            if prefix and not module.startswith(prefix):
                return ""
            stem = profile.module_path(module)
            if not stem:
                return ""
        except Exception:
            if not module:
                return ""
            stem = module.replace(".", "/")
        for relative in (stem + ".py", stem + "/__init__.py"):
            if (root / relative).is_file():
                return relative
        return ""

    def _source_bundle(self, root: Path, files: Sequence[str]) -> str:
        chunks: List[str] = []
        for relative in files:
            source = (root / relative).read_text(encoding="utf-8", errors="replace")
            if len(source) > 9000:
                source = source[:4500] + "\n...<file clipped>...\n" + source[-4500:]
            chunks.append(f"\n## FILE: {relative}\n```python\n{source}\n```")
        return "\n".join(chunks)

    def _symbol_catalog(
        self, root: Path, files: Sequence[str]
    ) -> Dict[str, List[str]]:
        catalog: Dict[str, List[str]] = {}
        for relative in files:
            if is_test_path(relative):
                continue
            path = root / relative
            if not path.is_file() or path.suffix != ".py":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            symbols: List[str] = []

            def visit(body: Sequence[ast.stmt], parents: List[str]) -> None:
                for node in body:
                    if isinstance(node, ast.ClassDef):
                        visit(node.body, parents + [node.name])
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append(".".join(parents + [node.name]))
                        visit(node.body, parents + [node.name])

            visit(tree.body, [])
            catalog[relative] = symbols
        return catalog

    def _mutation_source_bundle(self, root: Path, chain: Dict[str, Any]) -> str:
        requested: List[tuple[str, str]] = []
        for site in chain.get("mutation_sites") or []:
            if not isinstance(site, dict):
                continue
            path = str(site.get("file") or "")
            symbol = str(site.get("symbol") or "")
            if path and symbol:
                requested.append((path, symbol))
        chunks: List[str] = []
        for relative, symbol in requested:
            path = root / relative
            if not path.is_file() or path.suffix != ".py":
                return ""
            source = path.read_text(encoding="utf-8", errors="replace")
            span, error = self._symbol_span(source, symbol)
            if error:
                return ""
            start, end = span
            snippet = source[start:end].rstrip("\n")
            chunks.append(
                f"\n## EXACT SYMBOL: {relative}::{symbol}\n"
                f"```python\n{snippet}\n```"
            )
        return "\n".join(chunks) if chunks and len(chunks) == len(requested) else ""

    def _mutation_symbols_rejection(self, root: Path, chain: Dict[str, Any]) -> str:
        """Reject hallucinated plan symbols early enough for contract retries."""
        errors: List[str] = []
        for site in chain.get("mutation_sites") or []:
            if not isinstance(site, dict):
                continue
            relative = str(site.get("file") or "")
            symbol = str(site.get("symbol") or "")
            path = root / relative
            if not path.is_file() or path.suffix != ".py":
                errors.append(f"{relative}::{symbol}: production Python file not found")
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
            _, error = self._symbol_span(source, symbol)
            if error:
                errors.append(f"{relative}::{symbol}: {error}")
        if not errors:
            return ""
        return (
            "planned mutation symbols must exist exactly in the supplied source; "
            "replan using real class/function names: " + "; ".join(errors)
        )

    def _materialize_symbol_edits(
        self,
        root: Path,
        payload: Dict[str, Any],
        chain: Dict[str, Any],
        *,
        min_files: int,
        max_files: int,
        min_sites: int,
        max_sites: int,
    ) -> tuple[List[Dict[str, str]], str]:
        """Validate semantic edits and apply them transactionally.

        The model never controls diff headers, hunk ranges, or surrounding
        context. Each replacement must resolve uniquely inside its planned AST
        symbol; all resulting Python files are parsed before anything is written.
        """
        planned = {
            (str(site.get("file") or ""), str(site.get("symbol") or ""))
            for site in chain.get("mutation_sites") or []
            if isinstance(site, dict)
        }
        raw_edits = payload.get("edits")
        if not isinstance(raw_edits, list) or not min_sites <= len(raw_edits) <= max_sites:
            return [], f"edits count must be between {min_sites} and {max_sites}"
        if len(raw_edits) != len(planned):
            return [], f"edits must cover all {len(planned)} planned mutation sites exactly once"
        normalized: List[Dict[str, str]] = []
        files: set[str] = set()
        seen_sites: set[tuple[str, str]] = set()
        for index, raw in enumerate(raw_edits):
            if not isinstance(raw, dict):
                return [], f"edit {index} is not an object"
            edit = {
                key: str(raw.get(key) or "")
                for key in ("file", "symbol", "before", "after", "intent")
            }
            site = (edit["file"], edit["symbol"])
            if site not in planned:
                return [], f"edit {index} is not an exact planned file/symbol: {site}"
            if site in seen_sites:
                return [], f"duplicate edit for planned site: {site}"
            if (
                not is_safe_repo_path(edit["file"])
                or is_test_path(edit["file"])
                or not edit["before"]
                or edit["before"] == edit["after"]
            ):
                return [], f"edit {index} has an unsafe path, empty before, or no-op replacement"
            before_comments = {
                line.strip()
                for line in edit["before"].splitlines()
                if line.strip().startswith("#")
            }
            added_comments = [
                line.strip()
                for line in edit["after"].splitlines()
                if line.strip().startswith("#") and line.strip() not in before_comments
            ]
            if added_comments:
                return [], f"edit {index} adds comments that may reveal the generated regression"
            path = root / edit["file"]
            if not path.is_file() or path.suffix != ".py":
                return [], f"edit {index} target is not a Python production file"
            normalized.append(edit)
            files.add(edit["file"])
            seen_sites.add(site)
        if not min_files <= len(files) <= max_files:
            return [], f"edited file count must be between {min_files} and {max_files}"

        proposed_sources: Dict[str, str] = {}
        for relative in sorted(files):
            path = root / relative
            source = path.read_text(encoding="utf-8", errors="replace")
            replacements: List[tuple[int, int, str]] = []
            for edit in (row for row in normalized if row["file"] == relative):
                span, error = self._symbol_span(source, edit["symbol"])
                if error:
                    return [], f"{relative}::{edit['symbol']}: {error}"
                start, end = span
                symbol_source = source[start:end]
                if symbol_source.count(edit["before"]) != 1:
                    aligned = self._align_edit_indentation(
                        symbol_source, edit["before"], edit["after"]
                    )
                    if aligned is None:
                        return [], (
                            f"{relative}::{edit['symbol']}: before must occur exactly once "
                            "inside the AST symbol"
                        )
                    edit["before"], edit["after"] = aligned
                local = symbol_source.index(edit["before"])
                replacements.append(
                    (start + local, start + local + len(edit["before"]), edit["after"])
                )
            replacements.sort()
            for left, right in zip(replacements, replacements[1:]):
                if left[1] > right[0]:
                    return [], f"overlapping replacements in {relative}"
            changed = source
            for start, end, after in reversed(replacements):
                changed = changed[:start] + after + changed[end:]
            try:
                ast.parse(changed, filename=relative)
            except SyntaxError as exc:
                return [], f"{relative} is not valid Python after edits: {exc.msg} at line {exc.lineno}"
            proposed_sources[relative] = changed

        for relative, source in proposed_sources.items():
            (root / relative).write_text(source, encoding="utf-8")
        return normalized, ""

    def _align_edit_indentation(
        self,
        symbol_source: str,
        before: str,
        after: str,
    ) -> tuple[str, str] | None:
        """Correct only a uniform indentation offset, then require uniqueness."""

        def shift(text: str, delta: int) -> str | None:
            shifted: List[str] = []
            for line in text.splitlines(keepends=True):
                body = line.rstrip("\r\n")
                ending = line[len(body) :]
                if not body.strip():
                    shifted.append(line)
                    continue
                indent = len(body) - len(body.lstrip(" "))
                new_indent = indent + delta
                if new_indent < 0:
                    return None
                shifted.append(" " * new_indent + body[indent:] + ending)
            return "".join(shifted)

        matches: List[tuple[str, str]] = []
        for delta in range(-16, 33):
            if delta == 0:
                continue
            shifted_before = shift(before, delta)
            shifted_after = shift(after, delta)
            if (
                shifted_before is not None
                and shifted_after is not None
                and symbol_source.count(shifted_before) == 1
            ):
                matches.append((shifted_before, shifted_after))
        return matches[0] if len(matches) == 1 else None

    def _symbol_span(self, source: str, requested: str) -> tuple[tuple[int, int], str]:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return (0, 0), f"source is not valid Python: {exc.msg}"
        matches: List[ast.AST] = []

        def visit(body: Sequence[ast.stmt], parents: List[str]) -> None:
            for node in body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified = ".".join(parents + [node.name])
                    if requested == node.name or requested == qualified or requested.endswith("." + qualified):
                        matches.append(node)
                    visit(getattr(node, "body", []), parents + [node.name])

        visit(tree.body, [])
        if len(matches) != 1:
            return (0, 0), f"symbol resolved {len(matches)} times instead of once"
        node = matches[0]
        lines = source.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        start_line = int(getattr(node, "lineno", 1))
        end_line = int(getattr(node, "end_lineno", start_line))
        return (offsets[start_line - 1], offsets[end_line]), ""

    def _build_problem_statement(
        self, chain: Dict[str, Any], payload: Dict[str, Any]
    ) -> str:
        target_cases = [
            case
            for case in payload.get("contract_cases") or []
            if isinstance(case, dict) and not bool(case.get("compatibility_control", False))
        ]
        if target_cases:
            case = target_cases[0]
            scenario = str(case.get("name") or "affected workflow").replace("_", " ")
            playbook = str(case.get("playbook") or "").strip()
            expected = [str(value) for value in case.get("expected_output") or []]
            counts = dict(case.get("expected_counts") or {})
            expectations = "\n".join(f"- Output contains `{value}`." for value in expected)
            expectations += "".join(
                f"\n- `{token}` occurs exactly {count} time(s)."
                for token, count in counts.items()
            )
            return (
                f"# Incorrect public behavior in {scenario}\n\n"
                "## Description\n\n"
                "Running the following playbook through `ansible-playbook` does not "
                "preserve the expected end-to-end behavior. Nearby supported "
                "playbooks should remain compatible.\n\n"
                "## Reproduction\n\n"
                "```yaml\n"
                f"{playbook}\n"
                "```\n\n"
                "Run it against localhost with the local connection.\n\n"
                "## Expected behavior\n\n"
                f"{expectations}\n"
            )
        invariant = str(chain.get("root_invariant") or "behavior is preserved across the workflow").strip()
        cases = [
            str(case.get("name") or "").replace("_", " ").strip()
            for case in payload.get("contract_cases") or []
            if isinstance(case, dict) and not bool(case.get("compatibility_control", False))
        ]
        scenario = cases[0] if cases else "the affected end-to-end workflow"
        return (
            f"# Regression: {invariant[:1].upper() + invariant[1:]}\n\n"
            "## Description\n\n"
            f"When using {scenario}, the observable behavior becomes inconsistent as "
            "the operation moves through the related processing layers. The same "
            "invocation should retain one coherent meaning from entry to final output.\n\n"
            "## Expected behavior\n\n"
            f"{invariant.rstrip('.')} across the complete workflow. Existing nearby "
            "behavior that does not use this scenario must remain compatible.\n\n"
            "## Reproduction\n\n"
            f"Exercise {scenario} through the repository's public entry point and "
            "observe the final user-visible output. Repeating the operation with its "
            "supported variants should produce results consistent with the values "
            "provided for each invocation."
        )

    def _contract_test_taxonomy(
        self, payload: Dict[str, Any], test_files: Sequence[str]
    ) -> Dict[str, List[str]]:
        cases = payload.get("contract_cases") or []
        if cases and test_files:
            base = f"{test_files[0]}::test_generated_repo_contract"
            fail = [
                f"{base}[{case.get('name')}]"
                for case in cases
                if isinstance(case, dict) and not bool(case.get("compatibility_control", False))
            ]
            passed = [
                f"{base}[{case.get('name')}]"
                for case in cases
                if isinstance(case, dict) and bool(case.get("compatibility_control", False))
            ]
            passed.append(
                f"{test_files[0]}::test_generated_repo_cli_control"
            )
            return {"FAIL_TO_PASS": fail, "PASS_TO_PASS": passed}
        tests = payload.get("tests") or []
        nodeids: List[str] = []
        for row in tests:
            if not isinstance(row, dict):
                continue
            try:
                tree = ast.parse(str(row.get("content") or ""))
            except SyntaxError:
                continue
            nodeids.extend(
                f"{row.get('path')}::{node.name}"
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            )
        return {"FAIL_TO_PASS": nodeids, "PASS_TO_PASS": []}

    def _reverse_worktree_diff(
        self, workspace: str, changed_files: Sequence[str]
    ) -> str:
        result = subprocess.run(
            ["git", "-C", workspace, "diff", "-R", "HEAD", "--", *changed_files],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout if result.returncode == 0 else ""

    def _trajectory_evidence(self, plan: BugGenerationPlan) -> str:
        rows: List[str] = []
        signature = getattr(plan, "failure_signature", None)
        if signature is not None:
            rows.append(str(signature))
        for value in list(getattr(plan, "source_trajectory_ids", []) or [])[:3]:
            path = Path(value)
            if path.is_file():
                rows.append(
                    f"Prior trajectory artifact: {path.name} ({path.stat().st_size} bytes). "
                    "Its repository-specific bug story is intentionally withheld; use "
                    "only the normalized failure signature above."
                )
            else:
                rows.append(
                    "A prior trajectory was referenced but is unavailable locally; use "
                    "only the normalized failure signature above."
                )
        return "\n\n".join(rows) or "No prior trajectory is available; build a bootstrap task."

    def _source_patch(self, patch: str) -> str:
        return "".join(block for path, block in split_patch_by_file(patch) if not is_test_path(path))

    def _test_patch(self, patch: str) -> str:
        return "".join(block for path, block in split_patch_by_file(patch) if is_test_path(path))

    def _changed_test_files(self, patch: str) -> List[str]:
        return [path for path, _ in split_patch_by_file(patch) if is_test_path(path)]

    def _run_command(self, workspace: str, command: str, timeout: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=workspace,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout or 300)),
            check=False,
        )

    def _save_diagnostic(self, root: Path, name: str, result: subprocess.CompletedProcess[str]) -> None:
        (root / name).write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")

    def _remove_runtime_artifacts(self, root: Path) -> None:
        for path in root.rglob("__pycache__"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(root / ".pytest_cache", ignore_errors=True)

    def _candidate_id(self, plan: BugGenerationPlan, bug_patch: str, contract_patch: str) -> str:
        digest = hashlib.sha256(
            f"{plan.plan_id}:repo_chain:{bug_patch}:{contract_patch}".encode("utf-8")
        ).hexdigest()[:12]
        return f"cand_chain_{digest}"

    @staticmethod
    def _normalize(path: str) -> str:
        normalized = str(path).replace("\\", "/").strip()
        while normalized.startswith("./"):
            normalized = normalized[2:]
        pure = PurePosixPath(normalized)
        return pure.as_posix() if normalized and not pure.is_absolute() else ""


__all__ = ["RepoChainGenerator"]
