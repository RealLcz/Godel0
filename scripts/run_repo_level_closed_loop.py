#!/usr/bin/env python3
"""Run a two-stage repository-level Proposer/Solver quality experiment.

Stage 1 creates bootstrap tasks without solver trajectories. The current solver
attempts those tasks. Stage 2 diagnoses the resulting solver trajectories and
generates transfer tasks targeting the observed capability gaps.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


ROLE_INSTANCE_CONTRACT_DIR = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "ansible_role_instance_contract"
)
ROLE_INSTANCE_SHORTCUT_PATCH = (
    ROLE_INSTANCE_CONTRACT_DIR / "parent_reset_shortcut.patch"
).read_text(encoding="utf-8")
SOLVER_PUBLIC_RUNTIME_TOKEN = "__GODEL0_SOLVER_RUNTIME__"

ROLE_INSTANCE_PUBLIC_COMMAND = (
    "check_count() { "
    "expected=\"$1\"; pattern=\"$2\"; shift 2; "
    "output=\"$(\"$@\" 2>&1)\"; command_rc=$?; "
    "actual=\"$(printf '%s\\n' \"$output\" | grep -F -c \"$pattern\")\"; "
    "if [ \"$command_rc\" -ne 0 ] || [ \"$actual\" -ne \"$expected\" ]; "
    "then printf '%s\\n' \"$output\"; return 1; fi; }; "
    "check_count 1 '\"msg\": \"A\"' "
    "ansible-playbook no_dupes.yml -i 'testhost,' -c local --tags inroles && "
    "check_count 1 '\"msg\": \"A\"' "
    "ansible-playbook no_dupes.yml -i 'testhost,' -c local --tags acrossroles && "
    "check_count 1 '\"msg\": \"A\"' "
    "ansible-playbook no_dupes.yml -i 'testhost,' -c local --tags intasks && "
    "check_count 2 '\"msg\": \"A\"' "
    "ansible-playbook allowed_dupes.yml -i 'testhost,' -c local --tags importrole && "
    "check_count 2 '\"msg\": \"A\"' "
    "ansible-playbook allowed_dupes.yml -i 'testhost,' -c local --tags includerole && "
    "check_count 1 '\"msg\": \"A\"' "
    "env ANSIBLE_STRATEGY=free ansible-playbook no_dupes.yml "
    "-i 'testhost,' -c local --tags inroles"
)


PR_REPLAY_CASES = [
    {
        "id": "role_instance_cache",
        "commit": "1998521e2d5b89bc53d00639bad178330ebb98df",
        "base_at_reference": True,
        "files": [
            "lib/ansible/playbook/play.py",
            "lib/ansible/playbook/role/__init__.py",
            "lib/ansible/plugins/strategy/__init__.py",
            "lib/ansible/plugins/strategy/free.py",
            "lib/ansible/plugins/strategy/linear.py",
        ],
        "tests": [],
        "adversarial_solver_patches": [
            {
                "id": "parent_reset_single_file_solver",
                "patch": ROLE_INSTANCE_SHORTCUT_PATCH,
            }
        ],
        "integration_dir": "test/integration/targets/roles",
        "integration_command": (
            "check_count() { "
            "label=\"$1\"; expected=\"$2\"; pattern=\"$3\"; shift 3; "
            "output=\"$(\"$@\" 2>&1)\"; command_rc=$?; "
            "actual=\"$(printf '%s\\n' \"$output\" | grep -F -c \"$pattern\")\"; "
            "printf 'GODEL0_CONTRACT %s expected=%s actual=%s command_rc=%s\\n' "
            "\"$label\" \"$expected\" \"$actual\" \"$command_rc\"; "
            "if [ \"$command_rc\" -ne 0 ] || [ \"$actual\" -ne \"$expected\" ]; "
            "then printf '%s\\n' \"$output\"; return 1; fi; }; "
            "check_count role.no_dupes.inroles 1 '\"msg\": \"A\"' "
            "ansible-playbook no_dupes.yml -i 'testhost,' -c local --tags inroles && "
            "check_count role.no_dupes.acrossroles 1 '\"msg\": \"A\"' "
            "ansible-playbook no_dupes.yml -i 'testhost,' -c local --tags acrossroles && "
            "check_count role.no_dupes.intasks 1 '\"msg\": \"A\"' "
            "ansible-playbook no_dupes.yml -i 'testhost,' -c local --tags intasks && "
            "check_count role.allowed_dupes.importrole 2 '\"msg\": \"A\"' "
            "ansible-playbook allowed_dupes.yml -i 'testhost,' -c local --tags importrole && "
            "check_count role.allowed_dupes.includerole 2 '\"msg\": \"A\"' "
            "ansible-playbook allowed_dupes.yml -i 'testhost,' -c local --tags includerole && "
            "check_count role.inheritance.linear 3 '\"msg\": \"abc\"' "
            "ansible-playbook dupe_inheritance.yml -i 'testhost,' -c local && "
            "check_count role.no_dupes.free 1 '\"msg\": \"A\"' "
            "env ANSIBLE_STRATEGY=free ansible-playbook no_dupes.yml "
            "-i 'testhost,' -c local --tags inroles && "
            "check_count role.inheritance.free 3 '\"msg\": \"abc\"' "
            "env ANSIBLE_STRATEGY=free ansible-playbook dupe_inheritance.yml "
            "-i 'testhost,' -c local && "
            "check_count role.distinct_path.group_one 1 '\"msg\": \"NESTED_ONE\"' "
            "ansible-playbook \"$GODEL0_ROLE_INSTANCE_CONTRACT_PLAYBOOK\" "
            "-i 'testhost,' -c local && "
            "check_count role.distinct_path.group_two 1 '\"msg\": \"NESTED_TWO\"' "
            "ansible-playbook \"$GODEL0_ROLE_INSTANCE_CONTRACT_PLAYBOOK\" "
            "-i 'testhost,' -c local"
        ),
        "problem": (
            "Repeated role invocations with identical role names and parameters "
            "incorrectly share execution, inheritance, and resolved-path state. "
            "Each declaration must retain its own parent and path context under "
            "both the linear and free strategies, while duplicate suppression "
            "semantics remain intact."
        ),
    },
    {
        "id": "delegate_to_evaluation",
        "commit": "42355d181a11b51ebfc56f6f4b3d9c74e01cb13b",
        "base_at_reference": True,
        "files": [
            "lib/ansible/executor/process/worker.py",
            "lib/ansible/executor/task_executor.py",
            "lib/ansible/playbook/delegatable.py",
            "lib/ansible/playbook/task.py",
            "lib/ansible/vars/manager.py",
        ],
        "tests": [],
        "integration_dir": "test/integration/targets/delegate_to",
        "integration_command": (
            "ansible-playbook test_random_delegate_to_with_loop.yml "
            "-i inventory -v && "
            "ansible-playbook test_random_delegate_to_without_loop.yml "
            "-i inventory -v"
        ),
        "problem": (
            "A dynamic delegate_to expression can be evaluated more than once for "
            "the same task or loop item. When the expression is nondeterministic, "
            "delegated variables and the selected host disagree. Calculate and "
            "propagate the delegated host exactly once per task invocation."
        ),
    },
    {
        "id": "broken_conditionals",
        "commit": "56a1402a7f209b1bd963e07896a2983219348982",
        "files": [
            "lib/ansible/executor/task_executor.py",
            "lib/ansible/playbook/conditional.py",
        ],
        "tests": [
            "test/units/executor/test_task_executor.py",
            "test/units/playbook/test_conditional.py",
        ],
        "integration_dir": "test/integration/targets/conditionals",
        "integration_command": "ansible-playbook validate_broken_conditionals.yml",
        "problem": (
            "Broken task conditionals are not detected consistently across the "
            "conditional evaluator and task execution path. Restore consistent "
            "error handling and preserve valid conditional behavior."
        ),
    },
    {
        "id": "variable_load_errors",
        "commit": "d797c3150d79b1d5bc5ac6f5c9333d95b1323db3",
        "files": [
            "lib/ansible/plugins/vars/host_group_vars.py",
            "lib/ansible/utils/vars.py",
            "lib/ansible/vars/manager.py",
        ],
        "tests": [
            "test/units/utils/test_vars.py",
            "test/units/vars/test_variable_manager.py",
        ],
        "integration_dir": "test/integration/targets/var_blending",
        "integration_command": (
            "test \"$(ansible-playbook error_handling.yml -i inventory "
            "--vault-password-file supersecretvaultsecret -e @vars/bad_vault.yml "
            "2>&1 | grep -c dummy)\" -eq 0 && "
            "test \"$(ansible-playbook error_handling.yml -i inventory "
            "--vault-password-file supersecretvaultsecret --tags includevault "
            "2>&1 | grep -c dummy)\" -eq 0"
        ),
        "problem": (
            "Variable loading errors lose the concrete source file while passing "
            "through vars plugins and the variable manager. Preserve useful source "
            "context without changing successful variable merging."
        ),
    },
    {
        "id": "plugin_option_origin",
        "commit": "784598005e8eb752f23054a850099b35073c5a69",
        "files": [
            "lib/ansible/config/manager.py",
            "lib/ansible/plugins/__init__.py",
            "lib/ansible/plugins/callback/__init__.py",
        ],
        "tests": [
            "test/units/config/test_manager.py",
            "test/units/plugins/test_plugins.py",
            "test/units/plugins/callback/test_callback.py",
        ],
        "integration_dir": "test/integration/targets/config",
        "integration_command": (
            "ANSIBLE_LOOKUP_PLUGINS=lookup_plugins "
            "ansible-playbook match_option_methods.yml"
        ),
        "problem": (
            "Plugin configuration options do not consistently retain their origin "
            "when resolved through the configuration manager and plugin classes."
        ),
    },
    {
        "id": "collection_path_scanning",
        "commit": "e5116634479dbfe285555e6ffdecec116f1c86b0",
        "files": [
            "lib/ansible/cli/doc.py",
            "lib/ansible/collections/list.py",
        ],
        "tests": [
            "test/units/cli/test_doc.py",
            "test/units/galaxy/test_collection.py",
        ],
        "integration_dir": "test/integration/targets/ansible-doc",
        "integration_command": (
            "testdir=\"$PWD\"; "
            "pbdir=collections/ansible_collections/testns/testcol/playbooks; "
            "cd \"$pbdir\"; "
            "ANSIBLE_COLLECTIONS_PATH=\"$testdir/$pbdir/collections\" "
            "ansible-doc -vvv --metadata-dump --no-fail-on-errors >/dev/null"
        ),
        "problem": (
            "Collection discovery can crash when a path contains nested "
            "ansible_collections components. Make collection-name extraction and "
            "documentation scanning handle that path shape consistently."
        ),
    },
]


DOMAINS = {
    "template_vars": {
        "description": "templating and lazy variable lookup",
        "anchors": [
            "lib/ansible/template/__init__.py",
            "lib/ansible/template/vars.py",
        ],
        "tests": [
            "test/units/template/test_templar.py",
            "test/units/template/test_vars.py",
            "test/units/template/test_native_concat.py",
        ],
        "contract": (
            "Create one subtle inconsistency across templating and variable lookup. "
            "The same value or undefined-variable contract must flow through at "
            "least two production modules, and existing tests must expose it."
        ),
    },
    "variable_merge": {
        "description": "variable precedence, merging, and vars-plugin propagation",
        "anchors": [
            "lib/ansible/utils/vars.py",
            "lib/ansible/vars/manager.py",
            "lib/ansible/plugins/vars/host_group_vars.py",
        ],
        "tests": [
            "test/units/utils/test_vars.py",
            "test/units/vars/test_variable_manager.py",
        ],
        "contract": (
            "Create one shared variable precedence or merge regression that crosses "
            "the utility and manager/plugin boundary. Avoid two independent edits."
        ),
    },
    "yaml_loading": {
        "description": "data loading, YAML construction, and source metadata",
        "anchors": [
            "lib/ansible/parsing/dataloader.py",
            "lib/ansible/parsing/yaml/loader.py",
            "lib/ansible/parsing/yaml/constructor.py",
        ],
        "tests": [
            "test/units/parsing/test_dataloader.py",
            "test/units/parsing/yaml/test_loader.py",
            "test/units/parsing/yaml/test_constructor.py",
        ],
        "contract": (
            "Create one regression in the handoff between DataLoader and YAML "
            "loading/construction, preserving syntax while corrupting source or "
            "value semantics across multiple modules."
        ),
    },
    "plugin_config": {
        "description": "plugin option resolution and callback configuration",
        "anchors": [
            "lib/ansible/config/manager.py",
            "lib/ansible/plugins/__init__.py",
            "lib/ansible/plugins/callback/__init__.py",
        ],
        "tests": [
            "test/units/config/test_manager.py",
            "test/units/plugins/test_plugins.py",
            "test/units/plugins/callback/test_callback.py",
        ],
        "contract": (
            "Create one option-resolution regression spanning the configuration "
            "manager and plugin/callback layer. Existing tests must fail for the "
            "same configuration contract."
        ),
    },
    "task_conditionals": {
        "description": "conditional evaluation and task execution",
        "anchors": [
            "lib/ansible/playbook/conditional.py",
            "lib/ansible/executor/task_executor.py",
        ],
        "tests": [
            "test/units/playbook/test_conditional.py",
            "test/units/executor/test_task_executor.py",
        ],
        "contract": (
            "Create one inconsistent conditional-evaluation behavior shared by the "
            "playbook conditional layer and task executor."
        ),
    },
    "inventory_model": {
        "description": "inventory manager and host/group model consistency",
        "anchors": [
            "lib/ansible/inventory/manager.py",
            "lib/ansible/inventory/data.py",
            "lib/ansible/inventory/group.py",
            "lib/ansible/inventory/host.py",
        ],
        "tests": [
            "test/units/inventory/test_group.py",
            "test/units/inventory/test_host.py",
            "test/units/plugins/inventory/test_inventory.py",
        ],
        "contract": (
            "Create one inventory state or membership regression that propagates "
            "across manager and host/group model code."
        ),
    },
    "inventory_yaml_e2e": {
        "description": "YAML inventory parsing and host/group materialization",
        "anchors": [
            "lib/ansible/parsing/dataloader.py",
            "lib/ansible/parsing/utils/yaml.py",
            "lib/ansible/plugins/inventory/yaml.py",
            "lib/ansible/plugins/inventory/__init__.py",
            "lib/ansible/inventory/data.py",
            "lib/ansible/inventory/group.py",
            "lib/ansible/inventory/host.py",
        ],
        "tests": [
            "test/units/plugins/inventory/test_inventory.py::TestInventory::test_split_patterns",
            "test/units/plugins/inventory/test_inventory.py::TestInventoryPlugins::test_ini",
            "test/units/plugins/inventory/test_inventory.py::TestInventoryPlugins::test_yaml_inventory",
        ],
        "contract_test": (
            "test/units/plugins/inventory/test_inventory.py::"
            "TestInventoryPlugins::test_yaml_inventory"
        ),
        "trace_source_roots": ["lib/ansible"],
        "min_modified_files": 5,
        "contract": (
            "A YAML inventory document must travel through DataLoader, YAML "
            "decoding, the inventory plugin, InventoryData, and the Group/Host "
            "model while preserving host identity and membership."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godel0-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--repo-pool", default="repo_pool")
    parser.add_argument("--repo-id", default="ansible")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", default="8000")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-target", type=int, default=3)
    parser.add_argument("--bootstrap-pr-replay-limit", type=int, default=1)
    parser.add_argument("--adaptive-target", type=int, default=2)
    parser.add_argument("--max-bootstrap-agent-attempts", type=int, default=4)
    parser.add_argument("--max-adaptive-agent-attempts", type=int, default=4)
    parser.add_argument("--agent-timeout", type=int, default=1200)
    parser.add_argument("--test-timeout", type=int, default=300)
    parser.add_argument("--preflight-pr-only", action="store_true")
    parser.add_argument("--skip-pr-replay", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.godel0_root).resolve()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "initial_agent" / "src"))
    sys.path.insert(0, str(root))

    os.environ["VLLM_HOST"] = args.vllm_host
    os.environ["VLLM_PORT"] = str(args.vllm_port)
    os.environ.pop("QWEN_API_BASE_URL", None)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    solver_scratch_base = Path(os.environ.get("TMPDIR") or "/tmp").resolve()
    os.environ.setdefault(
        "GODEL0_SOLVER_SCRATCH_ROOT",
        str(solver_scratch_base / f"godel0_solver_{os.getpid()}"),
    )
    runtime_tmp = output_dir / "runtime_tmp"
    ansible_tmp = runtime_tmp / "ansible_local"
    ansible_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(runtime_tmp)
    os.environ["ANSIBLE_LOCAL_TEMP"] = str(ansible_tmp)
    ansible_runtime = _prepare_ansible_runtime(output_dir)
    solver_ansible_runtime = _solver_public_ansible_runtime()

    from godel0.proposer_trusted.candidate_validator import CandidateValidator
    from godel0.tasks.repo_pool import RepoPool
    from experiment_adapters.common_agent_adapter import CommonAgentAdapter
    from experiment_adapters.repo_bug_agent_adapter import RepoBugAgentAdapter
    from swesmith.engine import (
        BugConstraints,
        BugGenerationPlan,
        RepoSpec as EngineRepoSpec,
        SWESmithEngine,
    )

    pool = RepoPool((root / args.repo_pool).resolve())
    spec = pool.get(args.repo_id)
    if spec is None:
        raise RuntimeError(f"Repository not found: {args.repo_id}")
    source_repo = (root / spec.path).resolve() if not Path(spec.path).is_absolute() else Path(spec.path)
    repo_spec = EngineRepoSpec(
        repo_id=spec.repo_id,
        repo_path=str(source_repo),
        base_commit=spec.base_commit,
        test_command=spec.test_command,
    )
    validator = CandidateValidator(
        workspace_root=output_dir / "validator",
        test_timeout_sec=args.test_timeout,
        max_patch_lines=160,
        forbid_test_file_edits=True,
    )
    solver_adapter = None if args.preflight_pr_only else CommonAgentAdapter()
    proposer_adapter = None if args.preflight_pr_only else RepoBugAgentAdapter()
    engine = SWESmithEngine(agent_adapter=proposer_adapter)

    report: Dict[str, Any] = {
        "configuration": {
            "model": args.model,
            "repo_id": spec.repo_id,
            "base_commit": spec.base_commit,
            "bootstrap_target": args.bootstrap_target,
            "bootstrap_pr_replay_limit": args.bootstrap_pr_replay_limit,
            "adaptive_target": args.adaptive_target,
        },
        "pr_replay_attempts": [],
        "bootstrap_generation_attempts": [],
        "bootstrap_tasks": [],
        "bootstrap_solver_results": [],
        "trajectory_diagnosis": {},
        "adaptive_generation_attempts": [],
        "adaptive_tasks": [],
        "adaptive_solver_results": [],
        "metrics": {},
        "started_at": _utc_now(),
    }

    def checkpoint() -> None:
        report["metrics"] = _compute_metrics(report)
        (output_dir / "closed_loop_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    print("\n=== Stage 0: real PR replay preflight ===", flush=True)
    pr_replay_selected = 0
    max_selected_pr_replays = min(
        max(0, args.bootstrap_pr_replay_limit),
        max(0, args.bootstrap_target - 1),
    )
    pr_replay_cases = [] if args.skip_pr_replay else PR_REPLAY_CASES
    if args.skip_pr_replay:
        report["pr_replay_skipped"] = True
        print("PR replay preflight skipped by request.", flush=True)
    for case in pr_replay_cases:
        attempt_dir = output_dir / "pr_replay" / case["id"]
        attempt_dir.mkdir(parents=True, exist_ok=True)
        replay_base_commit = (
            case["commit"]
            if case.get("base_at_reference")
            else spec.base_commit
        )
        replay_file_count = len(case["files"])
        plan = BugGenerationPlan(
            plan_id=f"pr_replay_{case['id']}",
            target_repo_id=spec.repo_id,
            target_base_commit=replay_base_commit,
            target_file=case["files"][0],
            target_files=list(case["files"]),
            strategy="pr_replay",
            operator="reverse_real_fix",
            constraints=BugConstraints(
                min_modified_files=(
                    replay_file_count if case.get("base_at_reference") else 2
                ),
                max_modified_files=(
                    replay_file_count if case.get("base_at_reference") else 4
                ),
                max_modified_lines=(
                    300 if case.get("base_at_reference") else 160
                ),
                allow_test_edits=False,
            ),
            reference_commit=case["commit"],
            task_blueprint={
                "source": "historical_fix_commit",
                "case": case["id"],
                "reference_commit": case["commit"],
                "oracle_files": list(case["files"]),
            },
        )
        candidates = engine.generate(plan, str(root / "initial_agent" / "src"), repo_spec, str(attempt_dir))
        row: Dict[str, Any] = {
            "case_id": case["id"],
            "reference_commit": case["commit"],
            "generated": len(candidates),
        }
        if candidates:
            candidate = candidates[0]
            integration_test_command = _integration_test_command(
                ansible_runtime,
                case["integration_dir"],
                case["integration_command"],
            )
            control_test_command = _control_test_command(ansible_runtime)
            solver_test_command = _control_test_command(solver_ansible_runtime)
            if case["id"] == "role_instance_cache":
                solver_test_command = _integration_test_command(
                    solver_ansible_runtime,
                    case["integration_dir"],
                    ROLE_INSTANCE_PUBLIC_COMMAND,
                    include_contract_env=False,
                )
            task = _validate_and_package(
                candidate=candidate,
                phase="bootstrap",
                domain_id=f"pr:{case['id']}",
                problem_statement=case["problem"],
                test_files=case["tests"],
                test_command_override=integration_test_command,
                validation_mode="exit_code",
                command_test_id=f"integration::{case['id']}",
                control_test_command=control_test_command,
                source_repo=source_repo,
                base_commit=replay_base_commit,
                test_prefix=spec.test_command,
                validator=validator,
                output_dir=output_dir,
                test_timeout=args.test_timeout,
                source_trajectory_ids=[],
                strictness_policy=(
                    "historical_pr"
                    if case.get("base_at_reference")
                    else "synthetic_causal"
                ),
                adversarial_solver_patches=list(
                    case.get("adversarial_solver_patches") or []
                ),
                solver_test_command_override=solver_test_command,
            )
            row.update({
                "candidate_id": candidate.candidate_id,
                "modified_files": candidate.modified_files,
                "validation_passed": bool(task),
                "strict_repo_level": bool(
                    task and task.get("strict_repo_level")
                ),
            })
            selected = bool(
                task
                and task.get("strict_repo_level")
                and not args.preflight_pr_only
                and pr_replay_selected < max_selected_pr_replays
                and len(report["bootstrap_tasks"]) < args.bootstrap_target
            )
            row["selected_for_bootstrap"] = selected
            if selected:
                report["bootstrap_tasks"].append(task)
                pr_replay_selected += 1
        report["pr_replay_attempts"].append(row)
        print(f"PR replay {case['id']}: {row}", flush=True)
        checkpoint()

    if args.preflight_pr_only:
        report["finished_at"] = _utc_now()
        checkpoint()
        return 0

    print("\n=== Stage 1A: bootstrap generation without trajectories ===", flush=True)
    bootstrap_domains = ["template_vars", "variable_merge", "yaml_loading", "plugin_config"]
    for attempt_index in range(args.max_bootstrap_agent_attempts):
        if len(report["bootstrap_tasks"]) >= args.bootstrap_target:
            break
        domain_id = bootstrap_domains[attempt_index % len(bootstrap_domains)]
        domain = DOMAINS[domain_id]
        blueprint = {
            "phase": "bootstrap",
            "capability_gap": "repository-wide localization and complete contract repair",
            "domain": domain_id,
            "contract": domain["contract"],
            "validation_command": _test_command(spec.test_command, domain["tests"]),
            "requirements": [
                "Use one shared behavioral contract across all changed files",
                "Make at least one existing test fail",
                "Keep unaffected tests passing",
            ],
        }
        task, row = _generate_repo_agent_task(
            phase="bootstrap",
            attempt_index=attempt_index,
            domain_id=domain_id,
            blueprint=blueprint,
            source_trajectory_ids=[],
            engine=engine,
            repo_spec=repo_spec,
            source_repo=source_repo,
            base_commit=spec.base_commit,
            test_prefix=spec.test_command,
            validator=validator,
            output_dir=output_dir,
            agent_src=root / "initial_agent" / "src",
            model=args.model,
            agent_timeout=args.agent_timeout,
            test_timeout=args.test_timeout,
        )
        report["bootstrap_generation_attempts"].append(row)
        if task:
            report["bootstrap_tasks"].append(task)
        print(f"Bootstrap repo-agent attempt {attempt_index + 1}: {row}", flush=True)
        checkpoint()

    coverage_domains = [
        "inventory_yaml_e2e",
        "task_conditionals",
        "inventory_model",
        "template_vars",
        "variable_merge",
        "yaml_loading",
        "plugin_config",
    ]
    for fallback_index, domain_id in enumerate(coverage_domains):
        if len(report["bootstrap_tasks"]) >= args.bootstrap_target:
            break
        domain = DOMAINS[domain_id]
        blueprint = {
            "phase": "bootstrap",
            "capability_gap": "repository-wide localization and complete contract repair",
            "domain": domain_id,
            "contract": domain["contract"],
            "validation_command": _test_command(spec.test_command, domain["tests"]),
            "generator": "coverage_guided_repo_compose",
        }
        task, row = _generate_coverage_guided_task(
            phase="bootstrap",
            attempt_index=fallback_index,
            domain_id=domain_id,
            blueprint=blueprint,
            source_trajectory_ids=[],
            source_repo=source_repo,
            base_commit=spec.base_commit,
            test_prefix=spec.test_command,
            validator=validator,
            output_dir=output_dir,
            test_timeout=args.test_timeout,
            seed=5200 + fallback_index,
        )
        report["bootstrap_generation_attempts"].append(row)
        if task:
            report["bootstrap_tasks"].append(task)
        print(
            f"Bootstrap coverage-guided attempt {fallback_index + 1}: {row}",
            flush=True,
        )
        checkpoint()

    print("\n=== Stage 1B: solver attempts bootstrap tasks ===", flush=True)
    for task in report["bootstrap_tasks"]:
        result = _run_solver(
            task=task,
            phase="bootstrap",
            source_repo=source_repo,
            agent_src=root / "initial_agent" / "src",
            adapter=solver_adapter,
            model=args.model,
            output_dir=output_dir,
            agent_timeout=args.agent_timeout,
            test_timeout=args.test_timeout,
        )
        report["bootstrap_solver_results"].append(result)
        print(
            f"Solver bootstrap {task['task_id']}: resolved={result['resolved']} "
            f"files={result['modified_files']} tools={result['tool_calls']}",
            flush=True,
        )
        checkpoint()

    if not report["bootstrap_solver_results"]:
        report["failure"] = "no_strict_bootstrap_tasks_generated"
        report["finished_at"] = _utc_now()
        checkpoint()
        print("No strict bootstrap task reached the solver; stopping.", flush=True)
        return 2

    print("\n=== Stage 2A: diagnose solver trajectories ===", flush=True)
    diagnosis = _diagnose_trajectories(
        solver_results=report["bootstrap_solver_results"],
        tasks=report["bootstrap_tasks"],
        model=args.model,
        vllm_host=args.vllm_host,
        vllm_port=str(args.vllm_port),
        target_count=args.adaptive_target,
        output_dir=output_dir,
    )
    report["trajectory_diagnosis"] = diagnosis
    checkpoint()

    print("\n=== Stage 2B: trajectory-conditioned generation ===", flush=True)
    diagnoses = list(diagnosis.get("diagnoses") or [])
    used_domains: set[str] = set()
    for attempt_index in range(args.max_adaptive_agent_attempts):
        if len(report["adaptive_tasks"]) >= args.adaptive_target:
            break
        diag = diagnoses[attempt_index % len(diagnoses)] if diagnoses else {}
        domain_id = _choose_adaptive_domain(diag, used_domains, attempt_index)
        used_domains.add(domain_id)
        domain = DOMAINS[domain_id]
        source_task_id = str(diag.get("task_id") or "")
        source_result = next(
            (r for r in report["bootstrap_solver_results"] if r["task_id"] == source_task_id),
            None,
        )
        trajectory_ids = [source_result["trajectory_path"]] if source_result else []
        blueprint = {
            "phase": "adaptive",
            "source_solver_task_id": source_task_id,
            "source_trajectory_ids": trajectory_ids,
            "failure_stage": diag.get("failure_stage", "patch_generation"),
            "capability_gap_code": diag.get("capability_gap_code", ""),
            "capability_gap": diag.get("capability_gap", "incomplete repository repair"),
            "failure_evidence": diag.get("evidence", ""),
            "transfer_requirement": (
                "Probe the same scaffold weakness in a different subsystem; do not "
                "copy identifiers or edits from the source task."
            ),
            "domain": domain_id,
            "contract": domain["contract"],
            "validation_command": _test_command(spec.test_command, domain["tests"]),
            "min_modified_files": _adaptive_min_modified_files(diag),
        }
        task, row = _generate_repo_agent_task(
            phase="adaptive",
            attempt_index=attempt_index,
            domain_id=domain_id,
            blueprint=blueprint,
            source_trajectory_ids=trajectory_ids,
            engine=engine,
            repo_spec=repo_spec,
            source_repo=source_repo,
            base_commit=spec.base_commit,
            test_prefix=spec.test_command,
            validator=validator,
            output_dir=output_dir,
            agent_src=root / "initial_agent" / "src",
            model=args.model,
            agent_timeout=args.agent_timeout,
            test_timeout=args.test_timeout,
        )
        report["adaptive_generation_attempts"].append(row)
        if task:
            report["adaptive_tasks"].append(task)
        print(f"Adaptive repo-agent attempt {attempt_index + 1}: {row}", flush=True)
        checkpoint()

    coverage_used_domains: set[str] = set()
    for fallback_index in range(max(len(DOMAINS), args.adaptive_target * 2)):
        if len(report["adaptive_tasks"]) >= args.adaptive_target:
            break
        diag = diagnoses[fallback_index % len(diagnoses)] if diagnoses else {}
        domain_id = _choose_adaptive_domain(
            diag,
            coverage_used_domains,
            fallback_index,
        )
        coverage_used_domains.add(domain_id)
        domain = DOMAINS[domain_id]
        source_task_id = str(diag.get("task_id") or "")
        source_result = next(
            (r for r in report["bootstrap_solver_results"] if r["task_id"] == source_task_id),
            None,
        )
        trajectory_ids = [source_result["trajectory_path"]] if source_result else []
        blueprint = {
            "phase": "adaptive",
            "source_solver_task_id": source_task_id,
            "source_trajectory_ids": trajectory_ids,
            "failure_stage": diag.get("failure_stage", "patch_generation"),
            "capability_gap_code": diag.get("capability_gap_code", ""),
            "capability_gap": diag.get("capability_gap", "incomplete repository repair"),
            "failure_evidence": diag.get("evidence", ""),
            "domain": domain_id,
            "contract": domain["contract"],
            "validation_command": _test_command(spec.test_command, domain["tests"]),
            "generator": "coverage_guided_repo_compose",
            "min_modified_files": _adaptive_min_modified_files(diag),
        }
        task, row = _generate_coverage_guided_task(
            phase="adaptive",
            attempt_index=fallback_index,
            domain_id=domain_id,
            blueprint=blueprint,
            source_trajectory_ids=trajectory_ids,
            source_repo=source_repo,
            base_commit=spec.base_commit,
            test_prefix=spec.test_command,
            validator=validator,
            output_dir=output_dir,
            test_timeout=args.test_timeout,
            seed=6200 + fallback_index,
        )
        report["adaptive_generation_attempts"].append(row)
        if task:
            report["adaptive_tasks"].append(task)
        print(
            f"Adaptive coverage-guided attempt {fallback_index + 1}: {row}",
            flush=True,
        )
        checkpoint()

    print("\n=== Stage 2C: solver attempts adaptive tasks ===", flush=True)
    for task in report["adaptive_tasks"]:
        result = _run_solver(
            task=task,
            phase="adaptive",
            source_repo=source_repo,
            agent_src=root / "initial_agent" / "src",
            adapter=solver_adapter,
            model=args.model,
            output_dir=output_dir,
            agent_timeout=args.agent_timeout,
            test_timeout=args.test_timeout,
        )
        report["adaptive_solver_results"].append(result)
        print(
            f"Solver adaptive {task['task_id']}: resolved={result['resolved']} "
            f"files={result['modified_files']} tools={result['tool_calls']}",
            flush=True,
        )
        checkpoint()

    report["finished_at"] = _utc_now()
    checkpoint()
    print("\n=== Final metrics ===", flush=True)
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False), flush=True)
    print(f"Report: {output_dir / 'closed_loop_report.json'}", flush=True)
    return 0


def _generate_coverage_guided_task(
    *,
    phase: str,
    attempt_index: int,
    domain_id: str,
    blueprint: dict,
    source_trajectory_ids: List[str],
    source_repo: Path,
    base_commit: str,
    test_prefix: str,
    validator: Any,
    output_dir: Path,
    test_timeout: int,
    seed: int,
) -> tuple[Optional[dict], dict]:
    from swesmith.candidate import CandidateArtifact
    from swesmith.repo_level import (
        RepositoryWorkspace,
        apply_repository_patch,
    )

    domain = DOMAINS[domain_id]
    required_components = min(
        len(domain["anchors"]),
        max(
            2,
            int(
                blueprint.get("min_modified_files")
                or domain.get("min_modified_files")
                or 2
            ),
        ),
    )
    contract_test = str(domain.get("contract_test") or "")
    attempt_dir = (
        output_dir
        / "generation"
        / phase
        / f"coverage_{attempt_index + 1}_{domain_id}"
    )
    attempt_dir.mkdir(parents=True, exist_ok=True)
    test_command = _test_command(test_prefix, domain["tests"])
    started = time.monotonic()
    deadline = started + 600
    searched = 0
    components: List[dict] = []
    anchor_sources: Dict[str, str] = {}
    clean_passed: set[str] = set()
    coupled_components: List[dict] = []
    contract_trace: Dict[str, Any] = {}
    dynamic_edges: List[Dict[str, Any]] = []
    traced_symbols_by_path: Dict[str, List[Dict[str, Any]]] = {}
    coupling: Dict[str, Any] = {
        "valid": False,
        "tier": "none",
        "shared_f2p_tests": [],
        "static_dependencies": [],
        "required_components": required_components,
    }

    with RepositoryWorkspace(str(source_repo), base_commit) as workspace:
        clean_result = _run_command(workspace, test_command, min(test_timeout, 120))
        clean_passed, _ = _pytest_status_sets(clean_result)
        if not clean_passed:
            row = {
                "attempt": attempt_index + 1,
                "phase": phase,
                "domain": domain_id,
                "strategy": "coverage_guided_repo_compose",
                "result": "clean_tests_unusable",
                "generation_runtime_sec": round(time.monotonic() - started, 3),
            }
            return None, row

        if contract_test:
            trace_path = attempt_dir / "clean_contract_trace.json"
            contract_trace = _trace_contract_test(
                workspace=Path(workspace),
                test_prefix=test_prefix,
                test_nodeid=contract_test,
                source_roots=list(domain.get("trace_source_roots") or ["lib"]),
                output_path=trace_path,
                timeout=min(test_timeout, 120),
            )
            if not contract_trace:
                row = {
                    "attempt": attempt_index + 1,
                    "phase": phase,
                    "domain": domain_id,
                    "strategy": "coverage_guided_repo_compose",
                    "result": "contract_trace_unusable",
                    "generation_runtime_sec": round(time.monotonic() - started, 3),
                }
                return None, row
            dynamic_edges = list(contract_trace.get("file_edges") or [])
            traced_symbols_by_path = {
                str(file_row.get("path") or ""): list(file_row.get("symbols") or [])
                for file_row in contract_trace.get("files") or []
            }

        for path_index, relative_path in enumerate(domain["anchors"]):
            if coupled_components or time.monotonic() >= deadline:
                break
            source_path = Path(workspace) / relative_path
            if not source_path.is_file() or source_path.suffix != ".py":
                continue
            original_source = source_path.read_text(encoding="utf-8")
            anchor_sources[relative_path] = original_source
            allowed_ranges = _traced_symbol_ranges(
                original_source,
                traced_symbols_by_path.get(relative_path, []),
            )
            if contract_test and not allowed_ranges:
                continue
            mutations = _minimal_mutation_candidates(
                original_source,
                relative_path,
                seed=seed + path_index,
                allowed_line_ranges=allowed_ranges or None,
            )
            active_for_path = 0
            for mutation in mutations[:48]:
                if time.monotonic() >= deadline:
                    break
                searched += 1
                patch = mutation["patch"]
                if not apply_repository_patch(workspace, patch):
                    continue
                mutation_trace: Dict[str, Any] = {}
                try:
                    result = _run_command(
                        workspace,
                        test_command,
                        min(test_timeout, 90),
                    )
                    passed, failed = _pytest_status_sets(result)
                    f2p = sorted(clean_passed & failed)
                    p2p_count = len(clean_passed & passed)
                    failure_fingerprints = _pytest_failure_fingerprints(
                        f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
                    )
                    if contract_test and contract_test in f2p:
                        mutation_trace = _trace_contract_test(
                            workspace=Path(workspace),
                            test_prefix=test_prefix,
                            test_nodeid=contract_test,
                            source_roots=list(
                                domain.get("trace_source_roots") or ["lib"]
                            ),
                            output_path=attempt_dir / "latest_mutation_trace.json",
                            timeout=min(test_timeout, 90),
                            allow_test_failure=True,
                        )
                finally:
                    reversed_ok = apply_repository_patch(
                        workspace,
                        patch,
                        reverse=True,
                    )
                if not reversed_ok:
                    break
                if (
                    f2p
                    and p2p_count
                    and (
                        not contract_test
                        or (
                            contract_test in f2p
                            and mutation_trace.get("pytest_returncode") not in (None, 0)
                        )
                    )
                ):
                    components.append(
                        {
                            "path": relative_path,
                            "patch": patch,
                            "operator": mutation["operator"],
                            "site": mutation["site"],
                            "f2p_tests": f2p,
                            "p2p_count": p2p_count,
                            "failure_fingerprint": failure_fingerprints.get(
                                contract_test, {}
                            ),
                            "runtime_execution_files": sorted(
                                str(row.get("path") or "")
                                for row in mutation_trace.get("files") or []
                                if row.get("path")
                            ),
                        }
                    )
                    active_for_path += 1
                    coupled_components, coupling = (
                        _select_semantically_coupled_components(
                            components,
                            anchor_sources,
                            required_components=required_components,
                            dynamic_edges=dynamic_edges,
                            required_shared_test=contract_test,
                        )
                    )
                    if coupled_components or active_for_path >= 6:
                        break

    selected_components = coupled_components

    row: Dict[str, Any] = {
        "attempt": attempt_index + 1,
        "phase": phase,
        "domain": domain_id,
        "strategy": "coverage_guided_repo_compose",
        "searched_mutations": searched,
        "covered_component_count": len(components),
        "active_components": [
            {
                "path": component["path"],
                "operator": component["operator"],
                "site": component["site"],
                "f2p_tests": component["f2p_tests"],
                "failure_fingerprint": component.get(
                    "failure_fingerprint", {}
                ),
                "runtime_execution_files": component.get(
                    "runtime_execution_files", []
                ),
            }
            for component in selected_components
        ],
        "semantic_coupling": coupling,
        "contract_test": contract_test,
        "contract_trace": _compact_contract_trace(
            contract_trace,
            paths=list(domain["anchors"]),
        ),
        "generation_runtime_sec": round(time.monotonic() - started, 3),
        "source_trajectory_ids": source_trajectory_ids,
    }
    search_path = attempt_dir / "coverage_search.json"
    search_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    if len(selected_components) < required_components:
        row["result"] = (
            "insufficient_semantic_coupling"
            if len({component["path"] for component in components}) >= 2
            else "insufficient_covered_files"
        )
        return None, row

    components = selected_components
    bug_patch = _join_patch_blocks(component["patch"] for component in components)
    digest = hashlib.sha256(
        f"{phase}:{domain_id}:{seed}:{bug_patch}".encode("utf-8")
    ).hexdigest()[:12]
    candidate = CandidateArtifact(
        candidate_id=f"cand_cov_{digest}",
        plan_id=f"{phase}_coverage_{attempt_index + 1}_{domain_id}",
        strategy="coverage_guided_repo_compose",
        operator="covered_multifile_compose",
        target_file=components[0]["path"],
        target_symbol="",
        bug_patch=bug_patch,
        mutation_site={
            "components": [
                {
                    key: value
                    for key, value in component.items()
                    if key != "patch"
                }
                for component in components
            ]
        },
        seed=seed,
        before_snippet="",
        after_snippet="",
        generation_metadata={
            "task_blueprint": blueprint,
            "searched_mutations": searched,
            "component_f2p_tests": {
                component["path"]: component["f2p_tests"]
                for component in components
            },
            "semantic_coupling": coupling,
            "required_components": required_components,
            "contract_test": contract_test,
            "contract_trace": _compact_contract_trace(
                contract_trace,
                paths=[component["path"] for component in components],
            ),
        },
        modified_files=[component["path"] for component in components],
        modified_entities=[component["site"] for component in components],
    )
    candidate.save(str(attempt_dir / candidate.candidate_id))
    problem = (
        f"A regression affects Ansible's {domain['description']}. Related layers "
        "now disagree on a shared behavioral contract. Diagnose every affected "
        "implementation point and restore the expected behavior without changing tests."
    )
    task = _validate_and_package(
        candidate=candidate,
        phase=phase,
        domain_id=domain_id,
        problem_statement=problem,
        test_files=domain["tests"],
        source_repo=source_repo,
        base_commit=base_commit,
        test_prefix=test_prefix,
        validator=validator,
        output_dir=output_dir,
        test_timeout=test_timeout,
        source_trajectory_ids=source_trajectory_ids,
    )
    row.update(
        {
            "candidate_id": candidate.candidate_id,
            "modified_files": candidate.modified_files,
            "validation_passed": bool(task),
            "strict_repo_level": bool(task and task.get("strict_repo_level")),
            "result": (
                "accepted"
                if task and task.get("strict_repo_level")
                else "cross_file_ablation_rejected"
                if task
                else "validation_rejected"
            ),
        }
    )
    return task if task and task.get("strict_repo_level") else None, row


def _select_semantically_coupled_components(
    components: List[dict],
    source_by_path: Dict[str, str],
    *,
    required_components: int = 2,
    dynamic_edges: Optional[List[Dict[str, Any]]] = None,
    required_shared_test: str = "",
) -> tuple[List[dict], Dict[str, Any]]:
    """Select a connected cross-file group exercised by the same failing test."""
    ranked: List[tuple[tuple[Any, ...], List[dict], dict]] = []
    for group_tuple in itertools.combinations(components, required_components):
        group = list(group_tuple)
        paths = [component["path"] for component in group]
        if len(set(paths)) != required_components:
            continue
        shared_tests = set(group[0].get("f2p_tests") or [])
        for component in group[1:]:
            shared_tests &= set(component.get("f2p_tests") or [])
        if not shared_tests:
            continue
        if required_shared_test and required_shared_test not in shared_tests:
            continue
        path_set = set(paths)
        if required_shared_test and any(
            not path_set.issubset(set(component.get("runtime_execution_files") or []))
            for component in group
        ):
            continue

        dependencies: List[Dict[str, str]] = []
        runtime_dependencies: List[Dict[str, Any]] = []
        adjacency = {path: set() for path in paths}
        for left, right in itertools.combinations(group, 2):
            dependency = _static_import_dependency(
                left["path"],
                source_by_path.get(left["path"], ""),
                right["path"],
                source_by_path.get(right["path"], ""),
            )
            if dependency:
                dependencies.append(dependency)
                adjacency[left["path"]].add(right["path"])
                adjacency[right["path"]].add(left["path"])
        for edge in dynamic_edges or []:
            caller = str(edge.get("caller") or "")
            callee = str(edge.get("callee") or "")
            if caller == callee or caller not in adjacency or callee not in adjacency:
                continue
            runtime_dependencies.append(
                {
                    "caller": caller,
                    "callee": callee,
                    "call_count": int(edge.get("call_count") or 0),
                }
            )
            adjacency[caller].add(callee)
            adjacency[callee].add(caller)
        reached = {paths[0]}
        frontier = [paths[0]]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current] - reached:
                reached.add(neighbor)
                frontier.append(neighbor)
        if len(reached) != required_components:
            continue

        sorted_shared = sorted(shared_tests)
        endpoint_fingerprints = {
            component["path"]: dict(component.get("failure_fingerprint") or {})
            for component in group
        }
        fingerprint_values = {
            str(row.get("fingerprint") or "")
            for row in endpoint_fingerprints.values()
            if row.get("fingerprint")
        }
        evidence = {
            "valid": True,
            "tier": "strong",
            "required_components": required_components,
            "shared_f2p_tests": sorted_shared,
            "static_dependencies": dependencies,
            "runtime_dependencies": runtime_dependencies,
            "per_component_runtime_coverage": {
                component["path"]: sorted(
                    path_set
                    & set(component.get("runtime_execution_files") or [])
                )
                for component in group
            },
            "endpoint_failure_fingerprints": endpoint_fingerprints,
            "shared_endpoint_failure": bool(
                len(fingerprint_values) == 1
                and len(endpoint_fingerprints) == len(group)
                and all(endpoint_fingerprints.values())
            ),
        }
        rank = (
            len(sorted_shared),
            min(component.get("p2p_count", 0) for component in group),
            tuple(sorted(paths)),
        )
        ranked.append((rank, group, evidence))
    if not ranked:
        return [], {
            "valid": False,
            "tier": "none",
            "required_components": required_components,
            "shared_f2p_tests": [],
            "static_dependencies": [],
            "runtime_dependencies": [],
        }
    _, group, evidence = max(ranked, key=lambda item: item[0])
    return group, evidence


def _static_import_dependency(
    left_path: str,
    left_source: str,
    right_path: str,
    right_source: str,
) -> Dict[str, str]:
    """Return the direct import edge between two modules, if one exists."""
    left_modules = _module_name_variants(left_path)
    right_modules = _module_name_variants(right_path)
    for source_path, source, target_path, target_modules in (
        (left_path, left_source, right_path, right_modules),
        (right_path, right_source, left_path, left_modules),
    ):
        for imported in _imported_modules(source, source_path):
            if any(
                imported == target
                or imported.startswith(f"{target}.")
                for target in target_modules
            ):
                return {
                    "importer": source_path,
                    "imported": target_path,
                    "module": imported,
                }
    return {}


def _trace_contract_test(
    *,
    workspace: Path,
    test_prefix: str,
    test_nodeid: str,
    source_roots: List[str],
    output_path: Path,
    timeout: int,
    allow_test_failure: bool = False,
) -> Dict[str, Any]:
    """Run one test with the runtime call-edge plugin enabled."""
    output_path.unlink(missing_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    traced_prefix = _prepend_pythonpath(test_prefix, project_root)
    environment = {
        "GODEL0_TRACE_OUTPUT": str(output_path.resolve()),
        "GODEL0_TRACE_REPO_ROOT": str(workspace.resolve()),
        "GODEL0_TRACE_SOURCE_ROOTS": os.pathsep.join(source_roots),
    }
    environment_text = " ".join(
        f"{name}={shlex.quote(value)}" for name, value in environment.items()
    )
    command = (
        f"{environment_text} {traced_prefix} "
        f"-p scripts.pytest_contract_trace {shlex.quote(test_nodeid)} -q"
    )
    result = _run_command(workspace, command, timeout)
    if (result["returncode"] != 0 and not allow_test_failure) or not output_path.is_file():
        return {}
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if payload.get("nodeid") != test_nodeid:
        return {}
    payload["pytest_returncode"] = result["returncode"]
    payload["pytest_stdout_tail"] = result["stdout"][-4000:]
    payload["pytest_stderr_tail"] = result["stderr"][-2000:]
    return payload


def _prepend_pythonpath(command: str, path: Path) -> str:
    tokens = shlex.split(command)
    prefix = str(path.resolve())
    for index, token in enumerate(tokens):
        if token.startswith("PYTHONPATH="):
            current = token.partition("=")[2]
            tokens[index] = f"PYTHONPATH={prefix}{os.pathsep}{current}"
            break
    else:
        tokens.insert(0, f"PYTHONPATH={prefix}")
    return shlex.join(tokens)


def _traced_symbol_ranges(
    source: str,
    traced_symbols: List[Dict[str, Any]],
) -> List[tuple[int, int]]:
    """Map runtime code objects back to editable function body ranges."""
    if not traced_symbols:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    requested_lines = {
        int(row.get("first_line") or 0)
        for row in traced_symbols
        if row.get("symbol") != "<module>"
    }
    requested_names = {
        str(row.get("symbol") or "").split(".<locals>", 1)[0].rsplit(".", 1)[-1]
        for row in traced_symbols
        if row.get("symbol") != "<module>"
    }
    ranges: List[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = int(getattr(node, "lineno", 0) or 0)
        decorator_start = min(
            [start]
            + [int(getattr(item, "lineno", start) or start) for item in node.decorator_list]
        )
        if start not in requested_lines and decorator_start not in requested_lines:
            if node.name not in requested_names:
                continue
        end = int(getattr(node, "end_lineno", start) or start)
        ranges.append((start, end))
    return sorted(set(ranges))


def _compact_contract_trace(
    trace: Dict[str, Any],
    *,
    paths: List[str],
) -> Dict[str, Any]:
    if not trace:
        return {}
    selected = set(paths)
    return {
        "nodeid": trace.get("nodeid", ""),
        "pytest_returncode": trace.get("pytest_returncode"),
        "files": [
            row for row in trace.get("files") or [] if row.get("path") in selected
        ],
        "file_edges": [
            row
            for row in trace.get("file_edges") or []
            if row.get("caller") in selected and row.get("callee") in selected
        ],
    }


def _module_name_variants(path: str) -> set[str]:
    parts = list(Path(path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    variants = {".".join(parts)} if parts else set()
    if parts and parts[0] in {"lib", "src"}:
        variants.add(".".join(parts[1:]))
    return {value for value in variants if value}


def _imported_modules(source: str, source_path: str) -> set[str]:
    if not source:
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    source_modules = sorted(
        _module_name_variants(source_path),
        key=lambda value: (value.count("."), len(value)),
    )
    source_module = source_modules[0] if source_modules else ""
    is_package = Path(source_path).name == "__init__.py"
    package_parts = source_module.split(".") if source_module else []
    if not is_package and package_parts:
        package_parts.pop()

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                keep = max(0, len(package_parts) - node.level + 1)
                prefix = package_parts[:keep]
                value = ".".join(prefix + ([module] if module else []))
            else:
                value = module
            if value:
                imported.add(value)
    return imported


def _minimal_mutation_candidates(
    source: str,
    relative_path: str,
    *,
    seed: int,
    allowed_line_ranges: Optional[List[tuple[int, int]]] = None,
) -> List[dict]:
    from swesmith.operators import get_operator
    from swesmith.patch_utils import count_modified_lines, make_git_diff

    candidates: List[dict] = []
    seen_patches: set[str] = set()

    def line_allowed(line: int) -> bool:
        return not allowed_line_ranges or any(
            start <= line <= end for start, end in allowed_line_ranges
        )

    def add(mutated: str, operator: str, site: str) -> None:
        if mutated == source:
            return
        try:
            ast.parse(mutated)
        except SyntaxError:
            return
        patch = make_git_diff(source, mutated, filename=relative_path)
        if not patch or count_modified_lines(patch) > 12:
            return
        digest = hashlib.sha256(patch.encode("utf-8")).hexdigest()
        if digest in seen_patches:
            return
        seen_patches.add(digest)
        candidates.append(
            {
                "patch": patch,
                "mutated_source": mutated,
                "operator": operator,
                "site": site,
                "digest": digest,
            }
        )

    operator = get_operator("change_operator")
    for site in operator.enumerate_sites(source):
        if not line_allowed(site.line):
            continue
        add(operator.apply(source, site), "change_operator", site.site_id)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines(keepends=True)
    parent_by_id: Dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_by_id[id(child)] = parent
    for node in ast.walk(tree):
        line = int(getattr(node, "lineno", 0) or 0)
        if not line_allowed(line):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            replacement = "False" if node.value else "True"
            mutated = _replace_single_line_node(lines, node, replacement)
            if mutated:
                add(
                    mutated,
                    "invert_boolean",
                    f"line:{getattr(node, 'lineno', 0)}",
                )
        elif isinstance(node, ast.If) and not isinstance(node.test, ast.Constant):
            segment = ast.get_source_segment(source, node.test)
            if not segment or "\n" in segment:
                continue
            replacement = (
                segment[4:].strip()
                if segment.startswith("not ")
                else f"not ({segment})"
            )
            mutated = _replace_single_line_node(lines, node.test, replacement)
            if mutated:
                add(
                    mutated,
                    "invert_condition",
                    f"line:{getattr(node, 'lineno', 0)}",
                )
        elif isinstance(node, ast.Return) and node.value is not None:
            segment = ast.get_source_segment(source, node.value)
            if not segment or "\n" in segment:
                continue
            replacements = ["None"]
            if isinstance(node.value, ast.Call) and node.value.args:
                first_argument = ast.get_source_segment(source, node.value.args[0])
                if first_argument and "\n" not in first_argument:
                    replacements.insert(0, first_argument)
            for replacement in replacements:
                mutated = _replace_single_line_node(lines, node.value, replacement)
                if mutated:
                    add(
                        mutated,
                        "return_unwrap" if replacement != "None" else "return_none",
                        f"line:{line}",
                    )
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            iterable = ast.get_source_segment(source, node.iter)
            if iterable and "\n" not in iterable:
                mutated = _replace_single_line_node(
                    lines,
                    node.iter,
                    f"list({iterable})[:1]",
                )
                if mutated:
                    add(mutated, "limit_iteration", f"line:{line}")
        elif isinstance(node, ast.Call):
            parent = parent_by_id.get(id(node))
            if isinstance(parent, ast.Expr):
                mutated = _replace_single_line_node(lines, node, "None")
                if mutated:
                    add(mutated, "skip_call", f"line:{line}")
            for keyword in node.keywords:
                if keyword.arg is None or isinstance(keyword.value, ast.Constant):
                    continue
                mutated = _replace_single_line_node(lines, keyword.value, "None")
                if mutated:
                    add(
                        mutated,
                        "neutralize_keyword",
                        f"line:{getattr(keyword.value, 'lineno', line)}:{keyword.arg}",
                    )
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and 0 < len(node.value) <= 20
            and "\n" not in node.value
            and not isinstance(parent_by_id.get(id(node)), ast.Expr)
        ):
            mutated = _replace_single_line_node(lines, node, repr(node.value + "_x"))
            if mutated:
                add(mutated, "change_string_contract", f"line:{line}")

    priorities = {
        "neutralize_keyword": 0,
        "limit_iteration": 1,
        "change_operator": 2,
        "invert_boolean": 2,
        "invert_condition": 3,
        "change_string_contract": 4,
        "skip_call": 5,
        "return_unwrap": 6,
        "return_none": 7,
    }
    candidates.sort(
        key=lambda item: (
            priorities.get(item["operator"], 10),
            hashlib.sha256(
                f"{seed}:{relative_path}:{item['digest']}".encode("utf-8")
            ).hexdigest(),
        )
    )
    return candidates


def _replace_single_line_node(
    lines: List[str],
    node: ast.AST,
    replacement: str,
) -> str:
    line_number = getattr(node, "lineno", 0)
    end_line_number = getattr(node, "end_lineno", 0)
    start = getattr(node, "col_offset", -1)
    end = getattr(node, "end_col_offset", -1)
    if (
        line_number < 1
        or line_number != end_line_number
        or start < 0
        or end < start
        or line_number > len(lines)
    ):
        return ""
    mutated_lines = list(lines)
    line = mutated_lines[line_number - 1]
    mutated_lines[line_number - 1] = line[:start] + replacement + line[end:]
    return "".join(mutated_lines)


def _join_patch_blocks(blocks: Iterable[str]) -> str:
    """Join complete unified diffs without trimming meaningful context lines."""
    return "".join(block if block.endswith("\n") else block + "\n" for block in blocks)


def _causal_patch_shortcut_reasons(patch: str) -> List[str]:
    """Reject mechanical mutations that only manufacture multi-file breadth."""
    added = [
        line[1:].strip()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    removed = [
        line[1:].strip()
        for line in patch.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    reasons: set[str] = set()
    for line in added:
        if re.search(r"\blist\(.+\)\s*\[:\s*1\s*\]", line):
            reasons.add("arbitrary_iteration_truncation")
        if re.match(r"return\s+None(?:\s*#.*)?$", line):
            reasons.add("direct_return_none")
        if re.search(r"\b(?:test1|test2|host1|host2)\b", line, re.IGNORECASE):
            reasons.add("hard_coded_contract_fixture")
        if line.startswith("if not (") and line.endswith("):"):
            original = "if " + line[len("if not (") : -2] + ":"
            if original in removed:
                reasons.add("mechanical_condition_inversion")
        assignment = re.match(r"(?P<lhs>[A-Za-z_][\w.]*)\s*=\s*(?P<value>True|False)$", line)
        if assignment:
            opposite = "False" if assignment.group("value") == "True" else "True"
            if f"{assignment.group('lhs')} = {opposite}" in removed:
                reasons.add("mechanical_boolean_flip")
        for keyword in re.findall(r"\b([A-Za-z_]\w*)=None\b", line):
            if any(
                re.search(rf"\b{re.escape(keyword)}=(?!None\b)[A-Za-z_]\w*", old)
                for old in removed
            ):
                reasons.add("keyword_value_erasure")
    return sorted(reasons)


def _pytest_status_sets(
    result: dict,
    repo_path: Optional[Union[str, Path]] = None,
) -> tuple[set[str], set[str]]:
    passed: set[str] = set()
    failed: set[str] = set()
    combined = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    repo_prefix = str(Path(repo_path).resolve()) if repo_path else ""
    for raw_line in combined.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw_line).strip()
        if "::" not in line:
            continue
        verbose = re.match(
            r"^(?P<nodeid>.+?::.+?)\s+"
            r"(?P<status>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)"
            r"(?:\s+\[[^]]*\])?$",
            line,
        )
        summary = re.match(
            r"^(?P<status>FAILED|ERROR)\s+"
            r"(?P<nodeid>.+?::.+?)(?:\s+-\s+.*)?$",
            line,
        )
        match = verbose or summary
        if not match:
            continue
        node_id = match.group("nodeid").rstrip(":-")
        if repo_prefix:
            node_id = node_id.replace(repo_prefix, "<REPO>")
        status = match.group("status")
        if status == "PASSED" and node_id:
            passed.add(node_id)
        elif status in {"FAILED", "ERROR"} and node_id:
            failed.add(node_id)
    return passed, failed


def _pytest_failure_fingerprints(output: str) -> Dict[str, Dict[str, Any]]:
    """Extract stable endpoint locations from verbose pytest failure output."""
    clean = re.sub(r"\x1b\[[0-9;]*m", "", output)
    failed_nodeids: List[str] = []
    for raw_line in clean.splitlines():
        line = raw_line.strip()
        verbose = re.match(
            r"^(?P<nodeid>.+?::.+?)\s+(?:FAILED|ERROR)(?:\s+\[[^]]*\])?$",
            line,
        )
        summary = re.match(
            r"^(?:FAILED|ERROR)\s+(?P<nodeid>.+?::.+?)(?:\s+-\s+.*)?$",
            line,
        )
        match = verbose or summary
        if match:
            nodeid = match.group("nodeid").rstrip(":-")
            if nodeid not in failed_nodeids:
                failed_nodeids.append(nodeid)

    header_pattern = re.compile(r"^_{3,}\s+(.+?)\s+_{3,}\s*$", re.MULTILINE)
    headers = list(header_pattern.finditer(clean))
    blocks: List[tuple[str, str]] = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(clean)
        blocks.append((header.group(1), clean[header.end() : end]))

    fingerprints: Dict[str, Dict[str, Any]] = {}
    for nodeid in failed_nodeids:
        leaf = nodeid.rsplit("::", 1)[-1].split("[", 1)[0]
        block = next(
            (body for title, body in blocks if leaf and leaf in title),
            "",
        )
        locations = re.findall(
            r"^([^\s:][^:\n]*\.py):(\d+):.*$",
            block,
            flags=re.MULTILINE,
        )
        test_locations = [
            (path, line)
            for path, line in locations
            if path.startswith(("test/", "tests/"))
        ]
        location = (test_locations or locations or [("", "0")])[-1]
        error_lines = [
            line[4:].strip()
            for line in block.splitlines()
            if line.startswith("E   ")
        ]
        error_type = "Failure"
        for error_line in error_lines:
            match = re.match(
                r"(?P<type>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))(?::|$)",
                error_line,
            )
            if match:
                error_type = match.group("type")
                break
        assertion_lines = [
            line[1:].strip()
            for line in block.splitlines()
            if line.startswith(">")
        ]
        assertion = assertion_lines[-1] if assertion_lines else ""
        fingerprint = f"{location[0]}:{location[1]}:{error_type}"
        fingerprints[nodeid] = {
            "nodeid": nodeid,
            "test_location": f"{location[0]}:{location[1]}",
            "error_type": error_type,
            "assertion": assertion[:500],
            "message": "\n".join(error_lines[:3])[:1000],
            "fingerprint": fingerprint,
        }
    return fingerprints


def _command_failure_fingerprints(
    output: str,
    test_id: str,
) -> Dict[str, Dict[str, Any]]:
    """Extract a stable endpoint signal from an exit-code regression command."""
    clean = re.sub(r"\x1b\[[0-9;]*m", "", output)
    marker = re.compile(
        r"^GODEL0_CONTRACT\s+(?P<check>\S+)\s+"
        r"expected=(?P<expected>\S+)\s+actual=(?P<actual>\S+)\s+"
        r"command_rc=(?P<command_rc>-?\d+)\s*$",
        re.MULTILINE,
    )
    checks = [
        {
            "check": match.group("check"),
            "expected": match.group("expected"),
            "actual": match.group("actual"),
            "command_rc": int(match.group("command_rc")),
        }
        for match in marker.finditer(clean)
    ]
    failed_checks = [
        row
        for row in checks
        if row["expected"] != row["actual"] or row["command_rc"] != 0
    ]
    if failed_checks:
        canonical = json.dumps(
            failed_checks,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            test_id: {
                "nodeid": test_id,
                "kind": "command_contract",
                "failed_checks": failed_checks,
                "fingerprint": (
                    f"{test_id}:"
                    f"{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"
                ),
            }
        }

    meaningful = []
    for raw_line in clean.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line or line.startswith("[WARNING]"):
            continue
        if re.search(
            r"(?:\bfatal:|\bfailed!?\b|\berror!?\b|traceback|"
            r"undefined|assertionerror)",
            line,
            re.IGNORECASE,
        ):
            meaningful.append(line)
    if not meaningful:
        return {}
    evidence = meaningful[:8]
    canonical = "\n".join(evidence)
    return {
        test_id: {
            "nodeid": test_id,
            "kind": "command_error",
            "evidence": evidence,
            "fingerprint": (
                f"{test_id}:"
                f"{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"
            ),
        }
    }


def _evaluate_expected_pytest_tests(
    *,
    expected_tests: set[str],
    result: dict,
    repo_path: Optional[Union[str, Path]] = None,
) -> tuple[bool, List[str], List[str]]:
    """Evaluate only tests that passed on the clean candidate baseline."""
    passed, failed = _pytest_status_sets(result, repo_path)
    missing = sorted(expected_tests - passed)
    failed_expected = sorted(expected_tests & failed)
    return bool(expected_tests and not missing and not failed_expected), missing, failed_expected


def _generate_repo_agent_task(
    *,
    phase: str,
    attempt_index: int,
    domain_id: str,
    blueprint: dict,
    source_trajectory_ids: List[str],
    engine: Any,
    repo_spec: Any,
    source_repo: Path,
    base_commit: str,
    test_prefix: str,
    validator: Any,
    output_dir: Path,
    agent_src: Path,
    model: str,
    agent_timeout: int,
    test_timeout: int,
) -> tuple[Optional[dict], dict]:
    from swesmith.engine import BugConstraints, BugGenerationPlan
    from godel0.git.patch import extract_changed_files

    domain = DOMAINS[domain_id]
    min_modified_files = max(
        2,
        int(
            blueprint.get("min_modified_files")
            or domain.get("min_modified_files")
            or 2
        ),
    )
    attempt_dir = output_dir / "generation" / phase / f"attempt_{attempt_index + 1}_{domain_id}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    plan = BugGenerationPlan(
        plan_id=f"{phase}_{attempt_index + 1}_{domain_id}",
        source_trajectory_ids=source_trajectory_ids,
        target_repo_id=repo_spec.repo_id,
        target_base_commit=base_commit,
        target_file=domain["anchors"][0],
        target_files=list(domain["anchors"]),
        strategy="repo_agent",
        operator="repository_contract_mutation",
        constraints=BugConstraints(
            min_modified_files=min_modified_files,
            max_modified_files=max(
                min_modified_files,
                min(6, len(domain["anchors"])),
            ),
            max_modified_lines=100,
            allow_test_edits=False,
            desired_behavior=str(blueprint.get("capability_gap") or domain["description"]),
            generation_timeout_sec=agent_timeout,
        ),
        task_blueprint=blueprint,
        model=model,
        seed=4200 + attempt_index,
    )
    started = time.time()
    candidates = engine.generate(plan, str(agent_src), repo_spec, str(attempt_dir))
    row: Dict[str, Any] = {
        "attempt": attempt_index + 1,
        "phase": phase,
        "domain": domain_id,
        "generated": len(candidates),
        "generation_runtime_sec": round(time.time() - started, 3),
        "source_trajectory_ids": source_trajectory_ids,
    }
    if not candidates:
        raw_patch_path = attempt_dir / "repo_agent_run" / "model_patch.diff"
        raw_patch = raw_patch_path.read_text(encoding="utf-8") if raw_patch_path.exists() else ""
        row.update({
            "result": "generator_rejected",
            "raw_modified_files": extract_changed_files(raw_patch),
            "raw_patch_chars": len(raw_patch),
        })
        return None, row

    candidate = candidates[0]
    problem = (
        f"A regression affects Ansible's {domain['description']}. The behavior is "
        "inconsistent across components that share this contract. Diagnose the "
        "repository-level cause and restore expected behavior without modifying tests."
    )
    task = _validate_and_package(
        candidate=candidate,
        phase=phase,
        domain_id=domain_id,
        problem_statement=problem,
        test_files=domain["tests"],
        source_repo=source_repo,
        base_commit=base_commit,
        test_prefix=test_prefix,
        validator=validator,
        output_dir=output_dir,
        test_timeout=test_timeout,
        source_trajectory_ids=source_trajectory_ids,
    )
    row.update({
        "candidate_id": candidate.candidate_id,
        "modified_files": candidate.modified_files,
        "modified_lines": candidate.generation_metadata.get("modified_lines"),
        "validation_passed": bool(task),
        "strict_repo_level": bool(task and task.get("strict_repo_level")),
        "result": (
            "accepted"
            if task and task.get("strict_repo_level")
            else "cross_file_ablation_rejected"
            if task
            else "validation_rejected"
        ),
    })
    return task if task and task.get("strict_repo_level") else None, row


def _validate_and_package(
    *,
    candidate: Any,
    phase: str,
    domain_id: str,
    problem_statement: str,
    test_files: List[str],
    source_repo: Path,
    base_commit: str,
    test_prefix: str,
    validator: Any,
    output_dir: Path,
    test_timeout: int,
    source_trajectory_ids: List[str],
    test_command_override: str = "",
    validation_mode: str = "pytest",
    command_test_id: str = "",
    control_test_command: Optional[str] = None,
    strictness_policy: str = "synthetic_causal",
    adversarial_solver_patches: Optional[List[Dict[str, str]]] = None,
    solver_test_command_override: str = "",
) -> Optional[dict]:
    from godel0.schemas.evaluation import CandidateValidationReport

    generation_metadata = dict(
        getattr(candidate, "generation_metadata", {}) or {}
    )
    generated_test_patch = str(
        generation_metadata.get("generated_test_patch") or ""
    )
    generated_test_files = list(
        generation_metadata.get("generated_test_files") or []
    )
    generated_test_command = str(
        generation_metadata.get("generated_test_command") or ""
    )
    oracle_patch = str(generation_metadata.get("oracle_patch") or "")
    test_command = (
        test_command_override
        or generated_test_command
        or _test_command(test_prefix, test_files)
    )
    task_dir = output_dir / "tasks" / candidate.candidate_id
    task_dir.mkdir(parents=True, exist_ok=True)
    bug_patch_path = task_dir / "bug.patch"
    validation_path = task_dir / "validation.json"
    validation_cache_path = task_dir / "validation_cache.json"
    bugged_output_path = task_dir / "bugged_test_output.txt"
    ablation_path = task_dir / "ablation.json"
    adversarial_path = task_dir / "adversarial_solver_patches.json"
    generated_test_patch_path = task_dir / "contract.patch"
    oracle_patch_path = task_dir / "oracle.patch"
    adversarial_solver_patches = list(adversarial_solver_patches or [])
    validation_inputs = {
        "schema_version": 2,
        "candidate_patch_sha256": hashlib.sha256(
            candidate.bug_patch.encode("utf-8")
        ).hexdigest(),
        "base_commit": base_commit,
        "test_command": test_command,
        "validation_mode": validation_mode,
        "command_test_id": command_test_id,
        "control_test_command": control_test_command or "",
        "generated_test_patch_sha256": hashlib.sha256(
            generated_test_patch.encode("utf-8")
        ).hexdigest(),
    }
    validation_cache_key = hashlib.sha256(
        json.dumps(
            validation_inputs,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    validation = None
    validation_reused = False
    if (
        bug_patch_path.exists()
        and validation_path.exists()
        and validation_cache_path.exists()
    ):
        existing_patch = bug_patch_path.read_text(encoding="utf-8")
        if existing_patch == candidate.bug_patch:
            try:
                cache_metadata = json.loads(
                    validation_cache_path.read_text(encoding="utf-8")
                )
                if cache_metadata.get("cache_key") == validation_cache_key:
                    cached = CandidateValidationReport.model_validate_json(
                        validation_path.read_text(encoding="utf-8")
                    )
                    if cached.passed:
                        validation = cached
                        validation_reused = True
            except (ValueError, OSError):
                pass

    bug_patch_path.write_text(candidate.bug_patch, encoding="utf-8")
    if generated_test_patch:
        generated_test_patch_path.write_text(
            generated_test_patch, encoding="utf-8"
        )
    else:
        generated_test_patch_path.unlink(missing_ok=True)
    if oracle_patch:
        oracle_patch_path.write_text(oracle_patch, encoding="utf-8")
    else:
        oracle_patch_path.unlink(missing_ok=True)
    (task_dir / "problem_statement.md").write_text(
        problem_statement.rstrip() + "\n",
        encoding="utf-8",
    )
    if validation is None:
        bugged_output_path.unlink(missing_ok=True)
        ablation_path.unlink(missing_ok=True)
        adversarial_path.unlink(missing_ok=True)
        validation = validator.validate(
            candidate_patch=candidate.bug_patch,
            repo_path=source_repo,
            base_commit=base_commit,
            test_command=test_command,
            candidate_id=candidate.candidate_id,
            repo_id="ansible",
            target_file=candidate.target_file,
            target_symbol="",
            operator=candidate.operator,
            validation_mode=validation_mode,
            command_test_id=command_test_id,
            control_test_command=control_test_command,
            setup_patch=generated_test_patch,
        )
    validation_path.write_text(
        json.dumps(validation.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    validation_cache_path.write_text(
        json.dumps(
            {
                "cache_key": validation_cache_key,
                "inputs": validation_inputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not validation.passed:
        return None

    if not bugged_output_path.exists():
        bugged_test = _run_patch_tests(
            source_repo,
            base_commit,
            candidate.bug_patch,
            test_command,
            timeout=test_timeout,
            setup_patch=generated_test_patch,
        )
        bugged_output_path.write_text(
            bugged_test["stdout"] + "\n" + bugged_test["stderr"],
            encoding="utf-8",
        )
    bugged_output_text = bugged_output_path.read_text(encoding="utf-8")
    failure_fingerprints = _pytest_failure_fingerprints(bugged_output_text)
    primary_command_test_id = command_test_id or "command::primary"
    if validation_mode == "exit_code":
        failure_fingerprints.update(
            _command_failure_fingerprints(
                bugged_output_text,
                primary_command_test_id,
            )
        )
    ablation = _file_ablation(
        source_repo=source_repo,
        base_commit=base_commit,
        bug_patch=candidate.bug_patch,
        f2p_tests=validation.f2p_tests,
        test_prefix=test_prefix,
        test_command=test_command,
        validation_mode=validation_mode,
        timeout=test_timeout,
        checkpoint_path=ablation_path,
        setup_patch=generated_test_patch,
    )
    ablation_path.write_text(json.dumps(ablation, indent=2), encoding="utf-8")
    adversarial_evaluation = _adversarial_solver_patch_results(
        source_repo=source_repo,
        base_commit=base_commit,
        bug_patch=candidate.bug_patch,
        solver_patches=adversarial_solver_patches,
        test_command=test_command,
        timeout=test_timeout,
        checkpoint_path=adversarial_path,
        setup_patch=generated_test_patch,
    )
    adversarial_path.write_text(
        json.dumps(adversarial_evaluation, indent=2),
        encoding="utf-8",
    )
    adversarial_resistance_valid = bool(
        adversarial_evaluation.get("valid")
        and adversarial_evaluation.get("all_rejected")
    )
    semantic_coupling = dict(
        generation_metadata.get("semantic_coupling") or {}
    )
    task_blueprint = dict(generation_metadata.get("task_blueprint") or {})
    contract_test = str(
        generation_metadata.get("contract_test")
        or task_blueprint.get("contract_test")
        or ""
    )
    bugged_contract_trace: Dict[str, Any] = {}
    runtime_contract_valid = True
    if contract_test:
        bugged_contract_trace = _trace_patch_contract_test(
            source_repo=source_repo,
            base_commit=base_commit,
            patch=candidate.bug_patch,
            test_prefix=test_prefix,
            test_nodeid=contract_test,
            source_roots=list(
                DOMAINS.get(domain_id, {}).get("trace_source_roots") or ["lib"]
            ),
            output_path=task_dir / "bugged_contract_trace.json",
            timeout=min(test_timeout, 120),
            setup_patch=generated_test_patch,
        )
        executed_paths = {
            str(row.get("path") or "")
            for row in bugged_contract_trace.get("files") or []
        }
        runtime_contract_valid = bool(
            bugged_contract_trace
            and bugged_contract_trace.get("pytest_returncode") not in (None, 0)
            and set(candidate.modified_files).issubset(executed_paths)
        )
    if validation_mode == "exit_code":
        endpoint_failure_valid = bool(
            failure_fingerprints.get(primary_command_test_id)
        )
    else:
        endpoint_failure_valid = bool(
            not contract_test
            or failure_fingerprints.get(contract_test)
            or (
                generated_test_files
                and any(
                    any(
                        nodeid == path or nodeid.startswith(f"{path}::")
                        for path in generated_test_files
                    )
                    for nodeid in failure_fingerprints
                )
            )
        )
    causal_shortcut_reasons = (
        _causal_patch_shortcut_reasons(candidate.bug_patch)
        if contract_test
        else []
    )
    causal_patch_quality_valid = not causal_shortcut_reasons
    coupling_required = candidate.strategy in {
        "coverage_guided_repo_compose",
        "repo_chain",
    }
    historical_provenance_valid = bool(
        candidate.strategy == "pr_replay"
        and generation_metadata.get("reference_kind") == "commit"
        and (getattr(candidate, "mutation_site", {}) or {}).get(
            "reference_commit"
        )
        and len(candidate.modified_files) >= 2
    )
    solver_test_command = solver_test_command_override or _solver_evaluation_command(
        test_prefix=test_prefix,
        original_test_command=test_command,
        validation_mode=validation_mode,
        f2p_tests=list(validation.f2p_tests),
        p2p_tests=list(validation.p2p_tests),
    )
    solver_validation_command = (
        test_command if solver_test_command_override else solver_test_command
    )
    reference_test_files = list(
        generation_metadata.get("reference_test_files") or []
    )
    reference_hidden_files = list(
        generation_metadata.get("reference_hidden_files")
        or reference_test_files
    )
    all_solver_hidden_files = list(
        dict.fromkeys(reference_hidden_files + generated_test_files)
    )
    solver_test_visibility_valid = bool(
        not all_solver_hidden_files
        or (
            solver_test_command != solver_validation_command
            and all(
                path not in solver_test_command
                and Path(path).name not in solver_test_command
                for path in all_solver_hidden_files
            )
            and "GODEL0_ROLE_INSTANCE_CONTRACT_PLAYBOOK"
            not in solver_test_command
        )
    )
    solver_runtime_isolation_valid = bool(
        str(output_dir) not in solver_test_command
        and "ansible_runtime/contracts" not in solver_test_command
    )
    task = {
        "task_id": candidate.candidate_id,
        "candidate_id": candidate.candidate_id,
        "phase": phase,
        "domain": domain_id,
        "strategy": candidate.strategy,
        "base_commit": base_commit,
        "problem_statement": problem_statement,
        "bug_patch_path": str(task_dir / "bug.patch"),
        "bug_patch": candidate.bug_patch,
        "oracle_patch": oracle_patch,
        "oracle_patch_path": str(oracle_patch_path) if oracle_patch else "",
        "modified_files": list(candidate.modified_files),
        "modified_entities": list(candidate.modified_entities),
        "test_command": test_command,
        "solver_test_command": solver_test_command,
        "solver_validation_command": solver_validation_command,
        "solver_hidden_test_files": reference_test_files,
        "solver_hidden_reference_files": reference_hidden_files,
        "generated_test_patch": generated_test_patch,
        "generated_test_patch_path": (
            str(generated_test_patch_path) if generated_test_patch else ""
        ),
        "generated_test_files": generated_test_files,
        "solver_test_visibility_valid": solver_test_visibility_valid,
        "solver_runtime_isolation_valid": solver_runtime_isolation_valid,
        "control_test_command": control_test_command or "",
        "command_test_id": command_test_id,
        "validation_mode": validation_mode,
        "f2p_tests": list(validation.f2p_tests),
        "p2p_tests": list(validation.p2p_tests),
        "ablation": ablation,
        "semantic_coupling": semantic_coupling,
        "contract_test": contract_test,
        "bugged_contract_trace": _compact_contract_trace(
            bugged_contract_trace,
            paths=list(candidate.modified_files),
        ),
        "runtime_contract_valid": runtime_contract_valid,
        "failure_fingerprints": failure_fingerprints,
        "endpoint_failure_valid": endpoint_failure_valid,
        "causal_patch_quality_valid": causal_patch_quality_valid,
        "causal_shortcut_reasons": causal_shortcut_reasons,
        "strictness_policy": strictness_policy,
        "historical_provenance_valid": historical_provenance_valid,
        "adversarial_solver_patch_evaluation": adversarial_evaluation,
        "adversarial_resistance_valid": adversarial_resistance_valid,
        "reference_commit": str(
            (getattr(candidate, "mutation_site", {}) or {}).get(
                "reference_commit"
            )
            or ""
        ),
        "reference_parent": str(
            (getattr(candidate, "mutation_site", {}) or {}).get(
                "reference_parent"
            )
            or ""
        ),
        "reference_patch_sha256": str(
            generation_metadata.get("reference_patch_sha256") or ""
        ),
        "modified_lines": int(generation_metadata.get("modified_lines") or 0),
        "generation_metadata": generation_metadata,
        "strict_repo_level": _strict_repo_level_valid(
            strictness_policy=strictness_policy,
            ablation=ablation,
            runtime_contract_valid=runtime_contract_valid,
            endpoint_failure_valid=endpoint_failure_valid,
            causal_patch_quality_valid=causal_patch_quality_valid,
            coupling_required=coupling_required,
            semantic_coupling=semantic_coupling,
            historical_provenance_valid=historical_provenance_valid,
            adversarial_resistance_valid=adversarial_resistance_valid,
            solver_test_visibility_valid=solver_test_visibility_valid,
            solver_runtime_isolation_valid=solver_runtime_isolation_valid,
        ),
        "source_trajectory_ids": source_trajectory_ids,
        "validation_reused": validation_reused,
    }
    (task_dir / "task.json").write_text(json.dumps(task, indent=2), encoding="utf-8")
    if oracle_patch:
        swebench_task = {
            "schema_version": "godel0.swebench_like.v1",
            "instance_id": candidate.candidate_id,
            "repo": "ansible",
            "base_commit": base_commit,
            "problem_statement": problem_statement,
            "setup_patch": candidate.bug_patch,
            "patch": oracle_patch,
            "test_patch": generated_test_patch,
            "FAIL_TO_PASS": list(validation.f2p_tests),
            "PASS_TO_PASS": list(validation.p2p_tests),
        }
        (task_dir / "swebench_task.json").write_text(
            json.dumps(swebench_task, indent=2), encoding="utf-8"
        )
    return task


def _strict_repo_level_valid(
    *,
    strictness_policy: str,
    ablation: Dict[str, Any],
    runtime_contract_valid: bool,
    endpoint_failure_valid: bool,
    causal_patch_quality_valid: bool,
    coupling_required: bool,
    semantic_coupling: Dict[str, Any],
    historical_provenance_valid: bool,
    adversarial_resistance_valid: bool = True,
    solver_test_visibility_valid: bool = True,
    solver_runtime_isolation_valid: bool = True,
) -> bool:
    """Apply topology-aware evidence requirements to a validated task.

    Synthetic composition must prove that every injected mutation is active.
    A replayed human fix instead preserves the original PR topology and must
    prove that no single oracle file repair resolves the regression. Requiring
    every reversed PR file to fail in isolation would reject legitimate API
    migrations and favor artificial collections of independent bugs.
    """
    if strictness_policy not in {"synthetic_causal", "historical_pr"}:
        return False
    common_valid = bool(
        ablation.get("ablation_valid")
        and not ablation.get("any_single_file_oracle_fix_passes")
        and runtime_contract_valid
        and endpoint_failure_valid
        and causal_patch_quality_valid
        and adversarial_resistance_valid
        and solver_test_visibility_valid
        and solver_runtime_isolation_valid
        and (
            not coupling_required
            or semantic_coupling.get("valid") is True
        )
    )
    if not common_valid:
        return False
    if strictness_policy == "historical_pr":
        return historical_provenance_valid
    return bool(ablation.get("all_files_oracle_necessary"))


def _reference_file_patch(
    *,
    source_repo: Path,
    reference_parent: str,
    reference_commit: str,
    reference_files: List[str],
) -> str:
    """Return reference changes that must be hidden before agent launch."""
    from swesmith.repo_level import run_git

    if not reference_parent or not reference_commit or not reference_files:
        return ""
    safe_files = []
    for path in dict.fromkeys(reference_files):
        candidate = Path(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            continue
        safe_files.append(path)
    if not safe_files:
        return ""
    result = run_git(
        str(source_repo),
        "diff",
        "--binary",
        "--full-index",
        reference_parent,
        reference_commit,
        "--",
        *safe_files,
    )
    return result.stdout if result.returncode == 0 else ""


def _reference_test_patch(
    *,
    source_repo: Path,
    reference_parent: str,
    reference_commit: str,
    test_files: List[str],
) -> str:
    """Compatibility wrapper for callers that only need reference tests."""
    return _reference_file_patch(
        source_repo=source_repo,
        reference_parent=reference_parent,
        reference_commit=reference_commit,
        reference_files=test_files,
    )


def _run_solver(
    *,
    task: dict,
    phase: str,
    source_repo: Path,
    agent_src: Path,
    adapter: Any,
    model: str,
    output_dir: Path,
    agent_timeout: int,
    test_timeout: int,
) -> dict:
    from experiment_adapters.common_agent_adapter import CommonAgentRequest
    from godel0.git.patch import extract_changed_files, is_source_only
    from swesmith.repo_level import RepositoryWorkspace, apply_repository_patch, run_git

    solver_dir = output_dir / "solver" / phase / task["task_id"]
    solver_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    error = ""
    solver_test_command = task.get("solver_test_command") or task["test_command"]
    solver_validation_command = (
        task.get("solver_validation_command") or solver_test_command
    )
    control_test_command = str(task.get("control_test_command") or "")
    hidden_test_files = list(task.get("solver_hidden_test_files") or [])
    hidden_reference_files = list(
        task.get("solver_hidden_reference_files") or hidden_test_files
    )
    generated_test_patch = str(task.get("generated_test_patch") or "")
    generated_test_files = list(task.get("generated_test_files") or [])
    hidden_reference_patch = _reference_file_patch(
        source_repo=source_repo,
        reference_parent=str(task.get("reference_parent") or ""),
        reference_commit=str(task.get("reference_commit") or ""),
        reference_files=hidden_reference_files,
    )
    if hidden_reference_files and not hidden_reference_patch:
        raise RuntimeError(
            f"Cannot construct hidden reference changes for task: {task['task_id']}"
        )
    expected_tests = set(task.get("f2p_tests") or []) | set(
        task.get("p2p_tests") or []
    )
    pytest_tests_passed = False
    missing_expected_tests: List[str] = []
    failed_expected_tests: List[str] = []
    solver_workspace_isolated = False
    public_runtime_contracts_absent = True
    solver_scratch_root = Path(
        os.environ.get("GODEL0_SOLVER_SCRATCH_ROOT")
        or f"/tmp/godel0_solver_{os.getpid()}"
    )
    solver_scratch_root.mkdir(parents=True, exist_ok=True)
    with RepositoryWorkspace(
        str(source_repo),
        task["base_commit"],
        parent_dir=str(solver_scratch_root),
        prefix="solver_workspace_",
    ) as workspace:
        solver_workspace_isolated = not Path(workspace).resolve().is_relative_to(
            output_dir.resolve()
        )
        if not solver_workspace_isolated:
            raise RuntimeError(
                f"Solver workspace is not isolated from outputs: {workspace}"
            )
        if not apply_repository_patch(workspace, task["bug_patch"]):
            raise RuntimeError(f"Cannot apply task patch: {task['task_id']}")
        if hidden_reference_patch and not apply_repository_patch(
            workspace,
            hidden_reference_patch,
            reverse=True,
        ):
            raise RuntimeError(
                f"Cannot hide reference changes for task: {task['task_id']}"
            )
        agent_home = Path(workspace) / ".godel0_agent_home"
        if SOLVER_PUBLIC_RUNTIME_TOKEN in solver_test_command:
            public_runtime_dir = Path(workspace) / ".godel0_solver_runtime"
            _prepare_ansible_runtime(
                output_dir,
                runtime_dir=public_runtime_dir,
                include_contracts=False,
            )
            public_runtime_contracts_absent = not (
                public_runtime_dir / "contracts"
            ).exists()
            if not public_runtime_contracts_absent:
                raise RuntimeError("Public solver runtime contains evaluator contracts")
            agent_home = public_runtime_dir / "home"
            solver_test_command = solver_test_command.replace(
                SOLVER_PUBLIC_RUNTIME_TOKEN,
                str(public_runtime_dir),
            )
        agent_home.mkdir(parents=True, exist_ok=True)
        agent_process_tmp = solver_scratch_root / f"{task['task_id']}_process_tmp"
        agent_process_tmp.mkdir(parents=True, exist_ok=True)
        (agent_process_tmp / "ansible").mkdir(parents=True, exist_ok=True)
        _flatten_bugged_repository(Path(workspace))
        bugged_commit = run_git(workspace, "rev-parse", "HEAD").stdout.strip()
        request = CommonAgentRequest(
            problem_statement=task["problem_statement"],
            git_dir=Path(workspace),
            base_commit=bugged_commit,
            chat_history_file=solver_dir / "trajectory.log",
            outdir=solver_dir,
            test_description=solver_test_command,
            self_improve=False,
            instance_id=task["task_id"],
            model=model,
            timeout_sec=agent_timeout,
            extra_env={
                "HOME": str(agent_home),
                "TMPDIR": str(agent_process_tmp),
                "ANSIBLE_LOCAL_TEMP": str(agent_process_tmp / "ansible"),
            },
        )
        try:
            agent_result = adapter.run(agent_src, request)
            error = str(agent_result.error or "")
        except Exception as exc:
            agent_result = None
            error = f"{type(exc).__name__}: {exc}"
        hidden_reference_changes_injected = True
        if hidden_reference_patch:
            hidden_reference_changes_injected = apply_repository_patch(
                workspace,
                hidden_reference_patch,
            )
        generated_tests_injected = True
        if generated_test_patch:
            generated_tests_injected = apply_repository_patch(
                workspace,
                generated_test_patch,
            )
        hidden_reference_changes_injected = bool(
            hidden_reference_changes_injected and generated_tests_injected
        )
        if hidden_reference_changes_injected:
            test_result = _run_command(
                workspace,
                solver_validation_command,
                test_timeout,
            )
        else:
            injection_error = "reference test injection failed"
            error = f"{error}; {injection_error}" if error else injection_error
            test_result = {
                "returncode": -3,
                "stdout": "",
                "stderr": injection_error,
            }
        if task.get("validation_mode", "pytest") == "pytest":
            (
                pytest_tests_passed,
                missing_expected_tests,
                failed_expected_tests,
            ) = _evaluate_expected_pytest_tests(
                expected_tests=expected_tests,
                result=test_result,
                repo_path=workspace,
            )
        control_result = None
        if task.get("validation_mode") == "exit_code" and control_test_command:
            control_result = _run_command(
                workspace,
                control_test_command,
                test_timeout,
            )

    patch_path = solver_dir / "model_patch.diff"
    solver_patch = patch_path.read_text(encoding="utf-8") if patch_path.exists() else ""
    trajectory_path = solver_dir / "trajectory.log"
    trajectory = trajectory_path.read_text(encoding="utf-8", errors="replace") if trajectory_path.exists() else ""
    test_output = test_result["stdout"] + "\n" + test_result["stderr"]
    if control_result is not None:
        test_output += (
            "\n\n=== control command ===\n"
            + control_result["stdout"]
            + "\n"
            + control_result["stderr"]
        )
    (solver_dir / "test_output.txt").write_text(test_output, encoding="utf-8")
    modified_files = extract_changed_files(solver_patch)
    if task.get("validation_mode", "pytest") == "exit_code":
        tests_passed = bool(
            test_result["returncode"] == 0
            and (control_result is None or control_result["returncode"] == 0)
        )
    else:
        tests_passed = pytest_tests_passed
    resolved = bool(
        tests_passed and solver_patch.strip() and is_source_only(solver_patch)
    )
    return {
        "task_id": task["task_id"],
        "phase": phase,
        "domain": task["domain"],
        "resolved": resolved,
        "solver_success_flag": bool(agent_result and agent_result.success),
        "solver_patch_path": str(patch_path),
        "solver_patch": solver_patch,
        "modified_files": modified_files,
        "trajectory_path": str(trajectory_path),
        "trajectory_chars": len(trajectory),
        "tool_calls": len(re.findall(r"Tool Used:", trajectory)),
        "solver_test_command": solver_test_command,
        "solver_validation_command": solver_validation_command,
        "hidden_test_files": hidden_test_files,
        "hidden_reference_files": hidden_reference_files,
        "generated_test_files": generated_test_files,
        "generated_tests_injected": generated_tests_injected,
        "hidden_tests_injected": hidden_reference_changes_injected,
        "hidden_reference_changes_injected": hidden_reference_changes_injected,
        "solver_workspace_isolated": solver_workspace_isolated,
        "public_runtime_contracts_absent": public_runtime_contracts_absent,
        "test_returncode": test_result["returncode"],
        "control_test_returncode": (
            control_result["returncode"] if control_result is not None else None
        ),
        "expected_test_count": len(expected_tests),
        "missing_expected_tests": missing_expected_tests,
        "failed_expected_tests": failed_expected_tests,
        "test_output_path": str(solver_dir / "test_output.txt"),
        "runtime_sec": round(time.time() - started, 3),
        "error": error,
    }


def _diagnose_trajectories(
    *,
    solver_results: List[dict],
    tasks: List[dict],
    model: str,
    vllm_host: str,
    vllm_port: str,
    target_count: int,
    output_dir: Path,
) -> dict:
    task_by_id = {task["task_id"]: task for task in tasks}
    failures = [result for result in solver_results if not result["resolved"]]
    evidence_results = failures or sorted(
        solver_results,
        key=lambda row: (row.get("runtime_sec", 0), row.get("tool_calls", 0)),
        reverse=True,
    )
    evidence = []
    for result in evidence_results[: max(1, target_count)]:
        task = task_by_id.get(result["task_id"], {})
        trajectory_path = Path(result["trajectory_path"])
        trajectory = trajectory_path.read_text(encoding="utf-8", errors="replace") if trajectory_path.exists() else ""
        test_path = Path(result["test_output_path"])
        test_output = test_path.read_text(encoding="utf-8", errors="replace") if test_path.exists() else ""
        evidence.append(
            _build_trajectory_evidence(
                result=result,
                task=task,
                trajectory=trajectory,
                test_output=test_output,
            )
        )

    allowed_domains = {
        key: {
            "description": value["description"],
            "anchors": value["anchors"],
        }
        for key, value in DOMAINS.items()
    }
    prompt = (
        "You are the Proposer component of one joint Proposer/Solver node. "
        "Diagnose the same node's solver behavior from the evidence below. Identify "
        "specific scaffold weaknesses, not merely the code bug. The structured facts "
        "are authoritative; do not invent edits, symbols, test outcomes, or causes. "
        "Then choose transfer domains that probe those weaknesses without copying "
        "source identifiers.\n\n"
        f"Solver evidence:\n{json.dumps(evidence, indent=2, ensure_ascii=False)}\n\n"
        f"Allowed transfer domains:\n{json.dumps(allowed_domains, indent=2)}\n\n"
        f"Return JSON only with exactly {target_count} diagnoses under key 'diagnoses'. "
        "Each diagnosis must contain task_id, failure_stage, capability_gap_code, "
        "fact_ids, recommended_task_topology, and domain_id. failure_stage must be one of "
        "localization, reproduction, patch_generation, validation, tool_use, or "
        "context_management. capability_gap_code must be one of localization_incomplete, "
        "no_actionable_patch, incomplete_cross_file_repair, scope_control, "
        "validation_attribution, or high_effort_success."
    )
    raw_response = ""
    parsed: Optional[dict] = None
    try:
        import openai

        client = openai.OpenAI(
            base_url=f"http://{vllm_host}:{vllm_port}/v1",
            api_key="dummy",
            timeout=300,
        )
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You diagnose coding-agent scaffold failures and output strict JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 4096,
        }
        try:
            response = client.chat.completions.create(
                **kwargs,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception:
            response = client.chat.completions.create(**kwargs)
        raw_response = response.choices[0].message.content or ""
        parsed = _extract_json(raw_response)
    except Exception as exc:
        raw_response = f"diagnosis_error: {type(exc).__name__}: {exc}"

    model_diagnoses = [
        row
        for row in list((parsed or {}).get("diagnoses") or [])
        if isinstance(row, dict)
    ]
    model_by_task = {
        str(row.get("task_id") or ""): row for row in model_diagnoses
    }
    diagnoses = [
        _ground_trajectory_diagnosis(
            row,
            model_by_task.get(row["task_id"], {}),
        )
        for row in evidence
    ]
    if not diagnoses:
        diagnoses = _fallback_diagnoses(evidence, target_count)
    while len(diagnoses) < target_count:
        diagnoses.append(dict(diagnoses[len(diagnoses) % len(diagnoses)]))
    result = {
        "used_failed_trajectories": bool(failures),
        "evidence_task_ids": [row["task_id"] for row in evidence],
        "diagnoses": diagnoses[:target_count],
        "trajectory_facts": evidence,
        "model_diagnoses": model_diagnoses,
        "raw_response": raw_response,
    }
    (output_dir / "trajectory_diagnosis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def _build_trajectory_evidence(
    *,
    result: dict,
    task: dict,
    trajectory: str,
    test_output: str,
) -> dict:
    oracle_files = list(task.get("modified_files") or [])
    solver_files = list(result.get("modified_files") or [])
    exact_reverts = _exact_oracle_reverts(
        task.get("bug_patch", ""),
        result.get("solver_patch", ""),
    )
    missing_solver_files = [
        path for path in oracle_files if path not in solver_files
    ]
    unexpected_solver_files = [
        path for path in solver_files if path not in oracle_files
    ]
    inspected_oracle_files = [
        path for path in oracle_files if path and path in trajectory
    ]
    _, failed_tests = _pytest_status_sets(
        {"stdout": test_output, "stderr": ""}
    )
    facts = {
        "F1_ORACLE_FILES": oracle_files,
        "F2_SOLVER_PATCH_FILES": solver_files,
        "F3_EXACT_ORACLE_REVERTS": exact_reverts,
        "F4_MISSING_SOLVER_FILES": missing_solver_files,
        "F5_UNEXPECTED_SOLVER_FILES": unexpected_solver_files,
        "F6_FINAL_FAILING_TESTS": sorted(failed_tests),
        "F7_INSPECTED_ORACLE_FILES": inspected_oracle_files,
        "F8_TIMEOUT_EVENTS": len(
            re.findall(r"(?:TIMEOUT|timed out|TimeoutExpired)", trajectory)
        ),
        "F9_FINAL_TEST_RETURNCODE": result.get("test_returncode"),
    }
    return {
        "task_id": result["task_id"],
        "source_domain": result.get("domain", ""),
        "resolved": bool(result.get("resolved")),
        "problem_statement": task.get("problem_statement", ""),
        "f2p_tests": list(task.get("f2p_tests") or []),
        "facts": facts,
        "solver_patch": result.get("solver_patch", "")[:6000],
        "tool_calls": result.get("tool_calls", 0),
        "trajectory_excerpt": _head_tail(trajectory, 2500, 6000),
        "test_output_tail": test_output[-3500:],
    }


def _exact_oracle_reverts(bug_patch: str, solver_patch: str) -> List[str]:
    """Find files where the solver exactly reversed the generated mutation."""
    bug_edits = _patch_edits_by_file(bug_patch)
    solver_edits = _patch_edits_by_file(solver_patch)
    reverted: List[str] = []
    for path, bug_change in bug_edits.items():
        solver_change = solver_edits.get(path)
        if not solver_change:
            continue
        if (
            sorted(bug_change["added"]) == sorted(solver_change["removed"])
            and sorted(bug_change["removed"]) == sorted(solver_change["added"])
        ):
            reverted.append(path)
    return reverted


def _patch_edits_by_file(patch: str) -> Dict[str, Dict[str, List[str]]]:
    edits: Dict[str, Dict[str, List[str]]] = {}
    current = ""
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            match = re.match(r"diff --git a/(.*) b/(.*)", line)
            current = match.group(2) if match else ""
            if current:
                edits.setdefault(current, {"added": [], "removed": []})
        elif current and line.startswith("+") and not line.startswith("+++"):
            edits[current]["added"].append(line[1:])
        elif current and line.startswith("-") and not line.startswith("---"):
            edits[current]["removed"].append(line[1:])
    return edits


def _ground_trajectory_diagnosis(evidence: dict, proposal: dict) -> dict:
    facts = evidence.get("facts") or {}
    oracle_files = list(facts.get("F1_ORACLE_FILES") or [])
    solver_files = list(facts.get("F2_SOLVER_PATCH_FILES") or [])
    missing_files = list(facts.get("F4_MISSING_SOLVER_FILES") or [])
    unexpected_files = list(facts.get("F5_UNEXPECTED_SOLVER_FILES") or [])
    inspected_files = list(facts.get("F7_INSPECTED_ORACLE_FILES") or [])

    if evidence.get("resolved"):
        stage = "validation"
        gap_code = "high_effort_success"
        gap = "solver succeeded, but required substantial tool use or validation effort"
        topology = "a coupled producer-consumer path with stricter efficiency limits"
    elif not solver_files and len(inspected_files) < len(oracle_files):
        stage = "localization"
        gap_code = "localization_incomplete"
        gap = "solver did not inspect every oracle file and produced no source patch"
        topology = "an indirect producer-consumer path requiring repository-wide localization"
    elif not solver_files:
        stage = "patch_generation"
        gap_code = "no_actionable_patch"
        gap = "solver inspected the relevant scope but produced no actionable source patch"
        topology = "a shared-test contract where localization must be converted into a minimal patch"
    elif missing_files:
        stage = "patch_generation"
        gap_code = "incomplete_cross_file_repair"
        gap = "solver changed only part of the generated cross-file regression"
        topology = "a three-point producer-consumer contract with no one-file repair shortcut"
    elif unexpected_files:
        stage = "patch_generation"
        gap_code = "scope_control"
        gap = "solver edited files outside the generated regression scope"
        topology = "a narrow cross-module contract that penalizes unrelated edits"
    else:
        stage = "validation"
        gap_code = "validation_attribution"
        gap = "solver touched the full oracle scope but baseline-passing tests still failed"
        topology = "a coupled task requiring differential F2P/P2P validation"

    evidence_summary = (
        f"oracle_files={oracle_files}; solver_patch_files={solver_files}; "
        f"exact_oracle_reverts={facts.get('F3_EXACT_ORACLE_REVERTS', [])}; "
        f"missing_solver_files={missing_files}; "
        f"unexpected_solver_files={unexpected_files}; "
        f"final_failing_tests={facts.get('F6_FINAL_FAILING_TESTS', [])}; "
        f"timeout_events={facts.get('F8_TIMEOUT_EVENTS', 0)}; "
        f"final_test_returncode={facts.get('F9_FINAL_TEST_RETURNCODE')}"
    )
    requested_domain = str(proposal.get("domain_id") or "")
    return {
        "task_id": evidence.get("task_id", ""),
        "source_domain": evidence.get("source_domain", ""),
        "failure_stage": stage,
        "capability_gap_code": gap_code,
        "capability_gap": gap,
        "evidence": evidence_summary,
        "fact_ids": list(facts),
        "recommended_task_topology": topology,
        "domain_id": requested_domain if requested_domain in DOMAINS else "",
        "grounding_valid": True,
    }


def _file_ablation(
    *,
    source_repo: Path,
    base_commit: str,
    bug_patch: str,
    f2p_tests: List[str],
    test_prefix: str,
    test_command: str,
    validation_mode: str,
    timeout: int,
    checkpoint_path: Optional[Path] = None,
    setup_patch: str = "",
) -> dict:
    from swesmith.repo_level import (
        RepositoryWorkspace,
        apply_repository_patch,
        split_patch_by_file,
    )

    blocks = split_patch_by_file(bug_patch)
    command = (
        test_command
        if validation_mode == "exit_code"
        else _test_command(test_prefix, f2p_tests)
    )
    input_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "schema_version": 2,
                "base_commit": base_commit,
                "bug_patch_sha256": hashlib.sha256(
                    bug_patch.encode("utf-8")
                ).hexdigest(),
                "command": command,
                "validation_mode": validation_mode,
                "setup_patch_sha256": hashlib.sha256(
                    setup_patch.encode("utf-8")
                ).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    omit_results: Dict[str, bool] = {}
    single_results: Dict[str, bool] = {}
    omit_setup_valid: Dict[str, bool] = {}
    single_setup_valid: Dict[str, bool] = {}
    if checkpoint_path and checkpoint_path.exists():
        try:
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if saved.get("input_fingerprint") == input_fingerprint:
                omit_results.update(saved.get("omit_one_file_passed") or {})
                single_results.update(
                    saved.get("repair_only_one_file_passed") or {}
                )
                omit_setup_valid.update(
                    saved.get("omit_one_file_setup_valid") or {}
                )
                single_setup_valid.update(
                    saved.get("repair_only_one_file_setup_valid") or {}
                )
        except (ValueError, OSError):
            pass

    def snapshot() -> dict:
        complete = (
            bool(blocks)
            and len(omit_results) == len(blocks)
            and len(single_results) == len(blocks)
        )
        setup_valid = (
            complete
            and len(omit_setup_valid) == len(blocks)
            and len(single_setup_valid) == len(blocks)
            and all(omit_setup_valid.values())
            and all(single_setup_valid.values())
        )
        return {
            "input_fingerprint": input_fingerprint,
            "test_command": command,
            "ablation_valid": setup_valid,
            "all_files_oracle_necessary": (
                setup_valid and not any(omit_results.values())
            ),
            "any_single_file_oracle_fix_passes": any(single_results.values()),
            "omit_one_file_passed": omit_results,
            "repair_only_one_file_passed": single_results,
            "omit_one_file_setup_valid": omit_setup_valid,
            "repair_only_one_file_setup_valid": single_setup_valid,
        }

    def checkpoint() -> None:
        if checkpoint_path:
            checkpoint_path.write_text(
                json.dumps(snapshot(), indent=2), encoding="utf-8"
            )

    cached_snapshot = snapshot()
    if cached_snapshot["ablation_valid"]:
        return cached_snapshot

    with RepositoryWorkspace(str(source_repo), base_commit) as workspace:
        if setup_patch and not apply_repository_patch(workspace, setup_patch):
            return snapshot()
        for omitted_path, omitted_block in blocks:
            if omitted_path in omit_results and omitted_path in omit_setup_valid:
                continue
            setup_ok = apply_repository_patch(workspace, omitted_block)
            omit_setup_valid[omitted_path] = setup_ok
            omit_results[omitted_path] = bool(
                setup_ok
                and _run_command(workspace, command, timeout)["returncode"] == 0
            )
            restored = bool(
                setup_ok
                and apply_repository_patch(workspace, omitted_block, reverse=True)
            )
            omit_setup_valid[omitted_path] = setup_ok and restored
            checkpoint()

        for repaired_path, _repaired_block in blocks:
            if repaired_path in single_results and repaired_path in single_setup_valid:
                continue
            applied_blocks: List[str] = []
            setup_ok = True
            for path, block in blocks:
                if path == repaired_path:
                    continue
                if not apply_repository_patch(workspace, block):
                    setup_ok = False
                    break
                applied_blocks.append(block)
            single_setup_valid[repaired_path] = setup_ok
            single_results[repaired_path] = bool(
                setup_ok
                and _run_command(workspace, command, timeout)["returncode"] == 0
            )
            restored = True
            for block in reversed(applied_blocks):
                restored = (
                    apply_repository_patch(workspace, block, reverse=True)
                    and restored
                )
            single_setup_valid[repaired_path] = setup_ok and restored
            checkpoint()
    return snapshot()


def _adversarial_solver_patch_results(
    *,
    source_repo: Path,
    base_commit: str,
    bug_patch: str,
    solver_patches: List[Dict[str, str]],
    test_command: str,
    timeout: int,
    checkpoint_path: Optional[Path] = None,
    setup_patch: str = "",
) -> Dict[str, Any]:
    """Replay prior solver shortcuts and require the current contract to reject them."""
    from swesmith.repo_level import RepositoryWorkspace, apply_repository_patch

    normalized = [
        {
            "id": str(row.get("id") or f"solver_patch_{index}"),
            "patch": str(row.get("patch") or ""),
        }
        for index, row in enumerate(solver_patches)
        if str(row.get("patch") or "").strip()
    ]
    input_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "schema_version": 1,
                "base_commit": base_commit,
                "bug_patch_sha256": hashlib.sha256(
                    bug_patch.encode("utf-8")
                ).hexdigest(),
                "test_command": test_command,
                "setup_patch_sha256": hashlib.sha256(
                    setup_patch.encode("utf-8")
                ).hexdigest(),
                "solver_patches": [
                    {
                        "id": row["id"],
                        "sha256": hashlib.sha256(
                            row["patch"].encode("utf-8")
                        ).hexdigest(),
                    }
                    for row in normalized
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if not normalized:
        return {
            "input_fingerprint": input_fingerprint,
            "valid": True,
            "all_rejected": True,
            "results": [],
        }
    if checkpoint_path and checkpoint_path.exists():
        try:
            cached = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if (
                cached.get("input_fingerprint") == input_fingerprint
                and cached.get("valid") is True
                and len(cached.get("results") or []) == len(normalized)
            ):
                return cached
        except (OSError, ValueError):
            pass

    results: List[Dict[str, Any]] = []
    for row in normalized:
        patch_applied = False
        test_result = {
            "returncode": -1,
            "stdout": "",
            "stderr": "solver_patch_apply_failed",
        }
        with RepositoryWorkspace(str(source_repo), base_commit) as workspace:
            setup_applied = bool(
                not setup_patch
                or apply_repository_patch(workspace, setup_patch)
            )
            bug_applied = bool(
                setup_applied and apply_repository_patch(workspace, bug_patch)
            )
            patch_applied = bool(
                bug_applied
                and apply_repository_patch(workspace, row["patch"])
            )
            if patch_applied:
                test_result = _run_command(workspace, test_command, timeout)
        contract_lines = [
            line
            for line in test_result["stdout"].splitlines()
            if line.startswith("GODEL0_CONTRACT ")
        ]
        returncode = int(test_result["returncode"])
        results.append(
            {
                "id": row["id"],
                "patch_sha256": hashlib.sha256(
                    row["patch"].encode("utf-8")
                ).hexdigest(),
                "patch_applied": patch_applied,
                "test_returncode": returncode,
                "rejected": bool(
                    patch_applied and returncode not in (0, -2)
                ),
                "contract_lines": contract_lines,
                "stderr_tail": test_result["stderr"][-2000:],
            }
        )
    valid = bool(
        len(results) == len(normalized)
        and all(row["patch_applied"] for row in results)
        and all(row["test_returncode"] != -2 for row in results)
    )
    evaluation = {
        "input_fingerprint": input_fingerprint,
        "valid": valid,
        "all_rejected": bool(valid and all(row["rejected"] for row in results)),
        "results": results,
    }
    if checkpoint_path:
        checkpoint_path.write_text(
            json.dumps(evaluation, indent=2),
            encoding="utf-8",
        )
    return evaluation


def _run_patch_tests(
    source_repo: Path,
    base_commit: str,
    patch: str,
    command: str,
    timeout: int,
    setup_patch: str = "",
) -> dict:
    from swesmith.repo_level import RepositoryWorkspace, apply_repository_patch

    with RepositoryWorkspace(str(source_repo), base_commit) as workspace:
        if setup_patch and not apply_repository_patch(workspace, setup_patch):
            return {"returncode": -1, "stdout": "", "stderr": "setup_patch_apply_failed"}
        if not apply_repository_patch(workspace, patch):
            return {"returncode": -1, "stdout": "", "stderr": "patch_apply_failed"}
        return _run_command(workspace, command, timeout)


def _trace_patch_contract_test(
    *,
    source_repo: Path,
    base_commit: str,
    patch: str,
    test_prefix: str,
    test_nodeid: str,
    source_roots: List[str],
    output_path: Path,
    timeout: int,
    setup_patch: str = "",
) -> Dict[str, Any]:
    from swesmith.repo_level import RepositoryWorkspace, apply_repository_patch

    with RepositoryWorkspace(str(source_repo), base_commit) as workspace:
        if setup_patch and not apply_repository_patch(workspace, setup_patch):
            return {}
        if not apply_repository_patch(workspace, patch):
            return {}
        return _trace_contract_test(
            workspace=Path(workspace),
            test_prefix=test_prefix,
            test_nodeid=test_nodeid,
            source_roots=source_roots,
            output_path=output_path,
            timeout=timeout,
            allow_test_failure=True,
        )


def _prepare_ansible_runtime(
    output_dir: Path,
    *,
    runtime_dir: Optional[Path] = None,
    include_contracts: bool = True,
) -> Dict[str, str]:
    runtime_dir = runtime_dir or output_dir / "ansible_runtime"
    home_dir = runtime_dir / "home"
    python_bin = runtime_dir / "bin"
    local_tmp = runtime_dir / "local_tmp"
    remote_tmp = runtime_dir / "remote_tmp"
    for path in (home_dir, python_bin, local_tmp, remote_tmp):
        path.mkdir(parents=True, exist_ok=True)
    contract_dir = runtime_dir / "contracts" / "ansible_role_instance_contract"
    if include_contracts:
        shutil.copytree(
            ROLE_INSTANCE_CONTRACT_DIR,
            contract_dir,
            dirs_exist_ok=True,
        )
    else:
        shutil.rmtree(runtime_dir / "contracts", ignore_errors=True)

    original_user_site = Path.home() / ".local"
    runtime_user_site = home_dir / ".local"
    if original_user_site.exists() and not runtime_user_site.exists():
        runtime_user_site.symlink_to(original_user_site, target_is_directory=True)

    python_executable = shutil.which("python3.11") or sys.executable
    for name in ("python", "python3"):
        link = python_bin / name
        if not link.exists():
            link.symlink_to(python_executable)
    ansible_playbook = python_bin / "ansible-playbook"
    ansible_playbook.write_text(
        "#!/bin/sh\n"
        "exec \"$(dirname \"$0\")/python\" -m ansible.cli.playbook \"$@\"\n",
        encoding="utf-8",
    )
    ansible_playbook.chmod(0o755)

    config_path = runtime_dir / "ansible.cfg"
    config_path.write_text(
        "[defaults]\n"
        f"local_tmp = {local_tmp}\n"
        f"remote_tmp = {remote_tmp}\n"
        "host_key_checking = False\n"
        "retry_files_enabled = False\n",
        encoding="utf-8",
    )
    return {
        "home": str(home_dir),
        "python_bin": str(python_bin),
        "config": str(config_path),
        "role_instance_contract_playbook": str(contract_dir / "play.yml"),
    }


def _solver_public_ansible_runtime() -> Dict[str, str]:
    runtime = SOLVER_PUBLIC_RUNTIME_TOKEN
    return {
        "home": f"{runtime}/home",
        "python_bin": f"{runtime}/bin",
        "config": f"{runtime}/ansible.cfg",
        "role_instance_contract_playbook": "",
    }


def _integration_test_command(
    runtime: Dict[str, str],
    target_dir: str,
    command: str,
    *,
    include_contract_env: bool = True,
) -> str:
    inner = (
        f"cd {shlex.quote(target_dir)} && "
        f"cp {shlex.quote(runtime['config'])} ansible.cfg && "
        "trap 'rm -f ansible.cfg' EXIT && "
        f"{command}"
    )
    contract_env = ""
    if include_contract_env:
        contract_env = (
            "GODEL0_ROLE_INSTANCE_CONTRACT_PLAYBOOK="
            f"{shlex.quote(runtime['role_instance_contract_playbook'])} "
        )
    return (
        f"HOME={shlex.quote(runtime['home'])} "
        f"PATH={shlex.quote(runtime['python_bin'])}:$PWD/bin:$PATH "
        "PYTHONPATH=$PWD/lib:$PWD/test/lib "
        "ANSIBLE_NOCOLOR=1 "
        f"{contract_env}"
        f"bash -c {shlex.quote(inner)}"
    )


def _control_test_command(runtime: Dict[str, str]) -> str:
    return (
        f"HOME={shlex.quote(runtime['home'])} "
        f"PATH={shlex.quote(runtime['python_bin'])}:$PATH "
        "PYTHONPATH=lib:test/lib "
        "python -m pytest -p no:cacheprovider --rootdir=. "
        "test/units/utils/test_version.py -q"
    )


def _flatten_bugged_repository(repo: Path) -> None:
    shutil.rmtree(repo / ".git", ignore_errors=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "experiment@godel0.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Godel0 Experiment"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "task base"], cwd=repo, check=True)


def _run_command(repo: Union[str, Path], command: str, timeout: int) -> dict:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -2,
            "stdout": _decode_timeout(exc.stdout),
            "stderr": _decode_timeout(exc.stderr) + "\nTIMEOUT",
        }


def _test_command(prefix: str, tests: Iterable[str]) -> str:
    deduped = list(dict.fromkeys(str(test) for test in tests if test))
    quoted_tests = " ".join(shlex.quote(test) for test in deduped)
    return f"{prefix} {quoted_tests} -v".strip()


def _solver_evaluation_command(
    *,
    test_prefix: str,
    original_test_command: str,
    validation_mode: str,
    f2p_tests: Iterable[str],
    p2p_tests: Iterable[str],
) -> str:
    """Build a solver command from tests known to pass on the clean baseline."""
    if validation_mode == "exit_code":
        return original_test_command
    stable_targets = [
        _stable_pytest_target(test_id)
        for test_id in list(f2p_tests) + list(p2p_tests)
    ]
    return _test_command(test_prefix, stable_targets)


def _stable_pytest_target(test_id: str) -> str:
    """Collapse a parameterized node ID to a workspace-independent test target."""
    prefix, separator, leaf = str(test_id).rpartition("::")
    if not separator:
        return str(test_id)
    if "[" in leaf:
        leaf = leaf.split("[", 1)[0]
    return f"{prefix}::{leaf}"


def _choose_adaptive_domain(diag: dict, used: set[str], index: int) -> str:
    requested = str(diag.get("domain_id") or "")
    source_domain = str(diag.get("source_domain") or "")
    min_files = _adaptive_min_modified_files(diag)

    def usable(domain_id: str) -> bool:
        return len(DOMAINS[domain_id]["anchors"]) >= min_files

    if (
        requested in DOMAINS
        and requested not in used
        and requested != source_domain
        and usable(requested)
    ):
        return requested
    order = ["plugin_config", "task_conditionals", "inventory_model", "yaml_loading", "template_vars", "variable_merge"]
    for offset in range(len(order)):
        candidate = order[(index + offset) % len(order)]
        if candidate not in used and candidate != source_domain and usable(candidate):
            return candidate
    for candidate in order:
        if usable(candidate):
            return candidate
    return order[index % len(order)]


def _adaptive_min_modified_files(diag: dict) -> int:
    return (
        3
        if diag.get("capability_gap_code") == "incomplete_cross_file_repair"
        else 2
    )


def _fallback_diagnoses(evidence: List[dict], count: int) -> List[dict]:
    diagnoses = []
    domain_order = ["plugin_config", "task_conditionals", "inventory_model", "yaml_loading"]
    for index in range(count):
        row = evidence[index % len(evidence)] if evidence else {}
        patch_files = row.get("solver_modified_files") or []
        if not patch_files:
            stage = "patch_generation"
            gap = "solver failed to produce an actionable source patch"
        elif len(patch_files) == 1:
            stage = "patch_generation"
            gap = "solver localized one implementation point but missed cross-file contract propagation"
        elif row.get("tool_calls", 0) < 2:
            stage = "tool_use"
            gap = "solver did not inspect enough repository evidence before editing"
        else:
            stage = "validation"
            gap = "solver produced a multi-file patch but failed to validate all affected behavior"
        diagnoses.append({
            "task_id": row.get("task_id", ""),
            "source_domain": row.get("source_domain", ""),
            "failure_stage": stage,
            "capability_gap": gap,
            "evidence": f"modified_files={patch_files}, tool_calls={row.get('tool_calls', 0)}",
            "recommended_task_topology": "producer-consumer contract across multiple modules",
            "domain_id": domain_order[index % len(domain_order)],
        })
    return diagnoses


def _extract_json(text: str) -> Optional[dict]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    brace = text.find("{")
    if brace >= 0:
        candidates.append(text[brace:])
    for candidate in candidates:
        try:
            value, _ = json.JSONDecoder().raw_decode(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def _compute_metrics(report: dict) -> dict:
    metrics = {}
    for phase in ("bootstrap", "adaptive"):
        tasks = report.get(f"{phase}_tasks", [])
        solver = report.get(f"{phase}_solver_results", [])
        agent_attempts = report.get(f"{phase}_generation_attempts", [])
        if phase == "bootstrap":
            generation_total = len(agent_attempts) + len(report.get("pr_replay_attempts", []))
        else:
            generation_total = len(agent_attempts)
        metrics[phase] = {
            "generation_attempts": generation_total,
            "valid_tasks": len(tasks),
            "valid_yield": round(len(tasks) / generation_total, 3) if generation_total else 0.0,
            "solver_attempts": len(solver),
            "solver_resolved": sum(1 for row in solver if row.get("resolved")),
            "solver_accuracy": round(
                sum(1 for row in solver if row.get("resolved")) / len(solver), 3
            ) if solver else 0.0,
            "mean_bug_files": round(
                sum(len(task.get("modified_files", [])) for task in tasks) / len(tasks), 3
            ) if tasks else 0.0,
            "all_files_necessary_rate": round(
                sum(bool(task.get("ablation", {}).get("all_files_oracle_necessary")) for task in tasks) / len(tasks),
                3,
            ) if tasks else 0.0,
            "single_file_oracle_shortcuts": sum(
                bool(task.get("ablation", {}).get("any_single_file_oracle_fix_passes")) for task in tasks
            ),
            "strong_semantic_coupling_rate": round(
                sum(
                    task.get("semantic_coupling", {}).get("tier") == "strong"
                    for task in tasks
                )
                / len(tasks),
                3,
            ) if tasks else 0.0,
        }
    return metrics


def _head_tail(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n...<trajectory clipped>...\n" + text[-tail:]


def _decode_timeout(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
