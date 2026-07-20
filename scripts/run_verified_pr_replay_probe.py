#!/usr/bin/env python3
"""Replay and validate real multi-file Ansible fixes as repair tasks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROLE_INSTANCE_CONTRACT_DIR = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "ansible_role_instance_contract"
)
ROLE_INSTANCE_SHORTCUT_PATCH = (
    ROLE_INSTANCE_CONTRACT_DIR / "parent_reset_shortcut.patch"
).read_text(encoding="utf-8")


CASES: dict[str, dict[str, Any]] = {
    "role_instance_cache": {
        "commit": "1998521e2d5b89bc53d00639bad178330ebb98df",
        "files": [
            "lib/ansible/playbook/play.py",
            "lib/ansible/playbook/role/__init__.py",
            "lib/ansible/plugins/strategy/__init__.py",
            "lib/ansible/plugins/strategy/free.py",
            "lib/ansible/plugins/strategy/linear.py",
        ],
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
    "delegate_to_evaluation": {
        "commit": "42355d181a11b51ebfc56f6f4b3d9c74e01cb13b",
        "files": [
            "lib/ansible/executor/process/worker.py",
            "lib/ansible/executor/task_executor.py",
            "lib/ansible/playbook/delegatable.py",
            "lib/ansible/playbook/task.py",
            "lib/ansible/vars/manager.py",
        ],
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godel0-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--repo-pool", default="repo_pool")
    parser.add_argument("--repo-id", default="ansible")
    parser.add_argument("--case", choices=sorted(CASES), default="role_instance_cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-timeout", type=int, default=300)
    parser.add_argument("--run-solver", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--agent-timeout", type=int, default=2400)
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.godel0_root).resolve()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "initial_agent" / "src"))
    sys.path.insert(0, str(root))

    from experiment_adapters.common_agent_adapter import CommonAgentAdapter
    from godel0.proposer_trusted.candidate_validator import CandidateValidator
    from godel0.tasks.repo_pool import RepoPool
    from scripts import run_repo_level_closed_loop as closed_loop
    from swesmith.engine import (
        BugConstraints,
        BugGenerationPlan,
        RepoSpec as EngineRepoSpec,
        SWESmithEngine,
    )

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

    pool = RepoPool((root / args.repo_pool).resolve())
    pool_spec = pool.get(args.repo_id)
    if pool_spec is None:
        raise RuntimeError(f"Repository not found: {args.repo_id}")
    source_repo = Path(pool_spec.path)
    if not source_repo.is_absolute():
        source_repo = (root / source_repo).resolve()

    case = CASES[args.case]
    base_commit = str(case["commit"])
    repo_spec = EngineRepoSpec(
        repo_id=pool_spec.repo_id,
        repo_path=str(source_repo),
        base_commit=base_commit,
        test_command=pool_spec.test_command,
    )
    plan = BugGenerationPlan(
        plan_id=f"verified_pr_{args.case}",
        target_repo_id=pool_spec.repo_id,
        target_base_commit=base_commit,
        target_file=case["files"][0],
        target_files=list(case["files"]),
        strategy="pr_replay",
        operator="reverse_real_fix",
        reference_commit=base_commit,
        constraints=BugConstraints(
            min_modified_files=len(case["files"]),
            max_modified_files=len(case["files"]),
            max_modified_lines=300,
            allow_test_edits=False,
        ),
        task_blueprint={
            "source": "historical_fix_commit",
            "case": args.case,
            "reference_commit": base_commit,
            "oracle_files": list(case["files"]),
        },
    )
    generation_dir = output_dir / "generation" / args.case
    candidates = SWESmithEngine().generate(
        plan,
        str(root / "initial_agent" / "src"),
        repo_spec,
        str(generation_dir),
    )
    report: dict[str, Any] = {
        "case": args.case,
        "reference_commit": base_commit,
        "generated": len(candidates),
        "accepted_task": None,
        "solver_result": None,
    }
    report_path = output_dir / "verified_pr_replay_report.json"
    if not candidates:
        report["result"] = "generation_rejected"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report={report_path}", flush=True)
        return 2

    candidate = candidates[0]
    runtime = closed_loop._prepare_ansible_runtime(output_dir)
    solver_runtime = closed_loop._solver_public_ansible_runtime()
    test_command = closed_loop._integration_test_command(
        runtime,
        str(case["integration_dir"]),
        str(case["integration_command"]),
    )
    control_command = closed_loop._control_test_command(runtime)
    solver_test_command = closed_loop._control_test_command(solver_runtime)
    if args.case == "role_instance_cache":
        solver_test_command = closed_loop._integration_test_command(
            solver_runtime,
            str(case["integration_dir"]),
            closed_loop.ROLE_INSTANCE_PUBLIC_COMMAND,
            include_contract_env=False,
        )
    validator = CandidateValidator(
        workspace_root=output_dir / "validator",
        test_timeout_sec=args.test_timeout,
        max_patch_lines=300,
        forbid_test_file_edits=True,
    )
    task = closed_loop._validate_and_package(
        candidate=candidate,
        phase="verified_pr_probe_hidden_tests",
        domain_id=f"historical_pr:{args.case}",
        problem_statement=str(case["problem"]),
        test_files=[],
        source_repo=source_repo,
        base_commit=base_commit,
        test_prefix=pool_spec.test_command,
        validator=validator,
        output_dir=output_dir,
        test_timeout=args.test_timeout,
        source_trajectory_ids=[],
        test_command_override=test_command,
        validation_mode="exit_code",
        command_test_id=f"integration::{args.case}",
        control_test_command=control_command,
        strictness_policy="historical_pr",
        adversarial_solver_patches=list(
            case.get("adversarial_solver_patches") or []
        ),
        solver_test_command_override=solver_test_command,
    )
    report.update(
        {
            "candidate_id": candidate.candidate_id,
            "modified_files": candidate.modified_files,
            "modified_lines": candidate.generation_metadata.get("modified_lines"),
            "accepted_task": task,
            "result": (
                "accepted"
                if task and task.get("strict_repo_level")
                else "strict_gate_rejected"
                if task
                else "validation_rejected"
            ),
        }
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if task and task.get("strict_repo_level") and args.run_solver:
        report["solver_result"] = closed_loop._run_solver(
            task=task,
            phase="verified_pr_probe_hidden_tests",
            source_repo=source_repo,
            agent_src=root / "initial_agent" / "src",
            adapter=CommonAgentAdapter(),
            model=args.model,
            output_dir=output_dir,
            agent_timeout=args.agent_timeout,
            test_timeout=args.test_timeout,
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"case={args.case} result={report['result']} "
        f"files={candidate.modified_files} report={report_path}",
        flush=True,
    )
    return 0 if task and task.get("strict_repo_level") else 2


if __name__ == "__main__":
    raise SystemExit(main())
