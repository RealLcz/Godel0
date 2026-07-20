#!/usr/bin/env python3
"""Generate and validate one trajectory-conditioned Ansible repo-chain task."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


INCLUDE_LOOP_CONTEXT_FILES = [
    "lib/ansible/playbook/task.py",
    "lib/ansible/playbook/task_include.py",
    "lib/ansible/playbook/included_file.py",
    "lib/ansible/playbook/helpers.py",
    "lib/ansible/playbook/base.py",
    "lib/ansible/executor/task_executor.py",
    "lib/ansible/vars/manager.py",
    "lib/ansible/template/__init__.py",
    "test/units/playbook/test_helpers.py",
    "test/units/executor/test_task_executor.py",
]

HANDLER_NOTIFICATION_CONTEXT_FILES = [
    "lib/ansible/executor/task_executor.py",
    "lib/ansible/executor/task_result.py",
    "lib/ansible/executor/play_iterator.py",
    "lib/ansible/plugins/strategy/__init__.py",
    "lib/ansible/playbook/handler.py",
    "lib/ansible/playbook/notifiable.py",
    "lib/ansible/playbook/task.py",
    "lib/ansible/playbook/play.py",
    "test/units/executor/test_task_executor.py",
    "test/units/executor/test_play_iterator.py",
]


class VLLMChatAdapter:
    def __init__(self, host: str, port: int, model: str) -> None:
        import openai

        self.client = openai.OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="dummy",
            timeout=1800,
        )
        self.model = model

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0,
        max_tokens: int = 8192,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return str(response.choices[0].message.content or "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godel0-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--repo", default="repo_pool/ansible")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--trajectory", default="")
    parser.add_argument(
        "--scenario-profile",
        choices=("include_loop", "handler_notification"),
        default="include_loop",
    )
    parser.add_argument("--generation-timeout", type=int, default=900)
    parser.add_argument("--test-timeout", type=int, default=300)
    parser.add_argument("--run-solver", action="store_true")
    parser.add_argument("--agent-timeout", type=int, default=2400)
    return parser.parse_args()


def _trajectory_failure_summary(trajectory: str) -> tuple[str, str, str]:
    """Extract only domain-neutral solver behavior from a local trajectory."""
    path = Path(trajectory) if trajectory else None
    if path is None or not path.is_file():
        return (
            "context_management",
            "No usable prior trace was available; require full-chain verification.",
            "cross-layer invariant tracing",
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    tool_calls = text.count("Tool Used:")
    public_test_runs = text.count("python -m pytest")
    edited_files = set(
        match.group(1)
        for match in re.finditer(
            r"['\"]command['\"]\s*:\s*['\"]edit['\"].{0,500}?"
            r"['\"]path['\"]\s*:\s*['\"]([^'\"]+\.py)",
            text,
            re.DOTALL,
        )
    )
    edit_scope = len(edited_files)
    return (
        "premature_local_fix",
        (
            "The prior solver found one conspicuous local defect and stopped after "
            f"repeated public checks ({public_test_runs} observed test invocations, "
            f"{tool_calls} tool calls, {edit_scope or 1} apparent edit target). It "
            "did not verify every producer, carrier, identity boundary, and consumer "
            "participating in the stated invariant. Transfer this reasoning weakness "
            "only; do not transfer repository nouns or the original behavior."
        ),
        "premature single-site repair of a multi-layer invariant",
    )


def main() -> int:
    args = parse_args()
    root = Path(args.godel0_root).resolve()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "initial_agent" / "src"))
    sys.path.insert(0, str(root))

    from experiment_adapters.common_agent_adapter import CommonAgentAdapter
    from godel0.proposer_trusted.candidate_validator import CandidateValidator
    from scripts import run_repo_level_closed_loop as closed_loop
    from swesmith.engine import (
        BugConstraints,
        BugGenerationPlan,
        FailureSignature,
        RepoSpec,
        SWESmithEngine,
    )
    from swesmith.repo_level import run_git

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_repo = Path(args.repo)
    if not source_repo.is_absolute():
        source_repo = (root / source_repo).resolve()
    base_commit = run_git(str(source_repo), "rev-parse", "HEAD").stdout.strip()
    runtime = closed_loop._prepare_ansible_runtime(
        output_dir,
        include_contracts=False,
    )
    test_prefix = (
        f"HOME={runtime['home']} PATH={runtime['python_bin']}:$PATH "
        "PYTHONPATH=lib:test/lib python -m pytest -p no:cacheprovider --rootdir=."
    )
    repo_spec = RepoSpec(
        repo_id="ansible",
        repo_path=str(source_repo),
        base_commit=base_commit,
        test_command=test_prefix,
        source_dirs=["lib", "test/units"],
    )
    trajectories = [args.trajectory] if args.trajectory else []
    error_type, error_message, failure_pattern = _trajectory_failure_summary(
        args.trajectory
    )
    if args.scenario_profile == "handler_notification":
        context_files = HANDLER_NOTIFICATION_CONTEXT_FILES
        desired_behavior = (
            "Preserve a changed task's notification topic from task execution through "
            "result transport and strategy matching so a handler addressed through a "
            "listen alias runs exactly once, while direct handler-name notification "
            "remains compatible."
        )
        rationale = (
            "Transfer the prior solver's premature single-site stopping behavior to "
            "Ansible handler notification delivery without reusing include processing, "
            "loop variables, or variable templating."
        )
        blueprint = {
            "capability_gap": "trace one semantic identity across multiple ownership boundaries",
            "failure_stage": "premature_local_fix",
            "required_topology": (
                "task notification producer -> result carrier -> strategy matcher -> "
                "handler identity/deduplication -> observable execution"
            ),
            "forbidden_copy": (
                "include_tasks, loop variables, IncludedFile, apply_context, variable propagation"
            ),
            "forbidden_terms": [
                "include_tasks",
                "loop_var",
                "ansible_loop_var",
                "IncludedFile",
                "apply_context",
                "default_loop_value",
            ],
            "contract_scenario": (
                "A pytest test creates a temporary localhost playbook in which two "
                "changed tasks notify the same listen topic, runs ansible-playbook, "
                "and asserts that a unique handler marker occurs exactly once. Each "
                "notifying task must report changed=true. A "
                "separate compatibility case notifies a handler by its direct name."
            ),
            "contract_test_style": (
                "Use the public ansible-playbook CLI, localhost inventory, local "
                "connection, and temporary YAML. Do not instantiate or mock internals."
            ),
            "contract_test_renderer": "ansible_playbook_cli",
            "require_expected_counts": True,
            "generated_test_command": test_prefix + " {test_files}",
        }
    else:
        context_files = INCLUDE_LOOP_CONTEXT_FILES
        desired_behavior = (
            "Preserve a custom include_tasks loop variable and a value derived "
            "from it from task parsing through variable preparation, templating, "
            "and task execution without sharing, re-evaluating, or dropping it."
        )
        rationale = (
            "Transfer the prior solver's state-tracing weakness to Ansible task "
            "construction and execution, without copying the role-cache task."
        )
        blueprint = {
            "capability_gap": "cross-layer state and identity propagation",
            "failure_stage": "context_management",
            "required_topology": "include parser -> loop variable carrier -> templar -> executor",
            "forbidden_copy": "role cache, Role.load, duplicate role execution",
            "forbidden_terms": ["RoleInclude", "duplicate role", "role cache"],
            "contract_scenario": (
                "A pytest test creates a temporary playbook and included task file, "
                "runs ansible-playbook through subprocess, and asserts observable "
                "output for two include_tasks iterations using a custom loop_var."
            ),
            "contract_test_style": (
                "Use the public ansible-playbook CLI, localhost inventory, local "
                "connection, and temporary YAML files. Do not instantiate or mock "
                "Ansible internal classes."
            ),
            "contract_test_renderer": "ansible_playbook_cli",
            "generated_test_command": test_prefix + " {test_files}",
        }
    failure = FailureSignature(
        file="",
        symbol="",
        error_type=error_type,
        error_message=error_message,
        pattern=failure_pattern,
    )
    plan = BugGenerationPlan(
        plan_id="ansible_trajectory_repo_chain_probe",
        source_trajectory_ids=trajectories,
        failure_signature=failure,
        target_repo_id="ansible",
        target_base_commit=base_commit,
        target_file=context_files[0],
        target_files=context_files,
        strategy="repo_chain",
        operator="trajectory_conditioned_chain_mutation",
        constraints=BugConstraints(
            min_modified_files=3,
            max_modified_files=6,
            max_modified_lines=180,
            allow_test_edits=False,
            desired_behavior=desired_behavior,
            generation_timeout_sec=args.generation_timeout,
            context_file_budget=10,
            min_mutation_sites=3,
            max_mutation_sites=7,
            require_generated_tests=True,
        ),
        rationale=rationale,
        task_blueprint=blueprint,
        model=args.model,
        seed=7301,
    )
    adapter = VLLMChatAdapter(args.vllm_host, args.vllm_port, args.model)
    engine = SWESmithEngine(agent_adapter=adapter)
    started = time.time()
    candidates = engine.generate(
        plan,
        str(root / "initial_agent" / "src"),
        repo_spec,
        str(output_dir / "generation"),
    )
    report = {
        "base_commit": base_commit,
        "strategy": "repo_chain",
        "scenario_profile": args.scenario_profile,
        "failure_signature": {
            "error_type": error_type,
            "error_message": error_message,
            "pattern": failure_pattern,
        },
        "generated": len(candidates),
        "generator_rejection": engine.repo_chain.last_rejection,
        "generation_runtime_sec": round(time.time() - started, 3),
        "task": None,
        "solver_result": None,
    }
    report_path = output_dir / "repo_chain_probe_report.json"
    if not candidates:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return 2

    candidate = candidates[0]
    chain = dict(candidate.generation_metadata.get("chain_plan") or {})
    problem = str(candidate.generation_metadata.get("problem_statement") or "").strip()
    if not problem:
        raise RuntimeError("repo_chain candidate is missing its generated problem statement")
    validator = CandidateValidator(
        output_dir / "validator",
        test_timeout_sec=args.test_timeout,
        max_patch_lines=200,
        forbid_test_file_edits=True,
    )
    solver_runtime = closed_loop._solver_public_ansible_runtime()
    public_solver_command = closed_loop._control_test_command(solver_runtime)
    control_command = closed_loop._control_test_command(runtime)
    task = closed_loop._validate_and_package(
        candidate=candidate,
        phase="repo_chain_probe",
        domain_id="trajectory_repo_chain",
        problem_statement=problem,
        test_files=[],
        source_repo=source_repo,
        base_commit=base_commit,
        test_prefix=test_prefix,
        validator=validator,
        output_dir=output_dir,
        test_timeout=args.test_timeout,
        source_trajectory_ids=trajectories,
        control_test_command=control_command,
        strictness_policy="synthetic_causal",
        solver_test_command_override=public_solver_command,
    )
    report.update(
        {
            "candidate_id": candidate.candidate_id,
            "modified_files": candidate.modified_files,
            "modified_lines": candidate.generation_metadata.get("modified_lines"),
            "generated_test_files": candidate.generation_metadata.get("generated_test_files"),
            "chain_plan": chain,
            "causal_ablation": candidate.generation_metadata.get("causal_ablation"),
            "task": task,
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
        os.environ.setdefault(
            "GODEL0_SOLVER_SCRATCH_ROOT",
            str(Path(os.environ.get("TMPDIR") or "/tmp") / f"godel0_chain_solver_{os.getpid()}"),
        )
        report["solver_result"] = closed_loop._run_solver(
            task=task,
            phase="repo_chain_probe",
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
        f"result={report['result']} candidate={candidate.candidate_id} "
        f"files={candidate.modified_files} report={report_path}",
        flush=True,
    )
    return 0 if task and task.get("strict_repo_level") else 2


if __name__ == "__main__":
    raise SystemExit(main())
