#!/usr/bin/env python3
"""Generate and validate one trace-grounded repository-level repair task."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godel0-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--repo-pool", default="repo_pool")
    parser.add_argument("--repo-id", default="ansible")
    parser.add_argument("--domain", default="inventory_yaml_e2e")
    parser.add_argument(
        "--generator",
        choices=["procedural", "repo_agent"],
        default="procedural",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=7300)
    parser.add_argument("--test-timeout", type=int, default=180)
    parser.add_argument("--run-solver", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--agent-timeout", type=int, default=1800)
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", default="8000")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.godel0_root).resolve()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "initial_agent" / "src"))
    sys.path.insert(0, str(root))

    from godel0.proposer_trusted.candidate_validator import CandidateValidator
    from godel0.tasks.repo_pool import RepoPool
    from scripts import run_repo_level_closed_loop as closed_loop

    if args.domain not in closed_loop.DOMAINS:
        raise ValueError(f"Unknown domain: {args.domain}")
    domain = closed_loop.DOMAINS[args.domain]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_tmp = output_dir / "runtime_tmp"
    ansible_tmp = runtime_tmp / "ansible_local"
    ansible_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(runtime_tmp)
    os.environ["ANSIBLE_LOCAL_TEMP"] = str(ansible_tmp)
    os.environ["VLLM_HOST"] = args.vllm_host
    os.environ["VLLM_PORT"] = str(args.vllm_port)
    os.environ.pop("QWEN_API_BASE_URL", None)

    pool = RepoPool((root / args.repo_pool).resolve())
    spec = pool.get(args.repo_id)
    if spec is None:
        raise RuntimeError(f"Repository not found: {args.repo_id}")
    source_repo = Path(spec.path)
    if not source_repo.is_absolute():
        source_repo = (root / source_repo).resolve()

    validator = CandidateValidator(
        workspace_root=output_dir / "validator",
        test_timeout_sec=args.test_timeout,
        max_patch_lines=180,
        forbid_test_file_edits=True,
    )
    report: dict[str, Any] = {
        "domain": args.domain,
        "repo_id": spec.repo_id,
        "base_commit": spec.base_commit,
        "generator": args.generator,
        "attempts": [],
        "accepted_task": None,
        "solver_result": None,
    }
    report_path = output_dir / "causal_contract_report.json"

    engine = None
    repo_spec = None
    trace_evidence: dict[str, Any] = {}
    if args.generator == "repo_agent":
        from experiment_adapters.repo_bug_agent_adapter import RepoBugAgentAdapter
        from swesmith.engine import RepoSpec as EngineRepoSpec, SWESmithEngine
        from swesmith.repo_level import RepositoryWorkspace

        contract_test = str(domain.get("contract_test") or "")
        if not contract_test:
            raise ValueError("repo_agent causal generation requires a contract_test")
        trace_path = output_dir / "clean_contract_trace.json"
        with RepositoryWorkspace(str(source_repo), spec.base_commit) as workspace:
            clean_trace = closed_loop._trace_contract_test(
                workspace=Path(workspace),
                test_prefix=spec.test_command,
                test_nodeid=contract_test,
                source_roots=list(domain.get("trace_source_roots") or ["lib"]),
                output_path=trace_path,
                timeout=min(args.test_timeout, 120),
            )
        if not clean_trace:
            raise RuntimeError("Cannot trace the clean end-to-end contract test")
        trace_evidence = closed_loop._compact_contract_trace(
            clean_trace,
            paths=list(domain["anchors"]),
        )
        report["clean_contract_trace"] = trace_evidence
        repo_spec = EngineRepoSpec(
            repo_id=spec.repo_id,
            repo_path=str(source_repo),
            base_commit=spec.base_commit,
            test_command=spec.test_command,
        )
        engine = SWESmithEngine(
            agent_adapter=RepoBugAgentAdapter(
                max_llm_calls=40,
                max_output_tokens=2048,
                shell_timeout_sec=min(args.test_timeout, 120),
            )
        )

    for attempt_index in range(max(1, args.attempts)):
        seed = args.start_seed + attempt_index
        blueprint = {
            "phase": "causal_probe",
            "domain": args.domain,
            "contract": domain["contract"],
            "contract_test": domain.get("contract_test", ""),
            "min_modified_files": int(domain.get("min_modified_files") or 3),
            "generator": "trace_grounded_causal_compose",
        }
        if args.generator == "repo_agent":
            blueprint.update(
                {
                    "generator": "trace_grounded_causal_repo_agent",
                    "validation_command": closed_loop._test_command(
                        spec.test_command,
                        domain["tests"],
                    ),
                    "runtime_contract_trace": trace_evidence,
                    "causal_generation_rules": [
                        "All changed files must remain executed in the failing contract test",
                        "Use one representation, protocol, or value-handoff regression",
                        "Prefer an incomplete migration across producer and consumers",
                        "Do not truncate loops, toggle unrelated booleans, return None, or hard-code test values",
                        "A one-file repair must not restore the complete contract",
                    ],
                }
            )
            assert engine is not None and repo_spec is not None
            task, row = closed_loop._generate_repo_agent_task(
                phase="causal_probe",
                attempt_index=attempt_index,
                domain_id=args.domain,
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
        else:
            task, row = closed_loop._generate_coverage_guided_task(
                phase="causal_probe",
                attempt_index=attempt_index,
                domain_id=args.domain,
                blueprint=blueprint,
                source_trajectory_ids=[],
                source_repo=source_repo,
                base_commit=spec.base_commit,
                test_prefix=spec.test_command,
                validator=validator,
                output_dir=output_dir,
                test_timeout=args.test_timeout,
                seed=seed,
            )
        row["seed"] = seed
        report["attempts"].append(row)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"attempt={attempt_index + 1} seed={seed} "
            f"result={row.get('result')} searched={row.get('searched_mutations')} "
            f"files={row.get('modified_files', [])}",
            flush=True,
        )
        if task:
            report["accepted_task"] = task
            break

    if report["accepted_task"] and args.run_solver:
        from experiment_adapters.common_agent_adapter import CommonAgentAdapter

        report["solver_result"] = closed_loop._run_solver(
            task=report["accepted_task"],
            phase="causal_probe",
            source_repo=source_repo,
            agent_src=root / "initial_agent" / "src",
            adapter=CommonAgentAdapter(),
            model=args.model,
            output_dir=output_dir,
            agent_timeout=args.agent_timeout,
            test_timeout=args.test_timeout,
        )

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report={report_path}", flush=True)
    return 0 if report["accepted_task"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
