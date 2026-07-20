#!/usr/bin/env python3
"""Run the initial coding agent blindly against an existing Repo Chain candidate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godel0-root", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--agent-timeout", type=int, default=2400)
    parser.add_argument("--test-timeout", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.godel0_root).resolve()
    sys.path[:0] = [str(root / "src"), str(root / "initial_agent" / "src"), str(root)]

    from experiment_adapters.common_agent_adapter import CommonAgentAdapter
    from scripts import run_repo_level_closed_loop as closed_loop
    from swesmith.candidate import CandidateArtifact
    from swesmith.repo_level import run_git

    source_repo = Path(args.repo).resolve()
    candidate_dir = Path(args.candidate_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = CandidateArtifact.from_json(
        (candidate_dir / "candidate.json").read_text(encoding="utf-8")
    )
    metadata = dict(candidate.generation_metadata or {})
    problem_statement = (candidate_dir / "problem_statement.md").read_text(
        encoding="utf-8"
    )
    generated_test_files = list(metadata.get("generated_test_files") or [])
    actual_f2p = [
        f"{generated_test_files[0]}::test_generated_repo_contract[{case}]"
        for case in ("basic_loop_var_propagation", "basic_loop_var_compatibility")
    ] if generated_test_files else []
    runtime = closed_loop._prepare_ansible_runtime(output_dir, include_contracts=False)
    test_prefix = (
        f"HOME={runtime['home']} PATH={runtime['python_bin']}:$PATH "
        "PYTHONPATH=lib:test/lib python -m pytest -p no:cacheprovider --rootdir=."
    )
    public_runtime = closed_loop._solver_public_ansible_runtime()
    task = {
        "task_id": candidate.candidate_id,
        "domain": "trajectory_repo_chain_rejected_probe",
        "base_commit": run_git(str(source_repo), "rev-parse", "HEAD").stdout.strip(),
        "problem_statement": problem_statement,
        "bug_patch": candidate.bug_patch,
        "test_command": str(metadata.get("generated_test_command") or ""),
        "solver_test_command": closed_loop._control_test_command(public_runtime),
        "solver_validation_command": test_prefix + " " + " ".join(generated_test_files),
        "generated_test_patch": str(metadata.get("generated_test_patch") or ""),
        "generated_test_files": generated_test_files,
        "solver_hidden_test_files": [],
        "solver_hidden_reference_files": [],
        "f2p_tests": actual_f2p,
        "p2p_tests": [],
        "validation_mode": "pytest",
    }
    os.environ.setdefault(
        "GODEL0_SOLVER_SCRATCH_ROOT",
        str(Path(os.environ.get("TMPDIR") or "/tmp") / f"existing_chain_solver_{os.getpid()}"),
    )
    result = closed_loop._run_solver(
        task=task,
        phase="existing_repo_chain_probe",
        source_repo=source_repo,
        agent_src=root / "initial_agent" / "src",
        adapter=CommonAgentAdapter(),
        model=args.model,
        output_dir=output_dir,
        agent_timeout=args.agent_timeout,
        test_timeout=args.test_timeout,
    )
    report_path = output_dir / "solver_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result.get("resolved") else 2


if __name__ == "__main__":
    raise SystemExit(main())
