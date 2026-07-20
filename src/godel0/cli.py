"""Command-line interface for Godel0."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config, save_config


def _cmd_run(args: argparse.Namespace) -> int:
    from .controller.orchestrator import EvolutionOrchestrator

    overrides = {}
    if args.run_name:
        overrides["run.run_name"] = args.run_name
    if args.max_nodes:
        overrides["run.max_nodes"] = args.max_nodes
    if args.resume_from:
        overrides["run.resume_from"] = args.resume_from

    config = load_config(args.config, overrides=overrides)
    orchestrator = EvolutionOrchestrator.from_config(config)
    orchestrator.run()
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    from .controller.resume import ResumeManager

    rm = ResumeManager(Path(args.run_dir))
    rm.resume()
    return 0


def _cmd_export_solver_core(args: argparse.Namespace) -> int:
    from scripts.export_solver_core import export_solver_core

    export_solver_core(
        source_repo=Path(args.source_repo),
        source_commit=args.source_commit,
        output=Path(args.output),
    )
    return 0


def _cmd_verify_solver_core(args: argparse.Namespace) -> int:
    from scripts.verify_solver_core import verify_solver_core

    ok = verify_solver_core(
        code_dir=Path(args.code_dir),
        lock_file=Path(args.lock_file),
    )
    return 0 if ok else 1


def _cmd_validate_agent_codebase(args: argparse.Namespace) -> int:
    from scripts.validate_agent_codebase import validate_agent_codebase

    ok = validate_agent_codebase(Path(args.code_dir))
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="godel0",
        description="Godel0: Self-improving coding agent with built-in Proposer and SWE-smith",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the evolution loop")
    p_run.add_argument("--config", default="configs/default.yaml")
    p_run.add_argument("--run-name", default=None)
    p_run.add_argument("--max-nodes", type=int, default=None)
    p_run.add_argument("--resume-from", default=None)
    p_run.set_defaults(func=_cmd_run)

    p_resume = sub.add_parser("resume", help="Resume an interrupted run")
    p_resume.add_argument("--run-dir", required=True)
    p_resume.set_defaults(func=_cmd_resume)

    p_export = sub.add_parser("export-solver-core", help="Export Solver Core from HGM/DGM repo")
    p_export.add_argument("--source-repo", required=True)
    p_export.add_argument("--source-commit", default=None)
    p_export.add_argument("--output", default="initial_agent/src")
    p_export.set_defaults(func=_cmd_export_solver_core)

    p_verify = sub.add_parser("verify-solver-core", help="Verify Solver Core checksums")
    p_verify.add_argument("--code-dir", default="initial_agent/src")
    p_verify.add_argument("--lock-file", default="initial_agent/solver_core.lock.json")
    p_verify.set_defaults(func=_cmd_verify_solver_core)

    p_validate = sub.add_parser("validate-agent", help="Validate an agent codebase")
    p_validate.add_argument("--code-dir", default="initial_agent/src")
    p_validate.set_defaults(func=_cmd_validate_agent_codebase)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
