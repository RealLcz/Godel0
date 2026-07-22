from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

from .code_locator import RepoSpec
from .request import CandidateArtifact, ProposerRequest, ProposerResult, new_candidate_id
from .runner import ProposerRunner
from .schemas import BugGenerationPlan


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proposer_main",
        description="Godel0 Initial Proposer: generate bug candidates from solver "
        "trajectories for trusted validation.",
    )
    parser.add_argument(
        "--request",
        required=True,
        help="Path to the ProposerRequest JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        required=False,
        default=None,
        help="Directory to write proposer_result.json. Defaults to the request's output_dir.",
    )
    parser.add_argument(
        "--engine",
        default="swesmith",
        help="Engine to use for candidate generation. Default: swesmith.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run the pipeline without invoking the generation engine.",
    )
    return parser


def _build_runner(args: argparse.Namespace, request: ProposerRequest) -> ProposerRunner:
    agent_adapter = None
    try:
        from experiment_adapters.common_agent_adapter import CommonAgentAdapter

        agent_adapter = CommonAgentAdapter()
    except Exception:
        # Procedural generators remain usable without an LLM adapter.
        agent_adapter = None
    engine = None
    if not args.dry_run:
        engine = _load_engine(args.engine, request, agent_adapter)
    workflow_config = getattr(request, "workflow_config", None) or None
    return ProposerRunner(
        agent_adapter=agent_adapter,
        engine=engine,
        workflow_config=workflow_config,
        allow_workflow_fallback=bool(
            getattr(request, "allow_workflow_fallback", False)
        ),
        allow_human_curated_data=bool(
            getattr(request, "allow_human_curated_data", False)
        ),
    )


def _load_engine(name: str, request: ProposerRequest, agent_adapter=None):
    """Load an engine by name.

    The real SWESmithEngine lives in `swesmith/engine.py`. This loader
    imports it lazily so the proposer skeleton remains importable even
    before the engine is fully wired.
    """
    try:
        from swesmith.engine import SWESmithEngine  # type: ignore
    except Exception:
        return _StubEngine()
    try:
        return SWESmithEngine(agent_adapter=agent_adapter)
    except Exception:
        return SWESmithEngine(agent_adapter)


class _StubEngine:
    """Fallback engine used when SWE-smith is not yet wired.

    It emits a single empty-patch candidate per plan so the pipeline can
    be exercised end-to-end. Real generation is delegated to SWESmithEngine.
    """

    def generate(
        self,
        plan: BugGenerationPlan,
        node_code_dir: str,
        repo_spec: RepoSpec,
        output_dir: str,
    ) -> list:
        return [
            CandidateArtifact(
                candidate_id=new_candidate_id(),
                plan_id=plan.plan_id,
                repo_id=plan.target_repo_id,
                base_commit=plan.target_base_commit,
                file_path=plan.target_file,
                symbol_name=plan.target_symbol,
                strategy=plan.strategy,
                patch="",
                status="pending_validation",
            )
        ]


def main() -> int:
    parser = _build_argument_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.request):
        print(f"[proposer] request file not found: {args.request}", file=sys.stderr)
        return 2

    request = ProposerRequest.load(args.request)
    output_dir = args.output_dir or request.output_dir
    os.makedirs(output_dir, exist_ok=True)

    runner = _build_runner(args, request)
    result = runner.generate_batch(request)
    result_path = result.save(output_dir)
    print(f"[proposer] result saved to {result_path}")
    print(
        f"[proposer] accepted={len(result.accepted_candidates)} "
        f"rejected={len(result.rejected_candidates)} completed={result.completed}"
    )
    return 0 if result.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
