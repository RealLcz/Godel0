"""Resume manager for interrupted runs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..errors import ResumeError
from ..storage.jsonl import read_all_jsonl


NODE_STAGES = [
    "CREATED",
    "CYCLE_SUMMARY_DONE",
    "SPECIAL_ALERTS_DONE",
    "EVIDENCE_BUNDLE_DONE",
    "DIAGNOSIS_DONE",
    "SELF_EDIT_RUNNING",
    "SELF_EDIT_DONE",
    "TOOL_GATE_DONE",
    "LEVEL1_RUNNING",
    "LEVEL1_DONE",
    "PROPOSER_RUNNING",
    "PROPOSER_DONE",
    "LEVEL2_RUNNING",
    "LEVEL2_DONE",
    "COMPLETE",
]


class ResumeManager:
    """Manages resumption of interrupted runs."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.nodes_dir = self.run_dir / "nodes"

    def get_node_stage(self, node_id: str) -> str:
        """Determine the stage a node reached."""
        node_dir = self.nodes_dir / node_id
        if not node_dir.exists():
            return "UNKNOWN"

        if (node_dir / "scores.json").exists():
            return "COMPLETE"
        if (node_dir / "level2" / "result.json").exists():
            return "LEVEL2_DONE"
        if (node_dir / "proposer" / "generation_summary.json").exists():
            return "PROPOSER_DONE"
        if (node_dir / "level1" / "result.json").exists():
            return "LEVEL1_DONE"
        if (node_dir / "gates" / "patch_guard.json").exists():
            return "TOOL_GATE_DONE"
        if (node_dir / "self_evolve" / "final.patch").exists():
            return "SELF_EDIT_DONE"
        if (node_dir / "diagnosis" / "diagnosis.json").exists():
            return "DIAGNOSIS_DONE"
        if (node_dir / "diagnosis" / "evidence_bundle.json").exists():
            return "EVIDENCE_BUNDLE_DONE"
        if (node_dir / "diagnosis" / "special_alerts.json").exists():
            return "SPECIAL_ALERTS_DONE"
        if (node_dir / "diagnosis" / "cycle_summary.json").exists():
            return "CYCLE_SUMMARY_DONE"
        if (node_dir / "node.json").exists():
            return "CREATED"
        return "UNKNOWN"

    def resume(self) -> None:
        """Resume the run from where it left off."""
        if not self.run_dir.exists():
            raise ResumeError(f"Run directory not found: {self.run_dir}")

        events = self.run_dir / "events.jsonl"
        if events.exists():
            records = read_all_jsonl(events)
            print(f"Found {len(records)} events in run log")

        if self.nodes_dir.exists():
            for node_dir in self.nodes_dir.iterdir():
                if node_dir.is_dir():
                    stage = self.get_node_stage(node_dir.name)
                    print(f"Node {node_dir.name}: stage={stage}")

        config_path = self.run_dir / "config.resolved.yaml"
        if not config_path.exists():
            raise ResumeError(f"Resolved config not found: {config_path}")

        from ..config import load_config
        from .orchestrator import EvolutionOrchestrator

        config = load_config(
            config_path,
            overrides={
                "run.run_name": self.run_dir.name,
                "paths.runs": str(self.run_dir.parent),
            },
        )
        orchestrator = EvolutionOrchestrator.from_config(config)
        orchestrator.run()
