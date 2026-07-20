"""Run context: shared state for a single evolution run."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import Godel0Config
from ..storage.paths import RunPaths


@dataclass
class RunContext:
    """Context for a single evolution run."""
    run_id: str
    config: Godel0Config
    paths: RunPaths
    seed: int = 42

    @classmethod
    def create(cls, config: Godel0Config, runs_dir: Path) -> "RunContext":
        run_id = config.run.run_name or f"run_{uuid.uuid4().hex[:8]}"
        paths = RunPaths(runs_dir, run_id)
        paths.ensure_dirs()
        return cls(
            run_id=run_id,
            config=config,
            paths=paths,
            seed=config.run.seed,
        )
