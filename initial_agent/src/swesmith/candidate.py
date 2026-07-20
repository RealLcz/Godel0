from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class CandidateArtifact:
    candidate_id: str
    plan_id: str
    strategy: str
    operator: str
    target_file: str
    target_symbol: str
    bug_patch: str
    mutation_site: dict
    seed: int
    before_snippet: str
    after_snippet: str
    generation_metadata: dict = field(default_factory=dict)
    modified_files: List[str] = field(default_factory=list)
    modified_entities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateArtifact":
        return cls(
            candidate_id=data["candidate_id"],
            plan_id=data["plan_id"],
            strategy=data["strategy"],
            operator=data["operator"],
            target_file=data["target_file"],
            target_symbol=data.get("target_symbol", ""),
            bug_patch=data["bug_patch"],
            mutation_site=data.get("mutation_site", {}),
            seed=data.get("seed", 0),
            before_snippet=data.get("before_snippet", ""),
            after_snippet=data.get("after_snippet", ""),
            generation_metadata=data.get("generation_metadata", {}),
            modified_files=list(data.get("modified_files") or []),
            modified_entities=list(data.get("modified_entities") or []),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CandidateArtifact":
        return cls.from_dict(json.loads(json_str))

    def save(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        candidate_path = os.path.join(output_dir, "candidate.json")
        with open(candidate_path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

        patch_path = os.path.join(output_dir, "bug.patch")
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(self.bug_patch)

        return candidate_path

    def summary(self) -> str:
        return (
            f"CandidateArtifact(id={self.candidate_id}, plan={self.plan_id}, "
            f"strategy={self.strategy}, operator={self.operator}, "
            f"file={self.target_file}, symbol={self.target_symbol}, "
            f"modified_files={len(self.modified_files)})"
        )
