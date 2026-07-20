from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import List, Optional

from .candidate import CandidateArtifact
from .engine import BugGenerationPlan, RepoSpec
from .patch_utils import (
    patch_conflicts,
    count_modified_lines,
    extract_changed_files,
    apply_patch_to_string,
    make_git_diff,
)


@dataclass
class CombinedCandidateRef:
    candidate_id: str
    plan_id: str
    bug_patch: str
    target_file: str = ""
    target_symbol: str = ""
    failure_signature: str = ""


class CombineEngine:
    def __init__(self) -> None:
        pass

    def generate(
        self,
        plan: BugGenerationPlan,
        node_code_dir: str,
        repo_spec: RepoSpec,
        output_dir: str,
        candidates: Optional[List[CombinedCandidateRef]] = None,
    ) -> List[CandidateArtifact]:
        if not candidates:
            candidates = plan.constraints.__dict__.get("candidates", []) if hasattr(plan.constraints, "__dict__") else []
            if not candidates:
                return []

        if len(candidates) < 2:
            return []

        results: List[CandidateArtifact] = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a = candidates[i]
                b = candidates[j]
                if not self._compatible(a, b):
                    continue
                combined = self._combine(a, b)
                if combined:
                    results.append(combined)
                    if output_dir:
                        candidate_dir = os.path.join(output_dir, combined.candidate_id)
                        combined.save(candidate_dir)
        return results

    def _compatible(self, a: CombinedCandidateRef, b: CombinedCandidateRef) -> bool:
        if patch_conflicts(a.bug_patch, b.bug_patch):
            return False
        files_a = set(extract_changed_files(a.bug_patch))
        files_b = set(extract_changed_files(b.bug_patch))
        if files_a and files_b and files_a.isdisjoint(files_b):
            return True
        if not files_a or not files_b:
            return True
        if a.failure_signature and b.failure_signature:
            return a.failure_signature == b.failure_signature
        return True

    def _combine(self, a: CombinedCandidateRef, b: CombinedCandidateRef) -> Optional[CandidateArtifact]:
        combined_patch = a.bug_patch
        if b.bug_patch:
            combined_patch = combined_patch + "\n" + b.bug_patch

        total_lines = count_modified_lines(combined_patch)
        if total_lines > 40:
            return None

        candidate_id = self._make_combined_id(a, b)
        return CandidateArtifact(
            candidate_id=candidate_id,
            plan_id=f"{a.plan_id}+{b.plan_id}",
            strategy="combine",
            operator="combine",
            target_file=a.target_file or b.target_file,
            target_symbol=a.target_symbol or b.target_symbol,
            bug_patch=combined_patch,
            mutation_site={
                "source_candidates": [a.candidate_id, b.candidate_id],
                "total_modified_lines": total_lines,
            },
            seed=0,
            before_snippet="",
            after_snippet="",
            generation_metadata={
                "combined_from": [a.candidate_id, b.candidate_id],
                "application_order": [a.candidate_id, b.candidate_id],
            },
        )

    def _make_combined_id(self, a: CombinedCandidateRef, b: CombinedCandidateRef) -> str:
        raw = f"combine:{a.candidate_id}:{b.candidate_id}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        return f"cand_{digest}"

    def apply_patches_sequentially(
        self,
        original_source: str,
        patches: List[str],
    ) -> str:
        result = original_source
        for patch in patches:
            result = apply_patch_to_string(result, patch)
        return result
