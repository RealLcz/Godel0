from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .request import CandidateArtifact


@dataclass
class ValidationFeedback:
    """A single piece of feedback from the trusted validator."""

    candidate_id: str
    accepted: bool
    reason: str = ""
    notes: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationFeedback":
        return cls(
            candidate_id=str(data.get("candidate_id", "")),
            accepted=bool(data.get("accepted", False)),
            reason=str(data.get("reason", "")),
            notes=dict(data.get("notes", {})),
        )

    @classmethod
    def from_json(cls, path: str) -> "ValidationFeedback":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class CandidateFeedbackProcessor:
    """Processes validation feedback from the trusted validator.

    The proposer must NOT directly write to TaskStore or read trusted
    private inputs. It only interacts with the trusted validator through
    standard request/response files. This processor loads those response
    files and partitions candidates into accepted/rejected lists.
    """

    def load_feedback(self, feedback_dir: Optional[str]) -> List[ValidationFeedback]:
        if not feedback_dir or not os.path.isdir(feedback_dir):
            return []
        feedbacks: List[ValidationFeedback] = []
        for fname in sorted(os.listdir(feedback_dir)):
            if not fname.endswith(".json"):
                continue
            try:
                feedbacks.append(ValidationFeedback.from_json(os.path.join(feedback_dir, fname)))
            except (json.JSONDecodeError, OSError):
                continue
        return feedbacks

    def partition(
        self,
        candidates: List[CandidateArtifact],
        feedbacks: List[ValidationFeedback],
    ) -> Dict[str, List[CandidateArtifact]]:
        """Partition candidates into accepted/rejected based on feedback.

        Candidates without explicit feedback are treated as pending and
        placed in neither list. Returns a dict with keys "accepted" and
        "rejected".
        """
        verdict_by_id = {fb.candidate_id: fb for fb in feedbacks}
        accepted: List[CandidateArtifact] = []
        rejected: List[CandidateArtifact] = []
        for cand in candidates:
            fb = verdict_by_id.get(cand.candidate_id)
            if fb is None:
                continue
            if fb.accepted:
                cand.status = "accepted"
                accepted.append(cand)
            else:
                cand.status = "rejected"
                rejected.append(cand)
        return {"accepted": accepted, "rejected": rejected}

    def summarize(
        self,
        candidates: List[CandidateArtifact],
        feedbacks: List[ValidationFeedback],
    ) -> Dict[str, Any]:
        partitioned = self.partition(candidates, feedbacks)
        return {
            "total": len(candidates),
            "accepted": len(partitioned["accepted"]),
            "rejected": len(partitioned["rejected"]),
            "pending": len(candidates) - len(partitioned["accepted"]) - len(partitioned["rejected"]),
            "rejection_reasons": [
                {"candidate_id": fb.candidate_id, "reason": fb.reason}
                for fb in feedbacks
                if not fb.accepted
            ],
        }


__all__ = [
    "CandidateFeedbackProcessor",
    "ValidationFeedback",
]
