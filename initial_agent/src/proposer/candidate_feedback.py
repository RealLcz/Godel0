from __future__ import annotations

import json
import os
import re
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

    def scoped_rejections(
        self,
        feedbacks: List[ValidationFeedback],
        *,
        repo_id: str = "",
        base_commit: str = "",
        context_files: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return structured rejections relevant to the current repo context."""
        context = set(context_files or [])
        records: List[Dict[str, Any]] = []
        for feedback in feedbacks:
            if feedback.accepted:
                continue
            notes = dict(feedback.notes or {})
            feedback_repo = str(notes.get("repo") or notes.get("repo_id") or "")
            feedback_commit = str(notes.get("base_commit") or "")
            if repo_id and feedback_repo and feedback_repo != repo_id:
                continue
            if base_commit and feedback_commit and feedback_commit != base_commit:
                continue
            file_path = str(
                notes.get("file")
                or notes.get("file_path")
                or notes.get("target_file")
                or ""
            )
            reason_code = str(
                notes.get("reason_code")
                or self._reason_code(feedback.reason)
            )
            symbol = str(
                notes.get("symbol")
                or notes.get("qualified_name")
                or notes.get("target_symbol")
                or ""
            )
            parsed_sites = (
                [(file_path, symbol)]
                if file_path or symbol
                else self._sites_from_reason(feedback.reason)
            )
            if not parsed_sites:
                parsed_sites = [("", "")]
            for parsed_file, parsed_symbol in parsed_sites:
                if context and parsed_file and parsed_file not in context:
                    continue
                records.append(
                    {
                        "candidate_id": feedback.candidate_id,
                        "repo": feedback_repo,
                        "base_commit": feedback_commit,
                        "file": parsed_file,
                        "symbol": parsed_symbol,
                        "symbol_id": str(notes.get("symbol_id") or ""),
                        "reason_code": reason_code,
                        "reason": feedback.reason,
                        "notes": notes,
                    }
                )
        return records

    @staticmethod
    def _sites_from_reason(reason: str) -> List[tuple[str, str]]:
        """Recover exact Python sites from legacy engine rejection text."""
        return list(
            dict.fromkeys(
                re.findall(
                    r"([\w./-]+\.py)::"
                    r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)",
                    str(reason or ""),
                )
            )
        )

    @staticmethod
    def _reason_code(reason: str) -> str:
        text = str(reason or "").strip()
        if not text:
            return "validation_rejected"
        prefix = text.split(":", 1)[0].strip().lower()
        normalized = "".join(
            char if char.isalnum() else "_" for char in prefix
        ).strip("_")
        return normalized or "validation_rejected"

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
