"""Duplicate detection for candidate tasks."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from ..git.patch import patch_hash, extract_changed_files


@dataclass(frozen=True)
class DuplicateAssessment:
    """Structured duplicate result for admission and novelty reporting."""

    is_unique: bool
    classification: str = "unique"
    component_overlap_ratio: float = 0.0
    novelty_score: float = 1.0


@dataclass(frozen=True)
class _PatchIdentity:
    patch_fingerprint: str
    signature: str
    component_hashes: frozenset[str]
    changed_files: frozenset[str]


class DuplicateDetector:
    """Detects duplicate candidates based on patch content and metadata."""

    def __init__(self):
        self._seen_hashes: set[str] = set()
        self._seen_signatures: set[str] = set()
        self._seen_patches: list[_PatchIdentity] = []
        self._lock = RLock()

    def is_unique(
        self,
        patch: str,
        repo_id: str = "",
        target_file: str = "",
        target_symbol: str = "",
        operator: str = "",
    ) -> bool:
        """Return whether a candidate is unique without registering it."""
        assessment = self.assess(
            patch,
            repo_id=repo_id,
            target_file=target_file,
            target_symbol=target_symbol,
            operator=operator,
        )
        return assessment.is_unique

    def assess(
        self,
        patch: str,
        repo_id: str = "",
        target_file: str = "",
        target_symbol: str = "",
        operator: str = "",
    ) -> DuplicateAssessment:
        """Classify exact, near, and partial component overlap."""
        identity = self._identity(
            patch,
            repo_id=repo_id,
            target_file=target_file,
            target_symbol=target_symbol,
            operator=operator,
        )
        with self._lock:
            return self._assess_unlocked(identity)

    def record(
        self,
        patch: str,
        repo_id: str = "",
        target_file: str = "",
        target_symbol: str = "",
        operator: str = "",
    ) -> bool:
        """Atomically register a candidate if it is still unique."""
        identity = self._identity(
            patch,
            repo_id=repo_id,
            target_file=target_file,
            target_symbol=target_symbol,
            operator=operator,
        )
        with self._lock:
            assessment = self._assess_unlocked(identity)
            if not assessment.is_unique:
                return False
            self._record_unlocked(identity)
            return True

    def record_assessment(
        self,
        patch: str,
        repo_id: str = "",
        target_file: str = "",
        target_symbol: str = "",
        operator: str = "",
    ) -> DuplicateAssessment:
        """Atomically assess and register a unique candidate."""
        identity = self._identity(
            patch,
            repo_id=repo_id,
            target_file=target_file,
            target_symbol=target_symbol,
            operator=operator,
        )
        with self._lock:
            assessment = self._assess_unlocked(identity)
            if assessment.is_unique:
                self._record_unlocked(identity)
            return assessment

    def seed_from_patches(
        self,
        patches=None,
    ) -> int:
        """Register already-committed bug patches so resume rejects duplicates.

        ``patches`` is a list of ``(patch_text, repo_id)`` pairs.
        """
        seeded = 0
        for item in patches or []:
            if isinstance(item, (tuple, list)) and len(item) >= 1:
                patch = item[0]
                repo_id = item[1] if len(item) > 1 else ""
            else:
                continue
            if not patch:
                continue
            self.record(str(patch), repo_id=str(repo_id or ""))
            seeded += 1
        return seeded

    def check(
        self,
        patch: str,
        repo_id: str = "",
        target_file: str = "",
        target_symbol: str = "",
        operator: str = "",
    ) -> bool:
        """Check if a candidate is a duplicate.

        Returns True if NOT a duplicate (i.e., it's unique).
        """
        return self.record(
            patch,
            repo_id=repo_id,
            target_file=target_file,
            target_symbol=target_symbol,
            operator=operator,
        )

    def _identity(
        self,
        patch: str,
        *,
        repo_id: str,
        target_file: str,
        target_symbol: str,
        operator: str,
    ) -> _PatchIdentity:
        changed_files = extract_changed_files(patch)
        component_hashes = frozenset(
            patch_hash(block) for block in _split_diff_components(patch)
        )

        # A target/operator signature is useful for one-site procedural bugs, but
        # is too coarse for repository-level candidates sharing the same anchors.
        signature = ""
        if len(changed_files) <= 1 and any(
            [repo_id, target_file, target_symbol, operator]
        ):
            effective_target = (
                changed_files[0] if changed_files else target_file
            )
            signature = (
                f"{repo_id}|{effective_target}|{target_symbol}|{operator}"
            )
        return _PatchIdentity(
            patch_fingerprint=patch_hash(patch),
            signature=signature,
            component_hashes=component_hashes,
            changed_files=frozenset(changed_files),
        )

    def _assess_unlocked(
        self,
        identity: _PatchIdentity,
    ) -> DuplicateAssessment:
        if identity.patch_fingerprint in self._seen_hashes:
            return DuplicateAssessment(False, "full_duplicate", 1.0, 0.0)
        if identity.signature and identity.signature in self._seen_signatures:
            return DuplicateAssessment(False, "signature_duplicate", 1.0, 0.0)

        highest_overlap = 0.0
        near_duplicate = False
        for seen in self._seen_patches:
            shared = identity.component_hashes & seen.component_hashes
            if not shared:
                continue
            denominator = max(
                len(identity.component_hashes),
                len(seen.component_hashes),
                1,
            )
            overlap = len(shared) / denominator
            highest_overlap = max(highest_overlap, overlap)
            file_union = identity.changed_files | seen.changed_files
            file_similarity = (
                len(identity.changed_files & seen.changed_files) / len(file_union)
                if file_union
                else 0.0
            )
            # Component reuse is only a hard rejection when most of both
            # multi-file patches and their file sets are the same.
            if overlap >= 0.75 and file_similarity >= 0.75:
                near_duplicate = True

        if near_duplicate:
            return DuplicateAssessment(
                False,
                "near_duplicate",
                highest_overlap,
                max(0.0, 1.0 - highest_overlap),
            )
        if highest_overlap > 0.0:
            return DuplicateAssessment(
                True,
                "partial_component_reuse",
                highest_overlap,
                max(0.0, 1.0 - highest_overlap),
            )
        return DuplicateAssessment(True)

    def _record_unlocked(self, identity: _PatchIdentity) -> None:
        self._seen_hashes.add(identity.patch_fingerprint)
        if identity.signature:
            self._seen_signatures.add(identity.signature)
        self._seen_patches.append(identity)

    def reset(self) -> None:
        """Clear all seen entries."""
        with self._lock:
            self._seen_hashes.clear()
            self._seen_signatures.clear()
            self._seen_patches.clear()


def _split_diff_components(patch: str) -> list[str]:
    """Split a standard git patch into independently hashable file blocks."""
    blocks: list[str] = []
    current: list[str] = []
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current:
                blocks.append("".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append("".join(current))
    return blocks
