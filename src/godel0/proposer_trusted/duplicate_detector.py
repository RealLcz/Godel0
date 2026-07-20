"""Duplicate detection for candidate tasks."""

from __future__ import annotations

from threading import RLock

from ..git.patch import patch_hash, extract_changed_files


class DuplicateDetector:
    """Detects duplicate candidates based on patch content and metadata."""

    def __init__(self):
        self._seen_hashes: set[str] = set()
        self._seen_signatures: set[str] = set()
        self._seen_component_hashes: set[str] = set()
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
        identity = self._identity(
            patch,
            repo_id=repo_id,
            target_file=target_file,
            target_symbol=target_symbol,
            operator=operator,
        )
        with self._lock:
            return self._is_unique_unlocked(identity)

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
            if not self._is_unique_unlocked(identity):
                return False
            patch_fingerprint, signature, component_hashes = identity
            self._seen_hashes.add(patch_fingerprint)
            if signature:
                self._seen_signatures.add(signature)
            self._seen_component_hashes.update(component_hashes)
            return True

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
    ) -> tuple[str, str, frozenset[str]]:
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
        return patch_hash(patch), signature, component_hashes

    def _is_unique_unlocked(
        self,
        identity: tuple[str, str, frozenset[str]],
    ) -> bool:
        patch_fingerprint, signature, component_hashes = identity
        if patch_fingerprint in self._seen_hashes:
            return False
        if signature and signature in self._seen_signatures:
            return False
        if component_hashes & self._seen_component_hashes:
            return False
        return True

    def reset(self) -> None:
        """Clear all seen entries."""
        with self._lock:
            self._seen_hashes.clear()
            self._seen_signatures.clear()
            self._seen_component_hashes.clear()


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
