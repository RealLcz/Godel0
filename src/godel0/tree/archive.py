"""Node archive: persistent storage for all nodes."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..schemas.node import NodeRecord, NodeStatus
from ..storage.atomic import atomic_write_json, read_json
from ..storage.jsonl import append_jsonl, read_all_jsonl


class NodeArchive:
    """Persistent archive of all node records."""

    def __init__(self, archive_path: Path):
        self.archive_path = Path(archive_path)
        self._cache: dict[str, NodeRecord] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        for raw in read_all_jsonl(self.archive_path):
            record = NodeRecord(**raw)
            self._cache[record.node_id] = record
        self._loaded = True

    def add(self, record: NodeRecord) -> None:
        """Add a node to the archive."""
        self._ensure_loaded()
        self._cache[record.node_id] = record
        append_jsonl(self.archive_path, record.model_dump())

    def get(self, node_id: str) -> Optional[NodeRecord]:
        """Get a node by ID."""
        self._ensure_loaded()
        return self._cache.get(node_id)

    def update(self, record: NodeRecord) -> None:
        """Update an existing node record."""
        self._ensure_loaded()
        self._cache[record.node_id] = record
        self._rewrite()

    def children_of(self, node_id: str) -> List[NodeRecord]:
        """Get all children of a node."""
        self._ensure_loaded()
        return [
            r for r in self._cache.values()
            if r.parent_node_id == node_id
        ]

    def eligible_parents(self, min_solved: int = 3, scoring_mode: str = "joint") -> List[NodeRecord]:
        """Get all nodes eligible to be parents.

        In "hgm" scoring mode, additionally enforce the HGM-style proposer
        quality gate via ``NodeRecord.selection_eligible`` (BUG-12). Legacy
        behavior in "joint" mode keeps the old score-based check.
        """
        self._ensure_loaded()
        eligible = []
        for r in self._cache.values():
            if not r.is_eligible_parent(min_solved):
                continue
            if scoring_mode == "hgm":
                # BUG-12: the authoritative HGM quality gate is the explicit
                # selection_eligible flag set by compute_scores().eligible.
                # Do not substitute ``proposer_score > 0`` for the full gate.
                if not getattr(r, "selection_eligible", True):
                    continue
            eligible.append(r)
        return eligible

    def descendants_of(self, node_id: str) -> List[NodeRecord]:
        """Get all descendants (children, grandchildren, ...) of a node."""
        self._ensure_loaded()
        descendants: List[NodeRecord] = []
        pending = list(self.children_of(node_id))
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current.node_id in seen:
                continue
            seen.add(current.node_id)
            descendants.append(current)
            pending.extend(self.children_of(current.node_id))
        return descendants

    def all_nodes(self) -> List[NodeRecord]:
        """Get all nodes."""
        self._ensure_loaded()
        return list(self._cache.values())

    def complete_nodes(self) -> List[NodeRecord]:
        """Get all completed nodes."""
        self._ensure_loaded()
        return [r for r in self._cache.values() if r.status == NodeStatus.COMPLETE]

    def _rewrite(self) -> None:
        """Rewrite the entire archive."""
        records = list(self._cache.values())
        data = [r.model_dump() for r in records]
        import json
        lines = [json.dumps(r, default=str) for r in data]
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        self.archive_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    def save_node_json(self, record: NodeRecord, path: Path) -> None:
        """Save a single node record as JSON."""
        atomic_write_json(path, record.model_dump())
