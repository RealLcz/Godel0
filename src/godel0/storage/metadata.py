"""Metadata persistence for nodes and archives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ..schemas.node import NodeRecord
from .atomic import atomic_write_json, read_json
from .jsonl import append_jsonl, read_all_jsonl


class MetadataStore:
    """Persists node records and archive entries."""

    def __init__(self, archive_path: Path):
        self.archive_path = Path(archive_path)

    def append_node(self, record: NodeRecord) -> None:
        """Append a node record to the archive."""
        append_jsonl(self.archive_path, record.model_dump())

    def get_node(self, node_id: str) -> Optional[NodeRecord]:
        """Retrieve a node by ID."""
        for raw in read_all_jsonl(self.archive_path):
            if raw.get("node_id") == node_id:
                return NodeRecord(**raw)
        return None

    def all_nodes(self) -> List[NodeRecord]:
        """Return all nodes in the archive."""
        return [NodeRecord(**raw) for raw in read_all_jsonl(self.archive_path)]

    def update_node(self, record: NodeRecord) -> None:
        """Update a node record (rewrites archive)."""
        nodes = self.all_nodes()
        found = False
        for i, n in enumerate(nodes):
            if n.node_id == record.node_id:
                nodes[i] = record
                found = True
                break
        if not found:
            nodes.append(record)
        data = [n.model_dump() for n in nodes]
        atomic_write_json(self.archive_path.with_suffix(".tmp.json"), data)
        lines = [json.dumps(n, default=str) for n in data]
        Path(self.archive_path).write_text("\n".join(lines) + "\n" if lines else "")

    def save_node_json(self, record: NodeRecord, path: Path) -> None:
        """Save a single node record as JSON."""
        atomic_write_json(path, record.model_dump())
