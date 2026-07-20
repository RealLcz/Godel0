"""Tree node for the evolution tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..schemas.node import NodeRecord, NodeStatus


@dataclass
class Node:
    """Runtime tree node. Only holds lightweight references."""
    record: NodeRecord
    children_ids: List[str] = field(default_factory=list)

    @property
    def node_id(self) -> str:
        return self.record.node_id

    @property
    def parent_node_id(self) -> Optional[str]:
        return self.record.parent_node_id

    @property
    def score(self) -> float:
        return self.record.node_score or 0.0

    @property
    def is_complete(self) -> bool:
        return self.record.status == NodeStatus.COMPLETE

    @property
    def is_eligible_parent(self) -> bool:
        return self.record.is_eligible_parent()
