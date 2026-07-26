"""Budget manager for the evolution loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..schemas.node import NodeStatus


@dataclass
class Budget:
    """Track successful epochs separately from failed expansion attempts."""
    max_nodes: int = 200
    max_expansions: int = 200
    nodes_created: int = 0
    expansions_attempted: int = 0

    def can_expand(self) -> bool:
        return (
            self.expansions_attempted < self.max_expansions
            and self.nodes_created < self.max_nodes
        )

    def record_expansion(self) -> None:
        self.expansions_attempted += 1

    def record_node(self) -> None:
        self.nodes_created += 1

    def exhausted(self) -> bool:
        return not self.can_expand()

    def remaining(self) -> int:
        return max(0, self.max_nodes - self.nodes_created)

    @classmethod
    def from_archive(
        cls,
        archive,
        *,
        max_nodes: int,
        max_expansions: int,
        expansions_attempted: Optional[int] = None,
    ) -> "Budget":
        """Rebuild counters so a resumed run does not redo finished epochs.

        ``nodes_created`` counts completed *child* nodes (root is the seed,
        not an epoch). ``expansions_attempted`` defaults to the number of
        non-root nodes already present so failed children still consume the
        attempt budget the way a live run would have.
        """
        nodes = list(archive.all_nodes()) if archive is not None else []
        complete_children = [
            n
            for n in nodes
            if getattr(n, "parent_node_id", None) is not None
            and getattr(n, "status", None) == NodeStatus.COMPLETE
        ]
        non_root = [n for n in nodes if getattr(n, "parent_node_id", None) is not None]
        attempted = (
            int(expansions_attempted)
            if expansions_attempted is not None
            else len(non_root)
        )
        return cls(
            max_nodes=int(max_nodes),
            max_expansions=int(max_expansions),
            nodes_created=len(complete_children),
            expansions_attempted=max(len(complete_children), attempted),
        )
