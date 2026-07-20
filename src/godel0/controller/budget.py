"""Budget manager for the evolution loop."""

from __future__ import annotations

from dataclasses import dataclass


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
