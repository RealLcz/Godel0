"""Unit tests for NodeArchive."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from godel0.schemas.node import NodeRecord, NodeStatus
from godel0.tree.archive import NodeArchive
from godel0.tree.selection import EpsilonGreedySelector, ScoreProportionalSelector
from godel0.tree.node import Node

import random


def make_node(node_id: str, score: float = 0.5, status: NodeStatus = NodeStatus.COMPLETE) -> NodeRecord:
    return NodeRecord(
        node_id=node_id,
        code_commit="abc123",
        code_ref=f"refs/godel0/nodes/{node_id}",
        status=status,
        node_score=score,
        retention_rate=0.9,
        frontier_accuracy=0.5,
        solver_score=score * 0.7,
        proposer_score=score * 0.3,
        generated_task_batch_id="batch_1",
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )


class TestNodeArchive:
    def test_add_and_get(self, node_archive):
        node = make_node("node_1")
        node_archive.add(node)
        retrieved = node_archive.get("node_1")
        assert retrieved is not None
        assert retrieved.node_id == "node_1"

    def test_eligible_parents(self, node_archive):
        for i in range(3):
            node_archive.add(make_node(f"node_{i}", score=0.5 + i * 0.1))
        eligible = node_archive.eligible_parents()
        assert len(eligible) == 3

    def test_non_eligible(self, node_archive):
        node_archive.add(make_node("failed", score=0.0, status=NodeStatus.LEVEL1_FAILED))
        eligible = node_archive.eligible_parents()
        assert len(eligible) == 0

    def test_children_of(self, node_archive):
        parent = make_node("parent")
        node_archive.add(parent)
        child = make_node("child")
        child.parent_node_id = "parent"
        node_archive.add(child)
        children = node_archive.children_of("parent")
        assert len(children) == 1
        assert children[0].node_id == "child"

    def test_update(self, node_archive):
        node = make_node("node_1", score=0.3)
        node_archive.add(node)
        node.node_score = 0.8
        node_archive.update(node)
        retrieved = node_archive.get("node_1")
        assert retrieved.node_score == 0.8


class TestSelectors:
    def test_epsilon_greedy_selects_best(self, node_archive):
        for i in range(5):
            node_archive.add(make_node(f"node_{i}", score=0.1 * i))
        rng = random.Random(42)
        selector = EpsilonGreedySelector(epsilon=0.0)
        selected = selector.select(node_archive, rng)
        assert selected.node_score == 0.4

    def test_score_proportional(self, node_archive):
        for i in range(5):
            node_archive.add(make_node(f"node_{i}", score=0.1 * (i + 1)))
        rng = random.Random(42)
        selector = ScoreProportionalSelector()
        selected = selector.select(node_archive, rng)
        assert selected is not None


class TestNode:
    def test_node_properties(self):
        record = make_node("test", score=0.7)
        node = Node(record=record)
        assert node.node_id == "test"
        assert node.score == 0.7
        assert node.is_complete
        assert node.is_eligible_parent
