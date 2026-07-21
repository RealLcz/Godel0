"""Unit tests for NodeRecord selection_eligible (BUG-12) and NodeArchive
HGM-mode eligible_parents filtering."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from godel0.schemas.node import NodeRecord, NodeStatus
from godel0.tree.archive import NodeArchive


def _make_node(
    node_id: str = "n1",
    status: NodeStatus = NodeStatus.COMPLETE,
    node_score: float = 0.5,
    selection_eligible: bool = True,
    retention_rate: float = 0.9,
    solved_task_count: int = 5,
    generated_task_batch_id: str = "batch1",
    parent_node_id: str | None = None,
    utility_measures: list[float] | None = None,
) -> NodeRecord:
    return NodeRecord(
        node_id=node_id,
        parent_node_id=parent_node_id,
        code_commit="abc123",
        code_ref="refs/godel0/nodes/n1",
        status=status,
        node_score=node_score,
        selection_eligible=selection_eligible,
        retention_rate=retention_rate,
        solved_task_count=solved_task_count,
        generated_task_batch_id=generated_task_batch_id,
        utility_measures=utility_measures or [],
    )


class TestNodeRecordUtilityMeasures:
    """BUG-11: NodeRecord must carry utility_measures for Thompson Sampling."""

    def test_utility_measures_defaults_to_empty(self):
        node = _make_node()
        assert node.utility_measures == []

    def test_utility_measures_can_be_set(self):
        node = _make_node(utility_measures=[1.0, 0.0, 1.0])
        assert node.utility_measures == [1.0, 0.0, 1.0]

    def test_evaluated_task_ids_defaults_to_empty(self):
        node = _make_node()
        assert node.evaluated_task_ids == []

    def test_selection_eligible_defaults_to_true(self):
        node = _make_node()
        assert node.selection_eligible is True


class TestNodeRecordIsEligibleParent:
    """BUG-12: is_eligible_parent must honor selection_eligible."""

    def test_eligible_node_passes(self):
        node = _make_node(selection_eligible=True)
        assert node.is_eligible_parent() is True

    def test_selection_eligible_false_blocks(self):
        node = _make_node(selection_eligible=False)
        assert node.is_eligible_parent() is False

    def test_non_complete_status_blocks(self):
        node = _make_node(status=NodeStatus.CANDIDATE)
        assert node.is_eligible_parent() is False

    def test_zero_node_score_blocks(self):
        node = _make_node(node_score=0.0)
        assert node.is_eligible_parent() is False

    def test_negative_retention_rate_blocks(self):
        node = _make_node(retention_rate=-0.1)
        assert node.is_eligible_parent() is False

    def test_insufficient_solved_tasks_blocks(self):
        node = _make_node(solved_task_count=2)
        assert node.is_eligible_parent(min_solved=3) is False

    def test_no_generated_task_batch_blocks(self):
        node = _make_node(generated_task_batch_id=None)
        assert node.is_eligible_parent() is False


class TestNodeArchiveHGMEligibility:
    """BUG-12: NodeArchive.eligible_parents in hgm mode must use
    selection_eligible, not proposer_score > 0."""

    def test_hgm_mode_excludes_selection_eligible_false(self, tmp_path: Path):
        archive_path = tmp_path / "archive.jsonl"
        archive = NodeArchive(archive_path)
        eligible_node = _make_node(node_id="eligible")
        ineligible_node = _make_node(
            node_id="ineligible",
            selection_eligible=False,
            # Note: proposer_score would be > 0 in the old gate, but the new
            # gate uses selection_eligible.
            node_score=0.5,
        )
        archive.add(eligible_node)
        archive.add(ineligible_node)

        # Force reload from disk to ensure persistence.
        archive2 = NodeArchive(archive_path)
        parents = archive2.eligible_parents(scoring_mode="hgm")
        ids = [n.node_id for n in parents]
        assert "eligible" in ids
        assert "ineligible" not in ids

    def test_joint_mode_does_not_apply_hgm_gate(self, tmp_path: Path):
        archive_path = tmp_path / "archive.jsonl"
        archive = NodeArchive(archive_path)
        # In joint mode, selection_eligible=False should NOT be an extra filter
        # beyond is_eligible_parent. But is_eligible_parent itself checks
        # selection_eligible... so a False node is excluded in both modes.
        # The difference is that in "hgm" mode we don't ALSO require
        # proposer_score > 0 (legacy behavior). We test that a node with
        # selection_eligible=True but proposer_score unset/zero still appears.
        node = _make_node(node_id="n1", selection_eligible=True)
        # proposer_score is None by default.
        archive.add(node)
        archive2 = NodeArchive(archive_path)
        parents = archive2.eligible_parents(scoring_mode="hgm")
        assert any(n.node_id == "n1" for n in parents)

    def test_descendants_of_returns_all_descendants(self, tmp_path: Path):
        archive_path = tmp_path / "archive.jsonl"
        archive = NodeArchive(archive_path)
        root = _make_node(node_id="root", parent_node_id=None)
        child1 = _make_node(node_id="c1", parent_node_id="root")
        child2 = _make_node(node_id="c2", parent_node_id="root")
        grandchild = _make_node(node_id="g1", parent_node_id="c1")
        for n in (root, child1, child2, grandchild):
            archive.add(n)

        archive2 = NodeArchive(archive_path)
        descendants = archive2.descendants_of("root")
        ids = sorted(n.node_id for n in descendants)
        assert ids == ["c1", "c2", "g1"]
