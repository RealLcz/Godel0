"""Unit tests for the trusted proposer transport schema (BUG-25)."""
from __future__ import annotations

import pytest

from godel0.schemas.proposer_transport import (
    CandidateTransportV1,
    ProposerRequestV1,
    ProposerResultV1,
)


class TestCandidateTransportV1:
    def test_accepts_minimal_fields(self):
        cand = CandidateTransportV1(candidate_id="c1")
        assert cand.candidate_id == "c1"
        assert cand.patch == ""
        assert cand.generation_metadata == {}

    def test_accepts_extra_fields(self):
        cand = CandidateTransportV1(
            candidate_id="c1",
            patch="diff",
            custom_field="anything",  # extra=allow
        )
        assert cand.candidate_id == "c1"
        assert cand.patch == "diff"


class TestProposerResultV1:
    def test_from_dict_parses_minimal_result(self):
        data = {
            "run_id": "r1",
            "node_id": "n1",
            "completed": True,
            "accepted_candidates": [],
            "rejected_candidates": [],
            "pending_candidates": [],
        }
        result = ProposerResultV1.from_dict(data)
        assert result.run_id == "r1"
        assert result.node_id == "n1"
        assert result.completed is True
        assert result.accepted_candidates == []

    def test_from_dict_coerces_candidates(self):
        data = {
            "run_id": "r1",
            "node_id": "n1",
            "accepted_candidates": [
                {"candidate_id": "c1", "patch": "diff1", "strategy": "repo_chain"},
                {"candidate_id": "c2", "patch": "diff2"},
            ],
            "rejected_candidates": [],
            "pending_candidates": [],
        }
        result = ProposerResultV1.from_dict(data)
        assert len(result.accepted_candidates) == 2
        assert result.accepted_candidates[0].candidate_id == "c1"
        assert result.accepted_candidates[0].patch == "diff1"
        assert result.accepted_candidates[0].strategy == "repo_chain"
        assert isinstance(result.accepted_candidates[0], CandidateTransportV1)

    def test_from_dict_handles_missing_candidates_gracefully(self):
        data = {"run_id": "r1", "node_id": "n1"}
        result = ProposerResultV1.from_dict(data)
        assert result.accepted_candidates == []
        assert result.rejected_candidates == []
        assert result.pending_candidates == []

    def test_from_dict_rejects_non_dict(self):
        with pytest.raises(ValueError):
            ProposerResultV1.from_dict("not a dict")  # type: ignore[arg-type]


class TestProposerRequestV1:
    def test_accepts_split_trajectory_buckets(self):
        # BUG-08/09: the trusted transport carries the split trajectory buckets.
        req = ProposerRequestV1(
            node_id="n1",
            run_id="r1",
            agent_code_dir="/agent",
            repo_pool_dir="/repos",
            task_store_dir="/tasks",
            output_dir="/out",
            parent_failure_trajectories=["/traj/parent1.jsonl"],
            current_child_level1_trajectories=["/traj/child1.jsonl"],
        )
        assert req.parent_failure_trajectories == ["/traj/parent1.jsonl"]
        assert req.current_child_level1_trajectories == ["/traj/child1.jsonl"]
