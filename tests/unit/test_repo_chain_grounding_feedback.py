from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from proposer.candidate_feedback import CandidateFeedbackProcessor, ValidationFeedback
from proposer.code_locator import RepoIndex
from proposer.runner import ProposerRunner
from proposer.schemas import BugGenerationPlan, FailureSignature
from swesmith.repo_chain import RepoChainGenerator


def test_repo_index_assigns_stable_qualified_symbol_ids(tmp_path: Path):
    package = tmp_path / "pkg"
    package.mkdir()
    source = (
        "class First:\n"
        "    def run(self):\n"
        "        return 1\n\n"
        "class Second:\n"
        "    def run(self):\n"
        "        return 2\n"
    )
    (package / "mod.py").write_text(source)

    first = RepoIndex.build("repo", str(tmp_path))
    second = RepoIndex.build("repo", str(tmp_path))
    runs = [
        row
        for row in first.symbols
        if row.get("qualified_name", "").endswith(".run")
    ]

    assert {row["qualified_name"] for row in runs} == {"First.run", "Second.run"}
    assert len({row["symbol_id"] for row in runs}) == 2
    assert [row["symbol_id"] for row in first.symbols] == [
        row["symbol_id"] for row in second.symbols
    ]


def test_symbol_span_requires_exact_qualified_name():
    source = (
        "class First:\n"
        "    def run(self):\n"
        "        return 1\n\n"
        "class Second:\n"
        "    def run(self):\n"
        "        return 2\n"
    )
    generator = RepoChainGenerator()

    _, bare_error = generator._symbol_span(source, "run")
    first_span, exact_error = generator._symbol_span(source, "First.run")

    assert "resolved 0 times" in bare_error
    assert exact_error == ""
    assert source[first_span[0] : first_span[1]].startswith("    def run")


def test_contract_symbol_id_is_grounded_and_unknown_id_rejected(tmp_path: Path):
    path = tmp_path / "pkg.py"
    path.write_text("def target():\n    return 1\n")
    generator = RepoChainGenerator()
    catalog = generator._symbol_catalog(tmp_path, ["pkg.py"])
    symbol_id = catalog[0]["symbol_id"]
    payload = {
        "chain_plan": {
            "mutation_sites": [
                {"symbol_id": symbol_id, "role": "producer", "change": "change value"}
            ]
        }
    }

    grounded = generator._ground_contract_symbols(payload, catalog)
    site = grounded["chain_plan"]["mutation_sites"][0]
    assert site["file"] == "pkg.py"
    assert site["symbol"] == "target"

    site["symbol_id"] = "sym_unknown"
    error = generator._canonical_identity_rejection(
        tmp_path, grounded["chain_plan"], ["pkg.py"]
    )
    assert "unknown canonical symbol_id" in error


def test_plan_match_requires_complete_site_and_file_coverage():
    generator = RepoChainGenerator()
    chain = {
        "mutation_sites": [
            {"file": "a.py", "symbol": "A.run"},
            {"file": "b.py", "symbol": "build"},
        ]
    }
    edits = [
        {"file": "a.py", "symbol": "A.run"},
        {"file": "b.py", "symbol": "build"},
    ]

    assert generator._plan_matches_patch(chain, ["a.py", "b.py"], 2, edits)
    assert not generator._plan_matches_patch(chain, ["a.py"], 2, edits)
    assert not generator._plan_matches_patch(chain, ["a.py", "b.py"], 2, edits[:1])


def test_existing_tests_limit_catalog_to_statically_reachable_modules(
    tmp_path: Path,
):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "provider.py").write_text(
        "def token():\n    return 'ok'\n"
    )
    (tmp_path / "pkg" / "consumer.py").write_text(
        "from pkg.provider import token\n\n"
        "def accepted():\n    return token() == 'ok'\n"
    )
    (tmp_path / "pkg" / "unrelated.py").write_text(
        "def hidden():\n    return False\n"
    )
    (tmp_path / "tests" / "test_contract.py").write_text(
        "from pkg.consumer import accepted\n\n"
        "def test_contract():\n    assert accepted()\n"
    )
    generator = RepoChainGenerator()
    catalog = generator._symbol_catalog(
        tmp_path,
        [
            "pkg/provider.py",
            "pkg/consumer.py",
            "pkg/unrelated.py",
        ],
    )

    grounded = generator._symbols_grounded_by_tests(
        tmp_path, ["tests/test_contract.py"], catalog
    )

    assert {row["file_path"] for row in grounded} == {
        "pkg/provider.py",
        "pkg/consumer.py",
    }


def test_feedback_scope_preserves_notes_and_builds_forbidden_sites():
    feedbacks = [
        ValidationFeedback(
            candidate_id="bad",
            accepted=False,
            reason="existing_contract_did_not_fail:details",
            notes={
                "repo": "repo",
                "base_commit": "abc",
                "file": "pkg/a.py",
                "symbol": "A.run",
                "symbol_id": "sym_bad",
                "reason_code": "existing_contract_did_not_fail",
                "test_family": "units/a",
            },
        ),
        ValidationFeedback(
            candidate_id="other",
            accepted=False,
            reason="invalid",
            notes={"repo": "other", "file": "pkg/a.py"},
        ),
    ]
    records = CandidateFeedbackProcessor().scoped_rejections(
        feedbacks,
        repo_id="repo",
        base_commit="abc",
        context_files=["pkg/a.py"],
    )
    plan = BugGenerationPlan(
        plan_id="p",
        failure_signature=FailureSignature(signature_id="s"),
        target_repo_id="repo",
        target_base_commit="abc",
        target_file="pkg/a.py",
        target_files=["pkg/a.py"],
    )

    ProposerRunner(agent_adapter=SimpleNamespace())._attach_validation_feedback(
        plan, feedbacks
    )

    assert len(records) == 1
    assert records[0]["notes"]["test_family"] == "units/a"
    assert plan.task_blueprint["forbidden_mutation_sites"] == [
        {
            "file": "pkg/a.py",
            "symbol": "A.run",
            "symbol_id": "sym_bad",
            "reason_code": "existing_contract_did_not_fail",
        }
    ]


def test_legacy_engine_reason_recovers_all_invalid_symbol_sites():
    reason = (
        "invalid_chain_plan:planned mutation symbols must exist exactly: "
        "lib/ansible/cli/adhoc.py::AdHocCLI.name: symbol resolved 0 times; "
        "lib/ansible/cli/config.py::ConfigCLI.name: symbol resolved 0 times"
    )
    feedbacks = [
        ValidationFeedback(
            candidate_id="legacy-engine",
            accepted=False,
            reason=reason,
            notes={"attempt": 12, "stage": "contract_generation_failure"},
        )
    ]

    records = CandidateFeedbackProcessor().scoped_rejections(
        feedbacks,
        repo_id="ansible",
        context_files=[
            "lib/ansible/cli/adhoc.py",
            "lib/ansible/cli/config.py",
        ],
    )

    assert [
        (record["file"], record["symbol"], record["reason_code"])
        for record in records
    ] == [
        (
            "lib/ansible/cli/adhoc.py",
            "AdHocCLI.name",
            "invalid_chain_plan",
        ),
        (
            "lib/ansible/cli/config.py",
            "ConfigCLI.name",
            "invalid_chain_plan",
        ),
    ]


def test_repeated_failure_fingerprint_circuit_breaks_symbol(tmp_path: Path):
    (tmp_path / "a.py").write_text("def first():\n    return 1\n")
    generator = RepoChainGenerator()
    catalog = generator._symbol_catalog(tmp_path, ["a.py"])
    symbol = catalog[0]
    blueprint = {
        "trusted_validation_feedback": [
            {
                "file": "a.py",
                "symbol": "first",
                "symbol_id": symbol["symbol_id"],
                "reason_code": "existing_contract_did_not_fail",
            },
            {
                "file": "a.py",
                "symbol": "first",
                "symbol_id": symbol["symbol_id"],
                "reason_code": "existing_contract_did_not_fail",
            },
        ]
    }

    available = generator._available_symbol_catalog(
        tmp_path, ["a.py"], blueprint=blueprint
    )

    assert available == []
    fingerprint = blueprint["feedback_repair"]["failure_fingerprints"][0]
    assert fingerprint["count"] == 2
    assert fingerprint["circuit_broken"] is True
    assert "different canonical symbol" in blueprint["feedback_repair"][
        "required_action"
    ]


def test_bootstrap_receives_feedback_before_generation(tmp_path: Path):
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    (feedback_dir / "bad.json").write_text(
        json.dumps(
            {
                "candidate_id": "bad",
                "accepted": False,
                "reason": "invalid_symbol",
                "notes": {
                    "repo": "repo",
                    "base_commit": "abc",
                    "file": "pkg/a.py",
                    "symbol_id": "sym_bad",
                },
            }
        )
    )
    seen = {}

    class Workflow:
        def bootstrap(self, **kwargs):
            seen.update(kwargs)
            return [], []

    runner = ProposerRunner(
        agent_adapter=SimpleNamespace(),
        allow_workflow_fallback=True,
        workflow_config={"bootstrap_plans_per_call": 1},
    )
    runner._workflow = Workflow()
    request = SimpleNamespace(
        repo_specs=[
            SimpleNamespace(
                repo_id="repo",
                path=str(tmp_path),
                base_commit="abc",
                test_command="pytest",
            )
        ],
        output_dir=str(tmp_path / "out"),
        workflow_config={"bootstrap_plans_per_call": 1},
        target_batch_size=1,
        generation_attempt=0,
        feedback_dir=str(feedback_dir),
    )
    feedbacks = runner.feedback_processor.load_feedback(request.feedback_dir)

    runner._bootstrap_candidates(request, feedbacks=feedbacks)

    assert len(seen["validation_feedback"]) == 1
    assert seen["validation_feedback"][0].notes["symbol_id"] == "sym_bad"
