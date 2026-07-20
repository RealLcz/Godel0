from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .candidate import CandidateArtifact
from .operators import get_operator, OPERATORS
from .operators.base import MutationSite
from .patch_utils import make_git_diff, count_modified_lines
from .site_selector import SiteSelector
from .workspace import WorkspaceManager


@dataclass
class BugConstraints:
    min_modified_files: int = 1
    max_modified_files: int = 1
    max_modified_lines: int = 20
    allow_test_edits: bool = False
    require_syntax_valid: bool = True
    desired_behavior: str = ""
    generation_timeout_sec: int = 3600
    context_file_budget: int = 10
    min_mutation_sites: int = 2
    max_mutation_sites: int = 8
    require_generated_tests: bool = False


@dataclass
class FailureSignature:
    file: str = ""
    symbol: str = ""
    error_type: str = ""
    error_message: str = ""
    pattern: str = ""


@dataclass
class BugGenerationPlan:
    plan_id: str
    source_trajectory_ids: List[str] = field(default_factory=list)
    failure_signature: Optional[FailureSignature] = None
    target_repo_id: str = ""
    target_base_commit: str = ""
    target_file: str = ""
    target_symbol: str = ""
    target_files: List[str] = field(default_factory=list)
    target_symbols: List[str] = field(default_factory=list)
    strategy: str = "procedural"
    operator: Optional[str] = None
    constraints: BugConstraints = field(default_factory=BugConstraints)
    rationale: str = ""
    reference_commit: str = ""
    reference_parent: str = ""
    reference_patch: str = ""
    reference_patch_path: str = ""
    task_blueprint: Dict[str, Any] = field(default_factory=dict)
    model: str = ""
    seed: int = 0


@dataclass
class RepoSpec:
    """Specification of a repository checkout available to the engine.

    Fields align with the control layer's RepoSpec:
      - repo_path: path to the checked-out repository
      - base_commit: git commit SHA
      - test_command: command to run tests
    """
    repo_id: str = ""
    repo_path: str = ""
    base_commit: str = ""
    test_command: str = ""
    source_dirs: List[str] = field(default_factory=lambda: ["src", "."])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepoSpec":
        known = {fld for fld in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        # Accept repo_dir as alias for repo_path
        if "repo_dir" in data and "repo_path" not in filtered:
            filtered["repo_path"] = data["repo_dir"]
        return cls(**filtered)


class ProceduralEngine:
    def __init__(self) -> None:
        self.site_selector = SiteSelector()
        self.workspace_manager = WorkspaceManager()

    def generate(
        self,
        plan: BugGenerationPlan,
        node_code_dir: str,
        repo_spec: RepoSpec,
        output_dir: str,
    ) -> List[CandidateArtifact]:
        operator_name = plan.operator or "change_operator"
        if operator_name not in OPERATORS:
            return []

        operator = get_operator(operator_name)

        target_file = plan.target_file
        if not target_file:
            return []

        file_path = target_file
        if not os.path.isabs(file_path):
            file_path = os.path.join(repo_spec.repo_path or node_code_dir, target_file)

        if not os.path.exists(file_path):
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            original_source = f.read()

        sites = operator.enumerate_sites(original_source, target_symbol=plan.target_symbol)
        if not sites:
            return []

        seed = plan.seed if plan.seed else abs(hash((plan.plan_id, operator_name))) % (2 ** 31)
        site = self.site_selector.select(sites, seed=seed)
        if site is None:
            return []

        site.seed = seed
        mutated_source = operator.apply(original_source, site)

        if mutated_source == original_source:
            return []

        try:
            import ast as _ast
            _ast.parse(mutated_source)
        except SyntaxError:
            return []

        patch = make_git_diff(original_source, mutated_source, filename=plan.target_file)
        if not patch:
            return []

        if count_modified_lines(patch) > plan.constraints.max_modified_lines:
            return []

        candidate_id = self._make_candidate_id(plan, site)
        mutation_site_dict = {
            "site_id": site.site_id,
            "ast_node_type": site.ast_node_type,
            "ast_path": site.ast_path,
            "line": site.line,
            "col": site.col,
            "seed": site.seed,
            "before_snippet": site.before_snippet,
            "after_snippet": site.after_snippet,
            "metadata": site.metadata,
            "operator": operator_name,
        }

        artifact = CandidateArtifact(
            candidate_id=candidate_id,
            plan_id=plan.plan_id,
            strategy="procedural",
            operator=operator_name,
            target_file=plan.target_file,
            target_symbol=plan.target_symbol,
            bug_patch=patch,
            mutation_site=mutation_site_dict,
            seed=seed,
            before_snippet=site.before_snippet,
            after_snippet=site.after_snippet,
            generation_metadata={
                "num_sites": len(sites),
                "site_index": sites.index(site),
            },
        )

        if output_dir:
            candidate_dir = os.path.join(output_dir, candidate_id)
            artifact.save(candidate_dir)

        return [artifact]

    def _make_candidate_id(self, plan: BugGenerationPlan, site: MutationSite) -> str:
        raw = f"{plan.plan_id}:{site.site_id}:{site.seed}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        return f"cand_{digest}"

    def enumerate_all_sites(
        self,
        source: str,
        operator_name: str,
        target_symbol: str = "",
    ) -> List[MutationSite]:
        if operator_name not in OPERATORS:
            return []
        operator = get_operator(operator_name)
        return operator.enumerate_sites(source, target_symbol=target_symbol)


class SWESmithEngine:
    def __init__(self, agent_adapter: Any = None) -> None:
        self.agent_adapter = agent_adapter
        self.procedural = ProceduralEngine()
        self.lm_modify = None
        self.lm_rewrite = None
        self.combiner = None
        self.pr_mirror = None
        self.pr_replay = None
        self.repo_agent = None
        self.repo_chain = None

        try:
            from .lm_modify import LMModify
            self.lm_modify = LMModify(agent_adapter)
        except Exception:
            self.lm_modify = None

        try:
            from .lm_rewrite import LMRewrite
            self.lm_rewrite = LMRewrite(agent_adapter)
        except Exception:
            self.lm_rewrite = None

        try:
            from .combine import CombineEngine
            self.combiner = CombineEngine()
        except Exception:
            self.combiner = None

        try:
            from .pr_mirror import PRMirror
            self.pr_mirror = PRMirror(agent_adapter)
        except Exception:
            self.pr_mirror = None

        try:
            from .pr_replay import PRReplayGenerator
            self.pr_replay = PRReplayGenerator()
        except Exception:
            self.pr_replay = None

        try:
            from .repo_agent import RepoAgentGenerator
            self.repo_agent = RepoAgentGenerator(agent_adapter)
        except Exception:
            self.repo_agent = None

        try:
            from .repo_chain import RepoChainGenerator
            self.repo_chain = RepoChainGenerator(agent_adapter)
        except Exception:
            self.repo_chain = None

    def generate(
        self,
        plan: BugGenerationPlan,
        node_code_dir: str,
        repo_spec: RepoSpec,
        output_dir: str,
    ) -> List[CandidateArtifact]:
        strategy = plan.strategy

        if strategy == "lm_modify":
            if self.lm_modify is None:
                return []
            return self.lm_modify.generate(plan, node_code_dir, repo_spec, output_dir)
        elif strategy == "lm_rewrite":
            if self.lm_rewrite is None:
                return []
            return self.lm_rewrite.generate(plan, node_code_dir, repo_spec, output_dir)
        elif strategy == "procedural":
            return self.procedural.generate(plan, node_code_dir, repo_spec, output_dir)
        elif strategy == "combine":
            if self.combiner is None:
                return []
            return self.combiner.generate(plan, node_code_dir, repo_spec, output_dir)
        elif strategy == "pr_mirror":
            if self.pr_mirror is None:
                return []
            return self.pr_mirror.generate(plan, node_code_dir, repo_spec, output_dir)
        elif strategy == "pr_replay":
            if self.pr_replay is None:
                return []
            return self.pr_replay.generate(plan, node_code_dir, repo_spec, output_dir)
        elif strategy == "repo_agent":
            if self.repo_agent is None:
                return []
            return self.repo_agent.generate(plan, node_code_dir, repo_spec, output_dir)
        elif strategy == "repo_chain":
            if self.repo_chain is None:
                return []
            return self.repo_chain.generate(plan, node_code_dir, repo_spec, output_dir)
        else:
            return []

    def list_operators(self) -> List[str]:
        return list(OPERATORS.keys())

    def list_strategies(self) -> List[str]:
        return [
            "lm_modify",
            "lm_rewrite",
            "procedural",
            "combine",
            "pr_mirror",
            "pr_replay",
            "repo_agent",
            "repo_chain",
        ]
