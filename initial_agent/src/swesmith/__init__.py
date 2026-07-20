from __future__ import annotations

from .candidate import CandidateArtifact
from .engine import SWESmithEngine, ProceduralEngine
from .workspace import WorkspaceManager
from .patch_utils import (
    make_diff,
    apply_patch_to_string,
    extract_changed_files,
    patch_conflicts,
)
from .entity_index import EntityIndex, Entity
from .site_selector import SiteSelector
from .operators import (
    MutationSite,
    ProceduralOperator,
    OPERATORS,
    get_operator,
)
from .lm_modify import LMModify
from .lm_rewrite import LMRewrite
from .combine import CombineEngine
from .pr_mirror import PRMirror
from .pr_replay import PRReplayGenerator
from .repo_agent import RepoAgentGenerator
from .repo_chain import RepoChainGenerator

__all__ = [
    "CandidateArtifact",
    "SWESmithEngine",
    "ProceduralEngine",
    "WorkspaceManager",
    "make_diff",
    "apply_patch_to_string",
    "extract_changed_files",
    "patch_conflicts",
    "EntityIndex",
    "Entity",
    "SiteSelector",
    "MutationSite",
    "ProceduralOperator",
    "OPERATORS",
    "get_operator",
    "LMModify",
    "LMRewrite",
    "CombineEngine",
    "PRMirror",
    "PRReplayGenerator",
    "RepoAgentGenerator",
    "RepoChainGenerator",
]
