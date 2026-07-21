"""Proposer workflows package.

A workflow is the Proposer's task-generation paradigm. RepoChain is the default
workflow; it is NOT a mutation backend and NOT a SWESmith strategy. The
mutation backends (lm_modify, procedural, pr_replay, ...) live under
``swesmith/mutations/`` and are used by RepoChain internally.
"""

from __future__ import annotations

from .repo_chain.workflow import RepoChainWorkflow

__all__ = ["RepoChainWorkflow"]
