"""Mutation backends package.

These are the low-level mutation mechanisms that RepoChain uses internally:
  - lm_modify: LM-based mutation (modify existing code)
  - lm_rewrite: LM-based mutation (rewrite a function)
  - procedural: AST-operator-based deterministic mutation
  - pr_replay: PR-replay mutation
  - combine: combine multiple operators

RepoChain is NOT a mutation backend; it is a workflow that uses these backends.

The implementations currently live as top-level modules in swesmith/ for
backward compatibility. This package re-exports them so callers can use the
canonical ``swesmith.mutations.<backend>`` path.
"""

from __future__ import annotations

# Re-export the existing modules. They stay at swesmith/*.py for now; a later
# phase can physically relocate them once all imports are updated.
try:
    from .. import lm_modify as lm_modify_module  # type: ignore  # noqa: F401
except Exception:
    lm_modify_module = None
try:
    from .. import lm_rewrite as lm_rewrite_module  # type: ignore  # noqa: F401
except Exception:
    lm_rewrite_module = None
try:
    from .. import procedural as procedural_module  # type: ignore  # noqa: F401
except Exception:
    procedural_module = None
try:
    from .. import pr_replay as pr_replay_module  # type: ignore  # noqa: F401
except Exception:
    pr_replay_module = None
try:
    from .. import pr_mirror as pr_mirror_module  # type: ignore  # noqa: F401
except Exception:
    pr_mirror_module = None
try:
    from .. import combine as combine_module  # type: ignore  # noqa: F401
except Exception:
    combine_module = None

__all__ = [
    "lm_modify_module",
    "lm_rewrite_module",
    "procedural_module",
    "pr_replay_module",
    "pr_mirror_module",
    "combine_module",
]
