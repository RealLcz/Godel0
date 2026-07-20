"""Git node ref management."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .repository import create_ref, get_ref, run_git

NODE_REF_PREFIX = "refs/godel0/nodes"


def node_ref(node_id: str) -> str:
    """Get the git ref name for a node."""
    return f"{NODE_REF_PREFIX}/{node_id}"


def create_node_ref(agent_repo: Path, node_id: str, sha: str) -> None:
    """Create a git ref for a node."""
    create_ref(agent_repo, node_ref(node_id), sha)


def get_node_sha(agent_repo: Path, node_id: str) -> Optional[str]:
    """Get the commit SHA for a node, or None if not found."""
    return get_ref(agent_repo, node_ref(node_id))


def node_exists(agent_repo: Path, node_id: str) -> bool:
    """Check if a node ref exists."""
    return get_node_sha(agent_repo, node_id) is not None


def list_all_nodes(agent_repo: Path) -> list[str]:
    """List all node IDs from git refs."""
    result = run_git(
        agent_repo,
        "for-each-ref", "--format=%(refname)", NODE_REF_PREFIX,
        check=False,
    )
    if result.returncode != 0:
        return []
    nodes = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith(NODE_REF_PREFIX + "/"):
            node_id = line[len(NODE_REF_PREFIX) + 1:]
            nodes.append(node_id)
    return nodes
