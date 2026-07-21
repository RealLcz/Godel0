"""Stage 2: Repository Transfer.

Find a target in the base repository that requires the same abstract capability
as the identified weakness but belongs to a different code subsystem. The
transfer is of the *reasoning requirement*, not the business story.

Currently backed by ``CodeLocator.locate`` in proposer/code_locator.py.
"""

from __future__ import annotations


class RepositoryTransferStage:
    """Stage 2: transfer the abstract capability gap to a new code target."""

    def __init__(self, code_locator):
        self.code_locator = code_locator

    def run(self, signature, repo_index, used_symbols=None, max_results=5):
        return self.code_locator.locate(
            signature,
            repo_index,
            max_results=max_results,
            used_symbols=used_symbols,
        )


__all__ = ["RepositoryTransferStage"]
