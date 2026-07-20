from __future__ import annotations

import hashlib
from typing import List, Optional

from .operators import MutationSite
from .operators.base import make_rng


class SiteSelector:
    def select(
        self,
        sites: List[MutationSite],
        seed: int,
        index: Optional[int] = None,
    ) -> Optional[MutationSite]:
        if not sites:
            return None
        if index is not None and 0 <= index < len(sites):
            return sites[index]
        rng = make_rng(seed)
        return rng.choice(sites)

    def select_many(
        self,
        sites: List[MutationSite],
        seed: int,
        count: int,
    ) -> List[MutationSite]:
        if not sites:
            return []
        rng = make_rng(seed)
        n = min(count, len(sites))
        return rng.sample(sites, n)

    def select_deterministic(
        self,
        sites: List[MutationSite],
        plan_id: str,
        operator_name: str,
    ) -> Optional[MutationSite]:
        if not sites:
            return None
        digest = hashlib.md5(f"{plan_id}:{operator_name}".encode("utf-8")).hexdigest()
        idx = int(digest, 16) % len(sites)
        return sites[idx]

    def rank_sites(
        self,
        sites: List[MutationSite],
        target_symbol: str = "",
    ) -> List[MutationSite]:
        scored = []
        for i, site in enumerate(sites):
            score = 0
            if target_symbol and target_symbol in site.ast_path:
                score += 10
            score += min(len(site.before_snippet), 100) / 100.0
            score -= i * 0.001
            scored.append((score, i, site))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [s for _, _, s in scored]
