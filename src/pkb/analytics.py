"""Analytics engine for PKB bundle statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pkb.db.postgres import BundleRepository


class AnalyticsEngine:
    """Business-logic layer for analytics aggregation.

    Wraps BundleRepository aggregate queries with filtering,
    slicing, and overview assembly.
    """

    def __init__(self, *, repo: BundleRepository) -> None:
        self._repo = repo

    def domain_distribution(self, kb: str | None = None) -> list[dict]:
        """Bundle counts per domain."""
        return self._repo.count_bundles_by_domain(kb=kb)

    def topic_heatmap(self, top_n: int = 20, kb: str | None = None) -> list[dict]:
        """Top-N topics by bundle count."""
        all_topics = self._repo.count_bundles_by_topic(kb=kb)
        return all_topics[:top_n]

    def temporal_trend(self, months: int = 6, kb: str | None = None) -> list[dict]:
        """Monthly bundle creation trend."""
        return self._repo.count_bundles_by_month(kb=kb, months=months)

    def platform_distribution(self, kb: str | None = None) -> list[dict]:
        """Response counts per platform."""
        return self._repo.count_responses_by_platform(kb=kb)

    def knowledge_gaps(self, threshold: int = 3, kb: str | None = None) -> list[dict]:
        """Topics with fewer than threshold bundles."""
        all_topics = self._repo.count_bundles_by_topic(kb=kb)
        return [t for t in all_topics if t["count"] < threshold]

    def overview(self, kb: str | None = None) -> dict:
        """High-level stats overview."""
        bundle_ids = self._repo.list_all_bundle_ids(kb=kb)
        domains = self._repo.count_bundles_by_domain(kb=kb)
        topics = self._repo.count_bundles_by_topic(kb=kb)
        total_relations = self._repo.count_relations()
        return {
            "total_bundles": len(bundle_ids),
            "total_relations": total_relations,
            "domain_count": len(domains),
            "topic_count": len(topics),
        }
