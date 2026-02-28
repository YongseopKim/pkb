"""Search engine orchestrator — hybrid FTS + semantic search."""

from pkb.db.chromadb_client import ChunkStore
from pkb.db.postgres import BundleRepository
from pkb.search.models import BundleSearchResult, SearchMode, SearchQuery

# Hybrid scoring weights
FTS_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6
SINGLE_SOURCE_PENALTY = 0.7


class SearchEngine:
    """Orchestrates keyword, semantic, and hybrid search across PKB data stores."""

    def __init__(self, *, repo: BundleRepository, chunk_store: ChunkStore) -> None:
        self._repo = repo
        self._chunk_store = chunk_store

    def search(self, query: SearchQuery) -> list[BundleSearchResult]:
        """Execute search based on query mode."""
        if query.mode == SearchMode.KEYWORD:
            return self._keyword_search(query)
        elif query.mode == SearchMode.SEMANTIC:
            return self._semantic_search(query)
        else:
            return self._hybrid_search(query)

    def _keyword_search(self, query: SearchQuery) -> list[BundleSearchResult]:
        """FTS search via PostgreSQL tsvector."""
        rows = self._repo.search_fts(
            query=query.query, kb=query.kb,
            domains=query.domains or None, topics=query.topics or None,
            after=query.after, before=query.before, limit=query.limit,
            stance=query.stance,
            has_consensus=query.has_consensus,
            has_synthesis=query.has_synthesis,
        )
        if not rows:
            return []

        # Normalize ranks to 0-1
        ranks = [r["rank"] for r in rows]
        normalized = _min_max_normalize(ranks)

        results = []
        for row, norm_score in zip(rows, normalized):
            results.append(_row_to_result(row, score=norm_score, source="fts"))
        return results

    def _semantic_search(self, query: SearchQuery) -> list[BundleSearchResult]:
        """Semantic search via ChromaDB vectors."""
        where = {"kb": query.kb} if query.kb else None
        # Fetch more chunks to aggregate at bundle level
        chunk_results = self._chunk_store.search(
            query=query.query, n_results=query.limit * 3, where=where,
        )
        if not chunk_results:
            return []

        # Aggregate chunks per bundle — take best (min distance) per bundle
        bundle_best: dict[str, float] = {}
        for cr in chunk_results:
            bid = cr.metadata.get("bundle_id", "")
            if not bid:
                continue
            similarity = max(0.0, 1.0 - cr.distance)
            if bid not in bundle_best or similarity > bundle_best[bid]:
                bundle_best[bid] = similarity

        # Sort by similarity descending, limit
        sorted_bundles = sorted(bundle_best.items(), key=lambda x: x[1], reverse=True)
        sorted_bundles = sorted_bundles[:query.limit]

        # Normalize similarities to 0-1
        sims = [s for _, s in sorted_bundles]
        normalized = _min_max_normalize(sims)

        # Look up bundle metadata from repo
        results = []
        for (bid, _), norm_score in zip(sorted_bundles, normalized):
            bundle_data = self._repo.get_bundle_by_id(bid)
            if bundle_data is None:
                continue
            results.append(_row_to_result(bundle_data, score=norm_score, source="semantic"))
        return results

    def _hybrid_search(self, query: SearchQuery) -> list[BundleSearchResult]:
        """Combine FTS and semantic search with weighted average scoring."""
        fts_results = self._keyword_search(query)
        sem_results = self._semantic_search(query)

        # Build lookup maps by bundle_id
        fts_map: dict[str, BundleSearchResult] = {r.bundle_id: r for r in fts_results}
        sem_map: dict[str, BundleSearchResult] = {r.bundle_id: r for r in sem_results}

        all_ids = set(fts_map.keys()) | set(sem_map.keys())
        merged: list[BundleSearchResult] = []

        for bid in all_ids:
            fts_r = fts_map.get(bid)
            sem_r = sem_map.get(bid)

            if fts_r and sem_r:
                # Both sources — weighted average, no penalty
                score = FTS_WEIGHT * fts_r.score + SEMANTIC_WEIGHT * sem_r.score
                base = fts_r  # Use FTS data (has same metadata)
                merged.append(base.model_copy(update={"score": score, "source": "both"}))
            elif fts_r:
                # FTS only — apply single-source penalty
                score = fts_r.score * SINGLE_SOURCE_PENALTY
                merged.append(fts_r.model_copy(update={"score": score}))
            else:
                # Semantic only — apply single-source penalty
                score = sem_r.score * SINGLE_SOURCE_PENALTY  # type: ignore[union-attr]
                merged.append(
                    sem_r.model_copy(update={"score": score})  # type: ignore[union-attr]
                )

        # Sort by score descending, limit
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:query.limit]


def _min_max_normalize(values: list[float]) -> list[float]:
    """Normalize values to 0-1 range using min-max scaling."""
    if not values:
        return []
    if len(values) == 1:
        return [1.0]
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return [1.0] * len(values)
    return [(v - min_v) / (max_v - min_v) for v in values]


def _row_to_result(row: dict, *, score: float, source: str) -> BundleSearchResult:
    """Convert a repo row dict to BundleSearchResult."""
    domains_str = row.get("domains", "")
    topics_str = row.get("topics", "")
    return BundleSearchResult(
        bundle_id=row["bundle_id"],
        question=row["question"],
        summary=row.get("summary"),
        domains=[d for d in domains_str.split(",") if d] if domains_str else [],
        topics=[t for t in topics_str.split(",") if t] if topics_str else [],
        score=score,
        created_at=row["created_at"],
        source=source,
    )
