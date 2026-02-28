"""Knowledge graph relation detection between bundles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pkb.models.config import RelationConfig

if TYPE_CHECKING:
    from pkb.db.chromadb_client import ChunkStore
    from pkb.db.postgres import BundleRepository


class RelationBuilder:
    """Builds knowledge graph edges between bundles.

    Detects three relation types:
    - similar: content similarity via ChromaDB embeddings
    - related: shared topics/domains via PostgreSQL
    - sequel: same topic with temporal proximity (future)
    """

    def __init__(
        self,
        *,
        repo: BundleRepository,
        chunk_store: ChunkStore,
        config: RelationConfig,
    ) -> None:
        self._repo = repo
        self._chunk_store = chunk_store
        self._threshold = config.similarity_threshold
        self._max_relations = config.max_relations_per_bundle

    def find_similar(self, bundle_id: str) -> list[dict]:
        """Find similar bundles via embedding similarity.

        Returns list of relations: [{"source_bundle_id", "target_bundle_id",
        "relation_type", "score"}].
        """
        bundle = self._repo.get_bundle_by_id(bundle_id)
        if bundle is None:
            return []

        question = bundle["question"]
        if not question:
            return []
        kb = bundle.get("kb")

        where = {"kb": kb} if kb else None
        results = self._chunk_store.search(
            query=question, n_results=30, where=where,
        )

        # Group by bundle, take best similarity
        candidates: dict[str, float] = {}
        for r in results:
            other_id = r.metadata.get("bundle_id", "")
            if other_id == bundle_id:
                continue
            similarity = max(0.0, 1.0 - r.distance)
            if similarity >= self._threshold:
                if other_id not in candidates or similarity > candidates[other_id]:
                    candidates[other_id] = similarity

        # Sort by score descending, limit
        sorted_candidates = sorted(
            candidates.items(), key=lambda x: x[1], reverse=True,
        )[:self._max_relations]

        return [
            {
                "source_bundle_id": bundle_id,
                "target_bundle_id": other_id,
                "relation_type": "similar",
                "score": round(score, 4),
            }
            for other_id, score in sorted_candidates
        ]

    def find_related_by_topics(self, bundle_id: str) -> list[dict]:
        """Find related bundles via shared topics.

        Returns list of relations with score = shared_count / total_topics.
        Filters out pairs where overlap ratio is <= 0.3.
        """
        bundle = self._repo.get_bundle_by_id(bundle_id)
        if bundle is None:
            return []

        shared = self._repo.find_bundles_sharing_topics(bundle_id)

        return [
            {
                "source_bundle_id": bundle_id,
                "target_bundle_id": s["bundle_id"],
                "relation_type": "related",
                "score": round(s["shared_count"] / max(s["total_topics"], 1), 4),
            }
            for s in shared
            if s["shared_count"] / max(s["total_topics"], 1) > 0.3
        ]

    def scan_bundle(self, bundle_id: str) -> list[dict]:
        """Scan a single bundle for all relation types.

        Returns combined list of similar + related relations.
        """
        similar = self.find_similar(bundle_id)
        related = self.find_related_by_topics(bundle_id)
        return similar + related

    def scan(self, kb: str | None = None) -> dict:
        """Scan all bundles for relations.

        Returns stats: {"scanned": int, "new_relations": int}.
        """
        bundle_ids = self._repo.list_all_bundle_ids(kb=kb)
        total_new = 0
        skipped = 0

        for bid in bundle_ids:
            bundle = self._repo.get_bundle_by_id(bid)
            if bundle and not bundle.get("question"):
                skipped += 1
                continue

            relations = self.scan_bundle(bid)
            for rel in relations:
                self._repo.insert_relation(
                    rel["source_bundle_id"],
                    rel["target_bundle_id"],
                    rel["relation_type"],
                    rel["score"],
                )
                total_new += 1

        return {"scanned": len(bundle_ids), "new_relations": total_new, "skipped": skipped}
