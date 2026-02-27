"""Post-ingest automation pipeline.

Runs auto-relate, auto-dedup, and gap-update after a file is ingested.
Each sub-task is independently controlled by PostIngestConfig flags
and exceptions are caught so they never propagate to the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pkb.dedup import DuplicateDetector
from pkb.models.config import DedupConfig, PostIngestConfig, RelationConfig
from pkb.relations import RelationBuilder

if TYPE_CHECKING:
    from pkb.db.chromadb_client import ChunkStore
    from pkb.db.postgres import BundleRepository

logger = logging.getLogger(__name__)


@dataclass
class PostIngestResult:
    """Result of post-ingest processing for a single bundle."""

    new_relations: int = 0
    new_dedup_pairs: int = 0
    gap_topics: list[str] = field(default_factory=list)


class PostIngestProcessor:
    """Orchestrates post-ingest automation pipeline.

    After a file is ingested into PKB, this processor runs:
    1. Auto-relate: find relations between the new bundle and existing bundles
    2. Auto-dedup: detect if the new bundle is a duplicate of an existing one
    3. Gap-update: check if the bundle's topics are knowledge gaps (below threshold)
    """

    def __init__(
        self,
        *,
        repo: BundleRepository,
        chunk_store: ChunkStore,
        config: PostIngestConfig,
        relation_config: RelationConfig,
        dedup_config: DedupConfig,
        gap_threshold: int = 3,
    ) -> None:
        self._repo = repo
        self._config = config
        self._gap_threshold = gap_threshold

        self._relation_builder = RelationBuilder(
            repo=repo,
            chunk_store=chunk_store,
            config=relation_config,
        )
        self._dedup_detector = DuplicateDetector(
            repo=repo,
            chunk_store=chunk_store,
            config=dedup_config,
        )

    def process(self, bundle_id: str) -> PostIngestResult:
        """Run all enabled post-ingest sub-tasks for a bundle.

        Each sub-task is independent: a failure in one does not affect others.
        """
        result = PostIngestResult()

        if self._config.auto_relate:
            result.new_relations = self._auto_relate(bundle_id)

        if self._config.auto_dedup:
            result.new_dedup_pairs = self._auto_dedup(bundle_id)

        if self._config.gap_update:
            result.gap_topics = self._gap_update(bundle_id)

        return result

    def _auto_relate(self, bundle_id: str) -> int:
        """Scan for relations and insert them into the database.

        Returns the number of new relations inserted.
        Catches all exceptions, logs a warning, and returns 0.
        """
        try:
            relations = self._relation_builder.scan_bundle(bundle_id)
            for rel in relations:
                self._repo.insert_relation(
                    rel["source_bundle_id"],
                    rel["target_bundle_id"],
                    rel["relation_type"],
                    rel["score"],
                )
            return len(relations)
        except Exception:
            logger.warning(
                "auto_relate failed for bundle %s", bundle_id, exc_info=True,
            )
            return 0

    def _auto_dedup(self, bundle_id: str) -> int:
        """Scan for duplicates and insert them into the database.

        Returns the number of new duplicate pairs inserted.
        Catches all exceptions, logs a warning, and returns 0.
        """
        try:
            pairs = self._dedup_detector.scan_bundle(bundle_id)
            for pair in pairs:
                self._repo.insert_duplicate_pair(
                    pair["bundle_a"],
                    pair["bundle_b"],
                    pair["similarity"],
                )
            return len(pairs)
        except Exception:
            logger.warning(
                "auto_dedup failed for bundle %s", bundle_id, exc_info=True,
            )
            return 0

    def _gap_update(self, bundle_id: str) -> list[str]:
        """Check if the bundle's topics are knowledge gaps.

        Returns a list of topics where the bundle count is below gap_threshold.
        Catches all exceptions, logs a warning, and returns [].
        """
        try:
            bundle = self._repo.get_bundle_by_id(bundle_id)
            if bundle is None:
                return []

            raw_topics = bundle.get("topics", [])
            if isinstance(raw_topics, str):
                topics = [t.strip() for t in raw_topics.split(",") if t.strip()]
            else:
                topics = list(raw_topics)

            if not topics:
                return []

            counts = self._repo.count_bundles_for_topics(topics)

            return [
                topic for topic in topics
                if counts.get(topic, 0) < self._gap_threshold
            ]
        except Exception:
            logger.warning(
                "gap_update failed for bundle %s", bundle_id, exc_info=True,
            )
            return []
