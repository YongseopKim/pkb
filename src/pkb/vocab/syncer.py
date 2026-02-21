"""Topic syncer — syncs YAML topic changes to PostgreSQL."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pkb.db.postgres import BundleRepository


class TopicSyncer:
    """Syncs topic vocabulary changes (YAML) to PostgreSQL tables."""

    def __init__(self, repo: BundleRepository) -> None:
        self._repo = repo

    def sync_approve(self, canonical: str) -> None:
        """Sync a topic approval: update topic_vocab + mark bundle_topics as non-pending."""
        self._repo.upsert_topic_vocab(canonical=canonical, status="approved")
        self._repo.approve_pending_topic(canonical)

    def sync_merge(self, old: str, *, into: str) -> None:
        """Sync a topic merge: update topic_vocab + transfer bundle_topics references."""
        self._repo.upsert_topic_vocab(
            canonical=old, status="merged", merged_into=into,
        )
        self._repo.merge_topic_references(old, into)

    def sync_reject(self, canonical: str) -> None:
        """Sync a topic rejection: delete from topic_vocab + remove from bundle_topics."""
        self._repo.delete_topic_vocab(canonical)
        self._repo.remove_topic_from_bundles(canonical)
