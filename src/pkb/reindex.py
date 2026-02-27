"""Reindex bundles — sync _bundle.md frontmatter edits to DB + ChromaDB."""

import logging
from pathlib import Path

from pkb.db.chromadb_client import ChunkStore
from pkb.db.postgres import BundleRepository
from pkb.generator.chunker import chunk_text, prepare_chunks_for_chromadb
from pkb.generator.frontmatter_parser import parse_frontmatter, parse_md_body
from pkb.models.config import EmbeddingConfig
from pkb.models.meta import BundleFrontmatter

logger = logging.getLogger(__name__)


class Reindexer:
    """Synchronizes _bundle.md frontmatter edits to PostgreSQL + ChromaDB."""

    def __init__(
        self,
        *,
        repo: BundleRepository,
        chunk_store: ChunkStore,
        kb_path: Path,
        kb_name: str,
        embedding_config: EmbeddingConfig,
    ) -> None:
        self._repo = repo
        self._chunk_store = chunk_store
        self._kb_path = kb_path
        self._kb_name = kb_name
        self._embedding_config = embedding_config

    def reindex_bundle(self, bundle_id: str) -> dict:
        """Reindex a single bundle from its _bundle.md frontmatter.

        Returns:
            Dict with bundle_id and status (updated/skipped/error).
        """
        bundle_dir = self._kb_path / "bundles" / bundle_id

        if not bundle_dir.exists():
            return {
                "bundle_id": bundle_id,
                "status": "skipped",
                "reason": f"Bundle directory not found: {bundle_dir}",
            }

        bundle_md_path = bundle_dir / "_bundle.md"
        if not bundle_md_path.exists():
            return {
                "bundle_id": bundle_id,
                "status": "skipped",
                "reason": f"_bundle.md not found in {bundle_dir}",
            }

        # Parse and validate frontmatter
        try:
            fm_dict = parse_frontmatter(bundle_md_path)
            fm = BundleFrontmatter(**fm_dict)
        except (ValueError, Exception) as e:
            return {
                "bundle_id": bundle_id,
                "status": "error",
                "reason": str(e),
            }

        # Update PostgreSQL metadata
        self._repo.update_bundle_meta(
            bundle_id=bundle_id,
            summary=fm.summary,
            domains=fm.domains,
            topics=fm.topics,
            pending_topics=fm.pending_topics,
        )

        # Re-chunk platform MD files and update ChromaDB
        self._rechunk_bundle(bundle_id, bundle_dir, fm)

        return {"bundle_id": bundle_id, "status": "updated"}

    def reindex_full(self, progress_callback=None) -> dict:
        """Reindex all bundles and remove orphan DB records.

        Args:
            progress_callback: Optional callable(bundle_id, status) for progress.

        Returns:
            Dict with total, updated, skipped, errors, deleted counts.
        """
        bundles_dir = self._kb_path / "bundles"
        disk_bundles = set()
        if bundles_dir.exists():
            disk_bundles = {
                d.name for d in bundles_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            }

        stats = {"total": 0, "updated": 0, "skipped": 0, "errors": 0, "deleted": 0}

        # Reindex each bundle on disk
        for bundle_id in sorted(disk_bundles):
            stats["total"] += 1
            result = self.reindex_bundle(bundle_id)

            if result["status"] == "updated":
                stats["updated"] += 1
            elif result["status"] == "skipped":
                stats["skipped"] += 1
            else:
                stats["errors"] += 1

            if progress_callback:
                progress_callback(bundle_id, result["status"])

        # Delete orphan DB records (in DB but not on disk)
        db_bundle_ids = set(self._repo.list_all_bundle_ids(kb=self._kb_name))
        orphans = db_bundle_ids - disk_bundles

        for orphan_id in sorted(orphans):
            self._repo.delete_bundle(orphan_id)
            self._chunk_store.delete_by_bundle(orphan_id)
            stats["deleted"] += 1
            if progress_callback:
                progress_callback(orphan_id, "deleted")

        return stats

    def _rechunk_bundle(
        self, bundle_id: str, bundle_dir: Path, fm: BundleFrontmatter
    ) -> None:
        """Re-chunk platform MD files and replace ChromaDB data."""
        # Collect text from all platform .md files (exclude _bundle.md)
        texts = []
        for md_file in sorted(bundle_dir.glob("*.md")):
            if md_file.name.startswith("_"):
                continue
            try:
                body = parse_md_body(md_file)
                texts.append(body)
            except ValueError:
                # File without frontmatter — read whole content
                texts.append(md_file.read_text(encoding="utf-8", errors="replace"))

        full_text = "\n\n".join(texts)
        if not full_text.strip():
            return

        # Delete old chunks
        self._chunk_store.delete_by_bundle(bundle_id)

        # Generate new chunks
        chunks = chunk_text(
            full_text,
            chunk_size=self._embedding_config.chunk_size,
            overlap=self._embedding_config.chunk_overlap,
        )
        chunk_data = prepare_chunks_for_chromadb(
            chunks,
            metadata={
                "bundle_id": bundle_id,
                "kb": self._kb_name,
                "domains": ",".join(fm.domains),
                "topics": ",".join(fm.topics),
            },
        )
        self._chunk_store.upsert_chunks(chunk_data)
