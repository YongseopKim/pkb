"""Re-embedding engine for model migration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from pkb.generator.chunker import chunk_text, prepare_chunks_for_chromadb

if TYPE_CHECKING:
    from pkb.db.chromadb_client import ChunkStore
    from pkb.db.postgres import BundleRepository

logger = logging.getLogger(__name__)

SKIP_MD_NAMES = {"_bundle.md"}


class ReembedEngine:
    """Re-embed existing bundles with new embedding model.

    Reads MD files from disk, re-chunks, and upserts to ChromaDB.
    Does NOT touch PostgreSQL metadata.
    """

    def __init__(
        self,
        kb_path: Path,
        kb_name: str,
        chunk_store: ChunkStore,
        repo: BundleRepository,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
    ) -> None:
        self._kb_path = Path(kb_path)
        self._kb_name = kb_name
        self._chunk_store = chunk_store
        self._repo = repo
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def reembed_bundle(self, bundle_id: str) -> dict:
        """Re-embed a single bundle from its MD files on disk."""
        bundle_dir = self._kb_path / "bundles" / bundle_id
        if not bundle_dir.exists():
            return {"bundle_id": bundle_id, "status": "error",
                    "error": f"Bundle directory not found: {bundle_dir}"}

        md_files = [
            f for f in sorted(bundle_dir.iterdir())
            if f.suffix == ".md" and f.name not in SKIP_MD_NAMES
        ]
        if not md_files:
            return {"bundle_id": bundle_id, "status": "error",
                    "error": "No MD files found in bundle directory"}

        full_text = "\n\n".join(
            f.read_text(encoding="utf-8", errors="replace") for f in md_files
        )

        # Get metadata from DB for chunk metadata
        meta = self._repo.find_by_id(bundle_id)
        domains = meta.get("domains", []) if meta else []
        topics = meta.get("topics", []) if meta else []
        platform = meta.get("platform", "unknown") if meta else "unknown"

        chunks = chunk_text(full_text, self._chunk_size, self._chunk_overlap)
        chunk_data = prepare_chunks_for_chromadb(
            chunks,
            metadata={
                "bundle_id": bundle_id,
                "kb": self._kb_name,
                "platform": platform,
                "domains": ",".join(domains) if isinstance(domains, list) else str(domains),
                "topics": ",".join(topics) if isinstance(topics, list) else str(topics),
            },
        )

        self._chunk_store.delete_by_bundle(bundle_id)
        self._chunk_store.upsert_chunks(chunk_data)

        logger.info("Re-embedded %s: %d chunks", bundle_id, len(chunks))
        return {"bundle_id": bundle_id, "status": "reembedded", "chunks": len(chunks)}

    def reembed_all(
        self, progress_callback: Callable | None = None,
    ) -> dict:
        """Re-embed all bundles in the KB."""
        bundles_dir = self._kb_path / "bundles"
        bundle_ids: list[str] = []
        if bundles_dir.exists():
            bundle_ids = sorted(
                d.name for d in bundles_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )

        stats = {"total": 0, "reembedded": 0, "errors": 0}

        for bundle_id in bundle_ids:
            stats["total"] += 1
            result = self.reembed_bundle(bundle_id)
            if result["status"] == "reembedded":
                stats["reembedded"] += 1
            else:
                stats["errors"] += 1
                logger.warning("Failed to re-embed %s: %s",
                               bundle_id, result.get("error", "unknown"))

            if progress_callback:
                progress_callback(bundle_id, result["status"])

        return stats

    def reembed_collection_fresh(
        self, progress_callback: Callable | None = None,
    ) -> dict:
        """Drop collection, recreate, and re-embed all bundles.

        Use this when switching embedding models to avoid mixed embeddings.
        """
        logger.info("Dropping and recreating collection for fresh re-embedding")
        self._chunk_store.drop_and_recreate_collection()
        return self.reembed_all(progress_callback=progress_callback)
