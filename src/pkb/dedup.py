"""Duplicate bundle detection via embedding similarity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pkb.models.config import DedupConfig

if TYPE_CHECKING:
    from pkb.db.chromadb_client import ChunkStore
    from pkb.db.postgres import BundleRepository


class DuplicateDetector:
    """Detects duplicate bundles using ChromaDB embedding similarity."""

    def __init__(
        self,
        *,
        repo: BundleRepository,
        chunk_store: ChunkStore,
        config: DedupConfig,
    ) -> None:
        self._repo = repo
        self._chunk_store = chunk_store
        self._threshold = config.threshold

    def scan_bundle(self, bundle_id: str) -> list[dict]:
        """Scan a single bundle for duplicates against all other bundles.

        Returns list of detected pairs: [{"bundle_a", "bundle_b", "similarity"}].
        """
        bundle = self._repo.get_bundle_by_id(bundle_id)
        if bundle is None:
            return []

        question = bundle["question"]
        kb = bundle.get("kb")

        # Search ChromaDB for similar chunks
        where = {"kb": kb} if kb else None
        results = self._chunk_store.search(query=question, n_results=20, where=where)

        # Get existing pairs for this bundle to skip
        existing_pairs = self._repo.list_duplicate_pairs()
        existing_set = set()
        for p in existing_pairs:
            existing_set.add((p["bundle_a"], p["bundle_b"]))
            existing_set.add((p["bundle_b"], p["bundle_a"]))

        # Group by other bundle_id, take highest similarity
        candidates: dict[str, float] = {}
        for r in results:
            other_id = r.metadata.get("bundle_id", "")
            if other_id == bundle_id:
                continue
            similarity = 1.0 - r.distance
            if similarity >= self._threshold:
                if other_id not in candidates or similarity > candidates[other_id]:
                    candidates[other_id] = similarity

        # Filter out already-registered pairs
        pairs = []
        for other_id, similarity in candidates.items():
            a, b = sorted([bundle_id, other_id])
            if (a, b) not in existing_set:
                pairs.append({
                    "bundle_a": bundle_id,
                    "bundle_b": other_id,
                    "similarity": round(similarity, 4),
                })

        return pairs

    def scan(self, kb: str | None = None) -> dict:
        """Scan all bundles for duplicates.

        Returns stats: {"scanned": int, "new_pairs": int}.
        """
        bundle_ids = self._repo.list_all_bundle_ids(kb=kb)
        total_new = 0

        for bid in bundle_ids:
            pairs = self.scan_bundle(bid)
            for pair in pairs:
                self._repo.insert_duplicate_pair(
                    pair["bundle_a"], pair["bundle_b"], pair["similarity"],
                )
                total_new += 1

        return {"scanned": len(bundle_ids), "new_pairs": total_new}

    def list_pairs(self, status: str | None = None) -> list[dict]:
        """List duplicate pairs, optionally filtered by status."""
        return self._repo.list_duplicate_pairs(status=status)

    def dismiss_pair(self, pair_id: int) -> None:
        """Mark a pair as not duplicate."""
        self._repo.update_duplicate_status(pair_id, "dismissed")

    def confirm_pair(self, pair_id: int) -> None:
        """Confirm a pair as duplicate."""
        self._repo.update_duplicate_status(pair_id, "confirmed")
