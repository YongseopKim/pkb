"""Regenerate derived files from raw JSONL + re-run LLM meta extraction."""

import logging
from pathlib import Path

import yaml

from pkb.db.chromadb_client import ChunkStore
from pkb.db.postgres import BundleRepository
from pkb.generator.chunker import chunk_text, prepare_chunks_for_chromadb
from pkb.generator.md_generator import conversation_to_markdown, write_md_file
from pkb.generator.meta_gen import MetaGenerator
from pkb.ingest import compute_question_hash, compute_stable_id
from pkb.parser.directory import SUPPORTED_EXTENSIONS, parse_file

logger = logging.getLogger(__name__)


class Regenerator:
    """Regenerates all derived files from raw JSONL using current prompts/model."""

    def __init__(
        self,
        *,
        repo: BundleRepository,
        chunk_store: ChunkStore,
        meta_gen: MetaGenerator,
        kb_path: Path,
        kb_name: str,
        domains: list[str],
        topics: list[str],
        dry_run: bool = False,
    ) -> None:
        self._repo = repo
        self._chunk_store = chunk_store
        self._meta_gen = meta_gen
        self._kb_path = kb_path
        self._kb_name = kb_name
        self._domains = domains
        self._topics = topics
        self._dry_run = dry_run

    def regenerate_bundle(self, bundle_id: str) -> dict:
        """Regenerate all derived files for a single bundle.

        Returns:
            Dict with bundle_id and status (regenerated/error).
        """
        bundle_dir = self._kb_path / "bundles" / bundle_id
        if not bundle_dir.exists():
            return {
                "bundle_id": bundle_id,
                "status": "error",
                "reason": f"Bundle directory not found: {bundle_dir}",
            }

        # Find raw input files
        raw_dir = bundle_dir / "_raw"
        raw_files = []
        if raw_dir.exists():
            for ext in sorted(SUPPORTED_EXTENSIONS):
                raw_files.extend(sorted(raw_dir.glob(f"*{ext}")))
        if not raw_files:
            return {
                "bundle_id": bundle_id,
                "status": "error",
                "reason": f"No raw input files found in {raw_dir}",
            }

        # Parse input file (use first file)
        conv = parse_file(raw_files[0])
        question, question_hash = compute_question_hash(conv)

        # Preserve existing stable_id from DB (regenerate should not change identity)
        existing = self._repo.get_bundle_by_id(bundle_id)
        stable_id = (
            existing.get("stable_id") if existing else None
        ) or compute_stable_id(conv)

        # Generate bundle meta (re-run LLM)
        response_summaries = self._build_response_summaries(conv)
        bundle_meta = self._meta_gen.generate_bundle_meta(
            question=question,
            platforms=[conv.meta.platform],
            response_summaries=response_summaries,
            available_domains=self._domains,
            available_topics=self._topics,
        )

        # Generate response meta (re-run LLM)
        response_meta = self._meta_gen.generate_response_meta(
            platform=conv.meta.platform,
            content=self._get_assistant_content(conv),
        )

        # Overwrite platform MD
        frontmatter = response_meta.model_dump()
        md_path = bundle_dir / f"{conv.meta.platform}.md"
        write_md_file(conv, bundle_id, frontmatter, md_path)

        # Overwrite _bundle.md
        self._write_bundle_md(bundle_dir, bundle_id, bundle_meta, conv)

        # DB + ChromaDB operations (skip in dry-run)
        if not self._dry_run:
            self._repo.upsert_bundle(
                bundle_id=bundle_id,
                kb=self._kb_name,
                question=question,
                summary=bundle_meta.summary,
                created_at=conv.meta.exported_at,
                response_count=self._count_responses(conv),
                path=f"bundles/{bundle_id}",
                question_hash=question_hash,
                stable_id=stable_id,
                domains=bundle_meta.domains,
                topics=bundle_meta.topics,
                pending_topics=bundle_meta.pending_topics,
                responses=[{
                    "platform": conv.meta.platform,
                    "model": response_meta.model,
                    "turn_count": conv.turn_count,
                }],
            )

            # Re-chunk and update ChromaDB
            self._chunk_store.delete_by_bundle(bundle_id)
            full_text = conversation_to_markdown(conv, bundle_id)
            chunks = chunk_text(full_text)
            chunk_data = prepare_chunks_for_chromadb(
                chunks,
                metadata={
                    "bundle_id": bundle_id,
                    "kb": self._kb_name,
                    "platform": conv.meta.platform,
                    "domains": ",".join(bundle_meta.domains),
                    "topics": ",".join(bundle_meta.topics),
                },
            )
            self._chunk_store.upsert_chunks(chunk_data)

        return {"bundle_id": bundle_id, "status": "regenerated"}

    def regenerate_all(self, progress_callback=None) -> dict:
        """Regenerate all bundles in the KB.

        Args:
            progress_callback: Optional callable(bundle_id, status) for progress.

        Returns:
            Dict with total, regenerated, errors counts.
        """
        bundles_dir = self._kb_path / "bundles"
        bundle_ids = []
        if bundles_dir.exists():
            bundle_ids = sorted(
                d.name for d in bundles_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )

        stats = {"total": 0, "regenerated": 0, "errors": 0}

        for bundle_id in bundle_ids:
            stats["total"] += 1
            try:
                result = self.regenerate_bundle(bundle_id)
            except Exception:
                logger.exception("Failed to regenerate %s", bundle_id)
                result = {"bundle_id": bundle_id, "status": "error", "reason": "exception"}

            if result["status"] == "regenerated":
                stats["regenerated"] += 1
            else:
                stats["errors"] += 1

            if progress_callback:
                progress_callback(bundle_id, result["status"])

        return stats

    def _build_response_summaries(self, conv) -> str:
        parts = []
        for turn in conv.turns:
            if turn.role == "assistant":
                preview = turn.content[:500]
                if len(turn.content) > 500:
                    preview += "..."
                parts.append(f"{conv.meta.platform}: {preview}")
        return "\n".join(parts)

    def _get_assistant_content(self, conv) -> str:
        return "\n\n".join(
            turn.content for turn in conv.turns if turn.role == "assistant"
        )

    def _count_responses(self, conv) -> int:
        return sum(1 for t in conv.turns if t.role == "assistant")

    def _write_bundle_md(self, bundle_dir, bundle_id, bundle_meta, conv) -> None:
        meta_dict = bundle_meta.model_dump()
        meta_dict["id"] = bundle_id
        meta_dict["platforms"] = [conv.meta.platform]
        meta_dict["created_at"] = conv.meta.exported_at.isoformat()
        content = f"---\n{yaml.dump(meta_dict, allow_unicode=True, default_flow_style=False)}---\n"
        (bundle_dir / "_bundle.md").write_text(content, encoding="utf-8")
