"""PKB ingest pipeline — JSONL to organized bundle."""

import hashlib
import logging
import shutil
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml

from pkb.constants import DONE_DIR_NAME
from pkb.db.chromadb_client import ChunkStore
from pkb.db.postgres import BundleRepository
from pkb.generator.chunker import chunk_text, prepare_chunks_for_chromadb
from pkb.generator.md_generator import conversation_to_markdown, write_md_file
from pkb.generator.meta_gen import MetaGenerator
from pkb.models.jsonl import Conversation
from pkb.parser.directory import SUPPORTED_EXTENSIONS, parse_file

logger = logging.getLogger(__name__)

MIN_CONTENT_LENGTH = 50


def move_to_done(file_path: Path, watch_dir: Path, *, dry_run: bool = False) -> Path | None:
    """Move a successfully ingested file to the .done/ subdirectory.

    Supports files in subdirectories — preserves relative path structure.
    e.g. inbox/PKB/chatgpt.md → inbox/.done/PKB/chatgpt.md

    Args:
        file_path: Path to the ingested file.
        watch_dir: The inbox/watch directory.
        dry_run: If True, return destination path without moving.

    Returns:
        Destination path if moved (or would be moved), None otherwise.
    """
    resolved_file = file_path.resolve()
    resolved_watch = watch_dir.resolve()

    if not resolved_file.is_relative_to(resolved_watch):
        return None

    rel_path = resolved_file.relative_to(resolved_watch)
    if DONE_DIR_NAME in rel_path.parts:
        return None

    dest = watch_dir / DONE_DIR_NAME / rel_path

    if dry_run:
        return dest

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), str(dest))
    except OSError:
        logger.warning("Failed to move %s to %s", file_path, dest)
        return None

    # Clean up empty parent directories up to watch_dir
    parent = file_path.resolve().parent
    while parent != resolved_watch:
        try:
            parent.rmdir()  # Only removes if empty
        except OSError:
            break
        parent = parent.parent

    return dest


def generate_question_hash(question: str) -> str:
    """Generate a SHA-256 hash of the question for deduplication."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def compute_question_hash(conv: Conversation) -> tuple[str, str]:
    """Compute question text and its hash from a Conversation.

    For JSONL with user turns: uses first_user_message or title.
    For MD (no user turns, no title): uses first 500 chars of assistant content
    to avoid all MD files sharing the same SHA256("") hash.

    Returns:
        (question, question_hash) tuple.
    """
    question = conv.first_user_message or conv.meta.title or ""
    if question:
        return question, generate_question_hash(question)
    # Fallback: use assistant content prefix for hash (question stays "")
    assistant_text = ""
    for turn in conv.turns:
        if turn.role == "assistant":
            assistant_text = turn.content[:500]
            break
    hash_input = assistant_text if assistant_text else ""
    return "", generate_question_hash(hash_input)


def _normalize_url(url: str) -> str:
    """Normalize a URL for stable_id generation.

    - Strip query string and fragment
    - Strip trailing slash from path
    - Lowercase scheme and hostname
    - Preserve path case (conversation IDs are case-sensitive)
    """
    parsed = urlparse(url)
    # urlparse preserves case; explicitly lowercase scheme and netloc
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return f"{scheme}://{netloc}{path}"


def compute_stable_id(conv: Conversation) -> str:
    """Compute a stable conversation identity hash (64-char hex SHA-256).

    Priority 1: If conv.meta.url exists, SHA-256 of normalized URL.
    Priority 2 (fallback): First 5 turns, each formatted as
        '{role}:{content[:200]}', joined by newline, then SHA-256.
    """
    if conv.meta.url:
        normalized = _normalize_url(conv.meta.url)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # Fallback: fingerprint from first 5 turns
    parts = []
    for turn in conv.turns[:5]:
        parts.append(f"{turn.role}:{turn.content[:200]}")
    fingerprint = "\n".join(parts)
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def generate_bundle_id(*, date: datetime, slug: str, question: str) -> str:
    """Generate bundle ID: {YYYYMMDD}-{slug}-{hash4}."""
    date_str = date.strftime("%Y%m%d")
    hash4 = hashlib.sha256(question.encode("utf-8")).hexdigest()[:4]
    return f"{date_str}-{slug}-{hash4}"


class IngestPipeline:
    """Orchestrates the full ingest pipeline for a single JSONL file."""

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
        # Per-stable_id locks to prevent TOCTOU race in concurrent ingest
        self._hash_locks: dict[str, threading.Lock] = {}
        self._hash_locks_mutex = threading.Lock()

    def _get_stable_lock(self, stable_id: str) -> threading.Lock:
        """Get or create a per-stable_id lock for concurrent dedup safety."""
        with self._hash_locks_mutex:
            if stable_id not in self._hash_locks:
                self._hash_locks[stable_id] = threading.Lock()
            return self._hash_locks[stable_id]

    def ingest_file(self, file_path: Path, *, force: bool = False) -> dict | None:
        """Ingest a single input file (JSONL or MD) into the knowledge base.

        Args:
            file_path: Path to the input file.
            force: If True, bypass dedup check (for re-ingesting modified files).

        Returns:
            Dict with bundle_id and metadata.
            If updated (same stable_id + platform): dict includes "updated": True.
            If merged (same stable_id, diff platform): dict includes "merged": True.
            Skip dict with "status" key for graceful skips (skip_*).
        """
        from pkb.parser.exceptions import ParseError

        # 1. Parse input file (graceful skip on parse error)
        try:
            conv = parse_file(file_path)
        except ParseError as e:
            logger.warning("Parse error, skipping %s: %s", file_path.name, e)
            return {"status": "skip_parse_error", "reason": str(e)}

        question, question_hash = compute_question_hash(conv)
        stable_id = compute_stable_id(conv)

        # 1b. Content minimum validation
        total_content = sum(
            len(t.content) for t in conv.turns if t.role == "assistant"
        )
        if not conv.turns or total_content < MIN_CONTENT_LENGTH:
            logger.warning(
                "Insufficient content (%d chars), skipping %s",
                total_content, file_path.name,
            )
            return {
                "status": "skip_insufficient_content",
                "reason": f"Content too short ({total_content} chars)",
            }

        # 2. Dedup / merge / update check with per-stable_id lock
        #    (skipped when force=True or dry_run)
        if not force and not self._dry_run:
            lock = self._get_stable_lock(stable_id)
            with lock:
                return self._dedup_and_ingest(
                    file_path, conv, question, question_hash, stable_id,
                )

        return self._create_new_bundle(
            file_path, conv, question, question_hash, stable_id,
        )

    def _dedup_and_ingest(
        self,
        file_path: Path,
        conv: Conversation,
        question: str,
        question_hash: str,
        stable_id: str,
    ) -> dict | None:
        """Check for duplicates and either update, merge, or create new bundle.

        Must be called under the per-stable_id lock.

        - Same stable_id + same platform → UPDATE (re-run LLM, refresh DB/ChromaDB)
        - Same stable_id + different platform → MERGE (add platform to bundle)
        - No match → CREATE new bundle
        """
        existing = self._repo.find_bundle_by_stable_id(stable_id)
        if existing is not None:
            if conv.meta.platform in existing["platforms"]:
                return self._update_existing_bundle(
                    file_path, conv, existing, question, question_hash, stable_id,
                )
            return self.merge_file(file_path, conv, existing)
        return self._create_new_bundle(
            file_path, conv, question, question_hash, stable_id,
        )

    def _update_existing_bundle(
        self,
        file_path: Path,
        conv: Conversation,
        existing_bundle: dict,
        question: str,
        question_hash: str,
        stable_id: str,
    ) -> dict:
        """Update an existing bundle when same stable_id + same platform.

        Re-runs LLM for response_meta and bundle_meta, regenerates derived files
        (MD, _bundle.md), refreshes DB via upsert_bundle (ON CONFLICT UPDATE),
        and refreshes ChromaDB (delete old chunks + upsert new).

        All LLM calls happen before any disk writes to prevent partial state.
        """
        bundle_id = existing_bundle["bundle_id"]
        bundle_dir = self._kb_path / "bundles" / bundle_id
        raw_dir = bundle_dir / "_raw"

        logger.info(
            "Updating bundle %s from %s (platform=%s)",
            bundle_id, file_path.name, conv.meta.platform,
        )

        # 1. Build response summaries and validate
        response_summaries = self._build_response_summaries(conv)
        if len(response_summaries.strip()) < MIN_CONTENT_LENGTH:
            logger.warning(
                "Response summaries too short (%d chars), skipping update %s",
                len(response_summaries.strip()), file_path.name,
            )
            return {
                "status": "skip_insufficient_content",
                "reason": "Response summaries too short",
            }

        # 2. LLM: Generate bundle meta
        bundle_meta = self._meta_gen.generate_bundle_meta(
            question=question,
            platforms=[conv.meta.platform],
            response_summaries=response_summaries,
            available_domains=self._domains,
            available_topics=self._topics,
        )

        # 3. LLM: Generate response meta (before disk writes)
        response_meta = self._meta_gen.generate_response_meta(
            platform=conv.meta.platform,
            content=self._get_assistant_content(conv),
        )

        # 4. Disk writes: copy raw + write MD files
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, raw_dir / file_path.name)

        frontmatter = response_meta.model_dump()
        md_path = bundle_dir / f"{conv.meta.platform}.md"
        write_md_file(conv, bundle_id, frontmatter, md_path)

        self._write_bundle_md(bundle_dir, bundle_id, bundle_meta, conv)

        # 5. DB: upsert_bundle with existing bundle_id (ON CONFLICT UPDATE)
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
                    "source_path": str(file_path),
                }],
                source_path=str(file_path),
            )

            # 6. ChromaDB: delete old chunks, then upsert new
            self._chunk_store.delete_by_bundle_id(bundle_id)

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

        return {
            "bundle_id": bundle_id,
            "platform": conv.meta.platform,
            "updated": True,
            "summary": bundle_meta.summary,
            "domains": bundle_meta.domains,
            "topics": bundle_meta.topics,
        }

    def _create_new_bundle(
        self,
        file_path: Path,
        conv: Conversation,
        question: str,
        question_hash: str,
        stable_id: str | None = None,
    ) -> dict:
        """Create a new bundle from a parsed conversation.

        All LLM calls happen before any disk writes to prevent orphan directories.
        """
        logger.info(
            "Creating bundle from %s (platform=%s)",
            file_path.name, conv.meta.platform,
        )

        # 1. Build response summaries and validate
        response_summaries = self._build_response_summaries(conv)
        logger.debug("Response summaries: %d chars", len(response_summaries))
        if len(response_summaries.strip()) < MIN_CONTENT_LENGTH:
            logger.warning(
                "Response summaries too short (%d chars), skipping %s",
                len(response_summaries.strip()), file_path.name,
            )
            return {
                "status": "skip_insufficient_content",
                "reason": "Response summaries too short",
            }

        # 2. LLM: Generate bundle meta (for slug)
        bundle_meta = self._meta_gen.generate_bundle_meta(
            question=question,
            platforms=[conv.meta.platform],
            response_summaries=response_summaries,
            available_domains=self._domains,
            available_topics=self._topics,
        )

        # 2. LLM: Generate response meta (before mkdir — prevents orphan dirs)
        response_meta = self._meta_gen.generate_response_meta(
            platform=conv.meta.platform,
            content=self._get_assistant_content(conv),
        )

        # 3. Generate bundle_id
        bundle_id = generate_bundle_id(
            date=conv.meta.exported_at,
            slug=bundle_meta.slug,
            question=question,
        )
        logger.info("Bundle created: %s", bundle_id)

        # 4. Disk writes: create directory + copy raw + write MD files
        bundle_dir = self._kb_path / "bundles" / bundle_id
        raw_dir = bundle_dir / "_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, raw_dir / file_path.name)

        frontmatter = response_meta.model_dump()
        md_path = bundle_dir / f"{conv.meta.platform}.md"
        write_md_file(conv, bundle_id, frontmatter, md_path)

        self._write_bundle_md(bundle_dir, bundle_id, bundle_meta, conv)

        # 5. DB operations (skip in dry-run)
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
                    "source_path": str(file_path),
                }],
                source_path=str(file_path),
            )

            # 6. ChromaDB upsert
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

        return {
            "bundle_id": bundle_id,
            "platform": conv.meta.platform,
            "summary": bundle_meta.summary,
            "domains": bundle_meta.domains,
            "topics": bundle_meta.topics,
        }

    def merge_file(
        self, file_path: Path, conv: Conversation, existing_bundle: dict,
    ) -> dict:
        """Merge a new platform file into an existing bundle.

        Called when the same stable_id exists but with a different platform.
        Adds raw file, generates platform MD, regenerates _bundle.md with all
        platforms, and updates DB/ChromaDB.

        Args:
            file_path: Path to the new input file.
            conv: Parsed Conversation from the new file.
            existing_bundle: Dict from find_bundle_by_stable_id().

        Returns:
            Dict with bundle_id, platform, merged=True, and updated metadata.
        """
        bundle_id = existing_bundle["bundle_id"]
        bundle_dir = self._kb_path / "bundles" / bundle_id
        raw_dir = bundle_dir / "_raw"

        # 1. LLM: Generate response meta for new platform (before disk writes)
        response_meta = self._meta_gen.generate_response_meta(
            platform=conv.meta.platform,
            content=self._get_assistant_content(conv),
        )

        # 2. Collect existing summaries + new file's summary (new file not yet copied)
        existing_summaries = self._collect_all_response_summaries(raw_dir)
        new_summary = self._build_response_summaries(conv)
        all_summaries = "\n".join(
            part for part in [existing_summaries, new_summary] if part
        )
        all_platforms = existing_bundle["platforms"] + [conv.meta.platform]
        question, _ = compute_question_hash(conv)

        # 3. LLM: Regenerate bundle meta with all platforms (before disk writes)
        bundle_meta = self._meta_gen.generate_bundle_meta(
            question=question,
            platforms=all_platforms,
            response_summaries=all_summaries,
            available_domains=self._domains,
            available_topics=self._topics,
        )

        # 4. All LLM calls succeeded — now safe to write to disk
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, raw_dir / file_path.name)

        # 5. Write new platform MD
        frontmatter = response_meta.model_dump()
        md_path = bundle_dir / f"{conv.meta.platform}.md"
        write_md_file(conv, bundle_id, frontmatter, md_path)

        # 6. Rewrite _bundle.md with updated platforms
        self._write_bundle_md_multi(
            bundle_dir, bundle_id, bundle_meta, question, all_platforms, conv,
        )

        # 7. DB operations
        if not self._dry_run:
            self._repo.add_response_to_bundle(
                bundle_id=bundle_id,
                platform=conv.meta.platform,
                model=response_meta.model,
                turn_count=conv.turn_count,
                source_path=str(file_path),
            )
            self._repo.update_bundle_meta(
                bundle_id=bundle_id,
                summary=bundle_meta.summary,
                domains=bundle_meta.domains,
                topics=bundle_meta.topics,
                pending_topics=bundle_meta.pending_topics,
            )

            # 8. ChromaDB: upsert new platform chunks
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

        return {
            "bundle_id": bundle_id,
            "platform": conv.meta.platform,
            "merged": True,
            "summary": bundle_meta.summary,
            "domains": bundle_meta.domains,
            "topics": bundle_meta.topics,
        }

    def _collect_all_response_summaries(self, raw_dir: Path) -> str:
        """Collect response summaries from all raw files in a bundle."""
        parts = []
        for ext in sorted(SUPPORTED_EXTENSIONS):
            for raw_file in sorted(raw_dir.glob(f"*{ext}")):
                try:
                    raw_conv = parse_file(raw_file)
                    parts.append(self._build_response_summaries(raw_conv))
                except Exception:
                    logger.warning("Failed to parse raw file: %s", raw_file)
        return "\n".join(parts)

    def _write_bundle_md_multi(
        self,
        bundle_dir: Path,
        bundle_id: str,
        bundle_meta,
        question: str,
        platforms: list[str],
        conv: Conversation,
    ) -> None:
        """Write _bundle.md for multi-platform bundles."""
        meta_dict = bundle_meta.model_dump()
        meta_dict["id"] = bundle_id
        meta_dict["platforms"] = platforms
        meta_dict["created_at"] = conv.meta.exported_at.isoformat()
        content = f"---\n{yaml.dump(meta_dict, allow_unicode=True, default_flow_style=False)}---\n"
        (bundle_dir / "_bundle.md").write_text(content, encoding="utf-8")

    def _build_response_summaries(self, conv: Conversation) -> str:
        """Build a summary string of all assistant responses for bundle meta prompt."""
        parts = []
        for turn in conv.turns:
            if turn.role == "assistant":
                preview = turn.content[:500]
                if len(turn.content) > 500:
                    preview += "..."
                parts.append(f"{conv.meta.platform}: {preview}")
        return "\n".join(parts)

    def _get_assistant_content(self, conv: Conversation) -> str:
        """Get concatenated assistant content for response meta generation."""
        return "\n\n".join(
            turn.content for turn in conv.turns if turn.role == "assistant"
        )

    def _count_responses(self, conv: Conversation) -> int:
        """Count assistant turns."""
        return sum(1 for t in conv.turns if t.role == "assistant")

    def _write_bundle_md(
        self,
        bundle_dir: Path,
        bundle_id: str,
        bundle_meta,
        conv: Conversation,
    ) -> None:
        """Write _bundle.md with aggregate metadata."""
        meta_dict = bundle_meta.model_dump()
        meta_dict["id"] = bundle_id
        meta_dict["platforms"] = [conv.meta.platform]
        meta_dict["created_at"] = conv.meta.exported_at.isoformat()
        content = f"---\n{yaml.dump(meta_dict, allow_unicode=True, default_flow_style=False)}---\n"
        (bundle_dir / "_bundle.md").write_text(content, encoding="utf-8")
