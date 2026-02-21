"""E2E integration tests for the ingest pipeline.

Tests the full ingest flow with real PostgreSQL + ChromaDB and mocked LLM.
Verifies that JSONL and MD files are properly ingested, stored in both databases,
bundle directories are created with expected structure, and move_to_done works.

Requires:
    - docker compose -f docker/docker-compose.test.yml up -d
    - PKB_DB_INTEGRATION=1 environment variable
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pkb.generator.meta_gen import MetaGenerator
from pkb.ingest import IngestPipeline, move_to_done

SAMPLES_DIR = Path(__file__).parents[3] / "exporter-examples"
JSONL_SAMPLE = SAMPLES_DIR / "jsonl" / "PKB" / "claude.jsonl"
MD_SAMPLE = SAMPLES_DIR / "md" / "PKB" / "claude.md"


def _mock_meta_gen():
    """Create a mock MetaGenerator that returns predictable metadata."""
    mg = MagicMock(spec=MetaGenerator)
    mg.generate_bundle_meta.return_value = MagicMock(
        slug="test-bundle",
        summary="테스트 요약입니다",
        domains=["dev"],
        topics=["python"],
        pending_topics=[],
        model_dump=lambda: {
            "slug": "test-bundle",
            "summary": "테스트 요약입니다",
            "domains": ["dev"],
            "topics": ["python"],
            "pending_topics": [],
        },
    )
    mg.generate_response_meta.return_value = MagicMock(
        model="claude-haiku",
        model_dump=lambda: {"model": "claude-haiku", "summary": "응답 요약"},
    )
    return mg


def _make_pipeline(repo, chunk_store, test_kb_path):
    """Create an IngestPipeline with mocked LLM."""
    return IngestPipeline(
        repo=repo,
        chunk_store=chunk_store,
        meta_gen=_mock_meta_gen(),
        kb_path=test_kb_path,
        kb_name="test-kb",
        domains=["dev", "life"],
        topics=["python", "cooking"],
    )


class TestIngestJsonl:
    """E2E tests for JSONL file ingestion."""

    def test_ingest_jsonl_stores_in_postgres(self, repo, chunk_store, test_kb_path):
        """JSONL ingest stores bundle metadata in PostgreSQL."""
        if not JSONL_SAMPLE.exists():
            pytest.skip("Sample JSONL not found")

        # Copy sample to inbox
        dest = test_kb_path / "inbox" / "claude.jsonl"
        shutil.copy2(JSONL_SAMPLE, dest)

        pipeline = _make_pipeline(repo, chunk_store, test_kb_path)
        result = pipeline.ingest_file(dest)

        # Verify ingest succeeded
        assert result is not None
        assert "bundle_id" in result
        assert result["platform"] == "claude"

        # Verify bundle exists in PostgreSQL
        bundle = repo.get_bundle_by_id(result["bundle_id"])
        assert bundle is not None
        assert bundle["bundle_id"] == result["bundle_id"]
        assert bundle["kb"] == "test-kb"
        assert bundle["summary"] == "테스트 요약입니다"
        assert "dev" in bundle["domains"]
        assert "python" in bundle["topics"]

    def test_ingest_jsonl_stores_chunks_in_chromadb(
        self, repo, chunk_store, test_kb_path
    ):
        """JSONL ingest stores text chunks in ChromaDB."""
        if not JSONL_SAMPLE.exists():
            pytest.skip("Sample JSONL not found")

        dest = test_kb_path / "inbox" / "claude.jsonl"
        shutil.copy2(JSONL_SAMPLE, dest)

        pipeline = _make_pipeline(repo, chunk_store, test_kb_path)
        result = pipeline.ingest_file(dest)

        assert result is not None

        # Verify chunks exist in ChromaDB
        chunks = chunk_store.search(
            query=result["bundle_id"],
            n_results=10,
            where={"bundle_id": result["bundle_id"]},
        )
        assert len(chunks) > 0
        # Each chunk should have bundle metadata
        for chunk in chunks:
            assert chunk.metadata["bundle_id"] == result["bundle_id"]
            assert chunk.metadata["kb"] == "test-kb"
            assert chunk.metadata["platform"] == "claude"


class TestIngestMd:
    """E2E tests for MD file ingestion."""

    def test_ingest_md_stores_in_postgres(self, repo, chunk_store, test_kb_path):
        """MD ingest stores bundle metadata in PostgreSQL."""
        if not MD_SAMPLE.exists():
            pytest.skip("Sample MD not found")

        dest = test_kb_path / "inbox" / "claude.md"
        shutil.copy2(MD_SAMPLE, dest)

        pipeline = _make_pipeline(repo, chunk_store, test_kb_path)
        result = pipeline.ingest_file(dest)

        assert result is not None
        assert "bundle_id" in result
        assert result["platform"] == "claude"

        # Verify bundle exists in PostgreSQL
        bundle = repo.get_bundle_by_id(result["bundle_id"])
        assert bundle is not None
        assert bundle["bundle_id"] == result["bundle_id"]
        assert bundle["kb"] == "test-kb"
        assert bundle["summary"] == "테스트 요약입니다"


class TestIngestBundleDir:
    """E2E tests for bundle directory creation."""

    def test_ingest_creates_bundle_dir_and_files(
        self, repo, chunk_store, test_kb_path
    ):
        """Ingest creates bundle directory with _raw/, platform.md, and _bundle.md."""
        if not JSONL_SAMPLE.exists():
            pytest.skip("Sample JSONL not found")

        dest = test_kb_path / "inbox" / "claude.jsonl"
        shutil.copy2(JSONL_SAMPLE, dest)

        pipeline = _make_pipeline(repo, chunk_store, test_kb_path)
        result = pipeline.ingest_file(dest)

        assert result is not None

        # Verify bundle directory structure
        bundle_dir = test_kb_path / "bundles" / result["bundle_id"]
        assert bundle_dir.is_dir()

        # _raw/ directory should exist with the original file
        raw_dir = bundle_dir / "_raw"
        assert raw_dir.is_dir()
        raw_files = list(raw_dir.iterdir())
        assert len(raw_files) >= 1
        raw_names = [f.name for f in raw_files]
        assert "claude.jsonl" in raw_names

        # Platform MD file should exist
        platform_md = bundle_dir / "claude.md"
        assert platform_md.is_file()
        content = platform_md.read_text(encoding="utf-8")
        assert "---" in content  # Has frontmatter

        # _bundle.md should exist with YAML frontmatter
        bundle_md = bundle_dir / "_bundle.md"
        assert bundle_md.is_file()
        bundle_content = bundle_md.read_text(encoding="utf-8")
        assert "---" in bundle_content
        assert result["bundle_id"] in bundle_content


class TestMoveToDone:
    """E2E test for move_to_done with subdirectory structure preservation."""

    def test_move_to_done_preserves_structure(self, test_kb_path):
        """move_to_done preserves subdirectory structure in .done/."""
        inbox = test_kb_path / "inbox"
        # Create subdirectory structure like exporter output
        pkb_dir = inbox / "PKB"
        pkb_dir.mkdir(parents=True, exist_ok=True)

        test_file = pkb_dir / "test.jsonl"
        test_file.write_text('{"_meta": true}')

        result = move_to_done(test_file, inbox)

        assert result is not None
        expected = inbox / ".done" / "PKB" / "test.jsonl"
        assert result == expected
        assert result.exists()
        assert not test_file.exists()
        assert result.read_text() == '{"_meta": true}'
