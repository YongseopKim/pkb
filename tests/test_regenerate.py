"""Tests for Regenerator class."""

from unittest.mock import MagicMock

import pytest

from pkb.models.meta import BundleMeta, ResponseMeta
from pkb.regenerate import Regenerator


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.list_all_bundle_ids.return_value = []
    return repo


@pytest.fixture
def mock_chunk_store():
    return MagicMock()


@pytest.fixture
def mock_meta_gen():
    meta_gen = MagicMock()
    meta_gen.generate_bundle_meta.return_value = BundleMeta(
        summary="재생성된 요약",
        slug="test",
        domains=["dev"],
        topics=["python"],
        pending_topics=[],
    )
    meta_gen.generate_response_meta.return_value = ResponseMeta(
        platform="claude",
        model="haiku",
        summary="응답 요약",
    )
    return meta_gen


@pytest.fixture
def kb_dir(tmp_path):
    """Create a KB directory with a raw JSONL bundle."""
    bundle_dir = tmp_path / "bundles" / "20260101-test-abc1"
    raw_dir = bundle_dir / "_raw"
    raw_dir.mkdir(parents=True)

    # Write raw JSONL
    jsonl_content = (
        '{"_meta":true,"platform":"claude","url":"https://example.com",'
        '"exported_at":"2026-01-01T00:00:00.000Z","title":"테스트"}\n'
        '{"role":"user","content":"테스트 질문입니다"}\n'
        '{"role":"assistant","content":"테스트 답변입니다"}\n'
    )
    (raw_dir / "conv.jsonl").write_text(jsonl_content, encoding="utf-8")
    return tmp_path


@pytest.fixture
def regenerator(mock_repo, mock_chunk_store, mock_meta_gen, kb_dir):
    return Regenerator(
        repo=mock_repo,
        chunk_store=mock_chunk_store,
        meta_gen=mock_meta_gen,
        kb_path=kb_dir,
        kb_name="personal",
        domains=["dev", "ai"],
        topics=["python", "ml"],
    )


class TestRegenerateBundle:
    def test_regenerate_single_bundle(
        self, regenerator, mock_repo, mock_chunk_store, mock_meta_gen
    ):
        result = regenerator.regenerate_bundle("20260101-test-abc1")
        assert result["bundle_id"] == "20260101-test-abc1"
        assert result["status"] == "regenerated"
        mock_meta_gen.generate_bundle_meta.assert_called_once()
        mock_meta_gen.generate_response_meta.assert_called_once()
        mock_repo.upsert_bundle.assert_called_once()
        mock_chunk_store.delete_by_bundle.assert_called_once()
        mock_chunk_store.upsert_chunks.assert_called_once()

    def test_regenerate_writes_platform_md(self, regenerator, kb_dir):
        regenerator.regenerate_bundle("20260101-test-abc1")
        platform_md = kb_dir / "bundles" / "20260101-test-abc1" / "claude.md"
        assert platform_md.exists()
        content = platform_md.read_text(encoding="utf-8")
        assert "---" in content  # has frontmatter

    def test_regenerate_writes_bundle_md(self, regenerator, kb_dir):
        regenerator.regenerate_bundle("20260101-test-abc1")
        bundle_md = kb_dir / "bundles" / "20260101-test-abc1" / "_bundle.md"
        assert bundle_md.exists()
        content = bundle_md.read_text(encoding="utf-8")
        assert "재생성된 요약" in content

    def test_regenerate_bundle_md_no_question(self, regenerator, kb_dir):
        """regenerate로 생성되는 _bundle.md에 question 키가 없어야 함."""
        import yaml as yaml_mod

        regenerator.regenerate_bundle("20260101-test-abc1")
        bundle_md = kb_dir / "bundles" / "20260101-test-abc1" / "_bundle.md"
        content = bundle_md.read_text(encoding="utf-8")
        parts = content.split("---")
        meta = yaml_mod.safe_load(parts[1])
        assert "question" not in meta

    def test_regenerate_missing_raw(self, regenerator, kb_dir):
        """Bundle dir exists but no _raw/*.jsonl → error."""
        empty_dir = kb_dir / "bundles" / "20260102-empty-abc1"
        empty_dir.mkdir(parents=True)
        result = regenerator.regenerate_bundle("20260102-empty-abc1")
        assert result["status"] == "error"
        assert "raw" in result["reason"].lower() or "jsonl" in result["reason"].lower()

    def test_regenerate_missing_bundle_dir(self, regenerator):
        result = regenerator.regenerate_bundle("nonexistent-bundle")
        assert result["status"] == "error"
        assert "not found" in result["reason"].lower()

    def test_regenerate_increments_meta_version(self, regenerator, mock_repo):
        regenerator.regenerate_bundle("20260101-test-abc1")
        call_kwargs = mock_repo.upsert_bundle.call_args[1]
        # The bundle should be upserted (ON CONFLICT will update meta_version)
        assert call_kwargs["bundle_id"] == "20260101-test-abc1"

    def test_regenerate_dry_run(
        self, mock_repo, mock_chunk_store, mock_meta_gen, kb_dir
    ):
        regen = Regenerator(
            repo=mock_repo,
            chunk_store=mock_chunk_store,
            meta_gen=mock_meta_gen,
            kb_path=kb_dir,
            kb_name="personal",
            domains=["dev"],
            topics=["python"],
            dry_run=True,
        )
        result = regen.regenerate_bundle("20260101-test-abc1")
        assert result["status"] == "regenerated"
        # DB operations should be skipped in dry-run
        mock_repo.upsert_bundle.assert_not_called()
        mock_chunk_store.upsert_chunks.assert_not_called()


class TestRegenerateAll:
    def test_regenerate_all_scans_bundles(self, regenerator, mock_repo):
        result = regenerator.regenerate_all()
        assert result["regenerated"] >= 1
        assert "total" in result

    def test_regenerate_all_with_callback(self, regenerator):
        callback = MagicMock()
        regenerator.regenerate_all(progress_callback=callback)
        assert callback.call_count >= 1

    def test_regenerate_all_counts_errors(
        self, mock_repo, mock_chunk_store, mock_meta_gen, kb_dir
    ):
        # Add a bundle with no raw JSONL → should be counted as error
        bad_dir = kb_dir / "bundles" / "20260102-bad-abc1"
        bad_dir.mkdir(parents=True)

        regen = Regenerator(
            repo=mock_repo,
            chunk_store=mock_chunk_store,
            meta_gen=mock_meta_gen,
            kb_path=kb_dir,
            kb_name="personal",
            domains=["dev"],
            topics=["python"],
        )
        result = regen.regenerate_all()
        assert result["errors"] >= 1
