"""Tests for the ingest pipeline."""

import json
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import yaml

from pkb.constants import DONE_DIR_NAME
from pkb.ingest import (
    IngestPipeline,
    compute_question_hash,
    generate_bundle_id,
    generate_question_hash,
    move_to_done,
)

# Reusable test assistant content (>= 50 chars each)
_ASYNC_ANSWER = (
    "async는 Python 비동기 프로그래밍 핵심 키워드입니다. "
    "asyncio 라이브러리와 함께 사용합니다."
)
_GENERIC_ANSWER = (
    "답변입니다. 이것은 테스트를 위한 충분히 긴 "
    "응답 내용입니다. 최소 50자 이상이어야 합니다."
)
_CHATGPT_ANSWER = (
    "ChatGPT의 답변입니다. async/await는 비동기 "
    "프로그래밍의 핵심 패턴으로 유용합니다."
)
_NEW_ANSWER = (
    "새 답변입니다. 이것은 새로운 질문에 대한 "
    "충분히 긴 응답으로 50자 이상의 컨텐츠입니다."
)


class TestMoveToDone:
    """Tests for move_to_done() inbox cleanup function."""

    def test_moves_file_from_inbox(self, tmp_path):
        """inbox 직하 파일이 .done/으로 이동."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        f = inbox / "test.jsonl"
        f.write_text("data")

        result = move_to_done(f, inbox)

        assert result is not None
        assert result == inbox / DONE_DIR_NAME / "test.jsonl"
        assert result.exists()
        assert not f.exists()

    def test_ignores_file_outside_inbox(self, tmp_path):
        """inbox 외부 파일은 이동하지 않음."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        f = other / "test.jsonl"
        f.write_text("data")

        result = move_to_done(f, inbox)

        assert result is None
        assert f.exists()

    def test_dry_run_returns_path_without_moving(self, tmp_path):
        """dry_run=True이면 경로만 반환, 파일은 이동하지 않음."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        f = inbox / "test.jsonl"
        f.write_text("data")

        result = move_to_done(f, inbox, dry_run=True)

        assert result == inbox / DONE_DIR_NAME / "test.jsonl"
        assert f.exists()  # 원본 파일 그대로

    def test_overwrites_existing_file(self, tmp_path):
        """동일 파일명이 .done/에 있으면 덮어쓰기."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        done_dir = inbox / DONE_DIR_NAME
        done_dir.mkdir()
        (done_dir / "test.jsonl").write_text("old")
        f = inbox / "test.jsonl"
        f.write_text("new")

        result = move_to_done(f, inbox)

        assert result is not None
        assert result.read_text() == "new"
        assert not f.exists()

    def test_creates_done_dir_automatically(self, tmp_path):
        """.done/ 디렉토리가 없으면 자동 생성."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        f = inbox / "test.jsonl"
        f.write_text("data")

        assert not (inbox / DONE_DIR_NAME).exists()
        result = move_to_done(f, inbox)

        assert result is not None
        assert (inbox / DONE_DIR_NAME).is_dir()

    def test_handles_os_error_gracefully(self, tmp_path):
        """파일이 이미 없는 경우 등 OSError에도 안전."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        f = inbox / "nonexistent.jsonl"  # 존재하지 않는 파일

        result = move_to_done(f, inbox)

        assert result is None

    def test_moves_subdirectory_file_preserving_structure(self, tmp_path):
        """inbox 서브디렉토리의 파일을 .done/ 내 동일 구조로 이동."""
        inbox = tmp_path / "inbox"
        sub = inbox / "PKB"
        sub.mkdir(parents=True)
        f = sub / "chatgpt.md"
        f.write_text("data")

        result = move_to_done(f, inbox)

        assert result is not None
        assert result == inbox / DONE_DIR_NAME / "PKB" / "chatgpt.md"
        assert result.exists()
        assert not f.exists()

    def test_moves_deeply_nested_file(self, tmp_path):
        """깊이 중첩된 파일도 구조 보존하여 이동."""
        inbox = tmp_path / "inbox"
        deep = inbox / "a" / "b"
        deep.mkdir(parents=True)
        f = deep / "test.md"
        f.write_text("data")

        result = move_to_done(f, inbox)

        assert result == inbox / DONE_DIR_NAME / "a" / "b" / "test.md"
        assert result.exists()
        assert not f.exists()

    def test_dry_run_subdirectory(self, tmp_path):
        """dry_run에서 서브디렉토리 파일의 예상 경로 반환."""
        inbox = tmp_path / "inbox"
        sub = inbox / "PKB"
        sub.mkdir(parents=True)
        f = sub / "claude.jsonl"
        f.write_text("data")

        result = move_to_done(f, inbox, dry_run=True)

        assert result == inbox / DONE_DIR_NAME / "PKB" / "claude.jsonl"
        assert f.exists()

    def test_ignores_file_already_in_done(self, tmp_path):
        """.done/ 내부 파일은 이동하지 않음."""
        inbox = tmp_path / "inbox"
        done = inbox / DONE_DIR_NAME
        done.mkdir(parents=True)
        f = done / "old.jsonl"
        f.write_text("data")

        result = move_to_done(f, inbox)

        assert result is None
        assert f.exists()

    def test_cleans_up_empty_parent_dirs(self, tmp_path):
        """이동 후 빈 부모 디렉토리를 정리 (watch_dir까지만)."""
        inbox = tmp_path / "inbox"
        sub = inbox / "PKB"
        sub.mkdir(parents=True)
        f = sub / "only_file.jsonl"
        f.write_text("data")

        move_to_done(f, inbox)

        # PKB/ should be cleaned up since it's now empty
        assert not sub.exists()
        # inbox itself should still exist
        assert inbox.exists()


class TestComputeQuestionHash:
    """compute_question_hash(): MD 파일의 빈 question 대신 assistant content로 hash."""

    def _make_conv(self, *, user_msg=None, title=None, assistant_content="답변입니다."):
        """테스트용 Conversation 생성 헬퍼."""
        from pkb.models.jsonl import Conversation, ConversationMeta, Turn
        turns = []
        if user_msg:
            turns.append(Turn(role="user", content=user_msg, timestamp=None))
        turns.append(Turn(role="assistant", content=assistant_content, timestamp=None))
        return Conversation(
            meta=ConversationMeta(
                platform="claude",
                exported_at=datetime(2026, 2, 21, tzinfo=timezone.utc),
                title=title,
            ),
            turns=turns,
        )

    def test_with_user_message(self):
        """일반 JSONL: user message 기반 hash (기존 동작 동일)."""
        conv = self._make_conv(user_msg="파이썬에서 async가 뭐야?")
        question, qhash = compute_question_hash(conv)
        assert question == "파이썬에서 async가 뭐야?"
        assert qhash == generate_question_hash("파이썬에서 async가 뭐야?")

    def test_with_title_no_user_msg(self):
        """title만 있고 user message 없는 경우: title 기반 hash."""
        conv = self._make_conv(title="테스트 대화")
        question, qhash = compute_question_hash(conv)
        assert question == "테스트 대화"
        assert qhash == generate_question_hash("테스트 대화")

    def test_md_no_user_no_title(self):
        """MD 파일: user도 title도 없으면 assistant content 앞 500자 기반 hash."""
        conv = self._make_conv(assistant_content="이것은 LLM의 응답입니다.")
        question, qhash = compute_question_hash(conv)
        assert question == ""  # question 자체는 빈 문자열
        # hash는 빈 문자열이 아닌 assistant content 기반
        assert qhash != generate_question_hash("")
        assert qhash == generate_question_hash("이것은 LLM의 응답입니다.")

    def test_different_md_different_hash(self):
        """다른 내용의 MD 파일은 다른 hash."""
        conv1 = self._make_conv(assistant_content="응답 A")
        conv2 = self._make_conv(assistant_content="응답 B")
        _, h1 = compute_question_hash(conv1)
        _, h2 = compute_question_hash(conv2)
        assert h1 != h2

    def test_same_md_same_hash(self):
        """같은 내용의 MD 파일은 같은 hash (결정적)."""
        conv1 = self._make_conv(assistant_content="동일한 응답")
        conv2 = self._make_conv(assistant_content="동일한 응답")
        _, h1 = compute_question_hash(conv1)
        _, h2 = compute_question_hash(conv2)
        assert h1 == h2

    def test_long_content_uses_first_500_chars(self):
        """긴 content는 앞 500자만 사용."""
        long_content = "A" * 1000
        conv = self._make_conv(assistant_content=long_content)
        question, qhash = compute_question_hash(conv)
        assert question == ""
        assert qhash == generate_question_hash(long_content[:500])


class TestGenerateQuestionHash:
    def test_deterministic(self):
        h1 = generate_question_hash("파이썬에서 async가 뭐야?")
        h2 = generate_question_hash("파이썬에서 async가 뭐야?")
        assert h1 == h2

    def test_different_questions_different_hashes(self):
        h1 = generate_question_hash("질문 1")
        h2 = generate_question_hash("질문 2")
        assert h1 != h2

    def test_hash_length(self):
        h = generate_question_hash("test")
        assert len(h) == 64  # SHA-256 hex digest


class TestGenerateBundleId:
    def test_format(self):
        bid = generate_bundle_id(
            date=datetime(2026, 2, 21, tzinfo=timezone.utc),
            slug="pkb-system-design",
            question="test question",
        )
        assert bid.startswith("20260221-")
        assert "pkb-system-design" in bid
        # hash4 at the end
        parts = bid.split("-")
        assert len(parts[-1]) == 4

    def test_deterministic_hash(self):
        bid1 = generate_bundle_id(
            date=datetime(2026, 2, 21, tzinfo=timezone.utc),
            slug="test",
            question="same question",
        )
        bid2 = generate_bundle_id(
            date=datetime(2026, 2, 21, tzinfo=timezone.utc),
            slug="test",
            question="same question",
        )
        assert bid1 == bid2


class TestIngestPipeline:
    @pytest.fixture
    def mock_deps(self, tmp_path):
        """Create mocked dependencies for IngestPipeline."""
        repo = MagicMock()
        repo.find_bundle_by_question_hash.return_value = None  # no existing bundle
        chunk_store = MagicMock()
        meta_gen = MagicMock()
        meta_gen.generate_response_meta.return_value = MagicMock(
            platform="claude",
            model="claude-sonnet",
            summary="응답 요약",
            key_claims=["주장1"],
            stance="informative",
            model_dump=MagicMock(return_value={
                "platform": "claude",
                "model": "claude-sonnet",
                "summary": "응답 요약",
                "key_claims": ["주장1"],
                "stance": "informative",
            }),
        )
        meta_gen.generate_bundle_meta.return_value = MagicMock(
            summary="번들 요약",
            slug="test-slug",
            domains=["dev"],
            topics=["python"],
            pending_topics=[],
            consensus="합의점",
            divergence="차이점",
            model_dump=MagicMock(return_value={
                "summary": "번들 요약",
                "slug": "test-slug",
                "domains": ["dev"],
                "topics": ["python"],
                "pending_topics": [],
                "consensus": "합의점",
                "divergence": "차이점",
            }),
        )
        kb_path = tmp_path / "kb-test"
        kb_path.mkdir()
        return {
            "repo": repo,
            "chunk_store": chunk_store,
            "meta_gen": meta_gen,
            "kb_path": kb_path,
            "kb_name": "test",
        }

    @pytest.fixture
    def pipeline(self, mock_deps):
        return IngestPipeline(
            repo=mock_deps["repo"],
            chunk_store=mock_deps["chunk_store"],
            meta_gen=mock_deps["meta_gen"],
            kb_path=mock_deps["kb_path"],
            kb_name=mock_deps["kb_name"],
            domains=["dev", "invest"],
            topics=["python", "system-design"],
        )

    @pytest.fixture
    def sample_jsonl(self, tmp_path):
        """Create a sample JSONL file."""
        jsonl_path = tmp_path / "source" / "claude.jsonl"
        jsonl_path.parent.mkdir(parents=True)
        lines = [
            json.dumps({
                "_meta": True,
                "platform": "claude",
                "url": "https://claude.ai/chat/123",
                "exported_at": "2026-02-21T06:00:00.000Z",
                "title": "테스트 대화",
            }),
            json.dumps({
                "role": "user",
                "content": "파이썬에서 async가 뭐야?",
                "timestamp": "2026-02-21T06:00:01.000Z",
            }),
            json.dumps({
                "role": "assistant",
                "content": _ASYNC_ANSWER,
                "timestamp": "2026-02-21T06:00:02.000Z",
            }),
        ]
        jsonl_path.write_text("\n".join(lines), encoding="utf-8")
        return jsonl_path

    def test_ingest_single_file(self, pipeline, sample_jsonl, mock_deps):
        result = pipeline.ingest_file(sample_jsonl)
        assert result is not None
        assert result["bundle_id"] is not None
        # Repo should have been called to upsert
        mock_deps["repo"].upsert_bundle.assert_called_once()
        # ChunkStore should have been called
        mock_deps["chunk_store"].upsert_chunks.assert_called_once()

    def test_skip_duplicate(self, pipeline, sample_jsonl, mock_deps):
        # Same platform (claude) in existing bundle → SKIP
        mock_deps["repo"].find_bundle_by_question_hash.return_value = {
            "bundle_id": "20260221-existing-a3f2",
            "kb": "test",
            "path": "bundles/20260221-existing-a3f2",
            "platforms": ["claude"],
            "domains": ["dev"],
            "topics": ["python"],
        }
        result = pipeline.ingest_file(sample_jsonl)
        assert result is None  # Skipped
        mock_deps["repo"].upsert_bundle.assert_not_called()

    def test_creates_bundle_directory(self, pipeline, sample_jsonl, mock_deps):
        result = pipeline.ingest_file(sample_jsonl)
        bundle_dir = mock_deps["kb_path"] / "bundles" / result["bundle_id"]
        assert bundle_dir.exists()
        assert (bundle_dir / "_raw").is_dir()

    def test_copies_raw_jsonl(self, pipeline, sample_jsonl, mock_deps):
        result = pipeline.ingest_file(sample_jsonl)
        bundle_dir = mock_deps["kb_path"] / "bundles" / result["bundle_id"]
        raw_files = list((bundle_dir / "_raw").glob("*.jsonl"))
        assert len(raw_files) == 1

    def test_generates_md_file(self, pipeline, sample_jsonl, mock_deps):
        result = pipeline.ingest_file(sample_jsonl)
        bundle_dir = mock_deps["kb_path"] / "bundles" / result["bundle_id"]
        md_files = list(bundle_dir.glob("*.md"))
        assert len(md_files) >= 1  # At least platform MD + _bundle.md

    def test_force_bypasses_dedup(self, pipeline, sample_jsonl, mock_deps):
        """force=True이면 question_hash 중복이어도 ingest 진행."""
        mock_deps["repo"].find_bundle_by_question_hash.return_value = {
            "bundle_id": "20260221-existing-a3f2",
            "kb": "test",
            "path": "bundles/20260221-existing-a3f2",
            "platforms": ["claude"],
            "domains": ["dev"],
            "topics": ["python"],
        }
        result = pipeline.ingest_file(sample_jsonl, force=True)
        assert result is not None
        mock_deps["repo"].upsert_bundle.assert_called_once()

    def test_source_path_saved_to_db(self, pipeline, sample_jsonl, mock_deps):
        """ingest 시 source_path가 DB에 전달되어야 함."""
        result = pipeline.ingest_file(sample_jsonl)
        assert result is not None
        call_kwargs = mock_deps["repo"].upsert_bundle.call_args[1]
        assert "source_path" in call_kwargs
        assert call_kwargs["source_path"] == str(sample_jsonl)

    def test_source_path_in_responses(self, pipeline, sample_jsonl, mock_deps):
        """ingest 시 responses dict에도 source_path가 포함되어야 함."""
        result = pipeline.ingest_file(sample_jsonl)
        assert result is not None
        call_kwargs = mock_deps["repo"].upsert_bundle.call_args[1]
        responses = call_kwargs["responses"]
        assert len(responses) == 1
        assert "source_path" in responses[0]
        assert responses[0]["source_path"] == str(sample_jsonl)

    def test_force_false_default_skips_duplicate(self, pipeline, sample_jsonl, mock_deps):
        """force 기본값 False: 기존 동작과 동일하게 중복 시 None."""
        mock_deps["repo"].find_bundle_by_question_hash.return_value = {
            "bundle_id": "20260221-existing-a3f2",
            "kb": "test",
            "path": "bundles/20260221-existing-a3f2",
            "platforms": ["claude"],
            "domains": ["dev"],
            "topics": ["python"],
        }
        result = pipeline.ingest_file(sample_jsonl)
        assert result is None

    def test_dry_run_no_side_effects(self, mock_deps, sample_jsonl):
        pipeline = IngestPipeline(
            repo=mock_deps["repo"],
            chunk_store=mock_deps["chunk_store"],
            meta_gen=mock_deps["meta_gen"],
            kb_path=mock_deps["kb_path"],
            kb_name=mock_deps["kb_name"],
            domains=["dev"],
            topics=["python"],
            dry_run=True,
        )
        result = pipeline.ingest_file(sample_jsonl)
        assert result is not None
        # DB should NOT be called in dry-run
        mock_deps["repo"].upsert_bundle.assert_not_called()
        mock_deps["chunk_store"].upsert_chunks.assert_not_called()


class TestBundleMdNoQuestion:
    """_bundle.md frontmatter에 question 키가 없어야 함."""

    @pytest.fixture
    def mock_deps(self, tmp_path):
        repo = MagicMock()
        repo.find_bundle_by_question_hash.return_value = None
        chunk_store = MagicMock()
        meta_gen = MagicMock()
        meta_gen.generate_bundle_meta.return_value = MagicMock(
            summary="요약", slug="test-slug",
            domains=["dev"], topics=["python"], pending_topics=[],
            consensus=None, divergence=None,
            model_dump=MagicMock(return_value={
                "summary": "요약", "slug": "test-slug",
                "domains": ["dev"], "topics": ["python"],
                "pending_topics": [], "consensus": None, "divergence": None,
            }),
        )
        meta_gen.generate_response_meta.return_value = MagicMock(
            platform="claude", model="claude-3", summary="응답 요약",
            key_claims=[], stance="informative",
            model_dump=MagicMock(return_value={
                "platform": "claude", "model": "claude-3",
                "summary": "응답 요약", "key_claims": [], "stance": "informative",
            }),
        )
        kb_path = tmp_path / "kb-test"
        kb_path.mkdir()
        return {"repo": repo, "chunk_store": chunk_store, "meta_gen": meta_gen,
                "kb_path": kb_path, "kb_name": "test"}

    @pytest.fixture
    def pipeline(self, mock_deps):
        return IngestPipeline(
            repo=mock_deps["repo"], chunk_store=mock_deps["chunk_store"],
            meta_gen=mock_deps["meta_gen"], kb_path=mock_deps["kb_path"],
            kb_name=mock_deps["kb_name"], domains=["dev"], topics=["python"],
        )

    @pytest.fixture
    def sample_jsonl(self, tmp_path):
        lines = [
            json.dumps({"_meta": True, "platform": "claude",
                         "url": "https://claude.ai/chat/1",
                         "exported_at": "2026-02-21T06:00:00.000Z",
                         "title": "테스트"}),
            json.dumps({"role": "user", "content": "질문입니다",
                         "timestamp": "2026-02-21T06:00:01.000Z"}),
            json.dumps({"role": "assistant", "content": _GENERIC_ANSWER,
                         "timestamp": "2026-02-21T06:00:02.000Z"}),
        ]
        p = tmp_path / "test.jsonl"
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def test_bundle_md_has_no_question_key(self, pipeline, sample_jsonl, mock_deps):
        """새로 생성되는 _bundle.md에 question 키가 없어야 함."""
        result = pipeline.ingest_file(sample_jsonl)
        bundle_dir = mock_deps["kb_path"] / "bundles" / result["bundle_id"]
        content = (bundle_dir / "_bundle.md").read_text()
        parts = content.split("---")
        meta = yaml.safe_load(parts[1])
        assert "question" not in meta

    def test_ingest_return_has_no_question_key(self, pipeline, sample_jsonl, mock_deps):
        """ingest_file 반환 dict에 question 키가 없어야 함."""
        result = pipeline.ingest_file(sample_jsonl)
        assert "question" not in result
        assert "summary" in result


class TestOrphanDirectoryPrevention:
    """Bug 3: LLM 실패 시 orphan bundle 디렉토리가 생기면 안 됨."""

    @pytest.fixture
    def mock_deps(self, tmp_path):
        repo = MagicMock()
        repo.find_bundle_by_question_hash.return_value = None
        chunk_store = MagicMock()
        meta_gen = MagicMock()
        meta_gen.generate_bundle_meta.return_value = MagicMock(
            summary="요약", slug="test-slug",
            domains=["dev"], topics=["python"], pending_topics=[],
            model_dump=MagicMock(return_value={}),
        )
        meta_gen.generate_response_meta.return_value = MagicMock(
            platform="claude", model="model", summary="요약",
            model_dump=MagicMock(return_value={"platform": "claude"}),
        )
        kb_path = tmp_path / "kb-test"
        kb_path.mkdir()
        return {
            "repo": repo, "chunk_store": chunk_store, "meta_gen": meta_gen,
            "kb_path": kb_path, "kb_name": "test",
        }

    @pytest.fixture
    def pipeline(self, mock_deps):
        return IngestPipeline(
            repo=mock_deps["repo"], chunk_store=mock_deps["chunk_store"],
            meta_gen=mock_deps["meta_gen"], kb_path=mock_deps["kb_path"],
            kb_name=mock_deps["kb_name"], domains=["dev"], topics=["python"],
        )

    @pytest.fixture
    def sample_jsonl(self, tmp_path):
        jsonl_path = tmp_path / "source" / "claude.jsonl"
        jsonl_path.parent.mkdir(parents=True)
        lines = [
            json.dumps({"_meta": True, "platform": "claude",
                         "url": "https://claude.ai/chat/123",
                         "exported_at": "2026-02-21T06:00:00.000Z", "title": "테스트"}),
            json.dumps({"role": "user", "content": "질문?",
                         "timestamp": "2026-02-21T06:00:01.000Z"}),
            json.dumps({"role": "assistant", "content": _GENERIC_ANSWER,
                         "timestamp": "2026-02-21T06:00:02.000Z"}),
        ]
        jsonl_path.write_text("\n".join(lines), encoding="utf-8")
        return jsonl_path

    def test_response_meta_failure_no_orphan_dir(self, pipeline, sample_jsonl, mock_deps):
        """response_meta 실패 시 bundles/ 디렉토리에 아무것도 생기면 안 됨."""
        mock_deps["meta_gen"].generate_response_meta.side_effect = ValueError("LLM 빈 응답")

        with pytest.raises(ValueError, match="LLM 빈 응답"):
            pipeline.ingest_file(sample_jsonl)

        bundles_dir = mock_deps["kb_path"] / "bundles"
        if bundles_dir.exists():
            assert list(bundles_dir.iterdir()) == []

    def test_bundle_meta_failure_no_orphan_dir(self, pipeline, sample_jsonl, mock_deps):
        """bundle_meta 실패 시 bundles/ 디렉토리에 아무것도 생기면 안 됨."""
        mock_deps["meta_gen"].generate_bundle_meta.side_effect = ValueError("LLM 에러")

        with pytest.raises(ValueError, match="LLM 에러"):
            pipeline.ingest_file(sample_jsonl)

        bundles_dir = mock_deps["kb_path"] / "bundles"
        if bundles_dir.exists():
            assert list(bundles_dir.iterdir()) == []

    def test_llm_calls_before_mkdir(self, pipeline, sample_jsonl, mock_deps):
        """LLM 호출 순서: bundle_meta → response_meta 모두 mkdir 전."""
        call_order = []

        orig_bundle = mock_deps["meta_gen"].generate_bundle_meta
        orig_response = mock_deps["meta_gen"].generate_response_meta

        def track_bundle(*a, **kw):
            call_order.append("bundle_meta")
            return orig_bundle.return_value

        def track_response(*a, **kw):
            call_order.append("response_meta")
            return orig_response.return_value

        mock_deps["meta_gen"].generate_bundle_meta.side_effect = track_bundle
        mock_deps["meta_gen"].generate_response_meta.side_effect = track_response

        pipeline.ingest_file(sample_jsonl)

        assert "bundle_meta" in call_order
        assert "response_meta" in call_order
        # Both should be called (order: bundle_meta first, then response_meta)
        assert call_order.index("bundle_meta") < call_order.index("response_meta")


class TestMergeFile:
    """merge_file(): 같은 질문 + 다른 플랫폼 → 기존 번들에 merge."""

    @pytest.fixture
    def mock_deps(self, tmp_path):
        """Create mocked dependencies for IngestPipeline."""
        repo = MagicMock()
        repo.find_bundle_by_question_hash.return_value = None
        chunk_store = MagicMock()
        meta_gen = MagicMock()
        meta_gen.generate_response_meta.return_value = MagicMock(
            platform="chatgpt",
            model="gpt-4o",
            summary="ChatGPT 응답 요약",
            key_claims=["주장1"],
            stance="informative",
            model_dump=MagicMock(return_value={
                "platform": "chatgpt",
                "model": "gpt-4o",
                "summary": "ChatGPT 응답 요약",
                "key_claims": ["주장1"],
                "stance": "informative",
            }),
        )
        meta_gen.generate_bundle_meta.return_value = MagicMock(
            summary="멀티플랫폼 번들 요약",
            slug="test-slug",
            domains=["dev"],
            topics=["python"],
            pending_topics=[],
            consensus="합의점",
            divergence="차이점",
            model_dump=MagicMock(return_value={
                "summary": "멀티플랫폼 번들 요약",
                "slug": "test-slug",
                "domains": ["dev"],
                "topics": ["python"],
                "pending_topics": [],
                "consensus": "합의점",
                "divergence": "차이점",
            }),
        )
        kb_path = tmp_path / "kb-test"
        kb_path.mkdir()
        return {
            "repo": repo,
            "chunk_store": chunk_store,
            "meta_gen": meta_gen,
            "kb_path": kb_path,
            "kb_name": "test",
        }

    @pytest.fixture
    def pipeline(self, mock_deps):
        return IngestPipeline(
            repo=mock_deps["repo"],
            chunk_store=mock_deps["chunk_store"],
            meta_gen=mock_deps["meta_gen"],
            kb_path=mock_deps["kb_path"],
            kb_name=mock_deps["kb_name"],
            domains=["dev", "invest"],
            topics=["python", "system-design"],
        )

    @pytest.fixture
    def existing_bundle(self, mock_deps):
        """기존 claude 번들 디렉토리 + raw 파일 생성."""
        bundle_id = "20260221-test-slug-a3f2"
        bundle_dir = mock_deps["kb_path"] / "bundles" / bundle_id
        raw_dir = bundle_dir / "_raw"
        raw_dir.mkdir(parents=True)

        # 기존 claude raw 파일
        claude_lines = [
            json.dumps({
                "_meta": True, "platform": "claude",
                "url": "https://claude.ai/chat/123",
                "exported_at": "2026-02-21T06:00:00.000Z",
                "title": "테스트 대화",
            }),
            json.dumps({
                "role": "user", "content": "파이썬에서 async가 뭐야?",
                "timestamp": "2026-02-21T06:00:01.000Z",
            }),
            json.dumps({
                "role": "assistant", "content": _ASYNC_ANSWER,
                "timestamp": "2026-02-21T06:00:02.000Z",
            }),
        ]
        (raw_dir / "claude.jsonl").write_text("\n".join(claude_lines), encoding="utf-8")
        (bundle_dir / "claude.md").write_text("---\nplatform: claude\n---\n# Claude")
        (bundle_dir / "_bundle.md").write_text("---\nid: test\n---\n")

        return {
            "bundle_id": bundle_id,
            "kb": "test",
            "path": f"bundles/{bundle_id}",
            "platforms": ["claude"],
            "domains": ["dev"],
            "topics": ["python"],
        }

    @pytest.fixture
    def chatgpt_jsonl(self, tmp_path):
        """ChatGPT JSONL 파일 (같은 질문, 다른 플랫폼)."""
        jsonl_path = tmp_path / "source" / "chatgpt.jsonl"
        jsonl_path.parent.mkdir(parents=True)
        lines = [
            json.dumps({
                "_meta": True, "platform": "chatgpt",
                "url": "https://chatgpt.com/c/456",
                "exported_at": "2026-02-21T07:00:00.000Z",
                "title": "테스트 대화",
            }),
            json.dumps({
                "role": "user", "content": "파이썬에서 async가 뭐야?",
                "timestamp": "2026-02-21T07:00:01.000Z",
            }),
            json.dumps({
                "role": "assistant",
                "content": _ASYNC_ANSWER,
                "timestamp": "2026-02-21T07:00:02.000Z",
            }),
        ]
        jsonl_path.write_text("\n".join(lines), encoding="utf-8")
        return jsonl_path

    def test_merge_different_platform(self, pipeline, chatgpt_jsonl, existing_bundle, mock_deps):
        """같은 질문 + 다른 플랫폼 → merge_file 호출, merged=True."""
        # ingest_file에서 find_bundle_by_question_hash가 기존 번들 반환하도록 설정
        mock_deps["repo"].find_bundle_by_question_hash.return_value = existing_bundle

        result = pipeline.ingest_file(chatgpt_jsonl)

        assert result is not None
        assert result.get("merged") is True
        assert result["platform"] == "chatgpt"

    def test_skip_same_platform_same_question(self, pipeline, mock_deps, tmp_path):
        """같은 질문 + 같은 플랫폼 → None (진짜 SKIP)."""
        existing = {
            "bundle_id": "20260221-test-a3f2",
            "kb": "test",
            "path": "bundles/20260221-test-a3f2",
            "platforms": ["claude"],  # 같은 플랫폼
            "domains": ["dev"],
            "topics": ["python"],
        }
        mock_deps["repo"].find_bundle_by_question_hash.return_value = existing

        # claude JSONL 파일
        jsonl_path = tmp_path / "claude2.jsonl"
        lines = [
            json.dumps({
                "_meta": True, "platform": "claude",
                "url": "https://claude.ai/chat/789",
                "exported_at": "2026-02-21T08:00:00.000Z",
                "title": "테스트 대화",
            }),
            json.dumps({
                "role": "user", "content": "파이썬에서 async가 뭐야?",
                "timestamp": "2026-02-21T08:00:01.000Z",
            }),
            json.dumps({
                "role": "assistant", "content": _ASYNC_ANSWER,
                "timestamp": "2026-02-21T08:00:02.000Z",
            }),
        ]
        jsonl_path.write_text("\n".join(lines), encoding="utf-8")

        result = pipeline.ingest_file(jsonl_path)
        assert result is None  # SKIP

    def test_merge_copies_raw_file(self, pipeline, chatgpt_jsonl, existing_bundle, mock_deps):
        """merge 후 _raw/에 두 파일 존재."""
        mock_deps["repo"].find_bundle_by_question_hash.return_value = existing_bundle

        pipeline.ingest_file(chatgpt_jsonl)

        raw_dir = mock_deps["kb_path"] / "bundles" / existing_bundle["bundle_id"] / "_raw"
        raw_files = sorted(raw_dir.iterdir())
        assert len(raw_files) == 2
        names = {f.name for f in raw_files}
        assert "claude.jsonl" in names
        assert "chatgpt.jsonl" in names

    def test_merge_creates_platform_md(self, pipeline, chatgpt_jsonl, existing_bundle, mock_deps):
        """merge 후 새 {platform}.md 생성."""
        mock_deps["repo"].find_bundle_by_question_hash.return_value = existing_bundle

        pipeline.ingest_file(chatgpt_jsonl)

        bundle_dir = mock_deps["kb_path"] / "bundles" / existing_bundle["bundle_id"]
        assert (bundle_dir / "chatgpt.md").exists()
        assert (bundle_dir / "claude.md").exists()

    def test_merge_regenerates_bundle_md(self, pipeline, chatgpt_jsonl, existing_bundle, mock_deps):
        """merge 후 _bundle.md에 새 platforms 포함."""
        mock_deps["repo"].find_bundle_by_question_hash.return_value = existing_bundle

        pipeline.ingest_file(chatgpt_jsonl)

        bundle_dir = mock_deps["kb_path"] / "bundles" / existing_bundle["bundle_id"]
        import yaml as yaml_mod
        content = (bundle_dir / "_bundle.md").read_text()
        # Extract YAML frontmatter between ---
        parts = content.split("---")
        meta = yaml_mod.safe_load(parts[1])
        assert "chatgpt" in meta["platforms"]

    def test_merge_updates_db(self, pipeline, chatgpt_jsonl, existing_bundle, mock_deps):
        """merge 시 add_response_to_bundle + update_bundle_meta 호출."""
        mock_deps["repo"].find_bundle_by_question_hash.return_value = existing_bundle

        pipeline.ingest_file(chatgpt_jsonl)

        mock_deps["repo"].add_response_to_bundle.assert_called_once()
        mock_deps["repo"].update_bundle_meta.assert_called_once()

    def test_merge_passes_source_path(self, pipeline, chatgpt_jsonl, existing_bundle, mock_deps):
        """merge 시 add_response_to_bundle에 source_path 전달."""
        mock_deps["repo"].find_bundle_by_question_hash.return_value = existing_bundle

        pipeline.ingest_file(chatgpt_jsonl)

        call_kwargs = mock_deps["repo"].add_response_to_bundle.call_args[1]
        assert "source_path" in call_kwargs
        assert call_kwargs["source_path"] == str(chatgpt_jsonl)

    def test_merge_adds_chromadb_chunks(self, pipeline, chatgpt_jsonl, existing_bundle, mock_deps):
        """merge 시 새 플랫폼 chunks가 ChromaDB에 upsert."""
        mock_deps["repo"].find_bundle_by_question_hash.return_value = existing_bundle

        pipeline.ingest_file(chatgpt_jsonl)

        mock_deps["chunk_store"].upsert_chunks.assert_called_once()

    def test_merge_returns_merged_flag(self, pipeline, chatgpt_jsonl, existing_bundle, mock_deps):
        """merge 반환값에 merged=True, bundle_id, platform."""
        mock_deps["repo"].find_bundle_by_question_hash.return_value = existing_bundle

        result = pipeline.ingest_file(chatgpt_jsonl)

        assert result is not None
        assert result["merged"] is True
        assert result["bundle_id"] == existing_bundle["bundle_id"]
        assert result["platform"] == "chatgpt"

    def test_new_file_no_existing_bundle(self, pipeline, mock_deps, tmp_path):
        """새 파일 (기존 번들 없음) → 정상 ingest (merged 아님)."""
        mock_deps["repo"].find_bundle_by_question_hash.return_value = None

        jsonl_path = tmp_path / "new.jsonl"
        lines = [
            json.dumps({
                "_meta": True, "platform": "claude",
                "url": "https://claude.ai/chat/new",
                "exported_at": "2026-02-21T06:00:00.000Z",
                "title": "새 대화",
            }),
            json.dumps({
                "role": "user", "content": "새로운 질문",
                "timestamp": "2026-02-21T06:00:01.000Z",
            }),
            json.dumps({
                "role": "assistant", "content": _NEW_ANSWER,
                "timestamp": "2026-02-21T06:00:02.000Z",
            }),
        ]
        jsonl_path.write_text("\n".join(lines), encoding="utf-8")

        result = pipeline.ingest_file(jsonl_path)

        assert result is not None
        assert "merged" not in result or result.get("merged") is not True
        mock_deps["repo"].upsert_bundle.assert_called_once()


class TestMergeFilePartialFailure:
    """Bug 4: merge_file() LLM 실패 시 부분 상태 방지."""

    @pytest.fixture
    def mock_deps(self, tmp_path):
        repo = MagicMock()
        repo.find_bundle_by_question_hash.return_value = None
        chunk_store = MagicMock()
        meta_gen = MagicMock()
        meta_gen.generate_response_meta.return_value = MagicMock(
            platform="chatgpt", model="gpt-4o", summary="요약",
            model_dump=MagicMock(return_value={"platform": "chatgpt"}),
        )
        meta_gen.generate_bundle_meta.return_value = MagicMock(
            summary="번들요약", slug="test-slug",
            domains=["dev"], topics=["python"], pending_topics=[],
            model_dump=MagicMock(return_value={}),
        )
        kb_path = tmp_path / "kb-test"
        kb_path.mkdir()
        return {
            "repo": repo, "chunk_store": chunk_store, "meta_gen": meta_gen,
            "kb_path": kb_path, "kb_name": "test",
        }

    @pytest.fixture
    def pipeline(self, mock_deps):
        return IngestPipeline(
            repo=mock_deps["repo"], chunk_store=mock_deps["chunk_store"],
            meta_gen=mock_deps["meta_gen"], kb_path=mock_deps["kb_path"],
            kb_name=mock_deps["kb_name"], domains=["dev"], topics=["python"],
        )

    @pytest.fixture
    def existing_bundle(self, mock_deps):
        bundle_id = "20260221-test-slug-a3f2"
        bundle_dir = mock_deps["kb_path"] / "bundles" / bundle_id
        raw_dir = bundle_dir / "_raw"
        raw_dir.mkdir(parents=True)
        claude_lines = [
            json.dumps({"_meta": True, "platform": "claude",
                         "url": "https://claude.ai/chat/123",
                         "exported_at": "2026-02-21T06:00:00.000Z", "title": "테스트"}),
            json.dumps({"role": "user", "content": "질문?",
                         "timestamp": "2026-02-21T06:00:01.000Z"}),
            json.dumps({"role": "assistant", "content": _GENERIC_ANSWER,
                         "timestamp": "2026-02-21T06:00:02.000Z"}),
        ]
        (raw_dir / "claude.jsonl").write_text("\n".join(claude_lines), encoding="utf-8")
        return {
            "bundle_id": bundle_id, "kb": "test",
            "path": f"bundles/{bundle_id}",
            "platforms": ["claude"], "domains": ["dev"], "topics": ["python"],
        }

    @pytest.fixture
    def chatgpt_jsonl(self, tmp_path):
        jsonl_path = tmp_path / "source" / "chatgpt.jsonl"
        jsonl_path.parent.mkdir(parents=True)
        lines = [
            json.dumps({"_meta": True, "platform": "chatgpt",
                         "url": "https://chatgpt.com/c/456",
                         "exported_at": "2026-02-21T07:00:00.000Z", "title": "테스트"}),
            json.dumps({"role": "user", "content": "질문?",
                         "timestamp": "2026-02-21T07:00:01.000Z"}),
            json.dumps({"role": "assistant", "content": _CHATGPT_ANSWER,
                         "timestamp": "2026-02-21T07:00:02.000Z"}),
        ]
        jsonl_path.write_text("\n".join(lines), encoding="utf-8")
        return jsonl_path

    def test_response_meta_failure_no_raw_copy(
        self, pipeline, chatgpt_jsonl, existing_bundle, mock_deps,
    ):
        """response_meta 실패 시 새 raw 파일이 복사되면 안 됨."""
        mock_deps["meta_gen"].generate_response_meta.side_effect = ValueError("LLM 에러")
        from pkb.parser.directory import parse_file
        conv = parse_file(chatgpt_jsonl)

        with pytest.raises(ValueError, match="LLM 에러"):
            pipeline.merge_file(chatgpt_jsonl, conv, existing_bundle)

        raw_dir = mock_deps["kb_path"] / "bundles" / existing_bundle["bundle_id"] / "_raw"
        raw_files = list(raw_dir.iterdir())
        assert len(raw_files) == 1  # 기존 claude.jsonl만 있어야 함
        assert raw_files[0].name == "claude.jsonl"

    def test_bundle_meta_failure_no_raw_copy(
        self, pipeline, chatgpt_jsonl, existing_bundle, mock_deps,
    ):
        """bundle_meta 실패 시 새 raw 파일이 복사되면 안 됨."""
        mock_deps["meta_gen"].generate_bundle_meta.side_effect = ValueError("LLM 에러")
        from pkb.parser.directory import parse_file
        conv = parse_file(chatgpt_jsonl)

        with pytest.raises(ValueError, match="LLM 에러"):
            pipeline.merge_file(chatgpt_jsonl, conv, existing_bundle)

        raw_dir = mock_deps["kb_path"] / "bundles" / existing_bundle["bundle_id"] / "_raw"
        raw_files = list(raw_dir.iterdir())
        assert len(raw_files) == 1
        assert raw_files[0].name == "claude.jsonl"

    def test_merge_includes_new_file_summary_in_bundle_meta(
        self, pipeline, chatgpt_jsonl, existing_bundle, mock_deps,
    ):
        """merge 시 새 파일 summary가 bundle_meta 호출에 포함."""
        from pkb.parser.directory import parse_file
        conv = parse_file(chatgpt_jsonl)
        pipeline.merge_file(chatgpt_jsonl, conv, existing_bundle)

        call_kwargs = mock_deps["meta_gen"].generate_bundle_meta.call_args[1]
        # response_summaries should contain content from both platforms
        assert "chatgpt" in call_kwargs["response_summaries"].lower() or \
               "ChatGPT" in call_kwargs["response_summaries"]


class TestConcurrentDedup:
    """Bug 1: 동시 ingest에서 같은 question_hash → TOCTOU race 방지."""

    @pytest.fixture
    def mock_deps(self, tmp_path):
        repo = MagicMock()
        # Start with no existing bundle; after first upsert, return a bundle for merges
        call_count = {"n": 0}
        def _find_by_hash(qh):
            if call_count["n"] == 0:
                return None
            return {
                "bundle_id": "20260221-test-slug-bd80",
                "kb": "test",
                "path": "bundles/20260221-test-slug-bd80",
                "platforms": ["claude"],
                "domains": ["dev"],
                "topics": ["python"],
            }
        repo.find_bundle_by_question_hash.side_effect = _find_by_hash

        def _track_upsert(**kw):
            call_count["n"] += 1
        repo.upsert_bundle.side_effect = _track_upsert

        chunk_store = MagicMock()
        meta_gen = MagicMock()
        meta_gen.generate_response_meta.return_value = MagicMock(
            platform="claude", model="model", summary="요약",
            model_dump=MagicMock(return_value={"platform": "claude"}),
        )
        meta_gen.generate_bundle_meta.return_value = MagicMock(
            summary="번들요약", slug="test-slug",
            domains=["dev"], topics=["python"], pending_topics=[],
            model_dump=MagicMock(return_value={}),
        )
        kb_path = tmp_path / "kb-test"
        kb_path.mkdir()
        return {
            "repo": repo, "chunk_store": chunk_store, "meta_gen": meta_gen,
            "kb_path": kb_path, "kb_name": "test",
        }

    def _make_jsonl(self, tmp_path, name, *, user_content="같은 질문"):
        path = tmp_path / name
        lines = [
            json.dumps({"_meta": True, "platform": "claude",
                         "url": "https://claude.ai/chat/1",
                         "exported_at": "2026-02-21T06:00:00.000Z", "title": "테스트"}),
            json.dumps({"role": "user", "content": user_content,
                         "timestamp": "2026-02-21T06:00:01.000Z"}),
            json.dumps({"role": "assistant", "content": _GENERIC_ANSWER,
                         "timestamp": "2026-02-21T06:00:02.000Z"}),
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def test_concurrent_same_hash_only_one_created(self, mock_deps, tmp_path):
        """같은 question_hash 4개 동시 ingest → upsert_bundle 1번만 호출."""
        pipeline = IngestPipeline(
            repo=mock_deps["repo"], chunk_store=mock_deps["chunk_store"],
            meta_gen=mock_deps["meta_gen"], kb_path=mock_deps["kb_path"],
            kb_name=mock_deps["kb_name"], domains=["dev"], topics=["python"],
        )
        files = [self._make_jsonl(tmp_path, f"file{i}.jsonl") for i in range(4)]
        results = []
        errors = []

        def _ingest(f):
            try:
                r = pipeline.ingest_file(f)
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_ingest, args=(f,)) for f in files]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Unexpected errors: {errors}"
        # Only 1 upsert_bundle (new creation); rest should be SKIP/merge
        assert mock_deps["repo"].upsert_bundle.call_count == 1

    def test_concurrent_different_hash_independent(self, mock_deps, tmp_path):
        """다른 question_hash → 각각 독립 생성."""
        # Each file has different question → different hash → no lock contention
        mock_deps["repo"].find_bundle_by_question_hash.return_value = None

        pipeline = IngestPipeline(
            repo=mock_deps["repo"], chunk_store=mock_deps["chunk_store"],
            meta_gen=mock_deps["meta_gen"], kb_path=mock_deps["kb_path"],
            kb_name=mock_deps["kb_name"], domains=["dev"], topics=["python"],
        )
        files = [
            self._make_jsonl(tmp_path, f"file{i}.jsonl", user_content=f"질문 {i}")
            for i in range(3)
        ]
        results = []

        def _ingest(f):
            results.append(pipeline.ingest_file(f))

        threads = [threading.Thread(target=_ingest, args=(f,)) for f in files]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 3 should create new bundles
        assert mock_deps["repo"].upsert_bundle.call_count == 3

    def test_force_skips_lock(self, mock_deps, tmp_path):
        """force=True는 lock 사용 안 함 (dedup check 자체를 건너뜀)."""
        mock_deps["repo"].find_bundle_by_question_hash.return_value = None

        pipeline = IngestPipeline(
            repo=mock_deps["repo"], chunk_store=mock_deps["chunk_store"],
            meta_gen=mock_deps["meta_gen"], kb_path=mock_deps["kb_path"],
            kb_name=mock_deps["kb_name"], domains=["dev"], topics=["python"],
        )
        f = self._make_jsonl(tmp_path, "test.jsonl")
        result = pipeline.ingest_file(f, force=True)
        assert result is not None
        # No hash lock should have been created for force mode
        assert len(pipeline._hash_locks) == 0


class TestIngestContentValidation:
    def test_skip_when_no_turns(self, tmp_path):
        """turns가 비어있으면 skip."""
        repo = MagicMock()
        chunk_store = MagicMock()
        meta_gen = MagicMock()

        pipeline = IngestPipeline(
            repo=repo, chunk_store=chunk_store, meta_gen=meta_gen,
            kb_path=tmp_path, kb_name="test", domains=[], topics=[],
        )

        # MD with only a header, no meaningful content turns
        md_file = tmp_path / "chatgpt.md"
        md_file.write_text("# [ChatGPT](https://chatgpt.com/chat/abc)")

        result = pipeline.ingest_file(md_file)
        assert result is not None
        assert result["status"] == "skip_insufficient_content"

    def test_skip_when_content_too_short(self, tmp_path):
        """assistant 콘텐츠가 50자 미만이면 skip."""
        repo = MagicMock()
        chunk_store = MagicMock()
        meta_gen = MagicMock()

        pipeline = IngestPipeline(
            repo=repo, chunk_store=chunk_store, meta_gen=meta_gen,
            kb_path=tmp_path, kb_name="test", domains=[], topics=[],
        )

        md_file = tmp_path / "chatgpt.md"
        md_file.write_text(
            "# [ChatGPT](https://chatgpt.com/c/abc)\n\n## LLM 응답 1\n\nShort."
        )

        result = pipeline.ingest_file(md_file)
        assert result is not None
        assert result["status"] == "skip_insufficient_content"

    def test_accepts_sufficient_content(self, tmp_path):
        """50자 이상이면 정상 처리 진행 (meta_gen 호출)."""
        repo = MagicMock()
        repo.find_bundle_by_question_hash.return_value = None
        chunk_store = MagicMock()
        meta_gen = MagicMock()
        meta_gen.generate_bundle_meta.return_value = MagicMock(
            slug="test-slug", summary="테스트", domains=["dev"], topics=["python"],
            pending_topics=[], consensus=None, divergence=None,
            model_dump=lambda: {
                "slug": "test-slug", "summary": "테스트", "domains": ["dev"],
                "topics": ["python"], "pending_topics": [],
                "consensus": None, "divergence": None,
            },
        )
        meta_gen.generate_response_meta.return_value = MagicMock(
            model="gpt-4",
            model_dump=lambda: {
                "summary": "t", "key_claims": [], "stance": "neutral", "model": "gpt-4",
            },
        )

        pipeline = IngestPipeline(
            repo=repo, chunk_store=chunk_store, meta_gen=meta_gen,
            kb_path=tmp_path, kb_name="test", domains=["dev"], topics=["python"],
        )

        content = "A" * 60  # 60 chars, above threshold
        md_file = tmp_path / "chatgpt.md"
        md_file.write_text(
            f"# [ChatGPT](https://chatgpt.com/c/abc)\n\n## LLM 응답 1\n\n{content}"
        )

        result = pipeline.ingest_file(md_file)
        assert result is not None
        assert result.get("status") != "skip_insufficient_content"
        meta_gen.generate_bundle_meta.assert_called_once()


class TestIngestResponseSummaryValidation:
    def test_skip_when_response_summaries_too_short(self, tmp_path):
        """response_summaries가 50자 미만이면 skip (LLM 호출 방지)."""
        from unittest.mock import patch

        repo = MagicMock()
        repo.find_bundle_by_question_hash.return_value = None
        chunk_store = MagicMock()
        meta_gen = MagicMock()

        pipeline = IngestPipeline(
            repo=repo, chunk_store=chunk_store, meta_gen=meta_gen,
            kb_path=tmp_path, kb_name="test", domains=["dev"], topics=[],
        )

        # Create a valid MD file with enough content to pass content validation
        md_file = tmp_path / "chatgpt.md"
        md_file.write_text(
            "# [ChatGPT](https://chatgpt.com/c/abc)\n\n## LLM 응답 1\n\n" + "x" * 60
        )

        # Mock _build_response_summaries to return a short string
        # (simulates scenario where summary building strips content)
        with patch.object(pipeline, "_build_response_summaries", return_value="short"):
            result = pipeline.ingest_file(md_file)

        assert result is not None
        assert result["status"] == "skip_insufficient_content"
        # meta_gen should NOT have been called
        meta_gen.generate_bundle_meta.assert_not_called()


class TestIngestGracefulSkip:
    def test_parse_error_returns_skip_dict(self, tmp_path):
        """ParseError 발생 시 에러 대신 skip dict 반환."""
        repo = MagicMock()
        chunk_store = MagicMock()
        meta_gen = MagicMock()

        pipeline = IngestPipeline(
            repo=repo, chunk_store=chunk_store, meta_gen=meta_gen,
            kb_path=tmp_path, kb_name="test", domains=[], topics=[],
        )

        # Create empty .md file
        md_file = tmp_path / "empty.md"
        md_file.write_text("")

        result = pipeline.ingest_file(md_file)
        assert result is not None
        assert result["status"] == "skip_parse_error"

    def test_parse_error_does_not_raise(self, tmp_path):
        """ParseError가 exception으로 전파되지 않음."""
        repo = MagicMock()
        chunk_store = MagicMock()
        meta_gen = MagicMock()

        pipeline = IngestPipeline(
            repo=repo, chunk_store=chunk_store, meta_gen=meta_gen,
            kb_path=tmp_path, kb_name="test", domains=[], topics=[],
        )

        md_file = tmp_path / "bad.md"
        md_file.write_text("   \n  \n  ")

        # Should not raise
        result = pipeline.ingest_file(md_file)
        assert result["status"] == "skip_parse_error"
