"""Tests for meta data models (ResponseMeta, BundleMeta, BundleFrontmatter)."""

from pkb.models.meta import BundleFrontmatter, BundleMeta, ResponseMeta


class TestResponseMeta:
    def test_basic_construction(self):
        meta = ResponseMeta(
            platform="claude",
            model="claude-3.5-sonnet",
            summary="Python 비동기 처리 방법 설명",
            key_claims=["asyncio는 이벤트 루프 기반", "await로 코루틴 실행"],
            stance="informative",
        )
        assert meta.platform == "claude"
        assert meta.model == "claude-3.5-sonnet"
        assert len(meta.key_claims) == 2
        assert meta.stance == "informative"

    def test_optional_model(self):
        meta = ResponseMeta(
            platform="chatgpt",
            summary="답변 요약",
            key_claims=[],
            stance="neutral",
        )
        assert meta.model is None

    def test_defaults(self):
        meta = ResponseMeta(
            platform="gemini",
            summary="요약",
        )
        assert meta.key_claims == []
        assert meta.stance is None
        assert meta.model is None


class TestBundleMeta:
    def test_basic_construction(self):
        meta = BundleMeta(
            summary="PKB 시스템 설계에 대한 다중 LLM 토론",
            slug="pkb-system-design",
            domains=["dev"],
            topics=["system-design", "python"],
        )
        assert meta.slug == "pkb-system-design"
        assert "dev" in meta.domains
        assert len(meta.topics) == 2

    def test_optional_fields(self):
        meta = BundleMeta(
            summary="간단한 질문",
            slug="simple-question",
            domains=["general"],
            topics=[],
        )
        assert meta.consensus is None
        assert meta.divergence is None
        assert meta.pending_topics == []

    def test_with_consensus_and_divergence(self):
        meta = BundleMeta(
            summary="요약",
            slug="test-slug",
            domains=["dev"],
            topics=["python"],
            consensus="모든 LLM이 동의한 점",
            divergence="LLM간 의견 차이",
        )
        assert meta.consensus is not None
        assert meta.divergence is not None

    def test_pending_topics(self):
        meta = BundleMeta(
            summary="요약",
            slug="test",
            domains=["dev"],
            topics=["python"],
            pending_topics=["new-topic-1", "new-topic-2"],
        )
        assert len(meta.pending_topics) == 2


class TestBundleFrontmatter:
    def test_basic_construction(self):
        fm = BundleFrontmatter(
            id="20260101-test-abc1",
            question="테스트 질문",
            summary="테스트 요약",
            slug="test",
            domains=["dev"],
            topics=["python"],
            platforms=["claude"],
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert fm.id == "20260101-test-abc1"
        assert fm.question == "테스트 질문"
        assert fm.domains == ["dev"]
        assert fm.platforms == ["claude"]

    def test_defaults(self):
        fm = BundleFrontmatter(
            id="20260101-test-abc1",
            question="질문",
            summary="요약",
            slug="test",
            domains=["dev"],
            topics=[],
            platforms=["claude"],
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert fm.pending_topics == []
        assert fm.consensus is None
        assert fm.divergence is None

    def test_all_fields(self):
        fm = BundleFrontmatter(
            id="20260101-full-abc1",
            question="전체 필드",
            summary="전체 요약",
            slug="full-test",
            domains=["dev", "ai"],
            topics=["python", "ml"],
            pending_topics=["new-topic"],
            platforms=["claude", "chatgpt"],
            created_at="2026-01-01T12:00:00+00:00",
            consensus="합의 내용",
            divergence="차이점",
        )
        assert len(fm.domains) == 2
        assert len(fm.platforms) == 2
        assert fm.consensus == "합의 내용"

    def test_created_at_string_preserved(self):
        fm = BundleFrontmatter(
            id="20260101-test-abc1",
            question="질문",
            summary="요약",
            slug="test",
            domains=["dev"],
            topics=[],
            platforms=["claude"],
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert fm.created_at == "2026-01-01T00:00:00+00:00"

    def test_from_frontmatter_dict(self):
        """Test creating from a dict as returned by parse_frontmatter."""
        data = {
            "id": "20260101-dict-abc1",
            "question": "딕셔너리에서 생성",
            "summary": "요약",
            "slug": "dict-test",
            "domains": ["dev"],
            "topics": ["python"],
            "pending_topics": [],
            "platforms": ["claude"],
            "created_at": "2026-01-01T00:00:00+00:00",
            "consensus": None,
            "divergence": None,
        }
        fm = BundleFrontmatter(**data)
        assert fm.id == "20260101-dict-abc1"

    def test_question_optional(self):
        """question 필드 없이 BundleFrontmatter 생성 가능."""
        fm = BundleFrontmatter(
            id="20260101-no-q-abc1",
            summary="요약만 있는 번들",
            slug="no-question",
            domains=["dev"],
            topics=["python"],
            platforms=["claude"],
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert fm.question is None
        assert fm.summary == "요약만 있는 번들"

    def test_question_still_works_when_provided(self):
        """기존 _bundle.md (question 포함) 파싱도 여전히 작동."""
        fm = BundleFrontmatter(
            id="20260101-with-q-abc1",
            question="기존 질문",
            summary="요약",
            slug="with-question",
            domains=["dev"],
            topics=[],
            platforms=["claude"],
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert fm.question == "기존 질문"
