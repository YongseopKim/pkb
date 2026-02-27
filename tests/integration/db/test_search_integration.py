"""SearchEngine integration tests using real PostgreSQL + ChromaDB.

Tests keyword, semantic, and hybrid search modes, plus KB filtering,
against actual database instances.

Requires:
    - docker compose -f docker/docker-compose.test.yml up -d
    - PKB_DB_INTEGRATION=1 environment variable
"""

from datetime import datetime

from pkb.search.engine import SearchEngine
from pkb.search.models import SearchMode, SearchQuery


def _insert_bundle_with_chunks(repo, chunk_store, *, bundle_id, kb, question,
                               summary, domains, topics, chunks_text):
    """Helper: insert a bundle into PostgreSQL and its chunks into ChromaDB.

    Args:
        repo: BundleRepository fixture.
        chunk_store: ChunkStore fixture.
        bundle_id: Unique bundle ID.
        kb: Knowledge base name.
        question: Bundle question text.
        summary: Bundle summary text.
        domains: List of domain strings.
        topics: List of topic strings.
        chunks_text: List of chunk document strings to embed.
    """
    repo.upsert_bundle(
        bundle_id=bundle_id,
        kb=kb,
        question=question,
        summary=summary,
        created_at=datetime(2026, 2, 22, 12, 0),
        response_count=1,
        path=f"/bundles/{bundle_id}",
        question_hash=f"hash_{bundle_id}",
        stable_id=f"stable_{bundle_id}",
        domains=domains,
        topics=topics,
        responses=[{"platform": "claude", "model": "claude-3", "turn_count": 3}],
    )

    chunks = [
        {
            "id": f"{bundle_id}-chunk-{i}",
            "document": text,
            "metadata": {"bundle_id": bundle_id, "kb": kb, "chunk_index": i},
        }
        for i, text in enumerate(chunks_text)
    ]
    chunk_store.upsert_chunks(chunks)


class TestSearchEngine:
    """Integration tests for SearchEngine across keyword, semantic, and hybrid modes."""

    def test_keyword_search(self, repo, chunk_store):
        """Keyword (FTS) search finds bundle by matching text in question/summary."""
        _insert_bundle_with_chunks(
            repo, chunk_store,
            bundle_id="20260222-python-async-a1b2",
            kb="test-kb",
            question="Python에서 async/await 비동기 프로그래밍",
            summary="Python 비동기 프로그래밍 패턴과 asyncio 라이브러리 사용법",
            domains=["dev"],
            topics=["python"],
            chunks_text=[
                "Python의 async/await 구문을 사용하면 비동기 코드를 작성할 수 있습니다.",
                "asyncio 이벤트 루프는 코루틴을 스케줄링하고 실행합니다.",
            ],
        )

        engine = SearchEngine(repo=repo, chunk_store=chunk_store)
        results = engine.search(SearchQuery(
            query="Python",
            mode=SearchMode.KEYWORD,
        ))

        assert len(results) >= 1
        bundle_ids = [r.bundle_id for r in results]
        assert "20260222-python-async-a1b2" in bundle_ids

        # Verify result fields
        matched = [r for r in results if r.bundle_id == "20260222-python-async-a1b2"][0]
        assert matched.source == "fts"
        assert 0.0 <= matched.score <= 1.0
        assert matched.question == "Python에서 async/await 비동기 프로그래밍"
        assert "dev" in matched.domains
        assert "python" in matched.topics

    def test_semantic_search(self, repo, chunk_store):
        """Semantic search finds bundle using related but different query terms."""
        _insert_bundle_with_chunks(
            repo, chunk_store,
            bundle_id="20260222-docker-deploy-c3d4",
            kb="test-kb",
            question="Docker 컨테이너 배포 방법",
            summary="Docker를 사용한 애플리케이션 컨테이너화와 배포 전략",
            domains=["dev"],
            topics=["docker"],
            chunks_text=[
                "Docker는 애플리케이션을 컨테이너로 패키징하여 배포를 단순화합니다.",
                "Dockerfile을 작성하여 이미지를 빌드하고 docker-compose로 서비스를 구성합니다.",
                "컨테이너 오케스트레이션 도구인 Kubernetes와 함께 사용할 수 있습니다.",
            ],
        )

        engine = SearchEngine(repo=repo, chunk_store=chunk_store)
        # Use a semantically related but textually different query
        results = engine.search(SearchQuery(
            query="컨테이너 가상화 배포 자동화",
            mode=SearchMode.SEMANTIC,
        ))

        assert len(results) >= 1
        bundle_ids = [r.bundle_id for r in results]
        assert "20260222-docker-deploy-c3d4" in bundle_ids

        matched = [r for r in results if r.bundle_id == "20260222-docker-deploy-c3d4"][0]
        assert matched.source == "semantic"
        assert 0.0 <= matched.score <= 1.0

    def test_hybrid_search(self, repo, chunk_store):
        """Hybrid search combines FTS and semantic results with weighted scoring."""
        _insert_bundle_with_chunks(
            repo, chunk_store,
            bundle_id="20260222-react-hooks-e5f6",
            kb="test-kb",
            question="React hooks 사용법 정리",
            summary="React의 useState, useEffect 등 훅 패턴 가이드",
            domains=["dev"],
            topics=["react"],
            chunks_text=[
                "React hooks는 함수 컴포넌트에서 상태 관리를 가능하게 합니다.",
                "useState로 로컬 상태를 관리하고 useEffect로 사이드 이펙트를 처리합니다.",
            ],
        )

        engine = SearchEngine(repo=repo, chunk_store=chunk_store)
        results = engine.search(SearchQuery(
            query="React hooks",
            mode=SearchMode.HYBRID,
        ))

        assert len(results) >= 1
        bundle_ids = [r.bundle_id for r in results]
        assert "20260222-react-hooks-e5f6" in bundle_ids

        matched = [r for r in results if r.bundle_id == "20260222-react-hooks-e5f6"][0]
        # Hybrid results that appear in both FTS and semantic get source="both"
        # Results in only one source keep "fts" or "semantic"
        assert matched.source in ("fts", "semantic", "both")
        assert 0.0 <= matched.score <= 1.0

    def test_search_with_kb_filter(self, repo, chunk_store):
        """Search with kb filter returns only bundles from that knowledge base."""
        # Insert into kb-alpha
        _insert_bundle_with_chunks(
            repo, chunk_store,
            bundle_id="20260222-db-design-g7h8",
            kb="kb-alpha",
            question="데이터베이스 설계 원칙",
            summary="관계형 데이터베이스 정규화와 인덱스 설계",
            domains=["dev"],
            topics=["database"],
            chunks_text=[
                "데이터베이스 정규화는 데이터 중복을 줄이고 무결성을 보장합니다.",
                "적절한 인덱스 설계는 쿼리 성능에 큰 영향을 미칩니다.",
            ],
        )

        # Insert into kb-beta with similar content
        _insert_bundle_with_chunks(
            repo, chunk_store,
            bundle_id="20260222-db-perf-i9j0",
            kb="kb-beta",
            question="데이터베이스 성능 최적화",
            summary="데이터베이스 쿼리 튜닝과 캐싱 전략",
            domains=["dev"],
            topics=["database"],
            chunks_text=[
                "데이터베이스 쿼리 최적화를 위해 실행 계획을 분석합니다.",
                "캐시 레이어를 도입하여 반복 쿼리의 성능을 개선합니다.",
            ],
        )

        engine = SearchEngine(repo=repo, chunk_store=chunk_store)

        # Search with kb filter: only kb-alpha
        results_alpha = engine.search(SearchQuery(
            query="데이터베이스",
            mode=SearchMode.HYBRID,
            kb="kb-alpha",
        ))
        alpha_ids = [r.bundle_id for r in results_alpha]
        assert "20260222-db-design-g7h8" in alpha_ids
        assert "20260222-db-perf-i9j0" not in alpha_ids

        # Search with kb filter: only kb-beta
        results_beta = engine.search(SearchQuery(
            query="데이터베이스",
            mode=SearchMode.HYBRID,
            kb="kb-beta",
        ))
        beta_ids = [r.bundle_id for r in results_beta]
        assert "20260222-db-perf-i9j0" in beta_ids
        assert "20260222-db-design-g7h8" not in beta_ids
