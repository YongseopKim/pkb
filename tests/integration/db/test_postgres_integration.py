"""PostgreSQL integration tests for BundleRepository.

Tests CRUD operations, FTS search (including Korean text),
source_path lookup, and metadata updates against a real PostgreSQL instance.

Requires:
    - docker compose -f docker/docker-compose.test.yml up -d
    - PKB_DB_INTEGRATION=1 environment variable
"""

from datetime import datetime


class TestBundleUpsertAndGet:
    """Test upsert_bundle + get_bundle_by_id round-trip."""

    def test_upsert_and_get_basic(self, repo):
        """Basic upsert followed by get returns correct data."""
        repo.upsert_bundle(
            bundle_id="20260222-test-bundle-a1b2",
            kb="personal",
            question="Python에서 async/await 사용법",
            summary="Python 비동기 프로그래밍 가이드",
            created_at=datetime(2026, 2, 22, 12, 0),
            response_count=1,
            path="/bundles/20260222-test-bundle-a1b2",
            question_hash="hash1234",
            stable_id="stable_hash1234",
            domains=["dev"],
            topics=["python", "async"],
            responses=[{"platform": "claude", "model": "claude-3", "turn_count": 5}],
        )

        result = repo.get_bundle_by_id("20260222-test-bundle-a1b2")
        assert result is not None
        assert result["bundle_id"] == "20260222-test-bundle-a1b2"
        assert result["kb"] == "personal"
        assert result["question"] == "Python에서 async/await 사용법"
        assert result["summary"] == "Python 비동기 프로그래밍 가이드"
        assert "dev" in result["domains"]
        # Topics are comma-separated string
        assert "python" in result["topics"]
        assert "async" in result["topics"]

    def test_get_nonexistent_returns_none(self, repo):
        """Getting a non-existent bundle returns None."""
        result = repo.get_bundle_by_id("nonexistent-bundle-id")
        assert result is None

    def test_upsert_updates_existing(self, repo):
        """Upserting with same bundle_id updates the existing record."""
        repo.upsert_bundle(
            bundle_id="20260222-update-test-c3d4",
            kb="personal",
            question="원래 질문",
            summary="원래 요약",
            created_at=datetime(2026, 2, 22, 12, 0),
            response_count=1,
            path="/bundles/20260222-update-test-c3d4",
            question_hash="hash_orig",
            stable_id="stable_hash_orig",
            domains=["dev"],
            topics=["topic1"],
            responses=[{"platform": "chatgpt"}],
        )

        # Upsert again with updated summary and domains
        repo.upsert_bundle(
            bundle_id="20260222-update-test-c3d4",
            kb="personal",
            question="원래 질문",
            summary="업데이트된 요약",
            created_at=datetime(2026, 2, 22, 12, 0),
            response_count=2,
            path="/bundles/20260222-update-test-c3d4",
            question_hash="hash_updated",
            stable_id="stable_hash_orig",
            domains=["dev", "ai"],
            topics=["topic1", "topic2"],
            responses=[
                {"platform": "chatgpt"},
                {"platform": "claude"},
            ],
        )

        result = repo.get_bundle_by_id("20260222-update-test-c3d4")
        assert result is not None
        assert result["summary"] == "업데이트된 요약"
        assert "dev" in result["domains"]
        assert "ai" in result["domains"]
        assert "topic2" in result["topics"]


class TestFTS:
    """Test full-text search with Korean text and filters."""

    def _insert_sample_bundles(self, repo):
        """Insert sample bundles for FTS testing."""
        repo.upsert_bundle(
            bundle_id="20260222-python-guide-e5f6",
            kb="personal",
            question="Python 비동기 프로그래밍 가이드",
            summary="async/await 패턴 설명",
            created_at=datetime(2026, 2, 22, 12, 0),
            response_count=1,
            path="/bundles/20260222-python-guide-e5f6",
            question_hash="fts_hash1",
            stable_id="stable_fts_hash1",
            domains=["dev"],
            topics=["python"],
            responses=[{"platform": "claude"}],
        )
        repo.upsert_bundle(
            bundle_id="20260222-react-hooks-g7h8",
            kb="personal",
            question="React hooks 사용법 정리",
            summary="React의 useState와 useEffect 설명",
            created_at=datetime(2026, 2, 22, 13, 0),
            response_count=1,
            path="/bundles/20260222-react-hooks-g7h8",
            question_hash="fts_hash2",
            stable_id="stable_fts_hash2",
            domains=["dev"],
            topics=["react"],
            responses=[{"platform": "chatgpt"}],
        )
        repo.upsert_bundle(
            bundle_id="20260222-cooking-tips-i9j0",
            kb="work",
            question="요리 레시피 추천",
            summary="한식 요리 팁 모음",
            created_at=datetime(2026, 2, 22, 14, 0),
            response_count=1,
            path="/bundles/20260222-cooking-tips-i9j0",
            question_hash="fts_hash3",
            stable_id="stable_fts_hash3",
            domains=["life"],
            topics=["cooking"],
            responses=[{"platform": "gemini"}],
        )

    def test_fts_korean_text(self, repo):
        """FTS finds bundles matching Korean text query."""
        self._insert_sample_bundles(repo)

        results = repo.search_fts(query="Python")
        assert len(results) >= 1
        bundle_ids = [r["bundle_id"] for r in results]
        assert "20260222-python-guide-e5f6" in bundle_ids

    def test_fts_domain_filter(self, repo):
        """FTS with domain filter narrows results."""
        self._insert_sample_bundles(repo)

        # Search for "요리" with domain filter "life"
        results = repo.search_fts(query="요리", domains=["life"])
        assert len(results) == 1
        assert results[0]["bundle_id"] == "20260222-cooking-tips-i9j0"

        # Same query with domain filter "dev" should return nothing
        results_dev = repo.search_fts(query="요리", domains=["dev"])
        assert len(results_dev) == 0

    def test_fts_kb_filter(self, repo):
        """FTS with kb filter restricts to specific knowledge base."""
        self._insert_sample_bundles(repo)

        # "요리" exists only in kb="work"
        results = repo.search_fts(query="요리", kb="work")
        assert len(results) == 1
        assert results[0]["kb"] == "work"

        # Same query in kb="personal" should return nothing
        results_personal = repo.search_fts(query="요리", kb="personal")
        assert len(results_personal) == 0


class TestDeleteAndList:
    """Test delete_bundle cascading and list_all_bundle_ids."""

    def _insert_two_bundles(self, repo):
        """Insert two bundles in different KBs."""
        repo.upsert_bundle(
            bundle_id="20260222-del-test-k1l2",
            kb="personal",
            question="삭제 테스트 번들 1",
            summary="삭제될 번들",
            created_at=datetime(2026, 2, 22, 12, 0),
            response_count=1,
            path="/bundles/20260222-del-test-k1l2",
            question_hash="del_hash1",
            stable_id="stable_del_hash1",
            domains=["dev"],
            topics=["testing"],
            responses=[{"platform": "claude", "model": "claude-3", "turn_count": 3}],
        )
        repo.upsert_bundle(
            bundle_id="20260222-keep-test-m3n4",
            kb="work",
            question="유지 테스트 번들 2",
            summary="유지될 번들",
            created_at=datetime(2026, 2, 22, 13, 0),
            response_count=1,
            path="/bundles/20260222-keep-test-m3n4",
            question_hash="del_hash2",
            stable_id="stable_del_hash2",
            domains=["life"],
            topics=["general"],
            responses=[{"platform": "chatgpt"}],
        )

    def test_delete_bundle_cascades(self, repo):
        """Deleting a bundle removes it and its related data (domains, topics, responses)."""
        self._insert_two_bundles(repo)

        repo.delete_bundle("20260222-del-test-k1l2")

        assert repo.get_bundle_by_id("20260222-del-test-k1l2") is None
        # The other bundle should still exist
        assert repo.get_bundle_by_id("20260222-keep-test-m3n4") is not None

    def test_list_all_bundle_ids(self, repo):
        """list_all_bundle_ids returns all inserted bundle IDs."""
        self._insert_two_bundles(repo)

        ids = repo.list_all_bundle_ids()
        assert "20260222-del-test-k1l2" in ids
        assert "20260222-keep-test-m3n4" in ids
        assert len(ids) == 2

    def test_list_bundle_ids_with_kb_filter(self, repo):
        """list_all_bundle_ids with kb filter returns only matching KBs."""
        self._insert_two_bundles(repo)

        personal_ids = repo.list_all_bundle_ids(kb="personal")
        assert personal_ids == ["20260222-del-test-k1l2"]

        work_ids = repo.list_all_bundle_ids(kb="work")
        assert work_ids == ["20260222-keep-test-m3n4"]


class TestSourcePath:
    """Test find_by_source_path with bundles and responses tables."""

    def test_find_by_source_path_in_bundles(self, repo):
        """find_by_source_path finds bundle via bundles.source_path."""
        repo.upsert_bundle(
            bundle_id="20260222-src-bundle-o5p6",
            kb="personal",
            question="소스 경로 테스트",
            summary="번들 테이블 소스 경로",
            created_at=datetime(2026, 2, 22, 12, 0),
            response_count=1,
            path="/bundles/20260222-src-bundle-o5p6",
            question_hash="src_hash1",
            stable_id="stable_src_hash1",
            domains=["dev"],
            topics=["testing"],
            responses=[{"platform": "claude"}],
            source_path="/inbox/test-file.jsonl",
        )

        result = repo.find_by_source_path("/inbox/test-file.jsonl")
        assert result == "20260222-src-bundle-o5p6"

    def test_find_by_source_path_in_responses(self, repo):
        """find_by_source_path finds bundle via bundle_responses.source_path (priority)."""
        repo.upsert_bundle(
            bundle_id="20260222-src-resp-q7r8",
            kb="personal",
            question="응답 소스 경로 테스트",
            summary="응답 테이블 소스 경로",
            created_at=datetime(2026, 2, 22, 12, 0),
            response_count=2,
            path="/bundles/20260222-src-resp-q7r8",
            question_hash="src_hash2",
            stable_id="stable_src_hash2",
            domains=["dev"],
            topics=["testing"],
            responses=[
                {"platform": "claude", "source_path": "/inbox/claude-export.jsonl"},
                {"platform": "chatgpt", "source_path": "/inbox/chatgpt-export.md"},
            ],
            source_path="/inbox/original-file.jsonl",
        )

        # Should find via bundle_responses.source_path (checked first)
        result = repo.find_by_source_path("/inbox/chatgpt-export.md")
        assert result == "20260222-src-resp-q7r8"

        # Should also find via bundles.source_path (fallback)
        result_bundle = repo.find_by_source_path("/inbox/original-file.jsonl")
        assert result_bundle == "20260222-src-resp-q7r8"

    def test_find_by_source_path_not_found(self, repo):
        """find_by_source_path returns None when path does not exist."""
        result = repo.find_by_source_path("/nonexistent/path.jsonl")
        assert result is None


class TestCountAndDeleteByKb:
    """Test count_by_kb and delete_by_kb with real DB."""

    def _insert_bundles_in_two_kbs(self, repo):
        """Insert 2 bundles in personal, 1 in work."""
        repo.upsert_bundle(
            bundle_id="20260222-kb-del-p1a1",
            kb="personal",
            question="Personal 번들 1",
            summary="개인 번들 첫번째",
            created_at=datetime(2026, 2, 22, 12, 0),
            response_count=1,
            path="/bundles/20260222-kb-del-p1a1",
            question_hash="kbdel_hash1",
            stable_id="stable_kbdel_hash1",
            domains=["dev"],
            topics=["python"],
            responses=[{"platform": "claude"}],
        )
        repo.upsert_bundle(
            bundle_id="20260222-kb-del-p2b2",
            kb="personal",
            question="Personal 번들 2",
            summary="개인 번들 두번째",
            created_at=datetime(2026, 2, 22, 13, 0),
            response_count=1,
            path="/bundles/20260222-kb-del-p2b2",
            question_hash="kbdel_hash2",
            stable_id="stable_kbdel_hash2",
            domains=["ai"],
            topics=["ml"],
            responses=[{"platform": "chatgpt"}],
        )
        repo.upsert_bundle(
            bundle_id="20260222-kb-del-w1c3",
            kb="work",
            question="Work 번들 1",
            summary="업무 번들",
            created_at=datetime(2026, 2, 22, 14, 0),
            response_count=1,
            path="/bundles/20260222-kb-del-w1c3",
            question_hash="kbdel_hash3",
            stable_id="stable_kbdel_hash3",
            domains=["life"],
            topics=["general"],
            responses=[{"platform": "gemini"}],
        )

    def test_count_by_kb(self, repo):
        """count_by_kb returns correct count per KB."""
        self._insert_bundles_in_two_kbs(repo)
        assert repo.count_by_kb("personal") == 2
        assert repo.count_by_kb("work") == 1
        assert repo.count_by_kb("nonexistent") == 0

    def test_delete_by_kb_preserves_other_kb(self, repo):
        """delete_by_kb removes only target KB, other KB data intact."""
        self._insert_bundles_in_two_kbs(repo)

        deleted = repo.delete_by_kb("personal")
        assert deleted == 2

        # personal KB is empty
        assert repo.count_by_kb("personal") == 0
        assert repo.list_all_bundle_ids(kb="personal") == []

        # work KB is intact
        assert repo.count_by_kb("work") == 1
        result = repo.get_bundle_by_id("20260222-kb-del-w1c3")
        assert result is not None
        assert result["kb"] == "work"


class TestUpdateMeta:
    """Test update_bundle_meta for summary and tag changes."""

    def test_update_meta_changes_summary_and_tags(self, repo):
        """update_bundle_meta updates summary, domains, and topics."""
        repo.upsert_bundle(
            bundle_id="20260222-meta-upd-s9t0",
            kb="personal",
            question="메타 업데이트 테스트",
            summary="원래 요약",
            created_at=datetime(2026, 2, 22, 12, 0),
            response_count=1,
            path="/bundles/20260222-meta-upd-s9t0",
            question_hash="meta_hash1",
            stable_id="stable_meta_hash1",
            domains=["dev"],
            topics=["python"],
            responses=[{"platform": "claude"}],
        )

        repo.update_bundle_meta(
            bundle_id="20260222-meta-upd-s9t0",
            summary="업데이트된 요약 내용",
            domains=["dev", "ai"],
            topics=["python", "machine-learning"],
            pending_topics=["new-pending-topic"],
        )

        result = repo.get_bundle_by_id("20260222-meta-upd-s9t0")
        assert result is not None
        assert result["summary"] == "업데이트된 요약 내용"
        # Domains updated
        assert "dev" in result["domains"]
        assert "ai" in result["domains"]
        # Approved topics (non-pending) visible in get result
        assert "python" in result["topics"]
        assert "machine-learning" in result["topics"]
        # Pending topics are NOT included in get_bundle_by_id (filtered by is_pending=FALSE)
        assert "new-pending-topic" not in result["topics"]
