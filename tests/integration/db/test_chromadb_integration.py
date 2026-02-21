"""ChromaDB integration tests for ChunkStore.

Tests upsert, search (with metadata filtering), delete, and heartbeat
against a real ChromaDB instance.

Requires:
    - docker compose -f docker/docker-compose.test.yml up -d
    - PKB_DB_INTEGRATION=1 environment variable
"""


class TestChunkUpsertAndSearch:
    """Test upsert_chunks + search round-trip."""

    def test_upsert_and_search_returns_results(self, chunk_store):
        """Upsert 2 Korean chunks about Python venv, search returns correct bundle_id."""
        chunks = [
            {
                "id": "b1-chunk-0",
                "document": (
                    "파이썬에서 가상환경(venv)을 사용하면 "
                    "프로젝트별로 독립된 패키지를 관리할 수 있습니다."
                ),
                "metadata": {"bundle_id": "b1", "kb": "personal", "chunk_index": 0},
            },
            {
                "id": "b1-chunk-1",
                "document": (
                    "python -m venv .venv 명령어로 "
                    "가상환경을 생성하고 activate하여 사용합니다."
                ),
                "metadata": {"bundle_id": "b1", "kb": "personal", "chunk_index": 1},
            },
        ]
        chunk_store.upsert_chunks(chunks)

        results = chunk_store.search("파이썬 가상환경", n_results=5)
        assert len(results) >= 1
        bundle_ids = [r.metadata["bundle_id"] for r in results]
        assert "b1" in bundle_ids

    def test_search_returns_similarity_distance(self, chunk_store):
        """Search result distance is between 0 and 2.0 (ChromaDB L2 distance)."""
        chunks = [
            {
                "id": "dl-chunk-0",
                "document": (
                    "딥러닝은 신경망을 여러 층으로 쌓아 "
                    "복잡한 패턴을 학습하는 기계학습 기법입니다."
                ),
                "metadata": {"bundle_id": "dl-bundle", "kb": "personal", "chunk_index": 0},
            },
        ]
        chunk_store.upsert_chunks(chunks)

        results = chunk_store.search("딥러닝 신경망", n_results=5)
        assert len(results) >= 1
        for r in results:
            assert 0 <= r.distance <= 2.0

    def test_search_with_metadata_filter(self, chunk_store):
        """Search with where={kb: kb-a} returns only kb-a results."""
        chunks = [
            {
                "id": "ka-chunk-0",
                "document": "PostgreSQL은 강력한 오픈소스 관계형 데이터베이스입니다.",
                "metadata": {"bundle_id": "ka-bundle", "kb": "kb-a", "chunk_index": 0},
            },
            {
                "id": "kb-chunk-0",
                "document": (
                    "PostgreSQL에서 인덱스를 활용하면 "
                    "쿼리 성능을 크게 향상시킬 수 있습니다."
                ),
                "metadata": {"bundle_id": "kb-bundle", "kb": "kb-b", "chunk_index": 0},
            },
        ]
        chunk_store.upsert_chunks(chunks)

        results = chunk_store.search("PostgreSQL 데이터베이스", n_results=10, where={"kb": "kb-a"})
        assert len(results) >= 1
        for r in results:
            assert r.metadata["kb"] == "kb-a"

    def test_upsert_empty_list_is_noop(self, chunk_store):
        """Upserting an empty list does not raise."""
        chunk_store.upsert_chunks([])


class TestChunkDelete:
    """Test delete_by_bundle removes all chunks for a bundle."""

    def test_delete_by_bundle(self, chunk_store):
        """Upsert 3 chunks, delete by bundle_id, verify no results remain."""
        chunks = [
            {
                "id": f"del-bundle-chunk-{i}",
                "document": f"삭제 테스트 문서 {i}번째 청크입니다.",
                "metadata": {"bundle_id": "del-bundle", "kb": "personal", "chunk_index": i},
            }
            for i in range(3)
        ]
        chunk_store.upsert_chunks(chunks)

        # Verify chunks exist
        results_before = chunk_store.search(
            "삭제 테스트 문서", n_results=10, where={"bundle_id": "del-bundle"}
        )
        assert len(results_before) == 3

        # Delete
        chunk_store.delete_by_bundle("del-bundle")

        # Verify no results for that bundle_id
        results_after = chunk_store.search(
            "삭제 테스트 문서", n_results=10, where={"bundle_id": "del-bundle"}
        )
        assert len(results_after) == 0


class TestDeleteByKb:
    """Test delete_by_kb removes only target KB chunks."""

    def test_delete_by_kb_preserves_other_kb(self, chunk_store):
        """delete_by_kb('kb-a') removes kb-a chunks, kb-b chunks intact."""
        chunks = [
            {
                "id": "kba-chunk-0",
                "document": "KB-A에 속한 문서입니다. 파이썬 관련 내용.",
                "metadata": {"bundle_id": "kba-bundle", "kb": "kb-a", "chunk_index": 0},
            },
            {
                "id": "kba-chunk-1",
                "document": "KB-A의 두번째 청크. async 프로그래밍.",
                "metadata": {"bundle_id": "kba-bundle", "kb": "kb-a", "chunk_index": 1},
            },
            {
                "id": "kbb-chunk-0",
                "document": "KB-B에 속한 문서입니다. 자바스크립트 관련.",
                "metadata": {"bundle_id": "kbb-bundle", "kb": "kb-b", "chunk_index": 0},
            },
        ]
        chunk_store.upsert_chunks(chunks)

        # Delete kb-a
        chunk_store.delete_by_kb("kb-a")

        # kb-a chunks should be gone
        results_a = chunk_store.search(
            "파이썬 async", n_results=10, where={"kb": "kb-a"},
        )
        assert len(results_a) == 0

        # kb-b chunks should remain
        results_b = chunk_store.search(
            "자바스크립트", n_results=10, where={"kb": "kb-b"},
        )
        assert len(results_b) >= 1
        assert results_b[0].metadata["kb"] == "kb-b"

    def test_delete_by_kb_nonexistent_is_noop(self, chunk_store):
        """Deleting a non-existent KB does not raise."""
        chunk_store.delete_by_kb("nonexistent-kb")


class TestHeartbeat:
    """Test ChromaDB server connectivity."""

    def test_heartbeat(self, chunk_store):
        """heartbeat() returns a positive integer (nanosecond timestamp)."""
        result = chunk_store.heartbeat()
        assert isinstance(result, int)
        assert result > 0
