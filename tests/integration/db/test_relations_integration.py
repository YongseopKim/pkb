"""Integration tests for bundle_relations repository methods."""

from datetime import datetime

import pytest


@pytest.fixture
def _seed_bundles(repo):
    """Insert test bundles for relation tests."""
    for bid in [
        "20260101-alpha-abc1",
        "20260101-beta-def2",
        "20260101-gamma-ghi3",
    ]:
        repo.upsert_bundle(
            bundle_id=bid,
            kb="test",
            question=f"Question for {bid}",
            summary=f"Summary for {bid}",
            created_at=datetime(2026, 1, 1),
            response_count=1,
            path=f"/bundles/{bid}",
            question_hash=f"hash_{bid}",
            domains=["dev"],
            topics=["python"],
            responses=[{"platform": "claude", "model": "haiku", "turn_count": 1}],
        )


class TestInsertRelation:
    def test_insert_and_list(self, repo, _seed_bundles):
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.85,
        )
        rels = repo.list_relations("20260101-alpha-abc1")
        assert len(rels) == 1
        assert rels[0]["target_bundle_id"] == "20260101-beta-def2"
        assert rels[0]["score"] == pytest.approx(0.85, abs=0.01)

    def test_upsert_updates_score(self, repo, _seed_bundles):
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.80,
        )
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.90,
        )
        rels = repo.list_relations(
            "20260101-alpha-abc1", relation_type="similar",
        )
        assert len(rels) == 1
        assert rels[0]["score"] == pytest.approx(0.90, abs=0.01)

    def test_bidirectional_lookup(self, repo, _seed_bundles):
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.85,
        )
        # Should find when querying from the target side too
        rels = repo.list_relations("20260101-beta-def2")
        assert len(rels) == 1

    def test_multiple_relation_types(self, repo, _seed_bundles):
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.85,
        )
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "related", 0.60,
        )
        rels = repo.list_relations("20260101-alpha-abc1")
        assert len(rels) == 2


class TestDeleteRelations:
    def test_delete_removes_all(self, repo, _seed_bundles):
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.85,
        )
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-gamma-ghi3", "related", 0.70,
        )
        count = repo.delete_relations_for_bundle("20260101-alpha-abc1")
        assert count == 2
        assert repo.list_relations("20260101-alpha-abc1") == []


class TestListAllRelations:
    def test_list_all(self, repo, _seed_bundles):
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.85,
        )
        repo.insert_relation(
            "20260101-beta-def2", "20260101-gamma-ghi3", "related", 0.70,
        )
        all_rels = repo.list_all_relations()
        assert len(all_rels) == 2

    def test_filter_by_type(self, repo, _seed_bundles):
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.85,
        )
        repo.insert_relation(
            "20260101-beta-def2", "20260101-gamma-ghi3", "related", 0.70,
        )
        similar_only = repo.list_all_relations(relation_type="similar")
        assert len(similar_only) == 1

    def test_filter_by_kb(self, repo, _seed_bundles):
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.85,
        )
        # KB "test" should match
        results = repo.list_all_relations(kb="test")
        assert len(results) == 1
        # KB "nonexistent" should return empty
        results = repo.list_all_relations(kb="nonexistent")
        assert len(results) == 0


class TestFindBundlesSharingTopics:
    def test_finds_shared_topics(self, repo):
        from datetime import datetime
        repo.upsert_bundle(
            bundle_id="20260101-a-abc1", kb="test",
            question="Q1", summary="S1",
            created_at=datetime(2026, 1, 1),
            response_count=1, path="/a", question_hash="h1",
            domains=["dev"], topics=["python", "async"],
            responses=[{"platform": "claude", "model": "m", "turn_count": 1}],
        )
        repo.upsert_bundle(
            bundle_id="20260101-b-def2", kb="test",
            question="Q2", summary="S2",
            created_at=datetime(2026, 1, 1),
            response_count=1, path="/b", question_hash="h2",
            domains=["dev"], topics=["python", "testing"],
            responses=[{"platform": "claude", "model": "m", "turn_count": 1}],
        )

        shared = repo.find_bundles_sharing_topics("20260101-a-abc1")
        assert len(shared) == 1
        assert shared[0]["bundle_id"] == "20260101-b-def2"
        assert shared[0]["shared_count"] == 1  # "python" shared

    def test_excludes_self(self, repo):
        from datetime import datetime
        repo.upsert_bundle(
            bundle_id="20260101-a-abc1", kb="test",
            question="Q1", summary="S1",
            created_at=datetime(2026, 1, 1),
            response_count=1, path="/a", question_hash="h1",
            domains=["dev"], topics=["python"],
            responses=[{"platform": "claude", "model": "m", "turn_count": 1}],
        )
        shared = repo.find_bundles_sharing_topics("20260101-a-abc1")
        assert len(shared) == 0


class TestCountRelations:
    def test_count(self, repo, _seed_bundles):
        assert repo.count_relations() == 0
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.85,
        )
        assert repo.count_relations() == 1
        assert repo.count_relations(relation_type="similar") == 1
        assert repo.count_relations(relation_type="related") == 0
