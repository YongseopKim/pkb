"""Tests for BundleRepository relation methods (mock-based)."""

from unittest.mock import MagicMock

from pkb.db.postgres import BundleRepository


class TestInsertRelation:
    def test_method_exists(self):
        assert hasattr(BundleRepository, "insert_relation")

    def test_insert_relation_calls_execute(self):
        repo = MagicMock(spec=BundleRepository)
        repo.insert_relation = BundleRepository.insert_relation.__get__(repo)
        repo.insert_relation(
            source_bundle_id="20260101-test-abc1",
            target_bundle_id="20260101-test-def2",
            relation_type="similar",
            score=0.85,
        )
        repo._get_conn.assert_called_once()


class TestListRelations:
    def test_method_exists(self):
        assert hasattr(BundleRepository, "list_relations")


class TestDeleteRelations:
    def test_method_exists(self):
        assert hasattr(BundleRepository, "delete_relations_for_bundle")


class TestListAllRelations:
    def test_method_exists(self):
        assert hasattr(BundleRepository, "list_all_relations")


class TestFindBundlesSharingTopics:
    def test_method_exists(self):
        assert hasattr(BundleRepository, "find_bundles_sharing_topics")


class TestCountRelations:
    def test_method_exists(self):
        assert hasattr(BundleRepository, "count_relations")


class TestListBundlesByDomain:
    def test_method_exists(self):
        assert hasattr(BundleRepository, "list_bundles_by_domain")
