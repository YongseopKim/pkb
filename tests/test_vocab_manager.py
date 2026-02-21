"""Tests for topic vocabulary manager."""

from pathlib import Path

import yaml

from pkb.vocab.manager import TopicManager


def _create_topics_file(path: Path, topics: list[dict]) -> Path:
    topics_path = path / "topics.yaml"
    topics_path.write_text(
        yaml.dump({"topics": topics}, allow_unicode=True),
        encoding="utf-8",
    )
    return topics_path


class TestTopicManager:
    def test_list_all(self, tmp_path):
        _create_topics_file(tmp_path, [
            {"canonical": "python", "status": "approved"},
            {"canonical": "rust", "status": "approved"},
            {"canonical": "new-topic", "status": "pending"},
        ])
        mgr = TopicManager(tmp_path / "topics.yaml")
        all_topics = mgr.list_topics()
        assert len(all_topics) == 3

    def test_list_by_status(self, tmp_path):
        _create_topics_file(tmp_path, [
            {"canonical": "python", "status": "approved"},
            {"canonical": "new-topic", "status": "pending"},
            {"canonical": "old-topic", "status": "merged", "merged_into": "python"},
        ])
        mgr = TopicManager(tmp_path / "topics.yaml")
        pending = mgr.list_topics(status="pending")
        assert len(pending) == 1
        assert pending[0].canonical == "new-topic"

    def test_approve_topic(self, tmp_path):
        _create_topics_file(tmp_path, [
            {"canonical": "new-topic", "status": "pending"},
        ])
        mgr = TopicManager(tmp_path / "topics.yaml")
        mgr.approve("new-topic")
        # Reload and check
        reloaded = TopicManager(tmp_path / "topics.yaml")
        topics = reloaded.list_topics(status="approved")
        assert len(topics) == 1
        assert topics[0].canonical == "new-topic"

    def test_merge_topic(self, tmp_path):
        _create_topics_file(tmp_path, [
            {"canonical": "python", "status": "approved"},
            {"canonical": "py", "status": "pending"},
        ])
        mgr = TopicManager(tmp_path / "topics.yaml")
        mgr.merge("py", into="python")
        reloaded = TopicManager(tmp_path / "topics.yaml")
        py = next(t for t in reloaded.list_topics() if t.canonical == "py")
        assert py.status == "merged"
        assert py.merged_into == "python"

    def test_reject_topic(self, tmp_path):
        _create_topics_file(tmp_path, [
            {"canonical": "spam-topic", "status": "pending"},
        ])
        mgr = TopicManager(tmp_path / "topics.yaml")
        mgr.reject("spam-topic")
        reloaded = TopicManager(tmp_path / "topics.yaml")
        topics = reloaded.list_topics()
        assert not any(t.canonical == "spam-topic" for t in topics)

    def test_add_pending_topic(self, tmp_path):
        _create_topics_file(tmp_path, [
            {"canonical": "python", "status": "approved"},
        ])
        mgr = TopicManager(tmp_path / "topics.yaml")
        mgr.add_pending("new-topic")
        reloaded = TopicManager(tmp_path / "topics.yaml")
        topics = reloaded.list_topics()
        assert len(topics) == 2
        new = next(t for t in topics if t.canonical == "new-topic")
        assert new.status == "pending"

    def test_add_pending_duplicate_ignored(self, tmp_path):
        _create_topics_file(tmp_path, [
            {"canonical": "python", "status": "approved"},
        ])
        mgr = TopicManager(tmp_path / "topics.yaml")
        mgr.add_pending("python")
        topics = mgr.list_topics()
        assert len(topics) == 1  # Not duplicated
