"""Topic vocabulary manager — approve/merge/reject workflow."""

from pathlib import Path

import yaml

from pkb.models.vocab import Topic, TopicsVocab


class TopicManager:
    """Manages the L2 topic controlled vocabulary."""

    def __init__(self, topics_path: Path) -> None:
        self._path = topics_path
        self._vocab = self._load()

    def _load(self) -> TopicsVocab:
        raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        return TopicsVocab(**raw)

    def _save(self) -> None:
        data = {"topics": [t.model_dump(exclude_none=True) for t in self._vocab.topics]}
        self._path.write_text(
            yaml.dump(data, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )

    def list_topics(self, status: str | None = None) -> list[Topic]:
        """List topics, optionally filtered by status."""
        if status is None:
            return list(self._vocab.topics)
        return [t for t in self._vocab.topics if t.status == status]

    def approve(self, canonical: str) -> None:
        """Approve a pending topic."""
        for topic in self._vocab.topics:
            if topic.canonical == canonical:
                topic.status = "approved"
                break
        self._save()

    def merge(self, canonical: str, *, into: str) -> None:
        """Merge a topic into another (marks as 'merged')."""
        for topic in self._vocab.topics:
            if topic.canonical == canonical:
                topic.status = "merged"
                topic.merged_into = into
                break
        self._save()

    def reject(self, canonical: str) -> None:
        """Remove a topic from the vocabulary."""
        self._vocab.topics = [
            t for t in self._vocab.topics if t.canonical != canonical
        ]
        self._save()

    def add_pending(self, canonical: str) -> None:
        """Add a new pending topic (if not already present)."""
        existing = {t.canonical for t in self._vocab.topics}
        if canonical in existing:
            return
        self._vocab.topics.append(Topic(canonical=canonical, status="pending"))
        self._save()
