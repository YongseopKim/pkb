"""Tests for vocab data models."""

import pytest
from pydantic import ValidationError

from pkb.models.vocab import Domain, DomainsVocab, Topic, TopicsVocab


class TestDomain:
    def test_valid_domain(self):
        d = Domain(id="dev", label_ko="개발", label_en="Development")
        assert d.id == "dev"
        assert d.label_ko == "개발"

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            Domain(label_ko="코딩", label_en="Coding")


class TestDomainsVocab:
    def test_domain_list(self):
        vocab = DomainsVocab(
            domains=[
                Domain(id="dev", label_ko="개발", label_en="Development"),
                Domain(id="investing", label_ko="투자", label_en="Investing"),
            ]
        )
        assert len(vocab.domains) == 2

    def test_get_ids(self):
        vocab = DomainsVocab(
            domains=[
                Domain(id="dev", label_ko="개발", label_en="Development"),
            ]
        )
        assert vocab.get_ids() == {"dev"}


class TestTopic:
    def test_valid_topic(self):
        t = Topic(canonical="bitcoin", aliases=["비트코인", "btc"], status="approved")
        assert t.canonical == "bitcoin"
        assert len(t.aliases) == 2
        assert t.status == "approved"

    def test_default_status(self):
        t = Topic(canonical="new-topic", aliases=[])
        assert t.status == "approved"

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            Topic(canonical="test", aliases=[], status="invalid")

    def test_merged_topic(self):
        t = Topic(
            canonical="old-topic",
            aliases=[],
            status="merged",
            merged_into="new-topic",
        )
        assert t.merged_into == "new-topic"


class TestTopicsVocab:
    def _make_vocab(self):
        return TopicsVocab(
            topics=[
                Topic(canonical="bitcoin", aliases=["btc"], status="approved"),
                Topic(canonical="system-design", aliases=[], status="approved"),
                Topic(canonical="old-tag", aliases=[], status="merged", merged_into="bitcoin"),
                Topic(canonical="pending-tag", aliases=[], status="pending"),
            ]
        )

    def test_get_approved_canonicals(self):
        vocab = self._make_vocab()
        approved = vocab.get_approved_canonicals()
        assert approved == {"bitcoin", "system-design"}

    def test_all_canonicals(self):
        vocab = self._make_vocab()
        assert len(vocab.topics) == 4
