"""Tests for vocab data loading."""

from pathlib import Path

from pkb.vocab.loader import load_domains, load_topics

# Bundled data directory
DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "pkb" / "data"


class TestLoadDomains:
    def test_loads_bundled_domains(self):
        vocab = load_domains()
        assert len(vocab.domains) == 8

    def test_domain_ids(self):
        vocab = load_domains()
        expected_ids = {
            "dev", "investing", "learning", "health", "hobby",
            "self", "side-project", "work",
        }
        assert vocab.get_ids() == expected_ids

    def test_domain_labels(self):
        vocab = load_domains()
        dev = next(d for d in vocab.domains if d.id == "dev")
        assert dev.label_ko == "개발"
        assert dev.label_en == "Development"

    def test_custom_path(self, tmp_path: Path):
        """Can load from a custom path."""
        import yaml

        custom = tmp_path / "domains.yaml"
        custom.write_text(
            yaml.dump({
                "domains": [
                    {"id": "test", "label_ko": "테스트", "label_en": "Test"}
                ]
            }),
            encoding="utf-8",
        )
        vocab = load_domains(custom)
        assert len(vocab.domains) == 1
        assert vocab.domains[0].id == "test"


class TestLoadTopics:
    def test_loads_bundled_topics(self):
        vocab = load_topics()
        assert len(vocab.topics) >= 50

    def test_all_approved(self):
        """Seed topics should all be approved."""
        vocab = load_topics()
        for topic in vocab.topics:
            assert topic.status == "approved"

    def test_topic_has_aliases(self):
        vocab = load_topics()
        bitcoin = next(t for t in vocab.topics if t.canonical == "bitcoin")
        assert "비트코인" in bitcoin.aliases
        assert "btc" in bitcoin.aliases

    def test_domain_coverage(self):
        """Topics should cover all 8 domain areas."""
        vocab = load_topics()
        canonicals = {t.canonical for t in vocab.topics}
        # At least one topic from each expected domain area
        assert "python" in canonicals  # dev
        assert "bitcoin" in canonicals  # investing
        assert "rag" in canonicals  # learning
        assert "exercise" in canonicals  # health
        assert "guitar" in canonicals  # hobby
        assert "self-reflection" in canonicals  # self
        assert "pkb" in canonicals  # side-project
        assert "team-management" in canonicals  # work

    def test_custom_path(self, tmp_path: Path):
        import yaml

        custom = tmp_path / "topics.yaml"
        custom.write_text(
            yaml.dump({
                "topics": [
                    {"canonical": "test-topic", "aliases": ["테스트"], "status": "approved"}
                ]
            }),
            encoding="utf-8",
        )
        vocab = load_topics(custom)
        assert len(vocab.topics) == 1
