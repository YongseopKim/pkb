"""Tests for CLI commands (init, parse, search, reindex, regenerate)."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from pkb.batch import BatchProcessor
from pkb.cli import cli
from pkb.search.models import BundleSearchResult


@pytest.fixture
def runner():
    return CliRunner()


class TestInitCommand:
    def test_init_creates_pkb_home(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        pkb_home = tmp_path / ".pkb"
        monkeypatch.setenv("PKB_HOME", str(pkb_home))

        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0
        assert pkb_home.exists()
        assert (pkb_home / "config.yaml").is_file()
        assert (pkb_home / "vocab" / "domains.yaml").is_file()
        assert (pkb_home / "vocab" / "topics.yaml").is_file()
        assert (pkb_home / "index").is_dir()

    def test_init_already_exists(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        pkb_home = tmp_path / ".pkb"
        pkb_home.mkdir()
        (pkb_home / "config.yaml").write_text("existing")
        monkeypatch.setenv("PKB_HOME", str(pkb_home))

        result = runner.invoke(cli, ["init"])
        assert result.exit_code != 0

    def test_init_force(self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        pkb_home = tmp_path / ".pkb"
        pkb_home.mkdir()
        (pkb_home / "config.yaml").write_text("existing")
        monkeypatch.setenv("PKB_HOME", str(pkb_home))

        result = runner.invoke(cli, ["init", "--force"])
        assert result.exit_code == 0
        assert (pkb_home / "vocab" / "domains.yaml").is_file()


class TestParseCommand:
    def test_parse_single_file(self, runner: CliRunner, samples_dir: Path):
        result = runner.invoke(cli, ["parse", str(samples_dir / "claude.jsonl")])
        assert result.exit_code == 0
        assert "claude" in result.output

    def test_parse_directory(self, runner: CliRunner, samples_dir: Path):
        result = runner.invoke(cli, ["parse", str(samples_dir)])
        assert result.exit_code == 0
        # Should mention all 5 platforms
        for platform in ("chatgpt", "claude", "gemini", "grok", "perplexity"):
            assert platform in result.output

    def test_parse_nonexistent_path(self, runner: CliRunner):
        result = runner.invoke(cli, ["parse", "/nonexistent/path"])
        assert result.exit_code != 0

    def test_parse_file_shows_summary(self, runner: CliRunner, samples_dir: Path):
        result = runner.invoke(cli, ["parse", str(samples_dir / "chatgpt.jsonl")])
        assert result.exit_code == 0
        # Should show turn count
        assert "turn" in result.output.lower()

    def test_parse_label_first_message_not_question(self, runner: CliRunner, samples_dir: Path):
        """parse 출력에서 'Question:' 대신 'First message:' 사용."""
        result = runner.invoke(cli, ["parse", str(samples_dir / "claude.jsonl")])
        assert result.exit_code == 0
        assert "Question:" not in result.output
        assert "First message:" in result.output


def _make_search_results():
    """Helper to create mock search results."""
    return [
        BundleSearchResult(
            bundle_id="20260221-bitcoin-a3f2",
            question="Bitcoin halving 이후 가격 전망은?",
            summary="비트코인 반감기 분석, 과거 패턴 기반 예측",
            domains=["investing"],
            topics=["bitcoin", "crypto"],
            score=0.87,
            created_at=datetime(2026, 2, 21, tzinfo=timezone.utc),
            source="both",
        ),
        BundleSearchResult(
            bundle_id="20260215-python-b1c2",
            question="Python async 패턴?",
            summary="파이썬 비동기 패턴 설명",
            domains=["dev"],
            topics=["python"],
            score=0.65,
            created_at=datetime(2026, 2, 15, tzinfo=timezone.utc),
            source="fts",
        ),
    ]


class TestSearchCommand:
    """Tests for pkb search command.

    We patch at the import target (where the names are looked up inside the search function),
    which uses lazy imports from pkb.config, pkb.db.postgres, etc.
    """

    @patch("pkb.search.engine.SearchEngine")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_search_basic(
        self, mock_home, mock_load, mock_repo_cls, mock_store_cls, mock_engine_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        mock_engine = MagicMock()
        mock_engine.search.return_value = _make_search_results()
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(cli, ["search", "bitcoin halving"])
        assert result.exit_code == 0, result.output
        assert "bitcoin" in result.output.lower()
        assert "0.87" in result.output

    @patch("pkb.search.engine.SearchEngine")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_search_json_output(
        self, mock_home, mock_load, mock_repo_cls, mock_store_cls, mock_engine_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        mock_engine = MagicMock()
        mock_engine.search.return_value = _make_search_results()
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(cli, ["search", "bitcoin", "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["bundle_id"] == "20260221-bitcoin-a3f2"

    @patch("pkb.search.engine.SearchEngine")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_search_with_filters(
        self, mock_home, mock_load, mock_repo_cls, mock_store_cls, mock_engine_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        mock_engine = MagicMock()
        mock_engine.search.return_value = []
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(cli, [
            "search", "test",
            "--mode", "keyword",
            "--domain", "investing",
            "--topic", "bitcoin",
            "--kb", "personal",
            "--limit", "5",
        ])
        assert result.exit_code == 0, result.output
        # Verify the query was constructed correctly
        call_args = mock_engine.search.call_args[0][0]
        assert call_args.query == "test"
        assert call_args.mode.value == "keyword"
        assert call_args.domains == ["investing"]
        assert call_args.topics == ["bitcoin"]
        assert call_args.kb == "personal"
        assert call_args.limit == 5

    @patch("pkb.search.engine.SearchEngine")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_search_no_results(
        self, mock_home, mock_load, mock_repo_cls, mock_store_cls, mock_engine_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        mock_engine = MagicMock()
        mock_engine.search.return_value = []
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(cli, ["search", "nothing"])
        assert result.exit_code == 0, result.output
        assert "no results" in result.output.lower() or "0" in result.output

    @patch("pkb.search.engine.SearchEngine")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_search_date_filters(
        self, mock_home, mock_load, mock_repo_cls, mock_store_cls, mock_engine_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        mock_engine = MagicMock()
        mock_engine.search.return_value = []
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(cli, [
            "search", "test",
            "--after", "2026-01-01",
            "--before", "2026-12-31",
        ])
        assert result.exit_code == 0, result.output
        from datetime import date
        call_args = mock_engine.search.call_args[0][0]
        assert call_args.after == date(2026, 1, 1)
        assert call_args.before == date(2026, 12, 31)

    @patch("pkb.search.engine.SearchEngine")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_search_text_output_no_question(
        self, mock_home, mock_load, mock_repo_cls, mock_store_cls, mock_engine_cls,
        runner, tmp_path,
    ):
        """텍스트 출력에 Q: 줄이 없어야 함."""
        mock_home.return_value = tmp_path
        mock_load.return_value = MagicMock()
        mock_engine = MagicMock()
        mock_engine.search.return_value = _make_search_results()
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(cli, ["search", "bitcoin"])
        assert result.exit_code == 0, result.output
        assert "Q:" not in result.output

    @patch("pkb.search.engine.SearchEngine")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_search_json_output_no_question(
        self, mock_home, mock_load, mock_repo_cls, mock_store_cls, mock_engine_cls,
        runner, tmp_path,
    ):
        """JSON 출력에 question 키가 없어야 함."""
        mock_home.return_value = tmp_path
        mock_load.return_value = MagicMock()
        mock_engine = MagicMock()
        mock_engine.search.return_value = _make_search_results()
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(cli, ["search", "bitcoin", "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        for item in data:
            assert "question" not in item
            assert "summary" in item


def _create_bundle_on_disk(kb_path, bundle_id="20260101-test-abc1"):
    """Helper to create a bundle directory with _bundle.md and platform MD."""
    bundle_dir = kb_path / "bundles" / bundle_id
    bundle_dir.mkdir(parents=True)
    fm = {
        "id": bundle_id,
        "question": "테스트 질문",
        "summary": "테스트 요약",
        "slug": "test",
        "domains": ["dev"],
        "topics": ["python"],
        "pending_topics": [],
        "platforms": ["claude"],
        "created_at": "2026-01-01T00:00:00+00:00",
        "consensus": None,
        "divergence": None,
    }
    content = f"---\n{yaml.dump(fm, allow_unicode=True, default_flow_style=False)}---\n"
    (bundle_dir / "_bundle.md").write_text(content, encoding="utf-8")
    (bundle_dir / "claude.md").write_text(
        "---\nplatform: claude\n---\n\n# Content\nHello\n", encoding="utf-8"
    )
    return bundle_dir


class TestReindexCommand:
    @patch("pkb.reindex.Reindexer")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_reindex_single_bundle(
        self, mock_home, mock_load, mock_repo_cls, mock_store_cls, mock_reindexer_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_config = MagicMock()
        kb_entry = MagicMock()
        kb_entry.name = "personal"
        kb_entry.path = tmp_path / "kb"
        mock_config.knowledge_bases = [kb_entry]
        mock_load.return_value = mock_config

        mock_reindexer = MagicMock()
        mock_reindexer.reindex_bundle.return_value = {
            "bundle_id": "20260101-test-abc1",
            "status": "updated",
        }
        mock_reindexer_cls.return_value = mock_reindexer

        result = runner.invoke(
            cli, ["reindex", "20260101-test-abc1", "--kb", "personal"]
        )
        assert result.exit_code == 0, result.output
        assert "updated" in result.output.lower()

    @patch("pkb.reindex.Reindexer")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_reindex_full(
        self, mock_home, mock_load, mock_repo_cls, mock_store_cls, mock_reindexer_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_config = MagicMock()
        kb_entry = MagicMock()
        kb_entry.name = "personal"
        kb_entry.path = tmp_path / "kb"
        mock_config.knowledge_bases = [kb_entry]
        mock_load.return_value = mock_config

        mock_reindexer = MagicMock()
        mock_reindexer.reindex_full.return_value = {
            "total": 3, "updated": 2, "skipped": 1, "errors": 0, "deleted": 0,
        }
        mock_reindexer_cls.return_value = mock_reindexer

        result = runner.invoke(cli, ["reindex", "--full", "--kb", "personal"])
        assert result.exit_code == 0, result.output
        assert "2" in result.output  # updated count
        mock_reindexer.reindex_full.assert_called_once()

    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_reindex_missing_kb(self, mock_home, mock_load, runner, tmp_path):
        mock_home.return_value = tmp_path
        mock_config = MagicMock()
        mock_config.knowledge_bases = []
        mock_load.return_value = mock_config

        result = runner.invoke(cli, ["reindex", "test-id", "--kb", "nonexistent"])
        assert result.exit_code != 0


class TestRegenerateCommand:
    @patch("pkb.regenerate.Regenerator")
    @patch("pkb.generator.meta_gen.MetaGenerator")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.vocab.loader.load_topics")
    @patch("pkb.vocab.loader.load_domains")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_regenerate_single(
        self, mock_home, mock_load, mock_domains, mock_topics,
        mock_repo_cls, mock_store_cls, mock_meta_cls, mock_regen_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        (tmp_path / "vocab").mkdir()
        mock_config = MagicMock()
        kb_entry = MagicMock()
        kb_entry.name = "personal"
        kb_entry.path = tmp_path / "kb"
        mock_config.knowledge_bases = [kb_entry]
        mock_load.return_value = mock_config
        mock_domains.return_value = MagicMock(get_ids=MagicMock(return_value={"dev"}))
        mock_topics.return_value = MagicMock(
            get_approved_canonicals=MagicMock(return_value={"python"})
        )

        mock_regen = MagicMock()
        mock_regen.regenerate_bundle.return_value = {
            "bundle_id": "20260101-test-abc1", "status": "regenerated",
        }
        mock_regen_cls.return_value = mock_regen

        result = runner.invoke(
            cli, ["regenerate", "20260101-test-abc1", "--kb", "personal"]
        )
        assert result.exit_code == 0, result.output
        assert "regenerated" in result.output.lower()

    @patch("pkb.regenerate.Regenerator")
    @patch("pkb.generator.meta_gen.MetaGenerator")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.vocab.loader.load_topics")
    @patch("pkb.vocab.loader.load_domains")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_regenerate_all(
        self, mock_home, mock_load, mock_domains, mock_topics,
        mock_repo_cls, mock_store_cls, mock_meta_cls, mock_regen_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        (tmp_path / "vocab").mkdir()
        mock_config = MagicMock()
        kb_entry = MagicMock()
        kb_entry.name = "personal"
        kb_entry.path = tmp_path / "kb"
        mock_config.knowledge_bases = [kb_entry]
        mock_load.return_value = mock_config
        mock_domains.return_value = MagicMock(get_ids=MagicMock(return_value={"dev"}))
        mock_topics.return_value = MagicMock(
            get_approved_canonicals=MagicMock(return_value={"python"})
        )

        mock_regen = MagicMock()
        mock_regen.regenerate_all.return_value = {
            "total": 5, "regenerated": 4, "errors": 1,
        }
        mock_regen_cls.return_value = mock_regen

        result = runner.invoke(cli, ["regenerate", "--all", "--kb", "personal"])
        assert result.exit_code == 0, result.output
        assert "4" in result.output  # regenerated count
        mock_regen.regenerate_all.assert_called_once()

    @patch("pkb.regenerate.Regenerator")
    @patch("pkb.generator.meta_gen.MetaGenerator")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.vocab.loader.load_topics")
    @patch("pkb.vocab.loader.load_domains")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_regenerate_dry_run(
        self, mock_home, mock_load, mock_domains, mock_topics,
        mock_repo_cls, mock_store_cls, mock_meta_cls, mock_regen_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        (tmp_path / "vocab").mkdir()
        mock_config = MagicMock()
        kb_entry = MagicMock()
        kb_entry.name = "personal"
        kb_entry.path = tmp_path / "kb"
        mock_config.knowledge_bases = [kb_entry]
        mock_load.return_value = mock_config
        mock_domains.return_value = MagicMock(get_ids=MagicMock(return_value={"dev"}))
        mock_topics.return_value = MagicMock(
            get_approved_canonicals=MagicMock(return_value={"python"})
        )

        mock_regen = MagicMock()
        mock_regen.regenerate_bundle.return_value = {
            "bundle_id": "20260101-test-abc1", "status": "regenerated",
        }
        mock_regen_cls.return_value = mock_regen

        result = runner.invoke(
            cli, ["regenerate", "20260101-test-abc1", "--kb", "personal", "--dry-run"]
        )
        assert result.exit_code == 0, result.output
        # Should pass dry_run=True to Regenerator
        regen_kwargs = mock_regen_cls.call_args[1]
        assert regen_kwargs["dry_run"] is True


class TestTopicsCommand:
    def test_topics_default_lists_all(self, runner, tmp_path, monkeypatch):
        """pkb topics (no subcommand) should list all topics."""
        monkeypatch.setenv("PKB_HOME", str(tmp_path))
        vocab_dir = tmp_path / "vocab"
        vocab_dir.mkdir()
        topics_data = {
            "topics": [
                {"canonical": "python", "status": "approved"},
                {"canonical": "new-topic", "status": "pending"},
            ]
        }
        (vocab_dir / "topics.yaml").write_text(
            yaml.dump(topics_data, allow_unicode=True), encoding="utf-8"
        )

        result = runner.invoke(cli, ["topics"])
        assert result.exit_code == 0, result.output
        assert "python" in result.output
        assert "new-topic" in result.output

    def test_topics_list_subcommand(self, runner, tmp_path, monkeypatch):
        """pkb topics list --status pending should filter."""
        monkeypatch.setenv("PKB_HOME", str(tmp_path))
        vocab_dir = tmp_path / "vocab"
        vocab_dir.mkdir()
        topics_data = {
            "topics": [
                {"canonical": "python", "status": "approved"},
                {"canonical": "new-topic", "status": "pending"},
            ]
        }
        (vocab_dir / "topics.yaml").write_text(
            yaml.dump(topics_data, allow_unicode=True), encoding="utf-8"
        )

        result = runner.invoke(cli, ["topics", "list", "--status", "pending"])
        assert result.exit_code == 0, result.output
        assert "new-topic" in result.output
        assert "python" not in result.output

    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    def test_topics_approve(
        self, mock_load, mock_repo_cls, runner, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("PKB_HOME", str(tmp_path))
        vocab_dir = tmp_path / "vocab"
        vocab_dir.mkdir()
        topics_data = {
            "topics": [{"canonical": "new-topic", "status": "pending"}]
        }
        (vocab_dir / "topics.yaml").write_text(
            yaml.dump(topics_data, allow_unicode=True), encoding="utf-8"
        )
        # Create config.yaml so load_config works
        (tmp_path / "config.yaml").write_text("knowledge_bases: []\n")
        mock_config = MagicMock()
        mock_load.return_value = mock_config

        result = runner.invoke(cli, ["topics", "approve", "new-topic"])
        assert result.exit_code == 0, result.output
        assert "Approved" in result.output

        # Verify YAML was updated
        reloaded = yaml.safe_load(
            (vocab_dir / "topics.yaml").read_text(encoding="utf-8")
        )
        topic = reloaded["topics"][0]
        assert topic["status"] == "approved"

    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    def test_topics_approve_nonexistent(
        self, mock_load, mock_repo_cls, runner, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("PKB_HOME", str(tmp_path))
        vocab_dir = tmp_path / "vocab"
        vocab_dir.mkdir()
        topics_data = {"topics": [{"canonical": "python", "status": "approved"}]}
        (vocab_dir / "topics.yaml").write_text(
            yaml.dump(topics_data, allow_unicode=True), encoding="utf-8"
        )

        result = runner.invoke(cli, ["topics", "approve", "nonexistent"])
        assert result.exit_code != 0

    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    def test_topics_merge(
        self, mock_load, mock_repo_cls, runner, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("PKB_HOME", str(tmp_path))
        vocab_dir = tmp_path / "vocab"
        vocab_dir.mkdir()
        topics_data = {
            "topics": [
                {"canonical": "python", "status": "approved"},
                {"canonical": "py", "status": "pending"},
            ]
        }
        (vocab_dir / "topics.yaml").write_text(
            yaml.dump(topics_data, allow_unicode=True), encoding="utf-8"
        )
        (tmp_path / "config.yaml").write_text("knowledge_bases: []\n")
        mock_config = MagicMock()
        mock_load.return_value = mock_config

        result = runner.invoke(cli, ["topics", "merge", "py", "--into", "python"])
        assert result.exit_code == 0, result.output
        assert "Merged" in result.output

    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    def test_topics_reject(
        self, mock_load, mock_repo_cls, runner, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("PKB_HOME", str(tmp_path))
        vocab_dir = tmp_path / "vocab"
        vocab_dir.mkdir()
        topics_data = {
            "topics": [{"canonical": "spam", "status": "pending"}]
        }
        (vocab_dir / "topics.yaml").write_text(
            yaml.dump(topics_data, allow_unicode=True), encoding="utf-8"
        )
        (tmp_path / "config.yaml").write_text("knowledge_bases: []\n")
        mock_config = MagicMock()
        mock_load.return_value = mock_config

        result = runner.invoke(cli, ["topics", "reject", "spam"])
        assert result.exit_code == 0, result.output
        assert "Rejected" in result.output

        # Verify topic was removed from YAML
        reloaded = yaml.safe_load(
            (vocab_dir / "topics.yaml").read_text(encoding="utf-8")
        )
        assert len(reloaded["topics"]) == 0


class TestDedupCommand:
    @patch("pkb.dedup.DuplicateDetector")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_dedup_scan(
        self, mock_home, mock_load, mock_repo_cls, mock_store_cls, mock_detector_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        mock_detector = MagicMock()
        mock_detector.scan.return_value = {"scanned": 10, "new_pairs": 2}
        mock_detector_cls.return_value = mock_detector

        result = runner.invoke(cli, ["dedup", "scan"])
        assert result.exit_code == 0, result.output
        assert "10" in result.output
        assert "2" in result.output

    @patch("pkb.dedup.DuplicateDetector")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_dedup_list(
        self, mock_home, mock_load, mock_repo_cls, mock_detector_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        mock_detector = MagicMock()
        mock_detector.list_pairs.return_value = [
            {"id": 1, "bundle_a": "a", "bundle_b": "b", "similarity": 0.91, "status": "pending"},
        ]
        mock_detector_cls.return_value = mock_detector

        result = runner.invoke(cli, ["dedup", "list"])
        assert result.exit_code == 0, result.output
        assert "0.91" in result.output

    @patch("pkb.dedup.DuplicateDetector")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_dedup_dismiss(
        self, mock_home, mock_load, mock_repo_cls, mock_detector_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector

        result = runner.invoke(cli, ["dedup", "dismiss", "42"])
        assert result.exit_code == 0, result.output
        assert "Dismissed" in result.output


class TestWatchCommand:
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_watch_missing_kb(self, mock_home, mock_load, runner, tmp_path):
        mock_home.return_value = tmp_path
        mock_config = MagicMock()
        mock_config.knowledge_bases = []
        mock_load.return_value = mock_config

        result = runner.invoke(cli, ["watch", "--kb", "nonexistent"])
        assert result.exit_code != 0


class TestFindWatchDirForPath:
    """_find_watch_dir_for_path() 헬퍼 테스트."""

    def test_direct_child(self, tmp_path):
        from pkb.cli import _find_watch_dir_for_path

        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()
        resolved_map = {str(watch_dir.resolve()): "pipeline_a"}

        result = _find_watch_dir_for_path(watch_dir / "test.jsonl", resolved_map)
        assert result is not None
        assert result[1] == "pipeline_a"

    def test_subdirectory_file(self, tmp_path):
        from pkb.cli import _find_watch_dir_for_path

        watch_dir = tmp_path / "inbox"
        sub = watch_dir / "PKB"
        sub.mkdir(parents=True)
        resolved_map = {str(watch_dir.resolve()): "pipeline_a"}

        result = _find_watch_dir_for_path(sub / "chatgpt.md", resolved_map)
        assert result is not None
        assert result[1] == "pipeline_a"

    def test_unknown_path(self, tmp_path):
        from pkb.cli import _find_watch_dir_for_path

        resolved_map = {str((tmp_path / "inbox").resolve()): "pipeline_a"}
        result = _find_watch_dir_for_path(tmp_path / "other" / "test.jsonl", resolved_map)
        assert result is None

    def test_symlink_resolves(self, tmp_path):
        from pkb.cli import _find_watch_dir_for_path

        real_inbox = tmp_path / "real_inbox"
        real_inbox.mkdir()
        link_inbox = tmp_path / "link_inbox"
        link_inbox.symlink_to(real_inbox)

        # Map uses resolved path of symlink
        resolved_map = {str(link_inbox.resolve()): "pipeline_a"}
        # File uses real path
        result = _find_watch_dir_for_path(real_inbox / "test.jsonl", resolved_map)
        assert result is not None
        assert result[1] == "pipeline_a"


class TestWatchReingest:
    """watch _on_new_file 콜백: stable_id 기반 dedup 테스트.

    stable_id가 도입되면서 watch 콜백에서 수동 delete+reingest 로직이 제거됨.
    pipeline.ingest_file()이 CREATE/UPDATE/MERGE를 자동으로 처리.
    """

    def test_stable_id_handles_update_automatically(self, tmp_path):
        """수정된 파일: pipeline이 stable_id로 UPDATE를 자동 처리."""
        from pkb.cli import _build_watch_callback

        pipeline = MagicMock()
        pipeline.ingest_file.return_value = {
            "bundle_id": "20260221-old-bundle-a3f2",
            "updated": True,
        }

        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()

        callback = _build_watch_callback(
            pipelines={str(watch_dir): pipeline},
            kb_entries={str(watch_dir): MagicMock(path=tmp_path / "kb")},
        )

        file_path = watch_dir / "test.jsonl"
        callback(file_path)

        # stable_id 기반: force 없이 일반 ingest
        pipeline.ingest_file.assert_called_once_with(file_path)

    def test_new_file_ingest(self, tmp_path):
        """새 파일은 일반 ingest (no force)."""
        from pkb.cli import _build_watch_callback

        pipeline = MagicMock()
        pipeline.ingest_file.return_value = {"bundle_id": "20260221-new-a3f2"}

        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()

        callback = _build_watch_callback(
            pipelines={str(watch_dir): pipeline},
            kb_entries={str(watch_dir): MagicMock(path=tmp_path / "kb")},
        )

        file_path = watch_dir / "test.jsonl"
        callback(file_path)

        # stable_id 기반: force 없이 일반 ingest
        pipeline.ingest_file.assert_called_once_with(file_path)

    def test_duplicate_skip_output(self, tmp_path):
        """ingest가 None 반환 시 (duplicate) 에러 없이 처리."""
        from pkb.cli import _build_watch_callback

        pipeline = MagicMock()
        pipeline.ingest_file.return_value = None

        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()

        callback = _build_watch_callback(
            pipelines={str(watch_dir): pipeline},
            kb_entries={str(watch_dir): MagicMock(path=tmp_path / "kb")},
        )

        file_path = watch_dir / "test.jsonl"
        # Should not raise
        callback(file_path)

    def test_subdirectory_ingest(self, tmp_path):
        """서브디렉토리 파일도 stable_id 기반으로 정상 ingest."""
        from pkb.cli import _build_watch_callback

        pipeline = MagicMock()
        pipeline.ingest_file.return_value = {"bundle_id": "20260222-new-b4c5"}

        watch_dir = tmp_path / "inbox"
        sub = watch_dir / "PKB"
        sub.mkdir(parents=True)

        kb_entry = MagicMock()
        kb_entry.path = tmp_path / "kb"
        kb_entry.get_watch_dir.return_value = watch_dir

        callback = _build_watch_callback(
            pipelines={str(watch_dir): pipeline},
            kb_entries={str(watch_dir): kb_entry},
        )

        file_path = sub / "chatgpt.md"
        callback(file_path)

        # stable_id 기반: force 없이 일반 ingest
        pipeline.ingest_file.assert_called_once_with(file_path)


class TestBuildWatchIngestFn:
    """_build_watch_ingest_fn() 테스트: IngestEngine용 ingest_fn.

    stable_id 도입 후 수동 delete+reingest 로직 제거됨.
    pipeline.ingest_file()이 CREATE/UPDATE/MERGE를 자동 처리.
    """

    def test_update_via_ingest_fn(self, tmp_path):
        """수정된 파일: pipeline이 stable_id로 UPDATE를 자동 처리."""
        from pkb.cli import _build_watch_ingest_fn

        pipeline = MagicMock()
        pipeline.ingest_file.return_value = {
            "bundle_id": "20260221-old-a3f2",
            "updated": True,
        }

        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()

        ingest_fn = _build_watch_ingest_fn(
            pipelines={str(watch_dir): pipeline},
            kb_entries={str(watch_dir): MagicMock(path=tmp_path / "kb")},
        )

        file_path = watch_dir / "test.jsonl"
        result = ingest_fn(file_path)

        assert result == {"bundle_id": "20260221-old-a3f2", "updated": True}
        # stable_id 기반: force 없이 일반 ingest
        pipeline.ingest_file.assert_called_once_with(file_path)

    def test_new_file_via_ingest_fn(self, tmp_path):
        from pkb.cli import _build_watch_ingest_fn

        pipeline = MagicMock()
        pipeline.ingest_file.return_value = {"bundle_id": "20260221-new-a3f2"}

        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()

        ingest_fn = _build_watch_ingest_fn(
            pipelines={str(watch_dir): pipeline},
            kb_entries={str(watch_dir): MagicMock(path=tmp_path / "kb")},
        )

        file_path = watch_dir / "test.jsonl"
        result = ingest_fn(file_path)

        assert result == {"bundle_id": "20260221-new-a3f2"}
        pipeline.ingest_file.assert_called_once_with(file_path)

    def test_unknown_parent_returns_none(self, tmp_path):
        from pkb.cli import _build_watch_ingest_fn

        ingest_fn = _build_watch_ingest_fn(
            pipelines={},
            kb_entries={},
        )

        result = ingest_fn(tmp_path / "unknown" / "test.jsonl")
        assert result is None

    def test_symlinked_path_resolves_correctly(self, tmp_path):
        """symlink 경로에서도 pipeline이 올바르게 매칭되어야 함 (macOS /tmp 등)."""
        from pkb.cli import _build_watch_ingest_fn

        pipeline = MagicMock()
        pipeline.ingest_file.return_value = {"bundle_id": "20260222-test-a1b2"}

        # 실제 inbox 디렉토리 생성
        real_inbox = tmp_path / "real_inbox"
        real_inbox.mkdir()

        # symlink 생성: link_inbox -> real_inbox
        link_inbox = tmp_path / "link_inbox"
        link_inbox.symlink_to(real_inbox)

        # pipeline은 symlink 경로(unresolved)로 등록
        ingest_fn = _build_watch_ingest_fn(
            pipelines={str(link_inbox): pipeline},
            kb_entries={str(link_inbox): MagicMock(path=tmp_path / "kb")},
        )

        # watchdog은 resolved 경로로 파일을 전달 (macOS /tmp -> /private/tmp)
        resolved_file = real_inbox / "test.jsonl"
        result = ingest_fn(resolved_file)

        # symlink 해석 후 정상적으로 ingest되어야 함
        assert result == {"bundle_id": "20260222-test-a1b2"}
        pipeline.ingest_file.assert_called_once_with(resolved_file)

    def test_subdirectory_file_matches_pipeline(self, tmp_path):
        """서브디렉토리 파일이 올바른 pipeline에 매칭되어야 함."""
        from pkb.cli import _build_watch_ingest_fn

        pipeline = MagicMock()
        pipeline.ingest_file.return_value = {"bundle_id": "20260222-sub-a1b2"}

        watch_dir = tmp_path / "inbox"
        sub = watch_dir / "PKB"
        sub.mkdir(parents=True)

        kb_entry = MagicMock()
        kb_entry.path = tmp_path / "kb"
        kb_entry.get_watch_dir.return_value = watch_dir

        ingest_fn = _build_watch_ingest_fn(
            pipelines={str(watch_dir): pipeline},
            kb_entries={str(watch_dir): kb_entry},
        )

        file_path = sub / "chatgpt.md"
        result = ingest_fn(file_path)

        assert result == {"bundle_id": "20260222-sub-a1b2"}
        pipeline.ingest_file.assert_called_once_with(file_path)


class TestWatchMoveToDone:
    """watch 콜백에서 성공 후 move_to_done 호출 검증."""

    def test_callback_moves_to_done_on_success(self, tmp_path):
        """_build_watch_callback: ingest 성공 시 move_to_done 호출."""
        from pkb.cli import _build_watch_callback

        pipeline = MagicMock()
        pipeline.ingest_file.return_value = {"bundle_id": "20260221-new-a3f2"}

        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()
        jsonl = watch_dir / "test.jsonl"
        jsonl.write_text("{}")

        kb_entry = MagicMock()
        kb_entry.path = tmp_path / "kb"
        kb_entry.get_watch_dir.return_value = watch_dir

        callback = _build_watch_callback(
            pipelines={str(watch_dir): pipeline},
            kb_entries={str(watch_dir): kb_entry},
        )

        callback(jsonl)

        # 파일이 .done/으로 이동되었어야 함
        assert not jsonl.exists()
        assert (watch_dir / ".done" / "test.jsonl").exists()

    def test_ingest_fn_moves_to_done_on_success(self, tmp_path):
        """_build_watch_ingest_fn: ingest 성공 시 move_to_done 호출."""
        from pkb.cli import _build_watch_ingest_fn

        pipeline = MagicMock()
        pipeline.ingest_file.return_value = {"bundle_id": "20260221-new-a3f2"}

        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()
        jsonl = watch_dir / "test.jsonl"
        jsonl.write_text("{}")

        kb_entry = MagicMock()
        kb_entry.path = tmp_path / "kb"
        kb_entry.get_watch_dir.return_value = watch_dir

        ingest_fn = _build_watch_ingest_fn(
            pipelines={str(watch_dir): pipeline},
            kb_entries={str(watch_dir): kb_entry},
        )

        ingest_fn(jsonl)

        # 파일이 .done/으로 이동되었어야 함
        assert not jsonl.exists()
        assert (watch_dir / ".done" / "test.jsonl").exists()

    def test_callback_moves_subdirectory_file_to_done(self, tmp_path):
        """_build_watch_callback: 서브디렉토리 파일 .done/ 이동."""
        from pkb.cli import _build_watch_callback

        pipeline = MagicMock()
        pipeline.ingest_file.return_value = {"bundle_id": "20260222-sub-a1b2"}

        watch_dir = tmp_path / "inbox"
        sub = watch_dir / "PKB"
        sub.mkdir(parents=True)
        md_file = sub / "chatgpt.md"
        md_file.write_text("content")

        kb_entry = MagicMock()
        kb_entry.path = tmp_path / "kb"
        kb_entry.get_watch_dir.return_value = watch_dir

        callback = _build_watch_callback(
            pipelines={str(watch_dir): pipeline},
            kb_entries={str(watch_dir): kb_entry},
        )

        callback(md_file)

        assert not md_file.exists()
        assert (watch_dir / ".done" / "PKB" / "chatgpt.md").exists()

    def test_ingest_fn_moves_subdirectory_file_to_done(self, tmp_path):
        """_build_watch_ingest_fn: 서브디렉토리 파일 .done/ 이동."""
        from pkb.cli import _build_watch_ingest_fn

        pipeline = MagicMock()
        pipeline.ingest_file.return_value = {"bundle_id": "20260222-sub-a1b2"}

        watch_dir = tmp_path / "inbox"
        sub = watch_dir / "PKB"
        sub.mkdir(parents=True)
        md_file = sub / "claude.md"
        md_file.write_text("content")

        kb_entry = MagicMock()
        kb_entry.path = tmp_path / "kb"
        kb_entry.get_watch_dir.return_value = watch_dir

        ingest_fn = _build_watch_ingest_fn(
            pipelines={str(watch_dir): pipeline},
            kb_entries={str(watch_dir): kb_entry},
        )

        ingest_fn(md_file)

        assert not md_file.exists()
        assert (watch_dir / ".done" / "PKB" / "claude.md").exists()


class TestSkipMovesToDone:
    """Bug fix: SKIP(중복) 파일도 .done/으로 이동해야 함."""

    def test_watch_ingest_fn_moves_skip_to_done(self, tmp_path):
        """_build_watch_ingest_fn: SKIP 시에도 move_to_done 호출."""
        from pkb.cli import _build_watch_ingest_fn

        pipeline = MagicMock()
        pipeline.ingest_file.return_value = None  # SKIP (duplicate)

        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()
        jsonl = watch_dir / "test.jsonl"
        jsonl.write_text("{}")

        kb_entry = MagicMock()
        kb_entry.path = tmp_path / "kb"
        kb_entry.get_watch_dir.return_value = watch_dir

        ingest_fn = _build_watch_ingest_fn(
            pipelines={str(watch_dir): pipeline},
            kb_entries={str(watch_dir): kb_entry},
        )

        ingest_fn(jsonl)

        # SKIP이어도 파일이 .done/으로 이동되어야 함
        assert not jsonl.exists()
        assert (watch_dir / ".done" / "test.jsonl").exists()

    def test_watch_callback_moves_skip_to_done(self, tmp_path):
        """_build_watch_callback: SKIP 시에도 move_to_done 호출."""
        from pkb.cli import _build_watch_callback

        pipeline = MagicMock()
        pipeline.ingest_file.return_value = None  # SKIP (duplicate)

        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()
        jsonl = watch_dir / "test.jsonl"
        jsonl.write_text("{}")

        kb_entry = MagicMock()
        kb_entry.path = tmp_path / "kb"
        kb_entry.get_watch_dir.return_value = watch_dir

        callback = _build_watch_callback(
            pipelines={str(watch_dir): pipeline},
            kb_entries={str(watch_dir): kb_entry},
        )

        callback(jsonl)

        # SKIP이어도 파일이 .done/으로 이동되어야 함
        assert not jsonl.exists()
        assert (watch_dir / ".done" / "test.jsonl").exists()

    def test_sequential_batch_moves_skip_to_done(self, tmp_path):
        """BatchProcessor sequential 모드: SKIP 시에도 move_to_done 호출."""
        mock_pipeline = MagicMock()
        mock_pipeline.ingest_file.return_value = None  # SKIP

        inbox = tmp_path / "inbox"
        inbox.mkdir()
        jsonl = inbox / "test.jsonl"
        jsonl.write_text("{}")

        processor = BatchProcessor(
            pipeline=mock_pipeline,
            checkpoint_path=tmp_path / "checkpoint.yaml",
            watch_dir=inbox,
        )
        processor.process(inbox)

        # SKIP이어도 파일이 .done/으로 이동되어야 함
        assert not jsonl.exists()
        assert (inbox / ".done" / "test.jsonl").exists()

    def test_ingest_command_moves_skip_to_done(self, tmp_path):
        """CLI ingest 커맨드: SKIP 시에도 move_to_done 호출."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        jsonl = inbox / "test.jsonl"
        jsonl.write_text("{}")

        # mock pipeline.ingest_file returns None (SKIP)
        with (
            patch("pkb.config.get_pkb_home") as mock_home,
            patch("pkb.config.load_config") as mock_config,
            patch("pkb.db.postgres.BundleRepository"),
            patch("pkb.config.build_chunk_store"),
            patch("pkb.config.build_llm_router"),
            patch("pkb.generator.meta_gen.MetaGenerator"),
            patch("pkb.ingest.IngestPipeline") as mock_pipeline_cls,
            patch("pkb.vocab.loader.load_domains") as mock_domains,
            patch("pkb.vocab.loader.load_topics") as mock_topics,
        ):
            mock_home.return_value = tmp_path / ".pkb"
            (tmp_path / ".pkb").mkdir()
            config = MagicMock()
            kb_entry = MagicMock()
            kb_entry.name = "test"
            kb_entry.path = tmp_path / "kb"
            kb_entry.get_watch_dir.return_value = inbox
            config.knowledge_bases = [kb_entry]
            mock_config.return_value = config
            mock_domains.return_value = MagicMock(get_ids=MagicMock(return_value=[]))
            mock_topics.return_value = MagicMock(get_approved_canonicals=MagicMock(return_value=[]))

            pipeline_inst = MagicMock()
            pipeline_inst.ingest_file.return_value = None  # SKIP
            mock_pipeline_cls.return_value = pipeline_inst

            runner = CliRunner()
            result = runner.invoke(cli, ["ingest", str(jsonl), "--kb", "test"])
            assert result.exit_code == 0, result.output

        # SKIP이어도 파일이 .done/으로 이동되어야 함
        assert not jsonl.exists()
        assert (inbox / ".done" / "test.jsonl").exists()


class TestInitialScan:
    """_initial_scan() 시작 시 기존 파일 스캔 테스트."""

    @pytest.mark.asyncio
    async def test_finds_flat_files(self, tmp_path):
        """inbox 직하 파일을 EventCollector에 넣음."""
        from pkb.cli import _initial_scan
        from pkb.engine import EventCollector
        from pkb.models.config import ConcurrencyConfig

        inbox = tmp_path / "inbox"
        inbox.mkdir()
        (inbox / "test.jsonl").write_text("data")
        (inbox / "claude.md").write_text("data")

        config = ConcurrencyConfig(max_queue_size=100, batch_window=0.1, max_batch_size=10)
        collector = EventCollector(config)

        count = await _initial_scan([inbox], collector)
        assert count == 2

        batch = await collector.drain_batch()
        assert len(batch) == 2

    @pytest.mark.asyncio
    async def test_finds_subdirectory_files(self, tmp_path):
        """서브디렉토리 파일도 스캔."""
        from pkb.cli import _initial_scan
        from pkb.engine import EventCollector
        from pkb.models.config import ConcurrencyConfig

        inbox = tmp_path / "inbox"
        sub = inbox / "PKB"
        sub.mkdir(parents=True)
        (sub / "chatgpt.md").write_text("data")
        (sub / "claude.md").write_text("data")

        config = ConcurrencyConfig(max_queue_size=100, batch_window=0.1, max_batch_size=10)
        collector = EventCollector(config)

        count = await _initial_scan([inbox], collector)
        assert count == 2

    @pytest.mark.asyncio
    async def test_excludes_done_files(self, tmp_path):
        """.done/ 내부 파일은 제외."""
        from pkb.cli import _initial_scan
        from pkb.engine import EventCollector
        from pkb.models.config import ConcurrencyConfig

        inbox = tmp_path / "inbox"
        inbox.mkdir()
        (inbox / "active.jsonl").write_text("data")
        done = inbox / ".done"
        done.mkdir()
        (done / "old.jsonl").write_text("data")

        config = ConcurrencyConfig(max_queue_size=100, batch_window=0.1, max_batch_size=10)
        collector = EventCollector(config)

        count = await _initial_scan([inbox], collector)
        assert count == 1

    @pytest.mark.asyncio
    async def test_empty_inbox(self, tmp_path):
        """빈 inbox는 0."""
        from pkb.cli import _initial_scan
        from pkb.engine import EventCollector
        from pkb.models.config import ConcurrencyConfig

        inbox = tmp_path / "inbox"
        inbox.mkdir()

        config = ConcurrencyConfig(max_queue_size=100, batch_window=0.1, max_batch_size=10)
        collector = EventCollector(config)

        count = await _initial_scan([inbox], collector)
        assert count == 0


class TestIngestMoveToDone:
    """ingest 명령에서 성공 후 move_to_done 호출 검증."""

    @patch("pkb.ingest.move_to_done")
    @patch("pkb.ingest.IngestPipeline")
    @patch("pkb.config.build_llm_router")
    @patch("pkb.generator.meta_gen.MetaGenerator")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.vocab.loader.load_topics")
    @patch("pkb.vocab.loader.load_domains")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_ingest_calls_move_to_done_on_success(
        self, mock_home, mock_load, mock_domains, mock_topics,
        mock_repo_cls, mock_store_cls, mock_meta_cls, mock_build_router,
        mock_pipeline_cls, mock_move, runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        (tmp_path / "vocab").mkdir()
        mock_config = MagicMock()
        kb_entry = MagicMock()
        kb_entry.name = "personal"
        kb_entry.path = tmp_path / "kb"
        watch_dir = tmp_path / "inbox"
        kb_entry.get_watch_dir.return_value = watch_dir
        mock_config.knowledge_bases = [kb_entry]
        mock_load.return_value = mock_config
        mock_domains.return_value = MagicMock(get_ids=MagicMock(return_value={"dev"}))
        mock_topics.return_value = MagicMock(
            get_approved_canonicals=MagicMock(return_value={"python"})
        )

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_file.return_value = {"bundle_id": "20260101-test-abc1"}
        mock_pipeline_cls.return_value = mock_pipeline

        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text('{"_meta":true}\n')

        result = runner.invoke(cli, ["ingest", str(jsonl), "--kb", "personal"])
        assert result.exit_code == 0, result.output
        mock_move.assert_called_once_with(Path(str(jsonl)), watch_dir, dry_run=False)


class TestBuildLLMRouterWiring:
    """Verify CLI commands use build_llm_router for LLM initialization."""

    @patch("pkb.ingest.IngestPipeline")
    @patch("pkb.config.build_llm_router")
    @patch("pkb.generator.meta_gen.MetaGenerator")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.vocab.loader.load_topics")
    @patch("pkb.vocab.loader.load_domains")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_ingest_uses_build_llm_router(
        self, mock_home, mock_load, mock_domains, mock_topics,
        mock_repo_cls, mock_store_cls, mock_meta_cls, mock_build_router,
        mock_pipeline_cls, runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        (tmp_path / "vocab").mkdir()
        mock_config = MagicMock()
        kb_entry = MagicMock()
        kb_entry.name = "personal"
        kb_entry.path = tmp_path / "kb"
        mock_config.knowledge_bases = [kb_entry]
        mock_load.return_value = mock_config
        mock_domains.return_value = MagicMock(get_ids=MagicMock(return_value={"dev"}))
        mock_topics.return_value = MagicMock(
            get_approved_canonicals=MagicMock(return_value={"python"})
        )

        mock_router = MagicMock()
        mock_build_router.return_value = mock_router

        # Create a dummy JSONL file
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text('{"_meta":true,"platform":"claude","url":"","exported_at":"2026-01-01T00:00:00Z","title":"test"}\n')

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_file.return_value = {"bundle_id": "20260101-test-abc1"}
        mock_pipeline_cls.return_value = mock_pipeline

        result = runner.invoke(cli, ["ingest", str(jsonl), "--kb", "personal"])
        assert result.exit_code == 0, result.output
        mock_build_router.assert_called_once_with(mock_config)


class TestPeriodicRetryScan:
    """Bug 5: _periodic_retry_scan() 실패 파일 주기적 재시도."""

    @pytest.mark.asyncio
    async def test_rescans_after_interval(self, tmp_path):
        """interval 경과 후 inbox 파일이 다시 큐에 들어감."""
        import asyncio

        from pkb.cli import _periodic_retry_scan
        from pkb.engine import EventCollector
        from pkb.models.config import ConcurrencyConfig

        inbox = tmp_path / "inbox"
        inbox.mkdir()
        (inbox / "failed.jsonl").write_text("data")

        config = ConcurrencyConfig(max_queue_size=100, batch_window=0.1, max_batch_size=10)
        collector = EventCollector(config)
        shutdown = asyncio.Event()

        # Short interval for testing (0.2 seconds)
        task = asyncio.create_task(
            _periodic_retry_scan([inbox], collector, shutdown, interval_seconds=0.2)
        )

        # Wait for one rescan cycle
        await asyncio.sleep(0.4)
        shutdown.set()
        await task

        batch = await collector.drain_batch()
        names = {p.name for p in batch}
        assert "failed.jsonl" in names

    @pytest.mark.asyncio
    async def test_excludes_done_files(self, tmp_path):
        """.done/ 파일은 재큐하지 않음."""
        import asyncio

        from pkb.cli import _periodic_retry_scan
        from pkb.engine import EventCollector
        from pkb.models.config import ConcurrencyConfig

        inbox = tmp_path / "inbox"
        inbox.mkdir()
        done = inbox / ".done"
        done.mkdir()
        (done / "old.jsonl").write_text("data")
        (inbox / "pending.jsonl").write_text("data")

        config = ConcurrencyConfig(max_queue_size=100, batch_window=0.1, max_batch_size=10)
        collector = EventCollector(config)
        shutdown = asyncio.Event()

        task = asyncio.create_task(
            _periodic_retry_scan([inbox], collector, shutdown, interval_seconds=0.2)
        )

        await asyncio.sleep(0.4)
        shutdown.set()
        await task

        batch = await collector.drain_batch()
        names = {p.name for p in batch}
        assert "pending.jsonl" in names
        assert "old.jsonl" not in names

    @pytest.mark.asyncio
    async def test_shutdown_stops_immediately(self, tmp_path):
        """shutdown_event 설정 시 즉시 종료."""
        import asyncio

        from pkb.cli import _periodic_retry_scan
        from pkb.engine import EventCollector
        from pkb.models.config import ConcurrencyConfig

        inbox = tmp_path / "inbox"
        inbox.mkdir()

        config = ConcurrencyConfig(max_queue_size=100, batch_window=0.1, max_batch_size=10)
        collector = EventCollector(config)
        shutdown = asyncio.Event()
        shutdown.set()  # Already shutdown

        # Should return almost immediately
        await asyncio.wait_for(
            _periodic_retry_scan([inbox], collector, shutdown, interval_seconds=300),
            timeout=1.0,
        )


class TestRelateCommand:
    @patch("pkb.relations.RelationBuilder")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_relate_scan(
        self, mock_home, mock_load, mock_repo_cls, mock_store_cls, mock_builder_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        mock_builder = MagicMock()
        mock_builder.scan.return_value = {"scanned": 10, "new_relations": 5}
        mock_builder_cls.return_value = mock_builder

        result = runner.invoke(cli, ["relate", "scan"])
        assert result.exit_code == 0, result.output
        assert "10" in result.output
        assert "5" in result.output

    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_relate_list(
        self, mock_home, mock_load, mock_repo_cls, runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_load.return_value = MagicMock()
        mock_repo = MagicMock()
        mock_repo.list_all_relations.return_value = [
            {
                "id": 1,
                "source_bundle_id": "20260101-a-abc1",
                "target_bundle_id": "20260101-b-def2",
                "relation_type": "similar",
                "score": 0.85,
                "created_at": None,
            },
        ]
        mock_repo_cls.return_value = mock_repo

        result = runner.invoke(cli, ["relate", "list"])
        assert result.exit_code == 0, result.output
        assert "20260101-a-abc1" in result.output

    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_relate_show(
        self, mock_home, mock_load, mock_repo_cls, runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_load.return_value = MagicMock()
        mock_repo = MagicMock()
        mock_repo.list_relations.return_value = [
            {
                "id": 1,
                "source_bundle_id": "20260101-a-abc1",
                "target_bundle_id": "20260101-b-def2",
                "relation_type": "similar",
                "score": 0.85,
                "created_at": None,
            },
        ]
        mock_repo_cls.return_value = mock_repo

        result = runner.invoke(cli, ["relate", "show", "20260101-a-abc1"])
        assert result.exit_code == 0, result.output
        assert "20260101-b-def2" in result.output


class TestDigestCommand:
    def test_digest_help_exists(self, runner):
        result = runner.invoke(cli, ["digest", "--help"])
        assert result.exit_code == 0
        assert "topic" in result.output.lower() or "domain" in result.output.lower()

    def test_digest_requires_topic_or_domain(self, runner):
        result = runner.invoke(cli, ["digest"])
        assert result.exit_code != 0
        assert "topic" in result.output.lower() or "domain" in result.output.lower()


class TestStatsCommand:
    def test_stats_help(self, runner):
        result = runner.invoke(cli, ["stats", "--help"])
        assert result.exit_code == 0
        assert "--kb" in result.output
        assert "--json" in result.output

    @patch("pkb.analytics.AnalyticsEngine")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_stats_overview(
        self, mock_home, mock_load, mock_repo_cls, mock_engine_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_load.return_value = MagicMock()
        mock_engine = MagicMock()
        mock_engine.overview.return_value = {
            "total_bundles": 42,
            "total_relations": 10,
            "domain_count": 5,
            "topic_count": 20,
        }
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(cli, ["stats"])
        assert result.exit_code == 0, result.output
        assert "42" in result.output

    @patch("pkb.analytics.AnalyticsEngine")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_stats_json_output(
        self, mock_home, mock_load, mock_repo_cls, mock_engine_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_load.return_value = MagicMock()
        mock_engine = MagicMock()
        mock_engine.overview.return_value = {
            "total_bundles": 10,
            "total_relations": 3,
            "domain_count": 2,
            "topic_count": 5,
        }
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(cli, ["stats", "--json"])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert parsed["total_bundles"] == 10

    @patch("pkb.analytics.AnalyticsEngine")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_stats_domain_detail(
        self, mock_home, mock_load, mock_repo_cls, mock_engine_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_load.return_value = MagicMock()
        mock_engine = MagicMock()
        mock_engine.domain_distribution.return_value = [
            {"domain": "dev", "count": 15},
        ]
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(cli, ["stats", "--domain"])
        assert result.exit_code == 0, result.output
        assert "dev" in result.output
        assert "15" in result.output


class TestReportCommand:
    def test_report_help(self, runner):
        result = runner.invoke(cli, ["report", "--help"])
        assert result.exit_code == 0
        assert "--period" in result.output

    @patch("pkb.report.ReportGenerator")
    @patch("pkb.analytics.AnalyticsEngine")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_report_weekly(
        self, mock_home, mock_load, mock_repo_cls, mock_engine_cls,
        mock_gen_cls, runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_load.return_value = MagicMock()
        mock_gen = MagicMock()
        mock_gen.weekly.return_value = "# Weekly Report Content"
        mock_gen_cls.return_value = mock_gen

        result = runner.invoke(cli, ["report"])
        assert result.exit_code == 0, result.output
        assert "Weekly Report Content" in result.output

    @patch("pkb.report.ReportGenerator")
    @patch("pkb.analytics.AnalyticsEngine")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_report_monthly(
        self, mock_home, mock_load, mock_repo_cls, mock_engine_cls,
        mock_gen_cls, runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_load.return_value = MagicMock()
        mock_gen = MagicMock()
        mock_gen.monthly.return_value = "# Monthly Report Content"
        mock_gen_cls.return_value = mock_gen

        result = runner.invoke(cli, ["report", "--period", "monthly"])
        assert result.exit_code == 0, result.output
        assert "Monthly Report Content" in result.output

    @patch("pkb.report.ReportGenerator")
    @patch("pkb.analytics.AnalyticsEngine")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_report_save_to_file(
        self, mock_home, mock_load, mock_repo_cls, mock_engine_cls,
        mock_gen_cls, runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_load.return_value = MagicMock()
        mock_gen = MagicMock()
        mock_gen.weekly.return_value = "# Report"
        mock_gen_cls.return_value = mock_gen

        outfile = tmp_path / "report.md"
        result = runner.invoke(cli, ["report", "-o", str(outfile)])
        assert result.exit_code == 0, result.output
        assert outfile.exists()
        assert outfile.read_text() == "# Report"


class TestIngestPostIngestWiring:
    """ingest 명령에서 PostIngestProcessor 생성 및 IngestPipeline 전달 검증."""

    @patch("pkb.post_ingest.PostIngestProcessor")
    @patch("pkb.ingest.move_to_done")
    @patch("pkb.ingest.IngestPipeline")
    @patch("pkb.config.build_llm_router")
    @patch("pkb.generator.meta_gen.MetaGenerator")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.vocab.loader.load_topics")
    @patch("pkb.vocab.loader.load_domains")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_ingest_creates_post_ingest_processor(
        self, mock_home, mock_load, mock_domains, mock_topics,
        mock_repo_cls, mock_store_cls, mock_meta_cls, mock_build_router,
        mock_pipeline_cls, mock_move, mock_post_ingest_cls, runner, tmp_path,
    ):
        """ingest 명령이 PostIngestProcessor를 생성하고 IngestPipeline에 전달한다."""
        mock_home.return_value = tmp_path
        (tmp_path / "vocab").mkdir()
        mock_config = MagicMock()
        kb_entry = MagicMock()
        kb_entry.name = "personal"
        kb_entry.path = tmp_path / "kb"
        kb_entry.get_watch_dir.return_value = tmp_path / "inbox"
        mock_config.knowledge_bases = [kb_entry]
        mock_load.return_value = mock_config
        mock_domains.return_value = MagicMock(get_ids=MagicMock(return_value={"dev"}))
        mock_topics.return_value = MagicMock(
            get_approved_canonicals=MagicMock(return_value={"python"})
        )

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_file.return_value = {"bundle_id": "20260101-test-abc1"}
        mock_pipeline_cls.return_value = mock_pipeline

        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text('{"_meta":true}\n')

        result = runner.invoke(cli, ["ingest", str(jsonl), "--kb", "personal"])
        assert result.exit_code == 0, result.output

        # PostIngestProcessor가 생성되었는지 확인
        mock_post_ingest_cls.assert_called_once()
        call_kwargs = mock_post_ingest_cls.call_args[1]
        assert call_kwargs["config"] == mock_config.post_ingest
        assert call_kwargs["relation_config"] == mock_config.relations
        assert call_kwargs["dedup_config"] == mock_config.dedup
        assert call_kwargs["gap_threshold"] == mock_config.scheduler.gap_threshold

        # IngestPipeline에 post_ingest가 전달되었는지 확인
        pipeline_kwargs = mock_pipeline_cls.call_args[1]
        assert pipeline_kwargs["post_ingest"] == mock_post_ingest_cls.return_value

    @patch("pkb.ingest.IngestPipeline")
    @patch("pkb.config.build_llm_router")
    @patch("pkb.generator.meta_gen.MetaGenerator")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.vocab.loader.load_topics")
    @patch("pkb.vocab.loader.load_domains")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_ingest_dry_run_skips_post_ingest(
        self, mock_home, mock_load, mock_domains, mock_topics,
        mock_repo_cls, mock_store_cls, mock_meta_cls, mock_build_router,
        mock_pipeline_cls, runner, tmp_path,
    ):
        """dry-run 모드에서는 PostIngestProcessor를 생성하지 않는다."""
        mock_home.return_value = tmp_path
        (tmp_path / "vocab").mkdir()
        mock_config = MagicMock()
        kb_entry = MagicMock()
        kb_entry.name = "personal"
        kb_entry.path = tmp_path / "kb"
        kb_entry.get_watch_dir.return_value = tmp_path / "inbox"
        mock_config.knowledge_bases = [kb_entry]
        mock_load.return_value = mock_config
        mock_domains.return_value = MagicMock(get_ids=MagicMock(return_value={"dev"}))
        mock_topics.return_value = MagicMock(
            get_approved_canonicals=MagicMock(return_value={"python"})
        )

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_file.return_value = {"bundle_id": "20260101-test-abc1"}
        mock_pipeline_cls.return_value = mock_pipeline

        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text('{"_meta":true}\n')

        result = runner.invoke(
            cli, ["ingest", str(jsonl), "--kb", "personal", "--dry-run"],
        )
        assert result.exit_code == 0, result.output

        # IngestPipeline에 post_ingest=None이 전달되었는지 확인
        pipeline_kwargs = mock_pipeline_cls.call_args[1]
        assert pipeline_kwargs["post_ingest"] is None

    @patch("pkb.post_ingest.PostIngestProcessor")
    @patch("pkb.batch.BatchProcessor")
    @patch("pkb.ingest.IngestPipeline")
    @patch("pkb.config.build_llm_router")
    @patch("pkb.generator.meta_gen.MetaGenerator")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.vocab.loader.load_topics")
    @patch("pkb.vocab.loader.load_domains")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_batch_creates_post_ingest_processor(
        self, mock_home, mock_load, mock_domains, mock_topics,
        mock_repo_cls, mock_store_cls, mock_meta_cls, mock_build_router,
        mock_pipeline_cls, mock_batch_cls, mock_post_ingest_cls, runner, tmp_path,
    ):
        """batch 명령이 PostIngestProcessor를 생성하고 IngestPipeline에 전달한다."""
        mock_home.return_value = tmp_path
        (tmp_path / "vocab").mkdir()
        mock_config = MagicMock()
        kb_entry = MagicMock()
        kb_entry.name = "personal"
        kb_entry.path = tmp_path / "kb"
        kb_entry.get_watch_dir.return_value = tmp_path / "inbox"
        mock_config.knowledge_bases = [kb_entry]
        mock_load.return_value = mock_config
        mock_domains.return_value = MagicMock(get_ids=MagicMock(return_value={"dev"}))
        mock_topics.return_value = MagicMock(
            get_approved_canonicals=MagicMock(return_value={"python"})
        )

        mock_batch = MagicMock()
        mock_batch.process.return_value = {"success": 0, "skipped": 0, "errors": 0}
        mock_batch_cls.return_value = mock_batch

        source = tmp_path / "source"
        source.mkdir()
        (source / "test.jsonl").write_text('{"_meta":true}\n')

        result = runner.invoke(cli, ["batch", str(source), "--kb", "personal"])
        assert result.exit_code == 0, result.output

        # PostIngestProcessor가 생성되었는지 확인
        mock_post_ingest_cls.assert_called_once()

        # IngestPipeline에 post_ingest가 전달되었는지 확인
        pipeline_kwargs = mock_pipeline_cls.call_args[1]
        assert pipeline_kwargs["post_ingest"] == mock_post_ingest_cls.return_value


class TestWatchSchedulerWiring:
    """watch 명령에서 Scheduler 인스턴스 생성 검증."""

    @patch("pkb.scheduler.Scheduler")
    @patch("pkb.watcher.AsyncFileEventHandler")
    @patch("pkb.engine.EventCollector")
    @patch("pkb.engine.IngestEngine")
    @patch("pkb.ingest.IngestPipeline")
    @patch("pkb.config.build_llm_router")
    @patch("pkb.generator.meta_gen.MetaGenerator")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.vocab.loader.load_topics")
    @patch("pkb.vocab.loader.load_domains")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_watch_creates_scheduler(
        self, mock_home, mock_load, mock_domains, mock_topics,
        mock_repo_cls, mock_store_cls, mock_meta_cls, mock_build_router,
        mock_pipeline_cls, mock_engine_cls, mock_collector_cls,
        mock_handler_cls, mock_scheduler_cls, runner, tmp_path,
    ):
        """watch 명령이 Scheduler를 생성하고 due 상태를 로그한다."""
        mock_home.return_value = tmp_path
        (tmp_path / "vocab").mkdir()
        mock_config = MagicMock()
        kb_entry = MagicMock()
        kb_entry.name = "personal"
        kb_entry.path = tmp_path / "kb"
        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()
        kb_entry.get_watch_dir.return_value = watch_dir
        mock_config.knowledge_bases = [kb_entry]
        mock_load.return_value = mock_config
        mock_domains.return_value = MagicMock(get_ids=MagicMock(return_value={"dev"}))
        mock_topics.return_value = MagicMock(
            get_approved_canonicals=MagicMock(return_value={"python"})
        )

        mock_scheduler = MagicMock()
        mock_scheduler.is_weekly_digest_due.return_value = True
        mock_scheduler.is_monthly_report_due.return_value = False
        mock_scheduler_cls.return_value = mock_scheduler

        # Make asyncio.run raise KeyboardInterrupt to exit the watch loop
        with patch("asyncio.run", side_effect=KeyboardInterrupt):
            runner.invoke(cli, ["watch", "--kb", "personal"])

        # Scheduler가 생성되었는지 확인
        mock_scheduler_cls.assert_called_once()
        call_kwargs = mock_scheduler_cls.call_args[1]
        assert call_kwargs["config"] == mock_config.scheduler
        assert "scheduler_state.json" in str(call_kwargs["state_path"])
