"""Tests for pkb reembed CLI command."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from pkb.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def _setup_mocks(mock_home, mock_load, tmp_path, kb_name="personal"):
    mock_home.return_value = tmp_path
    mock_config = MagicMock()
    kb_entry = MagicMock()
    kb_entry.name = kb_name
    kb_entry.path = tmp_path / "kb"
    mock_config.knowledge_bases = [kb_entry]
    mock_config.embedding.chunk_size = 512
    mock_config.embedding.chunk_overlap = 50
    mock_load.return_value = mock_config
    return mock_config, kb_entry


class TestReembedCommand:
    @patch("pkb.reembed.ReembedEngine")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_reembed_single_bundle(
        self, mock_home, mock_load, mock_repo_cls, mock_store_cls, mock_engine_cls,
        runner, tmp_path,
    ):
        _setup_mocks(mock_home, mock_load, tmp_path)
        mock_engine = MagicMock()
        mock_engine.reembed_bundle.return_value = {
            "bundle_id": "20260101-test-abc1",
            "status": "reembedded",
            "chunks": 5,
        }
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(
            cli, ["reembed", "20260101-test-abc1", "--kb", "personal"]
        )
        assert result.exit_code == 0, result.output
        assert "reembedded" in result.output.lower()
        mock_engine.reembed_bundle.assert_called_once_with("20260101-test-abc1")

    @patch("pkb.reembed.ReembedEngine")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_reembed_all(
        self, mock_home, mock_load, mock_repo_cls, mock_store_cls, mock_engine_cls,
        runner, tmp_path,
    ):
        _setup_mocks(mock_home, mock_load, tmp_path)
        mock_engine = MagicMock()
        mock_engine.reembed_all.return_value = {
            "total": 10, "reembedded": 9, "errors": 1,
        }
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(cli, ["reembed", "--all", "--kb", "personal"])
        assert result.exit_code == 0, result.output
        assert "9" in result.output
        mock_engine.reembed_all.assert_called_once()

    @patch("pkb.reembed.ReembedEngine")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_reembed_fresh(
        self, mock_home, mock_load, mock_repo_cls, mock_store_cls, mock_engine_cls,
        runner, tmp_path,
    ):
        _setup_mocks(mock_home, mock_load, tmp_path)
        mock_engine = MagicMock()
        mock_engine.reembed_collection_fresh.return_value = {
            "total": 5, "reembedded": 5, "errors": 0,
        }
        mock_engine_cls.return_value = mock_engine

        # --fresh requires confirmation: provide KB name via input
        result = runner.invoke(
            cli, ["reembed", "--all", "--fresh", "--kb", "personal"],
            input="personal\n",
        )
        assert result.exit_code == 0, result.output
        mock_engine.reembed_collection_fresh.assert_called_once()

    @patch("pkb.reembed.ReembedEngine")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_reembed_fresh_wrong_confirmation(
        self, mock_home, mock_load, mock_repo_cls, mock_store_cls, mock_engine_cls,
        runner, tmp_path,
    ):
        _setup_mocks(mock_home, mock_load, tmp_path)

        result = runner.invoke(
            cli, ["reembed", "--all", "--fresh", "--kb", "personal"],
            input="wrong\n",
        )
        assert result.exit_code != 0
        assert "abort" in result.output.lower() or "mismatch" in result.output.lower()

    def test_reembed_no_args_error(self, runner):
        """bundle_id도 --all도 없으면 에러."""
        result = runner.invoke(cli, ["reembed", "--kb", "personal"])
        assert result.exit_code != 0

    def test_reembed_fresh_without_all_error(self, runner):
        """--fresh는 --all 없이 사용 불가."""
        result = runner.invoke(
            cli, ["reembed", "some-bundle", "--fresh", "--kb", "personal"]
        )
        assert result.exit_code != 0

    @patch("pkb.reembed.ReembedEngine")
    @patch("pkb.config.build_chunk_store")
    @patch("pkb.db.postgres.BundleRepository")
    @patch("pkb.config.load_config")
    @patch("pkb.config.get_pkb_home")
    def test_reembed_unknown_kb_error(
        self, mock_home, mock_load, mock_repo_cls, mock_store_cls, mock_engine_cls,
        runner, tmp_path,
    ):
        mock_home.return_value = tmp_path
        mock_config = MagicMock()
        mock_config.knowledge_bases = []
        mock_load.return_value = mock_config

        result = runner.invoke(
            cli, ["reembed", "--all", "--kb", "nonexistent"]
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()
