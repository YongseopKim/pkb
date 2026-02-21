"""Tests for `pkb kb` CLI commands."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from pkb.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_kb_config(tmp_path, monkeypatch):
    """Mock PKB config with 2 KBs and a fake DB."""
    pkb_home = tmp_path / ".pkb"
    pkb_home.mkdir()
    monkeypatch.setenv("PKB_HOME", str(pkb_home))

    kb_path = tmp_path / "kb-personal"
    kb_path.mkdir()

    config_yaml = pkb_home / "config.yaml"
    config_yaml.write_text(
        f"knowledge_bases:\n"
        f"  - name: personal\n"
        f"    path: {kb_path}\n"
        f"  - name: work\n"
        f"    path: {tmp_path / 'kb-work'}\n"
        f"database:\n"
        f"  postgres:\n"
        f"    host: testhost\n"
        f"    port: 5432\n"
        f"    database: testdb\n"
        f"    username: testuser\n"
        f"    password: testpw\n"
        f"  chromadb:\n"
        f"    host: testhost\n"
        f"    port: 8000\n"
        f"    collection: pkb_chunks\n"
    )
    return pkb_home


class TestKbGroup:
    def test_kb_help(self, runner):
        result = runner.invoke(cli, ["kb", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output

    def test_invoke_without_command_runs_list(self, runner, mock_kb_config):
        """pkb kb (no subcommand) = pkb kb list."""
        mock_repo = MagicMock()
        mock_repo.count_by_kb.return_value = 3

        with patch("pkb.db.postgres.psycopg.connect", return_value=MagicMock()):
            with patch(
                "pkb.db.postgres.BundleRepository.count_by_kb",
                return_value=3,
            ):
                result = runner.invoke(cli, ["kb"])

        assert result.exit_code == 0
        assert "personal" in result.output
        assert "work" in result.output


class TestKbList:
    def test_list_with_counts(self, runner, mock_kb_config):
        """정상 출력: KB 이름 + 번들 수."""
        with patch("pkb.db.postgres.psycopg.connect", return_value=MagicMock()):
            with patch(
                "pkb.db.postgres.BundleRepository.count_by_kb",
                side_effect=[5, 12],
            ):
                result = runner.invoke(cli, ["kb", "list"])

        assert result.exit_code == 0
        assert "personal" in result.output
        assert "5" in result.output
        assert "work" in result.output
        assert "12" in result.output

    def test_list_db_connection_failure(self, runner, mock_kb_config):
        """DB 연결 실패 시 ?로 표시."""
        with patch(
            "pkb.db.postgres.psycopg.connect",
            side_effect=Exception("Connection refused"),
        ):
            result = runner.invoke(cli, ["kb", "list"])

        assert result.exit_code == 0
        assert "personal" in result.output
        assert "?" in result.output

    def test_list_no_kbs(self, runner, tmp_path, monkeypatch):
        """KB 없을 때."""
        pkb_home = tmp_path / ".pkb"
        pkb_home.mkdir()
        monkeypatch.setenv("PKB_HOME", str(pkb_home))

        config_yaml = pkb_home / "config.yaml"
        config_yaml.write_text(
            "knowledge_bases: []\n"
            "database:\n"
            "  postgres:\n"
            "    host: testhost\n"
            "    port: 5432\n"
            "    database: testdb\n"
            "    username: testuser\n"
            "    password: testpw\n"
        )

        result = runner.invoke(cli, ["kb", "list"])
        assert result.exit_code == 0
        assert "No knowledge bases" in result.output
