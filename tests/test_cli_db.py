"""Tests for `pkb db` CLI commands."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from pkb.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_dsn(tmp_path, monkeypatch):
    """Mock PKB config loading to return a fake DSN."""
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
    return "postgresql://testuser:testpw@testhost:5432/testdb"


@pytest.fixture
def mock_reset_config(tmp_path, monkeypatch):
    """Mock PKB config with a KB for reset tests."""
    pkb_home = tmp_path / ".pkb"
    pkb_home.mkdir()
    monkeypatch.setenv("PKB_HOME", str(pkb_home))

    kb_path = tmp_path / "kb-personal"
    kb_path.mkdir()
    bundles_dir = kb_path / "bundles"
    bundles_dir.mkdir()
    # Create some fake bundle dirs
    (bundles_dir / "20260101-test-a1b2").mkdir()
    (bundles_dir / "20260102-test-c3d4").mkdir()
    inbox_done = kb_path / "inbox" / ".done"
    inbox_done.mkdir(parents=True)
    (inbox_done / "old.jsonl").write_text("fake")

    config_yaml = pkb_home / "config.yaml"
    config_yaml.write_text(
        f"knowledge_bases:\n"
        f"  - name: personal\n"
        f"    path: {kb_path}\n"
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
    return kb_path


class TestDbGroup:
    def test_db_help(self, runner):
        result = runner.invoke(cli, ["db", "--help"])
        assert result.exit_code == 0
        assert "upgrade" in result.output
        assert "downgrade" in result.output
        assert "current" in result.output
        assert "history" in result.output
        assert "stamp" in result.output
        assert "reset" in result.output


class TestDbUpgrade:
    @patch("pkb.db.migration_runner.run_upgrade")
    def test_upgrade_default_head(self, mock_upgrade, runner, mock_dsn):
        result = runner.invoke(cli, ["db", "upgrade"])
        assert result.exit_code == 0
        mock_upgrade.assert_called_once_with(mock_dsn, revision="head")

    @patch("pkb.db.migration_runner.run_upgrade")
    def test_upgrade_specific_revision(self, mock_upgrade, runner, mock_dsn):
        result = runner.invoke(cli, ["db", "upgrade", "--revision", "0001"])
        assert result.exit_code == 0
        mock_upgrade.assert_called_once_with(mock_dsn, revision="0001")


class TestDbDowngrade:
    def test_downgrade_requires_revision(self, runner, mock_dsn):
        result = runner.invoke(cli, ["db", "downgrade"])
        assert result.exit_code != 0

    @patch("pkb.db.migration_runner.run_downgrade")
    def test_downgrade_with_revision(self, mock_downgrade, runner, mock_dsn):
        result = runner.invoke(cli, ["db", "downgrade", "0001"])
        assert result.exit_code == 0
        mock_downgrade.assert_called_once_with(mock_dsn, revision="0001")


class TestDbCurrent:
    @patch("pkb.db.migration_runner.get_current")
    def test_current(self, mock_current, runner, mock_dsn):
        result = runner.invoke(cli, ["db", "current"])
        assert result.exit_code == 0
        mock_current.assert_called_once_with(mock_dsn)


class TestDbHistory:
    @patch("pkb.db.migration_runner.get_history")
    def test_history(self, mock_history, runner, mock_dsn):
        result = runner.invoke(cli, ["db", "history"])
        assert result.exit_code == 0
        mock_history.assert_called_once_with(mock_dsn)


class TestDbStamp:
    def test_stamp_requires_revision(self, runner, mock_dsn):
        result = runner.invoke(cli, ["db", "stamp"])
        assert result.exit_code != 0

    @patch("pkb.db.migration_runner.run_stamp")
    def test_stamp_with_revision(self, mock_stamp, runner, mock_dsn):
        result = runner.invoke(cli, ["db", "stamp", "head"])
        assert result.exit_code == 0
        mock_stamp.assert_called_once_with(mock_dsn, revision="head")


class TestDbReset:
    def test_reset_deletes_all(self, runner, mock_reset_config):
        """정상 삭제: DB + ChromaDB + FS 초기화."""
        kb_path = mock_reset_config
        with patch("pkb.db.postgres.psycopg.connect", return_value=MagicMock()):
            with patch(
                "pkb.db.postgres.BundleRepository.delete_by_kb", return_value=2,
            ):
                with patch(
                    "pkb.db.chromadb_client.chromadb.HttpClient",
                    return_value=MagicMock(),
                ):
                    with patch(
                        "pkb.db.chromadb_client.ChunkStore.delete_by_kb",
                    ):
                        result = runner.invoke(
                            cli, ["db", "reset", "--kb", "personal"],
                            input="personal\n",
                        )

        assert result.exit_code == 0
        assert "2" in result.output
        # bundles dir should be recreated (empty)
        assert (kb_path / "bundles").exists()
        assert not (kb_path / "bundles" / "20260101-test-a1b2").exists()
        # .done should be cleared
        assert not (kb_path / "inbox" / ".done" / "old.jsonl").exists()

    def test_reset_wrong_confirmation_aborts(self, runner, mock_reset_config):
        """잘못된 확인 입력 시 중단."""
        result = runner.invoke(
            cli, ["db", "reset", "--kb", "personal"],
            input="wrong-name\n",
        )
        assert result.exit_code == 0
        assert "Aborted" in result.output

    def test_reset_nonexistent_kb(self, runner, mock_dsn):
        """존재하지 않는 KB."""
        result = runner.invoke(
            cli, ["db", "reset", "--kb", "nonexistent"],
        )
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_reset_db_connection_failure(self, runner, mock_reset_config):
        """DB 연결 실패."""
        with patch(
            "pkb.db.postgres.psycopg.connect",
            side_effect=Exception("Connection refused"),
        ):
            result = runner.invoke(
                cli, ["db", "reset", "--kb", "personal"],
                input="personal\n",
            )

        assert result.exit_code != 0
        assert "Database connection failed" in result.output

    def test_reset_help(self, runner):
        """help에 reset 표시."""
        result = runner.invoke(cli, ["db", "reset", "--help"])
        assert result.exit_code == 0
        assert "--kb" in result.output


class TestDbMigrateDomain:
    def test_migrate_domain_renames(self, runner, mock_dsn):
        """정상 도메인 이름 변경."""
        with patch("pkb.db.postgres.psycopg.connect", return_value=MagicMock()):
            with patch(
                "pkb.db.postgres.BundleRepository.rename_domain", return_value=5,
            ) as mock_rename:
                result = runner.invoke(
                    cli, ["db", "migrate-domain", "coding", "dev"],
                )
        assert result.exit_code == 0
        mock_rename.assert_called_once_with("coding", "dev")
        assert "5" in result.output

    def test_migrate_domain_no_matches(self, runner, mock_dsn):
        """매칭 없으면 0 표시."""
        with patch("pkb.db.postgres.psycopg.connect", return_value=MagicMock()):
            with patch(
                "pkb.db.postgres.BundleRepository.rename_domain", return_value=0,
            ):
                result = runner.invoke(
                    cli, ["db", "migrate-domain", "nonexistent", "dev"],
                )
        assert result.exit_code == 0
        assert "0" in result.output

    def test_migrate_domain_requires_args(self, runner, mock_dsn):
        """old, new 인자 필수."""
        result = runner.invoke(cli, ["db", "migrate-domain"])
        assert result.exit_code != 0

    def test_migrate_domain_db_error(self, runner, mock_dsn):
        """DB 연결 실패."""
        with patch(
            "pkb.db.postgres.psycopg.connect",
            side_effect=Exception("Connection refused"),
        ):
            result = runner.invoke(
                cli, ["db", "migrate-domain", "coding", "dev"],
            )
        assert result.exit_code != 0
