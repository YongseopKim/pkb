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
    @patch("pkb.db.migration_runner.get_table_schema")
    @patch("pkb.db.migration_runner.get_current")
    def test_current(self, mock_current, mock_schema, runner, mock_dsn):
        mock_schema.return_value = {}
        result = runner.invoke(cli, ["db", "current"])
        assert result.exit_code == 0
        mock_current.assert_called_once_with(mock_dsn)
        mock_schema.assert_called_once_with(mock_dsn)

    @patch("pkb.db.migration_runner.get_table_schema")
    @patch("pkb.db.migration_runner.get_current")
    def test_current_shows_table_columns(self, mock_current, mock_schema, runner, mock_dsn):
        mock_schema.return_value = {
            "bundles": [
                {"column": "id", "type": "text", "nullable": "NO", "default": None},
                {"column": "kb", "type": "text", "nullable": "NO", "default": None},
                {"column": "stable_id", "type": "text", "nullable": "NO", "default": None},
            ],
            "bundle_responses": [
                {"column": "id", "type": "integer", "nullable": "NO", "default": "nextval(...)"},
            ],
        }
        result = runner.invoke(cli, ["db", "current"])
        assert result.exit_code == 0
        assert "bundles" in result.output
        assert "stable_id" in result.output
        assert "bundle_responses" in result.output
        assert "3 columns" in result.output
        assert "1 column" in result.output

    @patch("pkb.db.migration_runner.get_table_schema")
    @patch("pkb.db.migration_runner.get_current")
    def test_current_empty_schema(self, mock_current, mock_schema, runner, mock_dsn):
        mock_schema.return_value = {}
        result = runner.invoke(cli, ["db", "current"])
        assert result.exit_code == 0
        assert "No tables" in result.output


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


@pytest.fixture
def mock_stable_id_config(tmp_path, monkeypatch):
    """Mock PKB config with a KB that has bundles with _raw files."""
    pkb_home = tmp_path / ".pkb"
    pkb_home.mkdir()
    monkeypatch.setenv("PKB_HOME", str(pkb_home))

    kb_path = tmp_path / "kb-personal"
    kb_path.mkdir()

    # Create bundle with _raw directory and a JSONL file
    bundle_dir = kb_path / "bundles" / "20260101-test-a1b2"
    raw_dir = bundle_dir / "_raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "claude.jsonl").write_text(
        '{"_meta":true,"platform":"claude","url":"https://claude.ai/chat/abc123",'
        '"exported_at":"2026-01-01T00:00:00Z","title":"Test"}\n'
        '{"role":"user","content":"Hello","timestamp":"2026-01-01T00:00:01Z"}\n'
        '{"role":"assistant","content":"Hi there","timestamp":"2026-01-01T00:00:02Z"}\n'
    )

    # Create bundle without _raw directory
    bundle_no_raw = kb_path / "bundles" / "20260102-no-raw-c3d4"
    bundle_no_raw.mkdir(parents=True)

    # Create bundle with empty _raw directory
    bundle_empty_raw = kb_path / "bundles" / "20260103-empty-e5f6"
    raw_empty = bundle_empty_raw / "_raw"
    raw_empty.mkdir(parents=True)

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
    )
    return kb_path


class TestDbMigrateStableId:
    def test_dry_run_shows_preview(self, runner, mock_stable_id_config):
        """--dry-run shows preview without DB writes."""
        mock_conn = MagicMock()

        with patch("pkb.db.postgres.psycopg.connect", return_value=mock_conn):
            with patch(
                "pkb.db.postgres.BundleRepository.list_all_bundle_ids",
                return_value=["20260101-test-a1b2"],
            ):
                with patch(
                    "pkb.db.postgres.BundleRepository.get_bundle_by_id",
                    return_value={"bundle_id": "20260101-test-a1b2", "kb": "personal"},
                ):
                    result = runner.invoke(
                        cli,
                        ["db", "migrate-stable-id", "--kb", "personal", "--dry-run"],
                    )

        assert result.exit_code == 0
        assert "dry run" in result.output.lower()
        assert "20260101-test-a1b2" in result.output
        assert "1 updated" in result.output
        # No DB write should happen in dry run
        mock_conn.execute.assert_not_called()

    def test_updates_stable_id_for_bundles(self, runner, mock_stable_id_config):
        """Recomputes stable_id from raw files and writes to DB."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("pkb.db.postgres.psycopg.connect", return_value=mock_conn):
            with patch(
                "pkb.db.postgres.BundleRepository.list_all_bundle_ids",
                return_value=["20260101-test-a1b2"],
            ):
                with patch(
                    "pkb.db.postgres.BundleRepository.get_bundle_by_id",
                    return_value={"bundle_id": "20260101-test-a1b2", "kb": "personal"},
                ):
                    result = runner.invoke(
                        cli,
                        ["db", "migrate-stable-id", "--kb", "personal"],
                    )

        assert result.exit_code == 0
        assert "OK" in result.output
        assert "1 updated" in result.output
        # Verify SQL UPDATE was called
        mock_conn.execute.assert_called()
        call_args = mock_conn.execute.call_args
        assert "UPDATE bundles SET stable_id" in call_args[0][0]

    def test_skips_bundles_without_raw_dir(self, runner, mock_stable_id_config):
        """Bundles without _raw/ directory are skipped."""
        mock_conn = MagicMock()

        with patch("pkb.db.postgres.psycopg.connect", return_value=mock_conn):
            with patch(
                "pkb.db.postgres.BundleRepository.list_all_bundle_ids",
                return_value=["20260102-no-raw-c3d4"],
            ):
                with patch(
                    "pkb.db.postgres.BundleRepository.get_bundle_by_id",
                    return_value={"bundle_id": "20260102-no-raw-c3d4", "kb": "personal"},
                ):
                    result = runner.invoke(
                        cli,
                        ["db", "migrate-stable-id", "--kb", "personal"],
                    )

        assert result.exit_code == 0
        assert "SKIP" in result.output
        assert "1 skipped" in result.output

    def test_skips_bundles_with_empty_raw_dir(self, runner, mock_stable_id_config):
        """Bundles with empty _raw/ directory are skipped."""
        mock_conn = MagicMock()

        with patch("pkb.db.postgres.psycopg.connect", return_value=mock_conn):
            with patch(
                "pkb.db.postgres.BundleRepository.list_all_bundle_ids",
                return_value=["20260103-empty-e5f6"],
            ):
                with patch(
                    "pkb.db.postgres.BundleRepository.get_bundle_by_id",
                    return_value={"bundle_id": "20260103-empty-e5f6", "kb": "personal"},
                ):
                    result = runner.invoke(
                        cli,
                        ["db", "migrate-stable-id", "--kb", "personal"],
                    )

        assert result.exit_code == 0
        assert "SKIP" in result.output
        assert "no raw files" in result.output.lower()

    def test_skips_bundles_with_unknown_kb(self, runner, mock_stable_id_config):
        """Bundles whose KB is not in config are skipped."""
        mock_conn = MagicMock()

        with patch("pkb.db.postgres.psycopg.connect", return_value=mock_conn):
            with patch(
                "pkb.db.postgres.BundleRepository.list_all_bundle_ids",
                return_value=["20260101-test-a1b2"],
            ):
                with patch(
                    "pkb.db.postgres.BundleRepository.get_bundle_by_id",
                    return_value={"bundle_id": "20260101-test-a1b2", "kb": "unknown-kb"},
                ):
                    result = runner.invoke(
                        cli,
                        ["db", "migrate-stable-id"],
                    )

        assert result.exit_code == 0
        assert "1 skipped" in result.output

    def test_handles_parse_error(self, runner, mock_stable_id_config):
        """Parse errors are counted and reported."""
        kb_path = mock_stable_id_config
        # Write invalid JSONL to trigger a parse error
        raw_dir = kb_path / "bundles" / "20260101-test-a1b2" / "_raw"
        (raw_dir / "claude.jsonl").write_text("not valid json\n")

        mock_conn = MagicMock()

        with patch("pkb.db.postgres.psycopg.connect", return_value=mock_conn):
            with patch(
                "pkb.db.postgres.BundleRepository.list_all_bundle_ids",
                return_value=["20260101-test-a1b2"],
            ):
                with patch(
                    "pkb.db.postgres.BundleRepository.get_bundle_by_id",
                    return_value={"bundle_id": "20260101-test-a1b2", "kb": "personal"},
                ):
                    result = runner.invoke(
                        cli,
                        ["db", "migrate-stable-id", "--kb", "personal"],
                    )

        assert result.exit_code == 0
        assert "ERROR" in result.output
        assert "1 error" in result.output.lower()

    def test_db_connection_failure(self, runner, mock_stable_id_config):
        """DB 연결 실패."""
        with patch(
            "pkb.db.postgres.psycopg.connect",
            side_effect=Exception("Connection refused"),
        ):
            result = runner.invoke(
                cli, ["db", "migrate-stable-id"],
            )
        assert result.exit_code != 0
        assert "Database connection failed" in result.output

    def test_help_shows_migrate_stable_id(self, runner):
        """help에 migrate-stable-id 표시."""
        result = runner.invoke(cli, ["db", "--help"])
        assert result.exit_code == 0
        assert "migrate-stable-id" in result.output
