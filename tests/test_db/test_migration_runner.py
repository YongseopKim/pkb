"""Tests for migration_runner module (mock-based)."""

from unittest.mock import MagicMock, patch

from pkb.db.migration_runner import (
    _ensure_psycopg3_dialect,
    _make_alembic_config,
    _should_stamp,
    get_current,
    get_history,
    get_table_schema,
    run_downgrade,
    run_stamp,
    run_upgrade,
)


class TestEnsurePsycopg3Dialect:
    def test_converts_plain_postgresql(self):
        assert _ensure_psycopg3_dialect("postgresql://u:p@h/db") == "postgresql+psycopg://u:p@h/db"

    def test_leaves_already_converted(self):
        dsn = "postgresql+psycopg://u:p@h/db"
        assert _ensure_psycopg3_dialect(dsn) == dsn

    def test_leaves_other_schemes(self):
        dsn = "sqlite:///test.db"
        assert _ensure_psycopg3_dialect(dsn) == dsn


class TestMakeAlembicConfig:
    def test_returns_config_object(self):
        config = _make_alembic_config("postgresql://user:pw@host/db")
        assert config is not None

    def test_sets_sqlalchemy_url_with_psycopg_dialect(self):
        dsn = "postgresql://user:pw@host/db"
        config = _make_alembic_config(dsn)
        assert config.get_main_option("sqlalchemy.url") == "postgresql+psycopg://user:pw@host/db"

    def test_sets_script_location(self):
        config = _make_alembic_config("postgresql://user:pw@host/db")
        script_loc = config.get_main_option("script_location")
        assert script_loc is not None
        assert "migrations" in script_loc


class TestShouldStamp:
    def test_no_bundles_table_returns_false(self):
        """Fresh DB (no tables at all) → should NOT stamp."""
        conn = MagicMock()
        conn.execute.return_value.scalar.return_value = False
        assert _should_stamp(conn) is False

    def test_bundles_exists_no_alembic_returns_true(self):
        """Existing DB without alembic_version → should stamp."""
        conn = MagicMock()
        # First call: bundles exists? True
        # Second call: alembic_version exists? False
        conn.execute.return_value.scalar.side_effect = [True, False]
        assert _should_stamp(conn) is True

    def test_both_tables_exist_returns_false(self):
        """DB already has alembic_version → no stamp needed."""
        conn = MagicMock()
        conn.execute.return_value.scalar.side_effect = [True, True]
        assert _should_stamp(conn) is False


class TestRunUpgrade:
    @patch("pkb.db.migration_runner.command")
    @patch("pkb.db.migration_runner._auto_stamp_if_needed")
    def test_upgrade_to_head_default(self, mock_stamp, mock_cmd):
        run_upgrade("postgresql://user:pw@host/db")
        mock_cmd.upgrade.assert_called_once()
        args = mock_cmd.upgrade.call_args
        assert args[0][1] == "head"

    @patch("pkb.db.migration_runner.command")
    @patch("pkb.db.migration_runner._auto_stamp_if_needed")
    def test_upgrade_to_specific_revision(self, mock_stamp, mock_cmd):
        run_upgrade("postgresql://user:pw@host/db", revision="0001")
        args = mock_cmd.upgrade.call_args
        assert args[0][1] == "0001"

    @patch("pkb.db.migration_runner.command")
    @patch("pkb.db.migration_runner._auto_stamp_if_needed")
    def test_upgrade_calls_auto_stamp(self, mock_stamp, mock_cmd):
        run_upgrade("postgresql://user:pw@host/db")
        mock_stamp.assert_called_once()


class TestRunDowngrade:
    @patch("pkb.db.migration_runner.command")
    def test_downgrade_to_revision(self, mock_cmd):
        run_downgrade("postgresql://user:pw@host/db", revision="0001")
        mock_cmd.downgrade.assert_called_once()
        args = mock_cmd.downgrade.call_args
        assert args[0][1] == "0001"


class TestRunStamp:
    @patch("pkb.db.migration_runner.command")
    def test_stamp_revision(self, mock_cmd):
        run_stamp("postgresql://user:pw@host/db", revision="head")
        mock_cmd.stamp.assert_called_once()
        args = mock_cmd.stamp.call_args
        assert args[0][1] == "head"


class TestGetCurrent:
    @patch("pkb.db.migration_runner.command")
    def test_get_current(self, mock_cmd):
        get_current("postgresql://user:pw@host/db")
        mock_cmd.current.assert_called_once()


class TestGetHistory:
    @patch("pkb.db.migration_runner.command")
    def test_get_history(self, mock_cmd):
        get_history("postgresql://user:pw@host/db")
        mock_cmd.history.assert_called_once()


class TestGetTableSchema:
    @patch("pkb.db.migration_runner.create_engine")
    def test_returns_dict_of_tables(self, mock_engine_fn):
        """Returns {table_name: [columns]} for PKB tables."""
        mock_conn = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_engine_fn.return_value = mock_engine

        mock_conn.execute.return_value.fetchall.return_value = [
            ("bundles", "id", "text", "NO", None),
            ("bundles", "kb", "text", "NO", None),
            ("bundles", "stable_id", "text", "NO", None),
            ("bundle_responses", "id", "integer", "NO", "nextval(...)"),
        ]

        result = get_table_schema("postgresql://user:pw@host/db")

        assert "bundles" in result
        assert len(result["bundles"]) == 3
        assert result["bundles"][0]["column"] == "id"
        assert result["bundles"][0]["type"] == "text"
        assert result["bundles"][0]["nullable"] == "NO"
        assert "bundle_responses" in result
        assert len(result["bundle_responses"]) == 1

    @patch("pkb.db.migration_runner.create_engine")
    def test_empty_db_returns_empty_dict(self, mock_engine_fn):
        """No PKB tables → empty dict."""
        mock_conn = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_engine_fn.return_value = mock_engine

        mock_conn.execute.return_value.fetchall.return_value = []

        result = get_table_schema("postgresql://user:pw@host/db")
        assert result == {}

    @patch("pkb.db.migration_runner.create_engine")
    def test_excludes_alembic_version(self, mock_engine_fn):
        """alembic_version table should not appear in schema output."""
        mock_conn = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_engine_fn.return_value = mock_engine

        mock_conn.execute.return_value.fetchall.return_value = [
            ("bundles", "id", "text", "NO", None),
        ]

        result = get_table_schema("postgresql://user:pw@host/db")
        assert "alembic_version" not in result
