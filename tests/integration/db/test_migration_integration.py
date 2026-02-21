"""Alembic migration integration tests against real PostgreSQL.

Tests:
- upgrade to head creates all 6 tables
- tsvector trigger auto-populates tsv column
- downgrade to base + re-upgrade round-trip

These tests manage schema directly (DROP/CREATE) and do NOT use the
`repo` fixture. They use only `db_config` (session-scope).
"""

import psycopg

from pkb.db.migration_runner import run_downgrade, run_upgrade
from pkb.db.schema import DROP_TABLES_SQL, TABLE_NAMES


def _drop_everything(dsn: str) -> None:
    """Drop all PKB tables + alembic_version for a clean slate."""
    conn = psycopg.connect(dsn)
    conn.autocommit = True
    conn.execute(DROP_TABLES_SQL)
    conn.execute("DROP TABLE IF EXISTS alembic_version CASCADE")
    conn.close()


def _get_existing_tables(dsn: str) -> set[str]:
    """Return set of table names in the public schema."""
    conn = psycopg.connect(dsn)
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public'"
    ).fetchall()
    conn.close()
    return {row[0] for row in rows}


class TestMigrationUpgrade:
    """Test Alembic upgrade to head."""

    def test_upgrade_head_creates_all_tables(self, db_config):
        """Drop everything, run upgrade head, verify all 6 tables exist."""
        dsn = db_config.get_dsn()

        # 1. Clean slate
        _drop_everything(dsn)

        # 2. Upgrade to head
        run_upgrade(dsn, revision="head")

        # 3. Verify all tables exist
        tables = _get_existing_tables(dsn)
        for table_name in TABLE_NAMES:
            assert table_name in tables, f"Table '{table_name}' not found after upgrade"

        # alembic_version should also exist
        assert "alembic_version" in tables

    def test_tsvector_trigger_works(self, db_config):
        """Insert a row into bundles, verify tsv column is auto-populated."""
        dsn = db_config.get_dsn()

        # Ensure schema is up (re-upgrade is idempotent if already at head)
        _drop_everything(dsn)
        run_upgrade(dsn, revision="head")

        # Insert a bundle row directly
        conn = psycopg.connect(dsn)
        conn.autocommit = True
        conn.execute(
            "INSERT INTO bundles (id, kb, question, summary, path) "
            "VALUES (%s, %s, %s, %s, %s)",
            ("test-bundle-001", "test-kb", "Python asyncio 사용법",
             "비동기 프로그래밍 가이드", "/tmp/test"),
        )

        # Verify tsv is not NULL
        row = conn.execute(
            "SELECT tsv FROM bundles WHERE id = %s", ("test-bundle-001",)
        ).fetchone()
        conn.close()

        assert row is not None, "Bundle row not found"
        assert row[0] is not None, "tsv column is NULL — trigger did not fire"

        # Leave schema in working state
        _drop_everything(dsn)
        run_upgrade(dsn, revision="head")


class TestMigrationDowngradeUpgrade:
    """Test downgrade to base and re-upgrade round-trip."""

    def test_downgrade_to_base_then_upgrade(self, db_config):
        """Drop everything, upgrade, downgrade to base, upgrade again.

        Verifies all tables exist after the full round-trip.
        """
        dsn = db_config.get_dsn()

        # 1. Clean slate + upgrade
        _drop_everything(dsn)
        run_upgrade(dsn, revision="head")

        # 2. Downgrade to base (removes all tables)
        run_downgrade(dsn, revision="base")

        # Verify tables are gone after downgrade
        tables_after_downgrade = _get_existing_tables(dsn)
        for table_name in TABLE_NAMES:
            assert table_name not in tables_after_downgrade, (
                f"Table '{table_name}' still exists after downgrade to base"
            )

        # 3. Re-upgrade to head
        run_upgrade(dsn, revision="head")

        # 4. Verify all tables exist after round-trip
        tables_after_reupgrade = _get_existing_tables(dsn)
        for table_name in TABLE_NAMES:
            assert table_name in tables_after_reupgrade, (
                f"Table '{table_name}' not found after re-upgrade"
            )

        assert "alembic_version" in tables_after_reupgrade
