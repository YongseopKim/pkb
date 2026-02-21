"""Tests for Alembic migration files."""

import importlib
from pathlib import Path

import pytest

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "src" / "pkb" / "db" / "migrations"
VERSIONS_DIR = MIGRATIONS_DIR / "versions"


class TestMigrationStructure:
    """Migration directory structure and file existence."""

    def test_migrations_dir_exists(self):
        assert MIGRATIONS_DIR.is_dir()

    def test_versions_dir_exists(self):
        assert VERSIONS_DIR.is_dir()

    def test_env_py_exists(self):
        assert (MIGRATIONS_DIR / "env.py").is_file()

    def test_script_mako_exists(self):
        assert (MIGRATIONS_DIR / "script.py.mako").is_file()


class TestInitialMigration:
    """Tests for 0001_initial_schema migration."""

    @pytest.fixture
    def migration(self):
        """Load the initial migration module."""
        spec = importlib.util.spec_from_file_location(
            "initial_schema",
            VERSIONS_DIR / "0001_initial_schema.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_revision_id(self, migration):
        assert migration.revision == "0001"

    def test_down_revision_is_none(self, migration):
        """Initial migration has no parent."""
        assert migration.down_revision is None

    def test_has_upgrade_function(self, migration):
        assert callable(migration.upgrade)

    def test_has_downgrade_function(self, migration):
        assert callable(migration.downgrade)

    def test_upgrade_sql_contains_all_tables(self, migration):
        """upgrade() should create all 6 core tables."""
        expected_tables = [
            "bundles", "bundle_domains", "bundle_topics",
            "topic_vocab", "bundle_responses", "duplicate_pairs",
        ]
        # Read the source to verify SQL content
        source = (VERSIONS_DIR / "0001_initial_schema.py").read_text()
        for table in expected_tables:
            assert f"CREATE TABLE IF NOT EXISTS {table}" in source

    def test_upgrade_sql_has_tsv_trigger(self, migration):
        source = (VERSIONS_DIR / "0001_initial_schema.py").read_text()
        assert "bundles_tsv_trigger" in source

    def test_upgrade_sql_has_gin_index(self, migration):
        source = (VERSIONS_DIR / "0001_initial_schema.py").read_text()
        assert "idx_bundles_tsv" in source
        assert "GIN" in source

    def test_downgrade_sql_drops_all_tables(self, migration):
        source = (VERSIONS_DIR / "0001_initial_schema.py").read_text()
        assert "DROP TABLE IF EXISTS" in source

    def test_no_source_path_column(self, migration):
        """Initial migration should NOT include source_path (that's in 0002)."""
        source = (VERSIONS_DIR / "0001_initial_schema.py").read_text()
        assert "source_path" not in source


class TestSourcePathMigration:
    """Tests for 0002_add_source_path migration."""

    @pytest.fixture
    def migration(self):
        spec = importlib.util.spec_from_file_location(
            "add_source_path",
            VERSIONS_DIR / "0002_add_source_path.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_revision_id(self, migration):
        assert migration.revision == "0002"

    def test_down_revision_is_0001(self, migration):
        assert migration.down_revision == "0001"

    def test_has_upgrade_function(self, migration):
        assert callable(migration.upgrade)

    def test_has_downgrade_function(self, migration):
        assert callable(migration.downgrade)

    def test_upgrade_adds_source_path_column(self, migration):
        source = (VERSIONS_DIR / "0002_add_source_path.py").read_text()
        assert "source_path" in source
        assert "ADD COLUMN" in source

    def test_upgrade_adds_source_path_index(self, migration):
        source = (VERSIONS_DIR / "0002_add_source_path.py").read_text()
        assert "idx_bundles_source_path" in source

    def test_downgrade_drops_column(self, migration):
        source = (VERSIONS_DIR / "0002_add_source_path.py").read_text()
        assert "DROP COLUMN" in source


class TestResponseSourcePathMigration:
    """Tests for 0003_add_source_path_to_responses migration."""

    @pytest.fixture
    def migration(self):
        spec = importlib.util.spec_from_file_location(
            "add_source_path_to_responses",
            VERSIONS_DIR / "0003_add_source_path_to_responses.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_revision_id(self, migration):
        assert migration.revision == "0003"

    def test_down_revision_is_0002(self, migration):
        assert migration.down_revision == "0002"

    def test_has_upgrade_function(self, migration):
        assert callable(migration.upgrade)

    def test_has_downgrade_function(self, migration):
        assert callable(migration.downgrade)

    def test_upgrade_adds_source_path_to_bundle_responses(self, migration):
        source = (VERSIONS_DIR / "0003_add_source_path_to_responses.py").read_text()
        assert "bundle_responses" in source
        assert "source_path" in source
        assert "ADD COLUMN" in source

    def test_upgrade_adds_index(self, migration):
        source = (VERSIONS_DIR / "0003_add_source_path_to_responses.py").read_text()
        assert "idx_bundle_responses_source_path" in source

    def test_downgrade_drops_column(self, migration):
        source = (VERSIONS_DIR / "0003_add_source_path_to_responses.py").read_text()
        assert "DROP COLUMN" in source

    def test_downgrade_drops_index(self, migration):
        source = (VERSIONS_DIR / "0003_add_source_path_to_responses.py").read_text()
        assert "DROP INDEX" in source
