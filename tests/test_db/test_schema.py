"""Tests for PostgreSQL schema definitions."""

from pkb.db.schema import (
    CREATE_TABLES_SQL,
    DROP_TABLES_SQL,
    TABLE_NAMES,
)


class TestSchemaDefinitions:
    def test_table_names(self):
        expected = {
            "bundles",
            "bundle_domains",
            "bundle_topics",
            "topic_vocab",
            "bundle_responses",
            "duplicate_pairs",
        }
        assert TABLE_NAMES == expected

    def test_create_tables_sql_is_string(self):
        assert isinstance(CREATE_TABLES_SQL, str)
        assert len(CREATE_TABLES_SQL) > 100

    def test_create_tables_contains_all_tables(self):
        for table in TABLE_NAMES:
            assert f"CREATE TABLE IF NOT EXISTS {table}" in CREATE_TABLES_SQL

    def test_create_tables_has_tsvector(self):
        """PostgreSQL full-text search via tsvector + GIN index."""
        assert "tsvector" in CREATE_TABLES_SQL
        assert "gin" in CREATE_TABLES_SQL.lower()

    def test_create_tables_has_timestamptz(self):
        assert "TIMESTAMPTZ" in CREATE_TABLES_SQL

    def test_create_tables_has_jsonb_for_aliases(self):
        assert "JSONB" in CREATE_TABLES_SQL

    def test_drop_tables_sql(self):
        assert isinstance(DROP_TABLES_SQL, str)
        for table in TABLE_NAMES:
            assert table in DROP_TABLES_SQL

    def test_create_tables_has_cascade_on_delete(self):
        assert "ON DELETE CASCADE" in CREATE_TABLES_SQL

    def test_create_tables_has_source_path_column(self):
        """bundles 테이블에 source_path TEXT 컬럼이 있어야 함."""
        assert "source_path TEXT" in CREATE_TABLES_SQL

    def test_create_tables_has_source_path_index(self):
        """source_path 인덱스가 있어야 함."""
        assert "idx_bundles_source_path" in CREATE_TABLES_SQL

    def test_no_migrations_dict(self):
        """MIGRATIONS dict는 제거됨 (Alembic으로 전환)."""
        import pkb.db.schema as schema_mod
        assert not hasattr(schema_mod, "MIGRATIONS")
