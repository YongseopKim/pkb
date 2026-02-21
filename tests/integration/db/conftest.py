"""Fixtures for DB integration tests using local Docker containers.

Requires:
    - docker compose -f docker/docker-compose.test.yml up -d
    - PKB_DB_INTEGRATION=1 environment variable
"""

import os

import pytest

from pkb.db.migration_runner import run_upgrade
from pkb.db.schema import DROP_TABLES_SQL
from pkb.models.config import ChromaDBConfig, PostgresConfig

SKIP_REASON = "PKB_DB_INTEGRATION not set (need local Docker DB)"


def _skip_if_no_db():
    if not os.environ.get("PKB_DB_INTEGRATION"):
        pytest.skip(SKIP_REASON)


@pytest.fixture(scope="session")
def db_config():
    """PostgreSQL config for local test container (port 5433)."""
    _skip_if_no_db()
    return PostgresConfig(
        host="localhost",
        port=5433,
        database="pkb_test",
        username="pkb_test",
        password="pkb_test",
    )


@pytest.fixture(scope="session")
def chroma_config():
    """ChromaDB config for local test container (port 8001)."""
    _skip_if_no_db()
    return ChromaDBConfig(
        host="localhost",
        port=8001,
        collection="pkb_test",
    )


@pytest.fixture(scope="session")
def _create_schema(db_config):
    """Create DB schema via Alembic upgrade (once per session).

    Drops all tables first to ensure clean state, then runs upgrade.
    """
    import psycopg

    conn = psycopg.connect(db_config.get_dsn())
    conn.autocommit = True
    conn.execute(DROP_TABLES_SQL)
    conn.execute("DROP TABLE IF EXISTS alembic_version CASCADE")
    conn.close()

    run_upgrade(db_config.get_dsn(), revision="head")


@pytest.fixture
def repo(db_config, _create_schema):
    """BundleRepository connected to local test DB, truncated before each test."""
    from pkb.db.postgres import BundleRepository

    r = BundleRepository(db_config)
    r._conn.execute(
        "TRUNCATE bundles, bundle_domains, bundle_topics, "
        "topic_vocab, bundle_responses, duplicate_pairs CASCADE"
    )
    yield r
    r.close()


@pytest.fixture
def chunk_store(chroma_config, _create_schema):
    """ChunkStore connected to local test ChromaDB, cleared before each test."""
    from pkb.db.chromadb_client import ChunkStore

    store = ChunkStore(chroma_config)
    try:
        all_ids = store._collection.get()["ids"]
        if all_ids:
            store._collection.delete(ids=all_ids)
    except Exception:
        pass
    yield store


@pytest.fixture
def test_kb_path(tmp_path):
    """Temporary KB directory structure for E2E tests."""
    kb_path = tmp_path / "test-kb"
    inbox = kb_path / "inbox"
    bundles = kb_path / "bundles"
    inbox.mkdir(parents=True)
    bundles.mkdir(parents=True)
    return kb_path
