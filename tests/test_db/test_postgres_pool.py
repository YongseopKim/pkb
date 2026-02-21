"""Tests for BundleRepository connection pool support."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from pkb.models.config import ConcurrencyConfig, PostgresConfig


class TestBundleRepositoryPool:
    """Tests for from_pool() classmethod and _get_conn()."""

    def test_from_pool_creates_pool(self):
        """from_pool() should create a ConnectionPool and BundleRepository."""
        from pkb.db.postgres import BundleRepository

        config = PostgresConfig(host="localhost", password="test")
        concurrency = ConcurrencyConfig(db_pool_min=2, db_pool_max=4)

        mock_pool = MagicMock()
        with patch("pkb.db.postgres.ConnectionPool", return_value=mock_pool) as pool_cls:
            repo = BundleRepository.from_pool(config, concurrency)
            pool_cls.assert_called_once_with(
                conninfo=config.get_dsn(),
                min_size=2,
                max_size=4,
            )
        assert repo is not None
        assert repo._pool is mock_pool

    def test_get_conn_with_pool(self):
        """_get_conn() with pool should use pool.connection()."""
        from pkb.db.postgres import BundleRepository

        config = PostgresConfig(host="localhost", password="test")
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        @contextmanager
        def fake_connection():
            yield mock_conn

        mock_pool.connection = fake_connection

        with patch("pkb.db.postgres.ConnectionPool", return_value=mock_pool):
            repo = BundleRepository.from_pool(config, ConcurrencyConfig())

        with repo._get_conn() as conn:
            assert conn is mock_conn

    def test_get_conn_without_pool(self):
        """_get_conn() without pool should yield self._conn."""
        from pkb.db.postgres import BundleRepository

        config = PostgresConfig(host="localhost", password="test")
        mock_conn = MagicMock()

        with patch("pkb.db.postgres.psycopg.connect", return_value=mock_conn):
            repo = BundleRepository(config)

        with repo._get_conn() as conn:
            assert conn is mock_conn

    def test_close_with_pool(self):
        """close() with pool should close pool."""
        from pkb.db.postgres import BundleRepository

        config = PostgresConfig(host="localhost", password="test")
        mock_pool = MagicMock()

        with patch("pkb.db.postgres.ConnectionPool", return_value=mock_pool):
            repo = BundleRepository.from_pool(config, ConcurrencyConfig())

        repo.close()
        mock_pool.close.assert_called_once()

    def test_close_without_pool(self):
        """close() without pool should close conn."""
        from pkb.db.postgres import BundleRepository

        config = PostgresConfig(host="localhost", password="test")
        mock_conn = MagicMock()

        with patch("pkb.db.postgres.psycopg.connect", return_value=mock_conn):
            repo = BundleRepository(config)

        repo.close()
        mock_conn.close.assert_called_once()

    def test_pool_repo_can_execute_query(self):
        """Pool-based repo should be able to run queries via _get_conn()."""
        from pkb.db.postgres import BundleRepository

        config = PostgresConfig(host="localhost", password="test")
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None

        @contextmanager
        def fake_connection():
            yield mock_conn

        mock_pool.connection = fake_connection

        with patch("pkb.db.postgres.ConnectionPool", return_value=mock_pool):
            repo = BundleRepository.from_pool(config, ConcurrencyConfig())

        # bundle_exists uses _get_conn internally
        result = repo.bundle_exists("some_hash")
        assert result is False
        mock_conn.execute.assert_called()
