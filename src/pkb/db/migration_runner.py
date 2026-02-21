"""Alembic migration runner for PKB.

Wraps Alembic commands so the rest of PKB never imports Alembic directly.
Supports auto-stamp for existing databases that predate Alembic adoption.
"""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _ensure_psycopg3_dialect(dsn: str) -> str:
    """Convert postgresql:// to postgresql+psycopg:// for SQLAlchemy.

    PKB uses psycopg v3 (not psycopg2). SQLAlchemy defaults to psycopg2
    for plain postgresql:// URLs. The +psycopg suffix forces the v3 driver.
    """
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    return dsn


def _make_alembic_config(dsn: str) -> Config:
    """Create an Alembic Config object with the given DSN."""
    config = Config()
    config.set_main_option("script_location", str(_MIGRATIONS_DIR))
    config.set_main_option("sqlalchemy.url", _ensure_psycopg3_dialect(dsn))
    return config


def _should_stamp(connection) -> bool:
    """Determine if an existing DB needs auto-stamping.

    Returns True if bundles table exists but alembic_version does not.
    This indicates a pre-Alembic database that should be stamped at head.
    """
    bundles_exists = connection.execute(
        text(
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.tables "
            "  WHERE table_name = 'bundles'"
            ")"
        )
    ).scalar()

    if not bundles_exists:
        return False

    alembic_exists = connection.execute(
        text(
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.tables "
            "  WHERE table_name = 'alembic_version'"
            ")"
        )
    ).scalar()

    return not alembic_exists


def _auto_stamp_if_needed(dsn: str) -> bool:
    """Auto-stamp existing databases that predate Alembic.

    Returns True if stamping was performed.
    """
    engine = create_engine(_ensure_psycopg3_dialect(dsn))
    with engine.connect() as conn:
        if _should_stamp(conn):
            config = _make_alembic_config(dsn)
            command.stamp(config, "head")
            return True
    return False


def run_upgrade(dsn: str, *, revision: str = "head") -> None:
    """Upgrade database to a specific revision (default: head)."""
    _auto_stamp_if_needed(dsn)
    config = _make_alembic_config(dsn)
    command.upgrade(config, revision)


def run_downgrade(dsn: str, *, revision: str) -> None:
    """Downgrade database to a specific revision."""
    config = _make_alembic_config(dsn)
    command.downgrade(config, revision)


def run_stamp(dsn: str, *, revision: str) -> None:
    """Stamp the database with a revision without running migrations."""
    config = _make_alembic_config(dsn)
    command.stamp(config, revision)


def get_current(dsn: str) -> None:
    """Display the current revision of the database."""
    config = _make_alembic_config(dsn)
    command.current(config)


def get_history(dsn: str) -> None:
    """Display the migration history."""
    config = _make_alembic_config(dsn)
    command.history(config)
