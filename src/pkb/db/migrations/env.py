"""Alembic environment configuration for PKB.

Loads the database DSN from PKB config.yaml or from a direct sqlalchemy.url setting.
"""

from alembic import context
from sqlalchemy import create_engine


def _get_url() -> str:
    """Resolve database URL from Alembic config or PKB config.yaml."""
    config = context.config

    # Priority 1: Direct sqlalchemy.url (set programmatically or via alembic.ini)
    url = config.get_main_option("sqlalchemy.url")
    if url:
        return url

    # Priority 2: PKB config.yaml path
    config_path = config.get_main_option("pkb_config_path")
    if config_path:
        from pathlib import Path

        from pkb.config import load_config

        pkb_config = load_config(Path(config_path))
        dsn = pkb_config.database.postgres.get_dsn()
        # Ensure psycopg v3 dialect for SQLAlchemy
        if dsn.startswith("postgresql://"):
            dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
        return dsn

    raise RuntimeError(
        "No database URL configured. Set sqlalchemy.url or pkb_config_path."
    )


def run_migrations_online() -> None:
    """Run migrations with an online (live) database connection."""
    url = _get_url()
    connectable = create_engine(url)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
