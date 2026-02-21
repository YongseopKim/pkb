"""Add source_path column to bundles table.

Revision ID: 0002
Revises: 0001
Create Date: 2026-02-22
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE bundles ADD COLUMN IF NOT EXISTS source_path TEXT;"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_bundles_source_path ON bundles (source_path);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_bundles_source_path;")
    op.execute("ALTER TABLE bundles DROP COLUMN IF EXISTS source_path;")
