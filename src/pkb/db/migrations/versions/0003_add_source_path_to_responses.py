"""Add source_path column to bundle_responses table.

Tracks per-platform source file paths so that find_by_source_path()
can locate bundles via merged files (not just the first-ingested file).

Revision ID: 0003
Revises: 0002
Create Date: 2026-02-22
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE bundle_responses ADD COLUMN IF NOT EXISTS source_path TEXT;"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_bundle_responses_source_path "
        "ON bundle_responses (source_path);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_bundle_responses_source_path;")
    op.execute("ALTER TABLE bundle_responses DROP COLUMN IF EXISTS source_path;")
