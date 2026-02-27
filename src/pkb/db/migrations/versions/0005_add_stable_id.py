"""Add stable_id column to bundles table.

Provides a content-based stable identifier for bundles. Unlike bundle_id
(which includes date/slug that can change), stable_id is derived from
the conversation content hash and remains constant across re-ingestion.

Revision ID: 0005
Revises: 0004
Create Date: 2026-02-27
"""

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Step 1: Add column as nullable first
    op.execute(
        "ALTER TABLE bundles ADD COLUMN IF NOT EXISTS stable_id TEXT;"
    )
    # Step 2: Backfill from question_hash for existing rows
    op.execute(
        "UPDATE bundles SET stable_id = question_hash WHERE stable_id IS NULL;"
    )
    # Step 3: Set NOT NULL constraint
    op.execute(
        "ALTER TABLE bundles ALTER COLUMN stable_id SET NOT NULL;"
    )
    # Step 4: Create unique index
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_bundles_stable_id "
        "ON bundles (stable_id);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_bundles_stable_id;")
    op.execute("ALTER TABLE bundles DROP COLUMN IF EXISTS stable_id;")
