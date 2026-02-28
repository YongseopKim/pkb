"""Add metadata columns for Phase 8.

- bundles: consensus TEXT, divergence TEXT
- bundle_responses: key_claims JSONB, stance TEXT
- GIN index on key_claims for JSONB containment queries

Revision ID: 0006
Revises: 0005
Create Date: 2026-02-28
"""

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE bundles ADD COLUMN IF NOT EXISTS consensus TEXT;")
    op.execute("ALTER TABLE bundles ADD COLUMN IF NOT EXISTS divergence TEXT;")
    op.execute(
        "ALTER TABLE bundle_responses "
        "ADD COLUMN IF NOT EXISTS key_claims JSONB DEFAULT '[]'::JSONB;"
    )
    op.execute(
        "ALTER TABLE bundle_responses "
        "ADD COLUMN IF NOT EXISTS stance TEXT;"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_responses_key_claims "
        "ON bundle_responses USING GIN (key_claims);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_responses_key_claims;")
    op.execute("ALTER TABLE bundle_responses DROP COLUMN IF EXISTS stance;")
    op.execute("ALTER TABLE bundle_responses DROP COLUMN IF EXISTS key_claims;")
    op.execute("ALTER TABLE bundles DROP COLUMN IF EXISTS divergence;")
    op.execute("ALTER TABLE bundles DROP COLUMN IF EXISTS consensus;")
