"""Add bundle_relations table for knowledge graph.

Stores relationships between bundles: similar (embedding), related (topic),
sequel (temporal).

Revision ID: 0004
Revises: 0003
Create Date: 2026-02-23
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS bundle_relations (
            id SERIAL PRIMARY KEY,
            source_bundle_id TEXT NOT NULL REFERENCES bundles(id) ON DELETE CASCADE,
            target_bundle_id TEXT NOT NULL REFERENCES bundles(id) ON DELETE CASCADE,
            relation_type TEXT NOT NULL,
            score REAL NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(source_bundle_id, target_bundle_id, relation_type)
        );
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_relations_source "
        "ON bundle_relations (source_bundle_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_relations_target "
        "ON bundle_relations (target_bundle_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_relations_type "
        "ON bundle_relations (relation_type);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_relations_type;")
    op.execute("DROP INDEX IF EXISTS idx_relations_target;")
    op.execute("DROP INDEX IF EXISTS idx_relations_source;")
    op.execute("DROP TABLE IF EXISTS bundle_relations CASCADE;")
