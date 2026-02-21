"""Initial schema: 6 tables + indexes + tsvector trigger.

Revision ID: 0001
Revises: None
Create Date: 2026-02-22
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    CREATE TABLE IF NOT EXISTS bundles (
        id TEXT PRIMARY KEY,
        kb TEXT NOT NULL,
        question TEXT NOT NULL,
        summary TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ,
        response_count INTEGER,
        has_synthesis BOOLEAN DEFAULT FALSE,
        meta_version INTEGER DEFAULT 1,
        path TEXT NOT NULL,
        question_hash TEXT,
        tsv tsvector
    );

    CREATE INDEX IF NOT EXISTS idx_bundles_tsv ON bundles USING GIN (tsv);
    CREATE INDEX IF NOT EXISTS idx_bundles_kb ON bundles (kb);
    CREATE INDEX IF NOT EXISTS idx_bundles_question_hash ON bundles (question_hash);

    CREATE TABLE IF NOT EXISTS bundle_domains (
        bundle_id TEXT REFERENCES bundles(id) ON DELETE CASCADE,
        domain TEXT NOT NULL,
        PRIMARY KEY (bundle_id, domain)
    );

    CREATE TABLE IF NOT EXISTS bundle_topics (
        bundle_id TEXT REFERENCES bundles(id) ON DELETE CASCADE,
        topic TEXT NOT NULL,
        is_pending BOOLEAN DEFAULT FALSE,
        PRIMARY KEY (bundle_id, topic)
    );

    CREATE TABLE IF NOT EXISTS topic_vocab (
        canonical TEXT PRIMARY KEY,
        aliases JSONB DEFAULT '[]'::JSONB,
        status TEXT DEFAULT 'approved',
        merged_into TEXT
    );

    CREATE TABLE IF NOT EXISTS bundle_responses (
        bundle_id TEXT REFERENCES bundles(id) ON DELETE CASCADE,
        platform TEXT NOT NULL,
        model TEXT,
        turn_count INTEGER,
        PRIMARY KEY (bundle_id, platform)
    );

    CREATE TABLE IF NOT EXISTS duplicate_pairs (
        id SERIAL PRIMARY KEY,
        bundle_a TEXT REFERENCES bundles(id) ON DELETE CASCADE,
        bundle_b TEXT REFERENCES bundles(id) ON DELETE CASCADE,
        similarity REAL NOT NULL,
        status TEXT DEFAULT 'pending',
        resolved_at TIMESTAMPTZ,
        UNIQUE(bundle_a, bundle_b)
    );

    CREATE OR REPLACE FUNCTION bundles_tsv_trigger() RETURNS trigger AS $$
    BEGIN
        NEW.tsv := to_tsvector('simple',
            COALESCE(NEW.question, '') || ' ' || COALESCE(NEW.summary, ''));
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

    DROP TRIGGER IF EXISTS trg_bundles_tsv ON bundles;
    CREATE TRIGGER trg_bundles_tsv
        BEFORE INSERT OR UPDATE ON bundles
        FOR EACH ROW EXECUTE FUNCTION bundles_tsv_trigger();
    """)


def downgrade() -> None:
    op.execute("""
    DROP TRIGGER IF EXISTS trg_bundles_tsv ON bundles;
    DROP FUNCTION IF EXISTS bundles_tsv_trigger() CASCADE;
    DROP TABLE IF EXISTS duplicate_pairs CASCADE;
    DROP TABLE IF EXISTS bundle_responses CASCADE;
    DROP TABLE IF EXISTS bundle_topics CASCADE;
    DROP TABLE IF EXISTS bundle_domains CASCADE;
    DROP TABLE IF EXISTS topic_vocab CASCADE;
    DROP TABLE IF EXISTS bundles CASCADE;
    """)
