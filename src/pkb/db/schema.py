"""PostgreSQL schema definitions for PKB.

Schema evolution is managed by Alembic migrations in db/migrations/versions/.
CREATE_TABLES_SQL is kept as a reference for the current expected schema.
DROP_TABLES_SQL is used for testing/reset.

Converted from design-v1.md section 5.6 (SQLite) to PostgreSQL:
- TEXT → TEXT (same)
- BOOLEAN → BOOLEAN (native)
- JSON arrays → JSONB
- FTS5 → tsvector + GIN index
- TEXT dates → TIMESTAMPTZ
"""

TABLE_NAMES = frozenset({
    "bundles",
    "bundle_domains",
    "bundle_topics",
    "topic_vocab",
    "bundle_responses",
    "duplicate_pairs",
    "bundle_relations",
})

CREATE_TABLES_SQL = """
-- Bundle metadata (search/filter, unified across all KBs)
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
    stable_id TEXT NOT NULL,
    source_path TEXT,
    consensus TEXT,
    divergence TEXT,
    tsv tsvector
);

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_bundles_tsv ON bundles USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_bundles_kb ON bundles (kb);
CREATE INDEX IF NOT EXISTS idx_bundles_question_hash ON bundles (question_hash);
CREATE UNIQUE INDEX IF NOT EXISTS idx_bundles_stable_id ON bundles (stable_id);
CREATE INDEX IF NOT EXISTS idx_bundles_source_path ON bundles (source_path);

-- Domain mapping (L1, M:N)
CREATE TABLE IF NOT EXISTS bundle_domains (
    bundle_id TEXT REFERENCES bundles(id) ON DELETE CASCADE,
    domain TEXT NOT NULL,
    PRIMARY KEY (bundle_id, domain)
);

-- Topic mapping (L2, M:N)
CREATE TABLE IF NOT EXISTS bundle_topics (
    bundle_id TEXT REFERENCES bundles(id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    is_pending BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (bundle_id, topic)
);

-- Topic vocabulary management
CREATE TABLE IF NOT EXISTS topic_vocab (
    canonical TEXT PRIMARY KEY,
    aliases JSONB DEFAULT '[]'::JSONB,
    status TEXT DEFAULT 'approved',
    merged_into TEXT
);

-- Model/platform response mapping
CREATE TABLE IF NOT EXISTS bundle_responses (
    bundle_id TEXT REFERENCES bundles(id) ON DELETE CASCADE,
    platform TEXT NOT NULL,
    model TEXT,
    turn_count INTEGER,
    source_path TEXT,
    key_claims JSONB DEFAULT '[]'::JSONB,
    stance TEXT,
    PRIMARY KEY (bundle_id, platform)
);

CREATE INDEX IF NOT EXISTS idx_bundle_responses_source_path ON bundle_responses (source_path);
CREATE INDEX IF NOT EXISTS idx_responses_key_claims ON bundle_responses USING GIN (key_claims);

-- Duplicate pair tracking
CREATE TABLE IF NOT EXISTS duplicate_pairs (
    id SERIAL PRIMARY KEY,
    bundle_a TEXT REFERENCES bundles(id) ON DELETE CASCADE,
    bundle_b TEXT REFERENCES bundles(id) ON DELETE CASCADE,
    similarity REAL NOT NULL,
    status TEXT DEFAULT 'pending',
    resolved_at TIMESTAMPTZ,
    UNIQUE(bundle_a, bundle_b)
);

-- Bundle relations (knowledge graph edges)
CREATE TABLE IF NOT EXISTS bundle_relations (
    id SERIAL PRIMARY KEY,
    source_bundle_id TEXT NOT NULL REFERENCES bundles(id) ON DELETE CASCADE,
    target_bundle_id TEXT NOT NULL REFERENCES bundles(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    score REAL NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_bundle_id, target_bundle_id, relation_type)
);
CREATE INDEX IF NOT EXISTS idx_relations_source ON bundle_relations (source_bundle_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON bundle_relations (target_bundle_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON bundle_relations (relation_type);

-- Trigger to auto-update tsvector on insert/update
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
"""

DROP_TABLES_SQL = """
DROP TABLE IF EXISTS bundle_relations CASCADE;
DROP TABLE IF EXISTS duplicate_pairs CASCADE;
DROP TABLE IF EXISTS bundle_responses CASCADE;
DROP TABLE IF EXISTS bundle_topics CASCADE;
DROP TABLE IF EXISTS bundle_domains CASCADE;
DROP TABLE IF EXISTS topic_vocab CASCADE;
DROP TABLE IF EXISTS bundles CASCADE;
DROP FUNCTION IF EXISTS bundles_tsv_trigger() CASCADE;
"""
