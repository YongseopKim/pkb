# Stable ID Design

## Problem

PKB has two tracking mechanisms, both fragile:

- `source_path`: absolute file path — breaks when file moves
- `question_hash`: SHA-256 of first user message — breaks when content changes

When both file location and content change simultaneously, PKB loses track entirely and creates a duplicate bundle.

## Goal

Introduce a single `stable_id` that survives:
1. **File relocation** — same conversation at a different path
2. **Content extension** — re-exported conversation with more turns
3. **Re-ingestion** — same conversation ingested multiple times

## Design

### stable_id Generation

```python
def compute_stable_id(conv: Conversation) -> str:
    # Priority 1: URL (immutable per-conversation identifier)
    if conv.meta.url:
        normalized = _normalize_url(conv.meta.url)
        return hashlib.sha256(normalized.encode()).hexdigest()

    # Priority 2: initial turns fingerprint (first 5 turns)
    parts = []
    for turn in conv.turns[:5]:
        parts.append(f"{turn.role}:{turn.content[:200]}")
    fingerprint = "\n".join(parts)
    return hashlib.sha256(fingerprint.encode()).hexdigest()
```

**URL normalization**: strip query/fragment, trailing slash, lowercase hostname.

**Turn fingerprint**: first 5 turns, each `role:content[:200]`. Stable because conversation beginnings don't change when continued.

### DB Schema

Alembic migration `0005_add_stable_id`:

```sql
ALTER TABLE bundles ADD COLUMN stable_id TEXT;
UPDATE bundles SET stable_id = question_hash WHERE stable_id IS NULL;
ALTER TABLE bundles ALTER COLUMN stable_id SET NOT NULL;
CREATE UNIQUE INDEX idx_bundles_stable_id ON bundles (stable_id);
```

`question_hash` column retained (deprecated, backward compat). Removed in a future migration.

### Ingest Flow Change

```
parse_file() → Conversation
       │
compute_stable_id(conv) → stable_id
       │
find_bundle_by_stable_id(stable_id)
       │
   ┌───┴────────────────────┐
   │                        │
 Not found               Found
   │                  ┌─────┴──────┐
   ▼                  │            │
CREATE             Same platform  Different platform
new bundle            │            │
                      ▼            ▼
                   UPDATE        MERGE
               (refresh bundle) (add response)
```

Key change: same platform + same stable_id → **UPDATE** (was SKIP).
This enables conversation continuation and manual edits to be reflected.

**UPDATE logic:**
1. Re-generate derived files (MD, `_bundle.md`)
2. Upsert bundle meta in PostgreSQL
3. Delete + re-upsert chunks in ChromaDB
4. Update `source_path` to new file location

### Migration Command

```bash
pkb db migrate-stable-id --kb <name>   # specific KB
pkb db migrate-stable-id               # all KBs
pkb db migrate-stable-id --dry-run     # preview
```

Reads `_raw/` files for each bundle, parses them, computes `stable_id` from URL or initial turns, and updates the DB.

Conflict handling:
- No `_raw/` files → keep `question_hash` as `stable_id`
- Two bundles collide → warning + manual resolution needed

### Affected Components

| Component | File | Change |
|---|---|---|
| Identity | `ingest.py` | `compute_stable_id()`, `_normalize_url()` |
| Ingest flow | `ingest.py` | `_dedup_and_ingest()` → stable_id based, SKIP → UPDATE |
| DB repo | `db/postgres.py` | `find_bundle_by_stable_id()`, upsert with stable_id |
| DB schema | `db/schema.py` | `stable_id` column |
| Migration | `db/migrations/versions/0005_*` | Add stable_id + backfill |
| CLI | `cli.py` | `pkb db migrate-stable-id` command |
| Watch | `cli.py` | stable_id based identification |
| Dedup | `dedup.py` | Reference stable_id instead of question_hash |
| Regenerate | `regenerate.py` | Recompute stable_id |
| Tests | `tests/` | Bulk update (question_hash → stable_id) |

**Unchanged**: `bundle_id` format, filesystem structure, ChromaDB chunks, search, web, chat, MCP.

### Known Limitations

- MD files without URL where user edits the initial content → `stable_id` changes (unavoidable without URL)
- `bundle_id` generation still uses `question` for the hash4 component (unchanged, cosmetic only)

### Relationship: bundle_id vs stable_id

- `bundle_id` (`YYYYMMDD-slug-hash4`): filesystem path + human-readable identifier. Never changes.
- `stable_id` (SHA-256): logical conversation identity key. Used for dedup/merge/update decisions.
