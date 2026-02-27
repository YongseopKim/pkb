# Stable ID Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `question_hash` with `stable_id` (URL-first, initial-turns-fingerprint fallback) as the unified conversation identity key for dedup, merge, and update decisions.

**Architecture:** Add `compute_stable_id()` and `_normalize_url()` to `ingest.py`, add `stable_id` column to `bundles` table via Alembic migration, refactor `_dedup_and_ingest()` to use `stable_id` with UPDATE instead of SKIP for same-platform matches, and add `pkb db migrate-stable-id` CLI command for backfilling existing bundles.

**Tech Stack:** Python 3.11+, PostgreSQL (psycopg), Alembic, pytest, ruff

---

### Task 1: `compute_stable_id()` and `_normalize_url()` — Core Identity Functions

**Files:**
- Modify: `src/pkb/ingest.py` (add new functions after `compute_question_hash`)
- Test: `tests/test_ingest.py` (add new test class)

**Step 1: Write the failing tests**

In `tests/test_ingest.py`, add:

```python
from pkb.ingest import compute_stable_id, _normalize_url
from pkb.models.jsonl import Conversation, ConversationMeta, Turn
from datetime import datetime, timezone


class TestNormalizeUrl:
    """Tests for URL normalization used in stable_id generation."""

    def test_strips_query_string(self):
        assert _normalize_url("https://claude.ai/chat/abc?ref=x") == "https://claude.ai/chat/abc"

    def test_strips_fragment(self):
        assert _normalize_url("https://claude.ai/chat/abc#section") == "https://claude.ai/chat/abc"

    def test_strips_trailing_slash(self):
        assert _normalize_url("https://claude.ai/chat/abc/") == "https://claude.ai/chat/abc"

    def test_lowercases_hostname(self):
        assert _normalize_url("https://Claude.AI/chat/abc") == "https://claude.ai/chat/abc"

    def test_preserves_path_case(self):
        assert _normalize_url("https://claude.ai/chat/AbCdEf") == "https://claude.ai/chat/AbCdEf"

    def test_combined_normalization(self):
        url = "https://Claude.AI/chat/abc/?utm_source=x#top"
        assert _normalize_url(url) == "https://claude.ai/chat/abc"


class TestComputeStableId:
    """Tests for stable_id generation."""

    def _make_conv(self, *, url=None, turns=None):
        return Conversation(
            meta=ConversationMeta(
                platform="claude",
                url=url,
                exported_at=datetime(2026, 2, 27, tzinfo=timezone.utc),
                title=None,
            ),
            turns=turns or [],
        )

    def test_url_based_stable_id(self):
        """When URL is present, stable_id is SHA-256 of normalized URL."""
        conv = self._make_conv(url="https://claude.ai/chat/abc123")
        sid = compute_stable_id(conv)
        assert len(sid) == 64  # SHA-256 hex
        # Same URL with different query → same stable_id
        conv2 = self._make_conv(url="https://claude.ai/chat/abc123?ref=x")
        assert compute_stable_id(conv2) == sid

    def test_url_case_insensitive_host(self):
        conv1 = self._make_conv(url="https://Claude.AI/chat/abc")
        conv2 = self._make_conv(url="https://claude.ai/chat/abc")
        assert compute_stable_id(conv1) == compute_stable_id(conv2)

    def test_turn_fingerprint_fallback(self):
        """When no URL, stable_id uses first 5 turns."""
        turns = [
            Turn(role="user", content="Hello world", timestamp=None),
            Turn(role="assistant", content="Hi there", timestamp=None),
        ]
        conv = self._make_conv(turns=turns)
        sid = compute_stable_id(conv)
        assert len(sid) == 64

    def test_turn_fingerprint_stable_with_more_turns(self):
        """Adding turns after the first 5 doesn't change stable_id."""
        base_turns = [
            Turn(role="user", content=f"msg{i}", timestamp=None)
            for i in range(5)
        ]
        conv_short = self._make_conv(turns=base_turns)
        conv_long = self._make_conv(turns=base_turns + [
            Turn(role="assistant", content="extra turn", timestamp=None),
        ])
        assert compute_stable_id(conv_short) == compute_stable_id(conv_long)

    def test_turn_fingerprint_uses_content_prefix(self):
        """Only first 200 chars per turn are used."""
        long_content = "a" * 300
        turns = [Turn(role="assistant", content=long_content, timestamp=None)]
        conv = self._make_conv(turns=turns)
        sid1 = compute_stable_id(conv)

        # Different content after char 200 → same stable_id
        modified = "a" * 200 + "b" * 100
        turns2 = [Turn(role="assistant", content=modified, timestamp=None)]
        conv2 = self._make_conv(turns=turns2)
        assert compute_stable_id(conv2) == sid1

    def test_url_takes_priority_over_turns(self):
        """URL-based ID is different from turn-based ID for same conversation."""
        turns = [Turn(role="user", content="Hello", timestamp=None)]
        conv_with_url = self._make_conv(url="https://claude.ai/chat/abc", turns=turns)
        conv_no_url = self._make_conv(turns=turns)
        assert compute_stable_id(conv_with_url) != compute_stable_id(conv_no_url)

    def test_empty_conversation_fallback(self):
        """Empty conversation still produces a valid stable_id."""
        conv = self._make_conv(turns=[])
        sid = compute_stable_id(conv)
        assert len(sid) == 64
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest.py::TestNormalizeUrl -v`
Run: `pytest tests/test_ingest.py::TestComputeStableId -v`
Expected: FAIL with ImportError (functions don't exist yet)

**Step 3: Write minimal implementation**

In `src/pkb/ingest.py`, add after `compute_question_hash()`:

```python
from urllib.parse import urlparse, urlunparse


def _normalize_url(url: str) -> str:
    """Normalize a URL for stable_id generation.

    Strips query string, fragment, trailing slash. Lowercases hostname.
    Preserves path case (conversation IDs are case-sensitive).
    """
    parsed = urlparse(url)
    # Lowercase scheme and hostname
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    port = f":{parsed.port}" if parsed.port else ""
    # Strip trailing slash from path
    path = parsed.path.rstrip("/")
    # Reconstruct without query and fragment
    return f"{scheme}://{hostname}{port}{path}"


def compute_stable_id(conv: Conversation) -> str:
    """Compute a stable conversation identity hash.

    Priority:
      1. URL (immutable per-conversation identifier) → SHA-256 of normalized URL
      2. Initial turns fingerprint (first 5 turns, 200 chars each) → SHA-256

    Returns:
        64-char hex SHA-256 digest.
    """
    if conv.meta.url:
        normalized = _normalize_url(conv.meta.url)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # Fallback: initial turns fingerprint
    parts = []
    for turn in conv.turns[:5]:
        parts.append(f"{turn.role}:{turn.content[:200]}")
    fingerprint = "\n".join(parts)
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingest.py::TestNormalizeUrl tests/test_ingest.py::TestComputeStableId -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pkb/ingest.py tests/test_ingest.py
git commit -m "feat: add compute_stable_id() and _normalize_url()"
```

---

### Task 2: DB Schema — Alembic Migration for `stable_id` Column

**Files:**
- Create: `src/pkb/db/migrations/versions/0005_add_stable_id.py`
- Modify: `src/pkb/db/schema.py` (add `stable_id` column to `CREATE_TABLES_SQL`)
- Test: `tests/integration/db/test_stable_id_migration.py` (integration test, gated by PKB_DB_INTEGRATION)

**Step 1: Write the migration**

Create `src/pkb/db/migrations/versions/0005_add_stable_id.py`:

```python
"""Add stable_id column to bundles table.

Revision ID: 0005
Revises: 0004
Create Date: 2026-02-27
"""

revision = "0005"
down_revision = "0004"

from alembic import op


def upgrade():
    # Add nullable column first
    op.execute("ALTER TABLE bundles ADD COLUMN IF NOT EXISTS stable_id TEXT")
    # Backfill from question_hash (temporary — real values set by migrate-stable-id command)
    op.execute("UPDATE bundles SET stable_id = question_hash WHERE stable_id IS NULL")
    # Set NOT NULL
    op.execute("ALTER TABLE bundles ALTER COLUMN stable_id SET NOT NULL")
    # Unique index
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_bundles_stable_id ON bundles (stable_id)"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS idx_bundles_stable_id")
    op.execute("ALTER TABLE bundles DROP COLUMN IF EXISTS stable_id")
```

**Step 2: Update `schema.py` reference**

In `src/pkb/db/schema.py`, add `stable_id TEXT NOT NULL` to the bundles table in `CREATE_TABLES_SQL` (after `question_hash`), and add the unique index.

**Step 3: Run migration integration test (if DB available)**

Run: `PKB_DB_INTEGRATION=1 pytest tests/integration/db/ -v -k "stable_id or migration"`
Expected: PASS (migration applies cleanly)

**Step 4: Commit**

```bash
git add src/pkb/db/migrations/versions/0005_add_stable_id.py src/pkb/db/schema.py
git commit -m "feat: add stable_id column to bundles (migration 0005)"
```

---

### Task 3: DB Repository — `find_bundle_by_stable_id()` and `upsert_bundle` Update

**Files:**
- Modify: `src/pkb/db/postgres.py`
- Test: `tests/test_db/test_postgres.py`

**Step 1: Write the failing tests**

In `tests/test_db/test_postgres.py`, add:

```python
class TestFindBundleByStableId:
    """Tests for find_bundle_by_stable_id()."""

    def test_returns_none_when_not_found(self, repo):
        result = repo.find_bundle_by_stable_id("nonexistent_hash")
        assert result is None

    def test_finds_bundle_by_stable_id(self, repo):
        repo.upsert_bundle(
            bundle_id="20260227-test-a1b2",
            kb="test",
            question="test question",
            summary="test summary",
            created_at=datetime(2026, 2, 27, tzinfo=timezone.utc),
            response_count=1,
            path="bundles/20260227-test-a1b2",
            question_hash="qhash123",
            stable_id="stable_abc",
            domains=["dev"],
            topics=["python"],
            responses=[{"platform": "claude"}],
        )
        result = repo.find_bundle_by_stable_id("stable_abc")
        assert result is not None
        assert result["bundle_id"] == "20260227-test-a1b2"
        assert "claude" in result["platforms"]

    def test_returns_platforms_domains_topics(self, repo):
        repo.upsert_bundle(
            bundle_id="20260227-multi-c3d4",
            kb="test",
            question="q",
            summary="s",
            created_at=datetime(2026, 2, 27, tzinfo=timezone.utc),
            response_count=2,
            path="bundles/20260227-multi-c3d4",
            question_hash="qhash456",
            stable_id="stable_def",
            domains=["dev", "ai"],
            topics=["python", "ml"],
            responses=[
                {"platform": "claude"},
                {"platform": "chatgpt"},
            ],
        )
        result = repo.find_bundle_by_stable_id("stable_def")
        assert set(result["platforms"]) == {"claude", "chatgpt"}
        assert set(result["domains"]) == {"dev", "ai"}
        assert set(result["topics"]) == {"python", "ml"}
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_db/test_postgres.py::TestFindBundleByStableId -v`
Expected: FAIL (method doesn't exist, `stable_id` param not accepted)

**Step 3: Implement**

In `src/pkb/db/postgres.py`:

1. Add `stable_id: str | None = None` param to `upsert_bundle()`.
2. Add `stable_id` to INSERT and ON CONFLICT UPDATE.
3. Add `find_bundle_by_stable_id()` method (same structure as `find_bundle_by_question_hash`, querying `WHERE b.stable_id = %s`).

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_db/test_postgres.py::TestFindBundleByStableId -v`
Expected: ALL PASS

**Step 5: Run existing DB tests to ensure nothing broke**

Run: `pytest tests/test_db/ -v`
Expected: ALL PASS (existing tests still pass because `stable_id` is optional)

**Step 6: Commit**

```bash
git add src/pkb/db/postgres.py tests/test_db/test_postgres.py
git commit -m "feat: add find_bundle_by_stable_id() and stable_id upsert support"
```

---

### Task 4: Refactor Ingest Flow — `stable_id` Based Dedup with UPDATE

**Files:**
- Modify: `src/pkb/ingest.py` (refactor `_dedup_and_ingest`, `_create_new_bundle`, add `_update_existing_bundle`)
- Test: `tests/test_ingest.py` (update existing dedup tests, add UPDATE tests)

**Step 1: Write the failing tests for UPDATE behavior**

```python
class TestStableIdDedup:
    """Tests for stable_id based dedup: CREATE / UPDATE / MERGE."""

    def test_same_platform_same_stable_id_returns_update(self):
        """Same platform + same stable_id → UPDATE (not SKIP)."""
        # Setup: create a bundle, then ingest same conversation again
        # Assert: returns dict with "updated": True (not None/SKIP)
        ...

    def test_different_platform_same_stable_id_returns_merge(self):
        """Different platform + same stable_id → MERGE."""
        ...

    def test_no_match_creates_new_bundle(self):
        """No existing stable_id → CREATE new bundle."""
        ...
```

The exact test implementation will use the existing mock patterns from `tests/test_ingest.py`. The key assertion change: where old tests expected `None` for same-platform duplicates, new tests expect a dict with `"updated": True`.

**Step 2: Implement the changes**

In `src/pkb/ingest.py`:

1. Change `ingest_file()`:
   - Compute `stable_id` via `compute_stable_id(conv)` instead of `question_hash` for dedup.
   - Still compute `question, question_hash` for backward compat (passed to `_create_new_bundle`).
   - Lock key changes from `question_hash` to `stable_id`.

2. Change `_dedup_and_ingest()`:
   - Call `find_bundle_by_stable_id(stable_id)` instead of `find_bundle_by_question_hash`.
   - Same platform → call new `_update_existing_bundle()` instead of returning `None`.
   - Different platform → call `merge_file()` (unchanged).

3. Add `_update_existing_bundle()`:
   - Re-generate derived files (re-run LLM for response_meta, rewrite MD).
   - Upsert bundle in DB (existing bundle_id, updated content).
   - Delete + re-upsert ChromaDB chunks.
   - Update `source_path`.
   - Return dict with `"updated": True`.

4. Pass `stable_id` to `upsert_bundle()` calls in `_create_new_bundle()`.

**Step 3: Update existing tests**

- Tests that expected `None` for same-platform dedup → expect `{"updated": True, ...}`.
- Tests that mock `find_bundle_by_question_hash` → mock `find_bundle_by_stable_id`.
- All `question_hash` references in upsert calls → add `stable_id` param.

**Step 4: Run all ingest tests**

Run: `pytest tests/test_ingest.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pkb/ingest.py tests/test_ingest.py
git commit -m "feat: refactor ingest to use stable_id with UPDATE instead of SKIP"
```

---

### Task 5: Update Watch Mode — `stable_id` Based Identification

**Files:**
- Modify: `src/pkb/cli.py` (`_build_watch_ingest_fn`)
- Test: `tests/test_cli_commands.py` (update watch-related tests)

**Step 1: Refactor `_build_watch_ingest_fn()`**

The current watch flow:
1. `find_by_source_path()` → find old bundle → delete → reingest with `force=True`

New flow:
1. Just call `pipeline.ingest_file(file_path)` — the `stable_id` based dedup handles everything:
   - Same conversation → UPDATE (re-generate, upsert)
   - New conversation → CREATE
   - Different platform → MERGE
2. `source_path` tracking is now handled inside `_update_existing_bundle()`.
3. `force=True` is no longer needed (UPDATE path replaces it).

This **simplifies** `_build_watch_ingest_fn()` significantly — no more manual delete+reingest logic.

**Step 2: Update `_build_watch_ingest_fn`**

```python
def _build_watch_ingest_fn(*, pipelines, kb_entries, repo, chunk_store):
    from pkb.ingest import move_to_done

    resolved_pipelines = {str(Path(k).resolve()): v for k, v in pipelines.items()}
    resolved_kb_entries = {str(Path(k).resolve()): v for k, v in kb_entries.items()}

    def _ingest_fn(file_path):
        match = _find_watch_dir_for_path(file_path, resolved_pipelines)
        if match is None:
            return None
        _, pipeline = match

        kb_match = _find_watch_dir_for_path(file_path, resolved_kb_entries)
        kb_entry = kb_match[1] if kb_match else None

        # stable_id based dedup handles CREATE/UPDATE/MERGE automatically
        result = pipeline.ingest_file(file_path)

        if result and kb_entry and not result.get("status", "").startswith("skip_"):
            move_to_done(file_path, kb_entry.get_watch_dir())
        return result

    return _ingest_fn
```

**Step 3: Run watch tests**

Run: `pytest tests/test_cli_commands.py -v -k watch`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/pkb/cli.py tests/test_cli_commands.py
git commit -m "feat: simplify watch mode with stable_id based dedup"
```

---

### Task 6: Update Regenerate — Pass `stable_id`

**Files:**
- Modify: `src/pkb/regenerate.py`
- Test: `tests/test_regenerate.py`

**Step 1: Update regenerate to compute and pass `stable_id`**

In `src/pkb/regenerate.py`:
- Import `compute_stable_id` (already imports `compute_question_hash`).
- After parsing raw file, compute `stable_id = compute_stable_id(conv)`.
- Pass `stable_id=stable_id` to `upsert_bundle()`.

**Step 2: Update tests**

- Mock `upsert_bundle` expectations → include `stable_id` param.

**Step 3: Run tests**

Run: `pytest tests/test_regenerate.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/pkb/regenerate.py tests/test_regenerate.py
git commit -m "feat: regenerate computes and passes stable_id"
```

---

### Task 7: CLI — `pkb db migrate-stable-id` Command

**Files:**
- Modify: `src/pkb/cli.py` (add `migrate_stable_id` to `db` group)
- Test: `tests/test_cli_commands.py`

**Step 1: Write the failing test**

```python
class TestMigrateStableIdCommand:
    def test_dry_run_shows_bundles(self, runner, mock_config):
        result = runner.invoke(cli, ["db", "migrate-stable-id", "--dry-run", "--kb", "test"])
        assert result.exit_code == 0
        assert "dry run" in result.output.lower() or "preview" in result.output.lower()
```

**Step 2: Implement the command**

```python
@db_group.command("migrate-stable-id")
@click.option("--kb", default=None, help="Specific KB to migrate")
@click.option("--dry-run", is_flag=True, help="Preview without writing")
@click.pass_context
def migrate_stable_id(ctx, kb, dry_run):
    """Recompute stable_id for existing bundles from raw files."""
    # 1. List all bundle_ids (optionally filtered by kb)
    # 2. For each bundle: find _raw/ dir, parse first file, compute stable_id
    # 3. UPDATE bundles SET stable_id = ? WHERE id = ?
    # 4. Handle conflicts (two bundles → same stable_id)
```

**Step 3: Run tests**

Run: `pytest tests/test_cli_commands.py::TestMigrateStableIdCommand -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/pkb/cli.py tests/test_cli_commands.py
git commit -m "feat: add pkb db migrate-stable-id command"
```

---

### Task 8: Update Remaining Tests and Backward Compatibility

**Files:**
- Modify: `tests/test_ingest.py` — bulk update existing tests
- Modify: `tests/test_db/test_postgres.py` — add stable_id to test fixtures
- Modify: `tests/integration/db/` — update integration tests

**Step 1: Audit and fix all `question_hash` references in tests**

- `upsert_bundle()` calls in tests that don't pass `stable_id` → add `stable_id` param.
- Mocks of `find_bundle_by_question_hash` → update or add `find_bundle_by_stable_id` mocks.
- Assertions checking for `None` (SKIP) on same-platform dedup → update to expect UPDATE dict.

**Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 3: Run ruff**

Run: `ruff check src/ tests/`
Expected: No errors

**Step 4: Commit**

```bash
git add tests/
git commit -m "test: update all tests for stable_id migration"
```

---

### Task 9: Final Verification and Documentation

**Files:**
- Modify: `CLAUDE.md` — update architecture docs
- Modify: `docs/design-v1.md` — add stable_id section if design decisions changed

**Step 1: Run full test suite including integration**

Run: `PKB_DB_INTEGRATION=1 pytest -v`
Expected: ALL PASS

**Step 2: Run ruff lint**

Run: `ruff check src/ tests/`
Expected: No errors

**Step 3: Update documentation**

- `CLAUDE.md`: Add `stable_id` to architecture, update ingest flow description, note migration 0005.
- Remove `docs/plans/2026-02-27-stable-id-design.md` and `docs/plans/2026-02-27-stable-id-plan.md` (completed plans).

**Step 4: Final commit**

```bash
git add CLAUDE.md docs/
git commit -m "docs: update architecture docs for stable_id"
```
