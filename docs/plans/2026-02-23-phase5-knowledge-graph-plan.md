# Phase 5: Knowledge Graph Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a bundle relation system that automatically discovers and stores connections between bundles, enabling knowledge graph visualization and navigation.

**Architecture:** Follows the existing dedup module pattern — a `RelationBuilder` class uses ChromaDB embeddings + PostgreSQL topic/domain data to detect `similar`, `related`, and `sequel` relationships between bundles. Relations are stored in a new `bundle_relations` table. CLI commands (`pkb relate`) and Web UI routes (`/relations`) expose the data.

**Tech Stack:** PostgreSQL (bundle_relations table), ChromaDB (embedding similarity), Click (CLI), FastAPI + Jinja2 + htmx (Web UI), pytest (TDD)

---

### Task 1: Add RelationConfig to config models

**Files:**
- Modify: `src/pkb/models/config.py:82-86` (after DedupConfig)
- Modify: `src/pkb/models/config.py:138-148` (PKBConfig)
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# In tests/test_config.py — add to the existing test file

class TestRelationConfig:
    def test_defaults(self):
        from pkb.models.config import RelationConfig
        config = RelationConfig()
        assert config.similarity_threshold == 0.7
        assert config.max_relations_per_bundle == 20

    def test_pkbconfig_includes_relations(self):
        from pkb.models.config import PKBConfig
        config = PKBConfig()
        assert hasattr(config, "relations")
        assert config.relations.similarity_threshold == 0.7
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::TestRelationConfig -v`
Expected: FAIL with `ImportError` or `AttributeError`

**Step 3: Write minimal implementation**

In `src/pkb/models/config.py`, after `DedupConfig` (line ~86):

```python
class RelationConfig(BaseModel):
    """Configuration for knowledge graph relation detection."""

    similarity_threshold: float = 0.7
    max_relations_per_bundle: int = 20
```

In `PKBConfig` (line ~145), add field:

```python
class PKBConfig(BaseModel):
    ...
    dedup: DedupConfig = DedupConfig()
    relations: RelationConfig = RelationConfig()  # ADD
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::TestRelationConfig -v`
Expected: PASS

**Step 5: Commit**

```
feat(config): add RelationConfig for knowledge graph settings
```

---

### Task 2: Create bundle_relations DB migration (0004)

**Files:**
- Create: `src/pkb/db/migrations/versions/0004_bundle_relations.py`
- Modify: `src/pkb/db/schema.py` (add to TABLE_NAMES + CREATE/DROP SQL)
- Test: `tests/integration/db/test_migration_0004.py` (integration only)

**Step 1: Write the migration file**

```python
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
```

**Step 2: Update schema.py**

Add `"bundle_relations"` to `TABLE_NAMES` frozenset.

Add to end of `CREATE_TABLES_SQL` (before closing `"""`):
```sql
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
```

Add to `DROP_TABLES_SQL` (first line, before duplicate_pairs):
```sql
DROP TABLE IF EXISTS bundle_relations CASCADE;
```

**Step 3: Update integration test conftest.py**

In `tests/integration/db/conftest.py`, update the TRUNCATE in `repo` fixture:
```python
r._conn.execute(
    "TRUNCATE bundles, bundle_domains, bundle_topics, "
    "topic_vocab, bundle_responses, duplicate_pairs, bundle_relations CASCADE"
)
```

**Step 4: Run integration tests to verify migration works**

Run: `PKB_DB_INTEGRATION=1 pytest tests/integration/db/ -v`
Expected: All existing tests still pass (migration is additive)

**Step 5: Commit**

```
feat(db): add bundle_relations table migration (0004)
```

---

### Task 3: Add relation repository methods to BundleRepository

**Files:**
- Modify: `src/pkb/db/postgres.py` (add methods after `update_duplicate_status`)
- Test: `tests/test_db_relations.py` (new mock test file)

**Step 1: Write the failing tests**

Create `tests/test_db_relations.py`:

```python
"""Tests for BundleRepository relation methods."""

from unittest.mock import MagicMock, patch

import pytest


class TestInsertRelation:
    def test_insert_relation(self):
        from pkb.db.postgres import BundleRepository

        repo = MagicMock(spec=BundleRepository)
        repo.insert_relation = BundleRepository.insert_relation.__get__(repo)

        repo.insert_relation(
            source_bundle_id="20260101-test-abc1",
            target_bundle_id="20260101-test-def2",
            relation_type="similar",
            score=0.85,
        )
        # Verify SQL was called via _get_conn
        repo._get_conn.assert_called_once()


class TestListRelations:
    def test_list_relations_for_bundle(self):
        """list_relations should return relations for a given bundle."""
        from pkb.db.postgres import BundleRepository

        assert hasattr(BundleRepository, "list_relations")

    def test_list_relations_bidirectional(self):
        """list_relations should find relations in both directions."""
        from pkb.db.postgres import BundleRepository

        assert hasattr(BundleRepository, "list_relations")


class TestDeleteRelations:
    def test_delete_relations_for_bundle(self):
        """delete_relations_for_bundle should remove all relations."""
        from pkb.db.postgres import BundleRepository

        assert hasattr(BundleRepository, "delete_relations_for_bundle")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_db_relations.py -v`
Expected: FAIL — `insert_relation` method doesn't exist

**Step 3: Write implementation**

Add to `src/pkb/db/postgres.py` (after `update_duplicate_status` method):

```python
    def insert_relation(
        self,
        source_bundle_id: str,
        target_bundle_id: str,
        relation_type: str,
        score: float,
    ) -> None:
        """Insert a bundle relation (skip if already exists)."""
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO bundle_relations
                    (source_bundle_id, target_bundle_id, relation_type, score)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_bundle_id, target_bundle_id, relation_type)
                DO UPDATE SET score = EXCLUDED.score
                """,
                (source_bundle_id, target_bundle_id, relation_type, score),
            )

    def list_relations(
        self,
        bundle_id: str,
        relation_type: str | None = None,
    ) -> list[dict]:
        """List relations for a bundle (both directions)."""
        with self._get_conn() as conn:
            if relation_type:
                rows = conn.execute(
                    "SELECT id, source_bundle_id, target_bundle_id, "
                    "relation_type, score, created_at "
                    "FROM bundle_relations "
                    "WHERE (source_bundle_id = %s OR target_bundle_id = %s) "
                    "AND relation_type = %s "
                    "ORDER BY score DESC",
                    (bundle_id, bundle_id, relation_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, source_bundle_id, target_bundle_id, "
                    "relation_type, score, created_at "
                    "FROM bundle_relations "
                    "WHERE source_bundle_id = %s OR target_bundle_id = %s "
                    "ORDER BY score DESC",
                    (bundle_id, bundle_id),
                ).fetchall()
        return [
            {
                "id": row[0],
                "source_bundle_id": row[1],
                "target_bundle_id": row[2],
                "relation_type": row[3],
                "score": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]

    def delete_relations_for_bundle(self, bundle_id: str) -> int:
        """Delete all relations involving a bundle. Returns count deleted."""
        with self._get_conn() as conn:
            result = conn.execute(
                "DELETE FROM bundle_relations "
                "WHERE source_bundle_id = %s OR target_bundle_id = %s",
                (bundle_id, bundle_id),
            )
        return result.rowcount

    def list_all_relations(
        self,
        relation_type: str | None = None,
        kb: str | None = None,
    ) -> list[dict]:
        """List all relations, optionally filtered by type or KB."""
        with self._get_conn() as conn:
            conditions = []
            params: list = []
            if relation_type:
                conditions.append("br.relation_type = %s")
                params.append(relation_type)
            if kb:
                conditions.append(
                    "(bs.kb = %s OR bt.kb = %s)"
                )
                params.extend([kb, kb])
            where = "WHERE " + " AND ".join(conditions) if conditions else ""
            rows = conn.execute(
                f"SELECT br.id, br.source_bundle_id, br.target_bundle_id, "
                f"br.relation_type, br.score, br.created_at "
                f"FROM bundle_relations br "
                f"JOIN bundles bs ON bs.id = br.source_bundle_id "
                f"JOIN bundles bt ON bt.id = br.target_bundle_id "
                f"{where} "
                f"ORDER BY br.score DESC",
                params,
            ).fetchall()
        return [
            {
                "id": row[0],
                "source_bundle_id": row[1],
                "target_bundle_id": row[2],
                "relation_type": row[3],
                "score": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]

    def count_relations(self, relation_type: str | None = None) -> int:
        """Count total relations, optionally by type."""
        with self._get_conn() as conn:
            if relation_type:
                row = conn.execute(
                    "SELECT COUNT(*) FROM bundle_relations WHERE relation_type = %s",
                    (relation_type,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM bundle_relations"
                ).fetchone()
        return row[0] if row else 0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_db_relations.py -v`
Expected: PASS

**Step 5: Write integration tests**

Create `tests/integration/db/test_relations_integration.py`:

```python
"""Integration tests for bundle_relations repository methods."""

import pytest


@pytest.fixture
def _seed_bundles(repo):
    """Insert test bundles for relation tests."""
    from datetime import datetime
    for bid in ["20260101-alpha-abc1", "20260101-beta-def2", "20260101-gamma-ghi3"]:
        repo.upsert_bundle(
            bundle_id=bid,
            kb="test",
            question=f"Question for {bid}",
            summary=f"Summary for {bid}",
            created_at=datetime(2026, 1, 1),
            response_count=1,
            path=f"/bundles/{bid}",
            question_hash=f"hash_{bid}",
            domains=["dev"],
            topics=["python"],
            responses=[{"platform": "claude", "model": "haiku", "turn_count": 1}],
        )


class TestInsertRelation:
    def test_insert_and_list(self, repo, _seed_bundles):
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.85,
        )
        rels = repo.list_relations("20260101-alpha-abc1")
        assert len(rels) == 1
        assert rels[0]["target_bundle_id"] == "20260101-beta-def2"
        assert rels[0]["score"] == pytest.approx(0.85, abs=0.01)

    def test_upsert_updates_score(self, repo, _seed_bundles):
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.80,
        )
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.90,
        )
        rels = repo.list_relations("20260101-alpha-abc1", relation_type="similar")
        assert len(rels) == 1
        assert rels[0]["score"] == pytest.approx(0.90, abs=0.01)

    def test_bidirectional_lookup(self, repo, _seed_bundles):
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.85,
        )
        # Should find when querying from the target side too
        rels = repo.list_relations("20260101-beta-def2")
        assert len(rels) == 1

    def test_multiple_relation_types(self, repo, _seed_bundles):
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.85,
        )
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "related", 0.60,
        )
        rels = repo.list_relations("20260101-alpha-abc1")
        assert len(rels) == 2


class TestDeleteRelations:
    def test_delete_removes_all(self, repo, _seed_bundles):
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.85,
        )
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-gamma-ghi3", "related", 0.70,
        )
        count = repo.delete_relations_for_bundle("20260101-alpha-abc1")
        assert count == 2
        assert repo.list_relations("20260101-alpha-abc1") == []


class TestListAllRelations:
    def test_list_all(self, repo, _seed_bundles):
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.85,
        )
        repo.insert_relation(
            "20260101-beta-def2", "20260101-gamma-ghi3", "related", 0.70,
        )
        all_rels = repo.list_all_relations()
        assert len(all_rels) == 2

    def test_filter_by_type(self, repo, _seed_bundles):
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.85,
        )
        repo.insert_relation(
            "20260101-beta-def2", "20260101-gamma-ghi3", "related", 0.70,
        )
        similar_only = repo.list_all_relations(relation_type="similar")
        assert len(similar_only) == 1


class TestCountRelations:
    def test_count(self, repo, _seed_bundles):
        assert repo.count_relations() == 0
        repo.insert_relation(
            "20260101-alpha-abc1", "20260101-beta-def2", "similar", 0.85,
        )
        assert repo.count_relations() == 1
        assert repo.count_relations(relation_type="similar") == 1
        assert repo.count_relations(relation_type="related") == 0
```

**Step 6: Run integration tests**

Run: `PKB_DB_INTEGRATION=1 pytest tests/integration/db/test_relations_integration.py -v`
Expected: PASS

**Step 7: Commit**

```
feat(db): add bundle relation repository methods
```

---

### Task 4: Create RelationBuilder module

**Files:**
- Create: `src/pkb/relations.py`
- Test: `tests/test_relations.py`

**Step 1: Write the failing tests**

Create `tests/test_relations.py`:

```python
"""Tests for RelationBuilder — knowledge graph edge detection."""

from unittest.mock import MagicMock

import pytest

from pkb.models.config import RelationConfig


@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def mock_chunk_store():
    return MagicMock()


@pytest.fixture
def builder(mock_repo, mock_chunk_store):
    config = RelationConfig(similarity_threshold=0.7, max_relations_per_bundle=20)
    return RelationBuilder(
        repo=mock_repo,
        chunk_store=mock_chunk_store,
        config=config,
    )


# Import after fixture (will fail until implemented)
from pkb.relations import RelationBuilder


class TestFindSimilar:
    def test_finds_similar_bundle(self, builder, mock_repo, mock_chunk_store):
        """When embedding similarity > threshold, detect similar relation."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-src-abc1",
            "question": "Python async 패턴은?",
            "kb": "personal",
        }
        mock_chunk_store.search.return_value = [
            MagicMock(
                metadata={"bundle_id": "20260101-tgt-def2", "kb": "personal"},
                distance=0.20,  # similarity = 0.80 > 0.7
            ),
        ]
        mock_repo.list_relations.return_value = []

        relations = builder.find_similar("20260101-src-abc1")
        assert len(relations) == 1
        assert relations[0]["source_bundle_id"] == "20260101-src-abc1"
        assert relations[0]["target_bundle_id"] == "20260101-tgt-def2"
        assert relations[0]["relation_type"] == "similar"
        assert relations[0]["score"] >= 0.7

    def test_ignores_self(self, builder, mock_repo, mock_chunk_store):
        """Should not relate a bundle to itself."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-src-abc1",
            "question": "테스트",
            "kb": "personal",
        }
        mock_chunk_store.search.return_value = [
            MagicMock(
                metadata={"bundle_id": "20260101-src-abc1", "kb": "personal"},
                distance=0.0,
            ),
        ]
        mock_repo.list_relations.return_value = []

        relations = builder.find_similar("20260101-src-abc1")
        assert len(relations) == 0

    def test_below_threshold(self, builder, mock_repo, mock_chunk_store):
        """Below threshold should not be detected."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-src-abc1",
            "question": "Python",
            "kb": "personal",
        }
        mock_chunk_store.search.return_value = [
            MagicMock(
                metadata={"bundle_id": "20260101-other-xyz9", "kb": "personal"},
                distance=0.50,  # similarity = 0.50 < 0.7
            ),
        ]
        mock_repo.list_relations.return_value = []

        relations = builder.find_similar("20260101-src-abc1")
        assert len(relations) == 0

    def test_respects_max_relations(self, builder, mock_repo, mock_chunk_store):
        """Should not exceed max_relations_per_bundle."""
        builder._max_relations = 2
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-src-abc1",
            "question": "테스트",
            "kb": "personal",
        }
        mock_chunk_store.search.return_value = [
            MagicMock(
                metadata={"bundle_id": f"20260101-tgt-{i:04d}", "kb": "personal"},
                distance=0.10,
            )
            for i in range(5)
        ]
        mock_repo.list_relations.return_value = []

        relations = builder.find_similar("20260101-src-abc1")
        assert len(relations) <= 2

    def test_none_bundle_returns_empty(self, builder, mock_repo):
        """Non-existent bundle should return empty."""
        mock_repo.get_bundle_by_id.return_value = None
        relations = builder.find_similar("nonexistent")
        assert relations == []


class TestFindRelatedByTopics:
    def test_finds_shared_topics(self, builder, mock_repo):
        """Bundles sharing topics should be related."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-src-abc1",
            "question": "Python async",
            "kb": "personal",
        }
        # Return bundles that share topics
        mock_repo.find_bundles_sharing_topics.return_value = [
            {"bundle_id": "20260101-tgt-def2", "shared_count": 3, "total_topics": 4},
            {"bundle_id": "20260101-tgt-ghi3", "shared_count": 1, "total_topics": 5},
        ]

        relations = builder.find_related_by_topics("20260101-src-abc1")
        assert len(relations) >= 1
        assert all(r["relation_type"] == "related" for r in relations)


class TestScanBundle:
    def test_scan_bundle_combines_types(self, builder, mock_repo, mock_chunk_store):
        """scan_bundle should combine similar + related relations."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-src-abc1",
            "question": "Python async",
            "kb": "personal",
        }
        mock_chunk_store.search.return_value = [
            MagicMock(
                metadata={"bundle_id": "20260101-tgt-def2", "kb": "personal"},
                distance=0.15,
            ),
        ]
        mock_repo.list_relations.return_value = []
        mock_repo.find_bundles_sharing_topics.return_value = [
            {"bundle_id": "20260101-tgt-ghi3", "shared_count": 2, "total_topics": 3},
        ]

        relations = builder.scan_bundle("20260101-src-abc1")
        types = {r["relation_type"] for r in relations}
        assert "similar" in types
        assert "related" in types


class TestScanAll:
    def test_scan_all(self, builder, mock_repo, mock_chunk_store):
        """scan() should process all bundles and return stats."""
        mock_repo.list_all_bundle_ids.return_value = ["b1", "b2"]
        mock_repo.get_bundle_by_id.side_effect = [
            {"bundle_id": "b1", "question": "Q1", "kb": "personal"},
            {"bundle_id": "b1", "question": "Q1", "kb": "personal"},
            {"bundle_id": "b2", "question": "Q2", "kb": "personal"},
            {"bundle_id": "b2", "question": "Q2", "kb": "personal"},
        ]
        mock_chunk_store.search.return_value = []
        mock_repo.list_relations.return_value = []
        mock_repo.find_bundles_sharing_topics.return_value = []

        result = builder.scan(kb="personal")
        assert result["scanned"] == 2
        mock_repo.list_all_bundle_ids.assert_called_once_with(kb="personal")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_relations.py -v`
Expected: FAIL with `ImportError: cannot import name 'RelationBuilder'`

**Step 3: Write implementation**

Create `src/pkb/relations.py`:

```python
"""Knowledge graph relation detection between bundles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pkb.models.config import RelationConfig

if TYPE_CHECKING:
    from pkb.db.chromadb_client import ChunkStore
    from pkb.db.postgres import BundleRepository


class RelationBuilder:
    """Builds knowledge graph edges between bundles.

    Detects three relation types:
    - similar: content similarity via ChromaDB embeddings
    - related: shared topics/domains via PostgreSQL
    - sequel: same topic with temporal proximity
    """

    def __init__(
        self,
        *,
        repo: BundleRepository,
        chunk_store: ChunkStore,
        config: RelationConfig,
    ) -> None:
        self._repo = repo
        self._chunk_store = chunk_store
        self._threshold = config.similarity_threshold
        self._max_relations = config.max_relations_per_bundle

    def find_similar(self, bundle_id: str) -> list[dict]:
        """Find similar bundles via embedding similarity.

        Returns list of relations: [{"source_bundle_id", "target_bundle_id",
        "relation_type", "score"}].
        """
        bundle = self._repo.get_bundle_by_id(bundle_id)
        if bundle is None:
            return []

        question = bundle["question"]
        kb = bundle.get("kb")

        where = {"kb": kb} if kb else None
        results = self._chunk_store.search(
            query=question, n_results=30, where=where,
        )

        # Group by bundle, take best similarity
        candidates: dict[str, float] = {}
        for r in results:
            other_id = r.metadata.get("bundle_id", "")
            if other_id == bundle_id:
                continue
            similarity = max(0.0, 1.0 - r.distance)
            if similarity >= self._threshold:
                if other_id not in candidates or similarity > candidates[other_id]:
                    candidates[other_id] = similarity

        # Sort by score descending, limit
        sorted_candidates = sorted(
            candidates.items(), key=lambda x: x[1], reverse=True,
        )[:self._max_relations]

        return [
            {
                "source_bundle_id": bundle_id,
                "target_bundle_id": other_id,
                "relation_type": "similar",
                "score": round(score, 4),
            }
            for other_id, score in sorted_candidates
        ]

    def find_related_by_topics(self, bundle_id: str) -> list[dict]:
        """Find related bundles via shared topics.

        Returns list of relations with score = shared_count / total_topics.
        """
        bundle = self._repo.get_bundle_by_id(bundle_id)
        if bundle is None:
            return []

        shared = self._repo.find_bundles_sharing_topics(bundle_id)

        return [
            {
                "source_bundle_id": bundle_id,
                "target_bundle_id": s["bundle_id"],
                "relation_type": "related",
                "score": round(s["shared_count"] / max(s["total_topics"], 1), 4),
            }
            for s in shared
            if s["shared_count"] / max(s["total_topics"], 1) > 0.3
        ]

    def scan_bundle(self, bundle_id: str) -> list[dict]:
        """Scan a single bundle for all relation types.

        Returns combined list of similar + related relations.
        """
        similar = self.find_similar(bundle_id)
        related = self.find_related_by_topics(bundle_id)

        # Deduplicate: if same pair has both similar and related, keep both
        return similar + related

    def scan(self, kb: str | None = None) -> dict:
        """Scan all bundles for relations.

        Returns stats: {"scanned": int, "new_relations": int}.
        """
        bundle_ids = self._repo.list_all_bundle_ids(kb=kb)
        total_new = 0

        for bid in bundle_ids:
            relations = self.scan_bundle(bid)
            for rel in relations:
                self._repo.insert_relation(
                    rel["source_bundle_id"],
                    rel["target_bundle_id"],
                    rel["relation_type"],
                    rel["score"],
                )
                total_new += 1

        return {"scanned": len(bundle_ids), "new_relations": total_new}

    def list_relations(
        self, bundle_id: str, relation_type: str | None = None,
    ) -> list[dict]:
        """List relations for a bundle."""
        return self._repo.list_relations(bundle_id, relation_type=relation_type)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_relations.py -v`
Expected: PASS

**Step 5: Commit**

```
feat: add RelationBuilder for knowledge graph edge detection
```

---

### Task 5: Add find_bundles_sharing_topics to BundleRepository

**Files:**
- Modify: `src/pkb/db/postgres.py` (add method)
- Test: `tests/test_db_relations.py` (add test)
- Test: `tests/integration/db/test_relations_integration.py` (add test)

**Step 1: Write the failing test**

Add to `tests/test_db_relations.py`:

```python
class TestFindBundlesSharingTopics:
    def test_method_exists(self):
        from pkb.db.postgres import BundleRepository
        assert hasattr(BundleRepository, "find_bundles_sharing_topics")
```

Add to `tests/integration/db/test_relations_integration.py`:

```python
class TestFindBundlesSharingTopics:
    def test_finds_shared_topics(self, repo):
        from datetime import datetime
        # Insert two bundles with shared topics
        repo.upsert_bundle(
            bundle_id="20260101-a-abc1", kb="test",
            question="Q1", summary="S1",
            created_at=datetime(2026, 1, 1),
            response_count=1, path="/a", question_hash="h1",
            domains=["dev"], topics=["python", "async"],
            responses=[{"platform": "claude", "model": "m", "turn_count": 1}],
        )
        repo.upsert_bundle(
            bundle_id="20260101-b-def2", kb="test",
            question="Q2", summary="S2",
            created_at=datetime(2026, 1, 1),
            response_count=1, path="/b", question_hash="h2",
            domains=["dev"], topics=["python", "testing"],
            responses=[{"platform": "claude", "model": "m", "turn_count": 1}],
        )

        shared = repo.find_bundles_sharing_topics("20260101-a-abc1")
        assert len(shared) == 1
        assert shared[0]["bundle_id"] == "20260101-b-def2"
        assert shared[0]["shared_count"] == 1  # "python" shared

    def test_excludes_self(self, repo):
        from datetime import datetime
        repo.upsert_bundle(
            bundle_id="20260101-a-abc1", kb="test",
            question="Q1", summary="S1",
            created_at=datetime(2026, 1, 1),
            response_count=1, path="/a", question_hash="h1",
            domains=["dev"], topics=["python"],
            responses=[{"platform": "claude", "model": "m", "turn_count": 1}],
        )
        shared = repo.find_bundles_sharing_topics("20260101-a-abc1")
        assert len(shared) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_db_relations.py::TestFindBundlesSharingTopics -v`
Expected: FAIL

**Step 3: Write implementation**

Add to `src/pkb/db/postgres.py`:

```python
    def find_bundles_sharing_topics(self, bundle_id: str) -> list[dict]:
        """Find bundles that share topics with the given bundle.

        Returns list of dicts: [{"bundle_id", "shared_count", "total_topics"}].
        Excludes self. Sorted by shared_count descending.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT bt2.bundle_id,
                       COUNT(*) AS shared_count,
                       (SELECT COUNT(*) FROM bundle_topics bt3
                        WHERE bt3.bundle_id = bt2.bundle_id) AS total_topics
                FROM bundle_topics bt1
                JOIN bundle_topics bt2
                    ON bt1.topic = bt2.topic AND bt2.bundle_id != bt1.bundle_id
                WHERE bt1.bundle_id = %s
                GROUP BY bt2.bundle_id
                ORDER BY shared_count DESC
                """,
                (bundle_id,),
            ).fetchall()
        return [
            {
                "bundle_id": row[0],
                "shared_count": row[1],
                "total_topics": row[2],
            }
            for row in rows
        ]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_db_relations.py::TestFindBundlesSharingTopics -v`
Expected: PASS

Run: `PKB_DB_INTEGRATION=1 pytest tests/integration/db/test_relations_integration.py::TestFindBundlesSharingTopics -v`
Expected: PASS

**Step 5: Commit**

```
feat(db): add find_bundles_sharing_topics query
```

---

### Task 6: Add `pkb relate` CLI commands

**Files:**
- Modify: `src/pkb/cli.py` (add `relate` group with `scan`, `list`, `show` subcommands)
- Test: `tests/test_cli_commands.py` (add relation CLI tests)

**Step 1: Write the failing test**

Add to `tests/test_cli_commands.py` (following existing patterns):

```python
class TestRelateCommands:
    def test_relate_scan_invokes_builder(self):
        """pkb relate scan should call RelationBuilder.scan()."""
        from click.testing import CliRunner
        from pkb.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["relate", "scan", "--kb", "test"])
        # Will fail until implemented, but verifies CLI structure
        assert result.exit_code != 2  # Not "no such command"

    def test_relate_list_displays_relations(self):
        """pkb relate list should display relations."""
        from click.testing import CliRunner
        from pkb.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["relate", "list"])
        assert result.exit_code != 2

    def test_relate_show_displays_bundle_relations(self):
        """pkb relate show <bundle_id> should show relations for a bundle."""
        from click.testing import CliRunner
        from pkb.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["relate", "show", "20260101-test-abc1"])
        assert result.exit_code != 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_commands.py::TestRelateCommands -v`
Expected: FAIL — no such command "relate"

**Step 3: Write implementation**

Add to `src/pkb/cli.py` (after the `dedup` commands, before `search`):

```python
@cli.group()
def relate() -> None:
    """Knowledge graph: discover and manage bundle relations."""


@relate.command("scan")
@click.option("--kb", default=None, help="Knowledge base name filter.")
@click.option(
    "--type",
    "relation_type",
    type=click.Choice(["similar", "related", "all"]),
    default="all",
    help="Relation type to scan for.",
)
def relate_scan(kb: str | None, relation_type: str) -> None:
    """Scan bundles to discover relations (similar/related)."""
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.chromadb_client import ChunkStore
    from pkb.db.postgres import BundleRepository
    from pkb.relations import RelationBuilder

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
        chunk_store = ChunkStore(config.database.chromadb)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    builder = RelationBuilder(
        repo=repo, chunk_store=chunk_store, config=config.relations,
    )

    click.echo("Scanning for relations...")
    stats = builder.scan(kb=kb)
    click.echo(
        f"Done: {stats['scanned']} bundles scanned, "
        f"{stats['new_relations']} relations found."
    )
    repo.close()


@relate.command("list")
@click.option(
    "--type",
    "relation_type",
    type=click.Choice(["similar", "related", "all"]),
    default="all",
    help="Filter by relation type.",
)
@click.option("--kb", default=None, help="Knowledge base name filter.")
def relate_list(relation_type: str, kb: str | None) -> None:
    """List all discovered relations."""
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    filter_type = None if relation_type == "all" else relation_type
    relations = repo.list_all_relations(relation_type=filter_type, kb=kb)

    if not relations:
        click.echo("No relations found.")
    else:
        for r in relations:
            click.echo(
                f"  {r['source_bundle_id']} → {r['target_bundle_id']}  "
                f"({r['relation_type']}, score: {r['score']:.2f})"
            )
        click.echo(f"\nTotal: {len(relations)} relation(s)")
    repo.close()


@relate.command("show")
@click.argument("bundle_id")
@click.option(
    "--type",
    "relation_type",
    type=click.Choice(["similar", "related", "all"]),
    default="all",
    help="Filter by relation type.",
)
def relate_show(bundle_id: str, relation_type: str) -> None:
    """Show relations for a specific bundle."""
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    filter_type = None if relation_type == "all" else relation_type
    relations = repo.list_relations(bundle_id, relation_type=filter_type)

    if not relations:
        click.echo(f"No relations for {bundle_id}")
    else:
        click.echo(f"Relations for {bundle_id}:")
        for r in relations:
            other = (
                r["target_bundle_id"]
                if r["source_bundle_id"] == bundle_id
                else r["source_bundle_id"]
            )
            click.echo(
                f"  → {other}  ({r['relation_type']}, score: {r['score']:.2f})"
            )
        click.echo(f"\nTotal: {len(relations)} relation(s)")
    repo.close()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_commands.py::TestRelateCommands -v`
Expected: PASS (commands are recognized, though DB calls will be mocked/fail gracefully)

**Step 5: Commit**

```
feat(cli): add 'pkb relate' commands (scan/list/show)
```

---

### Task 7: Add relations Web UI route

**Files:**
- Create: `src/pkb/web/routes/relations.py`
- Create: `src/pkb/web/templates/relations/list.html`
- Create: `src/pkb/web/templates/relations/detail.html`
- Modify: `src/pkb/web/app.py` (register router)
- Test: `tests/test_web_relations.py`

**Step 1: Write the failing test**

Create `tests/test_web_relations.py`:

```python
"""Tests for relations web routes."""

from unittest.mock import MagicMock, patch

import pytest


class TestRelationsRoutes:
    def test_relations_list_route_exists(self):
        from pkb.web.routes.relations import router
        paths = [r.path for r in router.routes]
        assert "" in paths or "/" in paths

    def test_relations_detail_route_exists(self):
        from pkb.web.routes.relations import router
        paths = [r.path for r in router.routes]
        assert "/{bundle_id}" in paths
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_relations.py -v`
Expected: FAIL — module not found

**Step 3: Write implementation**

Create `src/pkb/web/routes/relations.py`:

```python
"""Web routes for bundle relations (knowledge graph)."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/relations", tags=["relations"])


@router.get("")
def relations_list(request: Request, relation_type: str = "all"):
    """List all bundle relations."""
    pkb = request.app.state.pkb
    templates = request.app.state.templates

    filter_type = None if relation_type == "all" else relation_type
    relations = pkb.repo.list_all_relations(relation_type=filter_type)
    count = pkb.repo.count_relations()

    return templates.TemplateResponse(request, "relations/list.html", {
        "relations": relations,
        "total": count,
        "current_type": relation_type,
    })


@router.get("/{bundle_id}")
def relations_detail(request: Request, bundle_id: str):
    """Show relations for a specific bundle."""
    pkb = request.app.state.pkb
    templates = request.app.state.templates

    relations = pkb.repo.list_relations(bundle_id)
    bundle = pkb.repo.get_bundle_by_id(bundle_id)

    return templates.TemplateResponse(request, "relations/detail.html", {
        "bundle_id": bundle_id,
        "bundle": bundle,
        "relations": relations,
    })


@router.get("/api/graph")
def relations_graph_json(request: Request, kb: str | None = None):
    """Return relation graph as JSON for D3.js visualization."""
    pkb = request.app.state.pkb

    relations = pkb.repo.list_all_relations(kb=kb)

    nodes = set()
    edges = []
    for r in relations:
        nodes.add(r["source_bundle_id"])
        nodes.add(r["target_bundle_id"])
        edges.append({
            "source": r["source_bundle_id"],
            "target": r["target_bundle_id"],
            "type": r["relation_type"],
            "score": r["score"],
        })

    return {
        "nodes": [{"id": n} for n in nodes],
        "edges": edges,
    }
```

Create `src/pkb/web/templates/relations/list.html`:

```html
{% extends "base.html" %}
{% block title %}Relations — PKB{% endblock %}
{% block content %}
<h1>Bundle Relations</h1>
<p>Total: {{ total }} relation(s)</p>

<div>
  <a href="/relations?relation_type=all">All</a> |
  <a href="/relations?relation_type=similar">Similar</a> |
  <a href="/relations?relation_type=related">Related</a>
</div>

<table>
  <thead>
    <tr>
      <th>Source</th>
      <th>Target</th>
      <th>Type</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    {% for r in relations %}
    <tr>
      <td><a href="/relations/{{ r.source_bundle_id }}">{{ r.source_bundle_id }}</a></td>
      <td><a href="/relations/{{ r.target_bundle_id }}">{{ r.target_bundle_id }}</a></td>
      <td>{{ r.relation_type }}</td>
      <td>{{ "%.2f"|format(r.score) }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
{% endblock %}
```

Create `src/pkb/web/templates/relations/detail.html`:

```html
{% extends "base.html" %}
{% block title %}Relations: {{ bundle_id }} — PKB{% endblock %}
{% block content %}
<h1>Relations for {{ bundle_id }}</h1>
{% if bundle %}
<p><strong>Question:</strong> {{ bundle.question }}</p>
{% endif %}

<table>
  <thead>
    <tr>
      <th>Related Bundle</th>
      <th>Type</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    {% for r in relations %}
    <tr>
      {% if r.source_bundle_id == bundle_id %}
      <td><a href="/relations/{{ r.target_bundle_id }}">{{ r.target_bundle_id }}</a></td>
      {% else %}
      <td><a href="/relations/{{ r.source_bundle_id }}">{{ r.source_bundle_id }}</a></td>
      {% endif %}
      <td>{{ r.relation_type }}</td>
      <td>{{ "%.2f"|format(r.score) }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>

<p><a href="/relations">← Back to all relations</a></p>
{% endblock %}
```

Modify `src/pkb/web/app.py` — register router:

```python
from pkb.web.routes.relations import router as relations_router
# ...
app.include_router(relations_router)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_web_relations.py -v`
Expected: PASS

**Step 5: Commit**

```
feat(web): add relations routes and templates
```

---

### Task 8: Ruff/lint pass + full test suite

**Files:**
- All modified files

**Step 1: Run ruff check**

Run: `ruff check src/ tests/`
Expected: No errors (fix any that appear)

**Step 2: Run full mock test suite**

Run: `pytest tests/ -v --ignore=tests/integration`
Expected: All pass

**Step 3: Run integration tests (if Docker available)**

Run: `PKB_DB_INTEGRATION=1 pytest tests/integration/db/ -v`
Expected: All pass

**Step 4: Commit**

```
chore: lint and test cleanup for Phase 5
```

---

### Task 9: Update documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/design-v1.md`

**Step 1: Update CLAUDE.md**

Add to "Development Commands" section:
```bash
pkb relate scan --kb <name>       # Scan for bundle relations (similar/related)
pkb relate list --type similar    # List discovered relations
pkb relate show <bundle_id>       # Show relations for a specific bundle
```

Add `bundle_relations` to the "Migrations" section.

Update test counts.

Update "Phased Implementation Plan":
```
- **Phase 5** ✓: Knowledge Graph — bundle relations (similar, related), relation CLI, web UI
```

**Step 2: Update docs/design-v1.md**

Add section for bundle_relations schema to the database section.

**Step 3: Commit**

```
docs: update CLAUDE.md and design-v1.md for Phase 5
```

---

## Summary

| Task | Description | New Files | Modified Files |
|------|-------------|-----------|----------------|
| 1 | RelationConfig | — | `models/config.py`, `tests/test_config.py` |
| 2 | DB migration 0004 | `migrations/versions/0004_*.py` | `db/schema.py`, `conftest.py` |
| 3 | Repo methods | `tests/test_db_relations.py`, `tests/integration/.../test_relations_integration.py` | `db/postgres.py` |
| 4 | RelationBuilder | `src/pkb/relations.py`, `tests/test_relations.py` | — |
| 5 | find_bundles_sharing_topics | — | `db/postgres.py`, tests |
| 6 | CLI commands | — | `cli.py`, `tests/test_cli_commands.py` |
| 7 | Web UI routes | `web/routes/relations.py`, templates | `web/app.py`, tests |
| 8 | Lint + test pass | — | various |
| 9 | Documentation | — | `CLAUDE.md`, `docs/design-v1.md` |
