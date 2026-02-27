"""PostgreSQL repository for PKB bundle metadata."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from typing import TYPE_CHECKING

import psycopg
from psycopg_pool import ConnectionPool

from pkb.db.schema import DROP_TABLES_SQL
from pkb.models.config import PostgresConfig

if TYPE_CHECKING:
    from collections.abc import Generator

    from pkb.models.config import ConcurrencyConfig


class BundleRepository:
    """Repository for bundle metadata in PostgreSQL.

    Schema management is handled by Alembic migrations (see `pkb db upgrade`).
    Supports both single-connection and connection-pool modes.
    """

    def __init__(self, config: PostgresConfig) -> None:
        self._conn = psycopg.connect(config.get_dsn())
        self._conn.autocommit = True
        self._pool: ConnectionPool | None = None

    @classmethod
    def from_pool(
        cls, config: PostgresConfig, concurrency: ConcurrencyConfig,
    ) -> BundleRepository:
        """Create a repository backed by a connection pool."""
        repo = cls.__new__(cls)
        repo._conn = None  # type: ignore[assignment]
        repo._pool = ConnectionPool(
            conninfo=config.get_dsn(),
            min_size=concurrency.db_pool_min,
            max_size=concurrency.db_pool_max,
        )
        return repo

    @contextmanager
    def _get_conn(self) -> Generator[psycopg.Connection, None, None]:
        """Get a database connection (from pool or single connection)."""
        if self._pool is not None:
            with self._pool.connection() as conn:
                conn.autocommit = True
                yield conn
        else:
            yield self._conn

    def drop_schema(self) -> None:
        """Drop all tables (for testing/reset)."""
        with self._get_conn() as conn:
            conn.execute(DROP_TABLES_SQL)

    def upsert_bundle(
        self,
        *,
        bundle_id: str,
        kb: str,
        question: str,
        summary: str | None,
        created_at: datetime,
        response_count: int,
        path: str,
        question_hash: str,
        domains: list[str],
        topics: list[str],
        responses: list[dict],
        pending_topics: list[str] | None = None,
        source_path: str | None = None,
        stable_id: str | None = None,
    ) -> None:
        """Insert or update a bundle and its related data."""
        with self._get_conn() as conn:
            # Upsert main bundle
            conn.execute(
                """
                INSERT INTO bundles (id, kb, question, summary, created_at, response_count, path,
                                     question_hash, stable_id, source_path, meta_version)
                VALUES (%(id)s, %(kb)s, %(question)s, %(summary)s, %(created_at)s,
                        %(response_count)s, %(path)s, %(question_hash)s, %(stable_id)s,
                        %(source_path)s, 1)
                ON CONFLICT (id) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    updated_at = NOW(),
                    response_count = EXCLUDED.response_count,
                    question_hash = EXCLUDED.question_hash,
                    stable_id = COALESCE(EXCLUDED.stable_id, bundles.stable_id),
                    source_path = EXCLUDED.source_path
                """,
                {
                    "id": bundle_id,
                    "kb": kb,
                    "question": question,
                    "summary": summary,
                    "created_at": created_at,
                    "response_count": response_count,
                    "path": path,
                    "question_hash": question_hash,
                    "stable_id": stable_id or question_hash,
                    "source_path": source_path,
                },
            )

            # Replace domains (deduplicate)
            conn.execute(
                "DELETE FROM bundle_domains WHERE bundle_id = %s", (bundle_id,)
            )
            for domain in dict.fromkeys(domains):
                conn.execute(
                    "INSERT INTO bundle_domains (bundle_id, domain) VALUES (%s, %s)",
                    (bundle_id, domain),
                )

            # Replace topics (deduplicate; pending that overlap with topics are dropped)
            conn.execute(
                "DELETE FROM bundle_topics WHERE bundle_id = %s", (bundle_id,)
            )
            unique_topics = list(dict.fromkeys(topics))
            topic_set = set(unique_topics)
            for topic in unique_topics:
                conn.execute(
                    "INSERT INTO bundle_topics (bundle_id, topic, is_pending) VALUES (%s, %s, %s)",
                    (bundle_id, topic, False),
                )
            for topic in dict.fromkeys(pending_topics or []):
                if topic not in topic_set:
                    conn.execute(
                        "INSERT INTO bundle_topics (bundle_id, topic, is_pending)"
                        " VALUES (%s, %s, %s)",
                        (bundle_id, topic, True),
                    )

            # Replace responses
            conn.execute(
                "DELETE FROM bundle_responses WHERE bundle_id = %s", (bundle_id,)
            )
            for resp in responses:
                conn.execute(
                    """INSERT INTO bundle_responses
                       (bundle_id, platform, model, turn_count, source_path)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (bundle_id, resp["platform"], resp.get("model"),
                     resp.get("turn_count"), resp.get("source_path")),
                )

    def search_fts(
        self,
        *,
        query: str,
        kb: str | None = None,
        domains: list[str] | None = None,
        topics: list[str] | None = None,
        after: date | None = None,
        before: date | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Full-text search on bundles using tsvector.

        Returns list of dicts with: bundle_id, kb, question, summary,
        created_at, domains (comma-separated), topics (comma-separated), rank.
        """
        # Build tsquery: split on whitespace, join with &
        words = query.strip().split()
        tsquery_str = " & ".join(words)

        conditions = ["b.tsv @@ to_tsquery('simple', %(tsquery)s)"]
        params: dict = {"tsquery": tsquery_str, "limit": limit}

        if kb:
            conditions.append("b.kb = %(kb)s")
            params["kb"] = kb
        if domains:
            conditions.append(
                "EXISTS (SELECT 1 FROM bundle_domains bd "
                "WHERE bd.bundle_id = b.id AND bd.domain = ANY(%(domains)s))"
            )
            params["domains"] = domains
        if topics:
            conditions.append(
                "EXISTS (SELECT 1 FROM bundle_topics bt "
                "WHERE bt.bundle_id = b.id AND bt.topic = ANY(%(topics)s))"
            )
            params["topics"] = topics
        if after:
            conditions.append("b.created_at >= %(after)s")
            params["after"] = after
        if before:
            conditions.append("b.created_at <= %(before)s")
            params["before"] = before

        where_clause = " AND ".join(conditions)

        sql = f"""
            SELECT b.id, b.kb, b.question, b.summary, b.created_at,
                   COALESCE(
                       (SELECT string_agg(bd.domain, ',') FROM bundle_domains bd
                        WHERE bd.bundle_id = b.id), ''
                   ) AS domains,
                   COALESCE(
                       (SELECT string_agg(bt.topic, ',') FROM bundle_topics bt
                        WHERE bt.bundle_id = b.id AND bt.is_pending = FALSE), ''
                   ) AS topics,
                   ts_rank(b.tsv, to_tsquery('simple', %(tsquery)s)) AS rank
            FROM bundles b
            WHERE {where_clause}
            ORDER BY rank DESC
            LIMIT %(limit)s
        """

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "bundle_id": row[0],
                "kb": row[1],
                "question": row[2],
                "summary": row[3],
                "created_at": row[4],
                "domains": row[5],
                "topics": row[6],
                "rank": row[7],
            }
            for row in rows
        ]

    def get_bundle_by_id(self, bundle_id: str) -> dict | None:
        """Get a bundle by its ID with domains and topics."""
        sql = """
            SELECT b.id, b.kb, b.question, b.summary, b.created_at,
                   COALESCE(
                       (SELECT string_agg(bd.domain, ',') FROM bundle_domains bd
                        WHERE bd.bundle_id = b.id), ''
                   ) AS domains,
                   COALESCE(
                       (SELECT string_agg(bt.topic, ',') FROM bundle_topics bt
                        WHERE bt.bundle_id = b.id AND bt.is_pending = FALSE), ''
                   ) AS topics
            FROM bundles b
            WHERE b.id = %s
        """
        with self._get_conn() as conn:
            row = conn.execute(sql, (bundle_id,)).fetchone()
        if row is None:
            return None
        return {
            "bundle_id": row[0],
            "kb": row[1],
            "question": row[2],
            "summary": row[3],
            "created_at": row[4],
            "domains": row[5],
            "topics": row[6],
        }

    def list_all_bundle_ids(self, kb: str | None = None) -> list[str]:
        """List all bundle IDs, optionally filtered by KB."""
        with self._get_conn() as conn:
            if kb:
                rows = conn.execute(
                    "SELECT id FROM bundles WHERE kb = %s ORDER BY id", (kb,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id FROM bundles ORDER BY id"
                ).fetchall()
        return [row[0] for row in rows]

    def count_by_kb(self, kb: str) -> int:
        """Count bundles belonging to a KB."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT count(*) FROM bundles WHERE kb = %s", (kb,)
            ).fetchone()
        return row[0]

    def delete_by_kb(self, kb: str) -> int:
        """Delete all bundles for a KB (CASCADE handles related tables).

        Returns the number of deleted bundles.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                "DELETE FROM bundles WHERE kb = %s RETURNING id", (kb,)
            ).fetchall()
        return len(rows)

    def rename_domain(self, old: str, new: str) -> int:
        """Rename a domain in bundle_domains. Returns affected row count.

        If a bundle already has the new domain, the old row is deleted
        instead of updated (to avoid unique constraint violation).
        """
        with self._get_conn() as conn:
            # First: delete old-domain rows where bundle already has new-domain
            conn.execute(
                """
                DELETE FROM bundle_domains
                WHERE domain = %s
                  AND bundle_id IN (
                      SELECT bundle_id FROM bundle_domains WHERE domain = %s
                  )
                """,
                (old, new),
            )
            # Then: rename remaining old → new
            result = conn.execute(
                "UPDATE bundle_domains SET domain = %s WHERE domain = %s",
                (new, old),
            )
        return result.rowcount

    def delete_bundle(self, bundle_id: str) -> None:
        """Delete a bundle and all related data (CASCADE)."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM bundles WHERE id = %s", (bundle_id,))

    def update_bundle_meta(
        self,
        *,
        bundle_id: str,
        summary: str,
        domains: list[str],
        topics: list[str],
        pending_topics: list[str] | None = None,
        question: str | None = None,
        question_hash: str | None = None,
        source_path: str | None = None,
    ) -> None:
        """Update bundle metadata (summary, domains, topics) from frontmatter edits.

        Optional params (question, question_hash, source_path) are for the
        reingest UPDATE path — when None, existing values are preserved.
        """
        with self._get_conn() as conn:
            conn.execute(
                """
                UPDATE bundles SET
                    summary = %(summary)s,
                    question = COALESCE(%(question)s, question),
                    question_hash = COALESCE(%(question_hash)s, question_hash),
                    source_path = COALESCE(%(source_path)s, source_path),
                    updated_at = NOW()
                WHERE id = %(id)s
                """,
                {
                    "id": bundle_id,
                    "summary": summary,
                    "question": question,
                    "question_hash": question_hash,
                    "source_path": source_path,
                },
            )

            # Replace domains (deduplicate)
            conn.execute(
                "DELETE FROM bundle_domains WHERE bundle_id = %s", (bundle_id,)
            )
            for domain in dict.fromkeys(domains):
                conn.execute(
                    "INSERT INTO bundle_domains (bundle_id, domain) VALUES (%s, %s)",
                    (bundle_id, domain),
                )

            # Replace topics (deduplicate; pending that overlap with topics are dropped)
            conn.execute(
                "DELETE FROM bundle_topics WHERE bundle_id = %s", (bundle_id,)
            )
            unique_topics = list(dict.fromkeys(topics))
            topic_set = set(unique_topics)
            for topic in unique_topics:
                conn.execute(
                    "INSERT INTO bundle_topics (bundle_id, topic, is_pending) VALUES (%s, %s, %s)",
                    (bundle_id, topic, False),
                )
            for topic in dict.fromkeys(pending_topics or []):
                if topic not in topic_set:
                    conn.execute(
                        "INSERT INTO bundle_topics (bundle_id, topic, is_pending)"
                        " VALUES (%s, %s, %s)",
                        (bundle_id, topic, True),
                    )

    def upsert_topic_vocab(
        self,
        *,
        canonical: str,
        status: str,
        merged_into: str | None = None,
    ) -> None:
        """Insert or update a topic vocabulary entry."""
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO topic_vocab (canonical, status, merged_into)
                VALUES (%(canonical)s, %(status)s, %(merged_into)s)
                ON CONFLICT (canonical) DO UPDATE SET
                    status = EXCLUDED.status,
                    merged_into = EXCLUDED.merged_into
                """,
                {"canonical": canonical, "status": status, "merged_into": merged_into},
            )

    def delete_topic_vocab(self, canonical: str) -> None:
        """Delete a topic from the vocabulary table."""
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM topic_vocab WHERE canonical = %s", (canonical,)
            )

    def merge_topic_references(self, old_canonical: str, new_canonical: str) -> None:
        """Transfer bundle_topics references from old topic to new topic."""
        with self._get_conn() as conn:
            # Update existing references, skip if target already exists (PK conflict)
            conn.execute(
                """
                UPDATE bundle_topics SET topic = %s, is_pending = FALSE
                WHERE topic = %s
                AND NOT EXISTS (
                    SELECT 1 FROM bundle_topics bt2
                    WHERE bt2.bundle_id = bundle_topics.bundle_id AND bt2.topic = %s
                )
                """,
                (new_canonical, old_canonical, new_canonical),
            )
            # Remove any remaining old references (duplicates that couldn't be updated)
            conn.execute(
                "DELETE FROM bundle_topics WHERE topic = %s", (old_canonical,)
            )

    def approve_pending_topic(self, canonical: str) -> None:
        """Mark a topic as non-pending in all bundle_topics rows."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE bundle_topics SET is_pending = FALSE "
                "WHERE topic = %s AND is_pending = TRUE",
                (canonical,),
            )

    def remove_topic_from_bundles(self, canonical: str) -> None:
        """Remove all bundle_topics references for a topic."""
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM bundle_topics WHERE topic = %s", (canonical,)
            )

    def insert_duplicate_pair(
        self, bundle_a: str, bundle_b: str, similarity: float,
    ) -> None:
        """Insert a duplicate pair (skip if already exists)."""
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO duplicate_pairs (bundle_a, bundle_b, similarity)
                VALUES (%s, %s, %s)
                ON CONFLICT (bundle_a, bundle_b) DO NOTHING
                """,
                (bundle_a, bundle_b, similarity),
            )

    def list_duplicate_pairs(self, status: str | None = None) -> list[dict]:
        """List duplicate pairs, optionally filtered by status."""
        with self._get_conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT id, bundle_a, bundle_b, similarity, status, resolved_at "
                    "FROM duplicate_pairs WHERE status = %s ORDER BY id",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, bundle_a, bundle_b, similarity, status, resolved_at "
                    "FROM duplicate_pairs ORDER BY id"
                ).fetchall()
        return [
            {
                "id": row[0],
                "bundle_a": row[1],
                "bundle_b": row[2],
                "similarity": row[3],
                "status": row[4],
                "resolved_at": row[5],
            }
            for row in rows
        ]

    def update_duplicate_status(self, pair_id: int, status: str) -> None:
        """Update the status of a duplicate pair."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE duplicate_pairs SET status = %s, resolved_at = NOW() WHERE id = %s",
                (status, pair_id),
            )

    # ── Bundle Relations (Knowledge Graph) ──────────────────────────

    def insert_relation(
        self,
        source_bundle_id: str,
        target_bundle_id: str,
        relation_type: str,
        score: float,
    ) -> None:
        """Insert a bundle relation (upsert: update score if exists)."""
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
            conditions: list[str] = []
            params: list = []
            if relation_type:
                conditions.append("br.relation_type = %s")
                params.append(relation_type)
            if kb:
                conditions.append("(bs.kb = %s OR bt.kb = %s)")
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
                    "SELECT COUNT(*) FROM bundle_relations "
                    "WHERE relation_type = %s",
                    (relation_type,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM bundle_relations"
                ).fetchone()
        return row[0] if row else 0

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

    def list_bundles_by_domain(
        self,
        domain: str,
        kb: str | None = None,
    ) -> list[dict]:
        """List bundles belonging to a domain, optionally filtered by KB."""
        with self._get_conn() as conn:
            if kb:
                rows = conn.execute(
                    "SELECT b.id AS bundle_id, b.kb, b.question, b.summary, "
                    "b.created_at "
                    "FROM bundles b "
                    "JOIN bundle_domains bd ON bd.bundle_id = b.id "
                    "WHERE bd.domain = %s AND b.kb = %s "
                    "ORDER BY b.created_at DESC",
                    (domain, kb),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT b.id AS bundle_id, b.kb, b.question, b.summary, "
                    "b.created_at "
                    "FROM bundles b "
                    "JOIN bundle_domains bd ON bd.bundle_id = b.id "
                    "WHERE bd.domain = %s "
                    "ORDER BY b.created_at DESC",
                    (domain,),
                ).fetchall()
        return [
            {
                "bundle_id": row[0],
                "kb": row[1],
                "question": row[2],
                "summary": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    def find_by_source_path(self, source_path: str) -> str | None:
        """Find a bundle ID by its source file path.

        Checks bundle_responses first (tracks per-platform merged files),
        then falls back to bundles table (tracks first-ingested file).

        Returns the bundle_id if found, None otherwise.
        """
        with self._get_conn() as conn:
            # Check bundle_responses first (merged files have source_path here)
            row = conn.execute(
                "SELECT bundle_id FROM bundle_responses "
                "WHERE source_path = %s LIMIT 1",
                (source_path,),
            ).fetchone()
            if row:
                return row[0]
            # Fallback to bundles table (first-ingested file)
            row = conn.execute(
                "SELECT id FROM bundles WHERE source_path = %s LIMIT 1",
                (source_path,),
            ).fetchone()
        return row[0] if row else None

    def bundle_exists(self, question_hash: str) -> bool:
        """Check if a bundle with this question hash already exists."""
        with self._get_conn() as conn:
            result = conn.execute(
                "SELECT 1 FROM bundles WHERE question_hash = %s LIMIT 1",
                (question_hash,),
            ).fetchone()
        return result is not None

    def find_bundle_by_question_hash(self, question_hash: str) -> dict | None:
        """Find a bundle by question_hash, including its platforms.

        Used for merge logic: when a file has the same question_hash but different
        platform, we merge into the existing bundle instead of skipping.

        Returns dict with bundle_id, kb, path, platforms (list), domains, topics,
        or None if not found.
        """
        sql = """
            SELECT b.id, b.kb, b.path,
                   COALESCE(
                       (SELECT string_agg(br.platform, ',') FROM bundle_responses br
                        WHERE br.bundle_id = b.id), ''
                   ) AS platforms,
                   COALESCE(
                       (SELECT string_agg(bd.domain, ',') FROM bundle_domains bd
                        WHERE bd.bundle_id = b.id), ''
                   ) AS domains,
                   COALESCE(
                       (SELECT string_agg(bt.topic, ',') FROM bundle_topics bt
                        WHERE bt.bundle_id = b.id AND bt.is_pending = FALSE), ''
                   ) AS topics
            FROM bundles b
            LEFT JOIN bundle_responses br ON br.bundle_id = b.id
            WHERE b.question_hash = %s
            LIMIT 1
        """
        with self._get_conn() as conn:
            row = conn.execute(sql, (question_hash,)).fetchone()
        if row is None:
            return None
        platforms = [p for p in row[3].split(",") if p] if row[3] else []
        domains = [d for d in row[4].split(",") if d] if row[4] else []
        topics = [t for t in row[5].split(",") if t] if row[5] else []
        return {
            "bundle_id": row[0],
            "kb": row[1],
            "path": row[2],
            "platforms": platforms,
            "domains": domains,
            "topics": topics,
        }

    def find_bundle_by_stable_id(self, stable_id: str) -> dict | None:
        """Find a bundle by stable_id, including its platforms.

        Used for content-based bundle lookup: stable_id is derived from
        conversation content and remains constant across re-ingestion.

        Returns dict with bundle_id, kb, path, platforms (list), domains, topics,
        or None if not found.
        """
        sql = """
            SELECT b.id, b.kb, b.path,
                   COALESCE(
                       (SELECT string_agg(br.platform, ',') FROM bundle_responses br
                        WHERE br.bundle_id = b.id), ''
                   ) AS platforms,
                   COALESCE(
                       (SELECT string_agg(bd.domain, ',') FROM bundle_domains bd
                        WHERE bd.bundle_id = b.id), ''
                   ) AS domains,
                   COALESCE(
                       (SELECT string_agg(bt.topic, ',') FROM bundle_topics bt
                        WHERE bt.bundle_id = b.id AND bt.is_pending = FALSE), ''
                   ) AS topics
            FROM bundles b
            WHERE b.stable_id = %s
            LIMIT 1
        """
        with self._get_conn() as conn:
            row = conn.execute(sql, (stable_id,)).fetchone()
        if row is None:
            return None
        platforms = [p for p in row[3].split(",") if p] if row[3] else []
        domains = [d for d in row[4].split(",") if d] if row[4] else []
        topics = [t for t in row[5].split(",") if t] if row[5] else []
        return {
            "bundle_id": row[0],
            "kb": row[1],
            "path": row[2],
            "platforms": platforms,
            "domains": domains,
            "topics": topics,
        }

    def add_response_to_bundle(
        self,
        *,
        bundle_id: str,
        platform: str,
        model: str | None = None,
        turn_count: int = 0,
        source_path: str | None = None,
    ) -> None:
        """Add a new platform response to an existing bundle (merge).

        Inserts into bundle_responses and updates response_count.
        """
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO bundle_responses
                       (bundle_id, platform, model, turn_count, source_path)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (bundle_id, platform) DO UPDATE SET
                       model = EXCLUDED.model,
                       turn_count = EXCLUDED.turn_count,
                       source_path = EXCLUDED.source_path""",
                (bundle_id, platform, model, turn_count, source_path),
            )
            conn.execute(
                """UPDATE bundles SET
                       response_count = (
                           SELECT COUNT(*) FROM bundle_responses WHERE bundle_id = %s
                       ),
                       updated_at = NOW()
                   WHERE id = %s""",
                (bundle_id, bundle_id),
            )

    # ─── Analytics aggregates ──────────────────────────────

    def count_bundles_for_topics(self, topics: list[str]) -> dict[str, int]:
        """Count bundles for specific topics.

        Returns a dict mapping topic name to its bundle count.
        Topics with 0 bundles are omitted from the result.
        Empty input returns empty dict.
        """
        if not topics:
            return {}
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT topic, COUNT(DISTINCT bundle_id) AS cnt "
                "FROM bundle_topics "
                "WHERE topic = ANY(%(topics)s) "
                "GROUP BY topic",
                {"topics": topics},
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def count_bundles_by_domain(self, kb: str | None = None) -> list[dict]:
        """Count bundles per domain, ordered by count descending."""
        with self._get_conn() as conn:
            if kb:
                rows = conn.execute(
                    "SELECT bd.domain, COUNT(*) AS cnt "
                    "FROM bundle_domains bd "
                    "JOIN bundles b ON b.id = bd.bundle_id "
                    "WHERE b.kb = %s "
                    "GROUP BY bd.domain ORDER BY cnt DESC",
                    (kb,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT domain, COUNT(*) AS cnt "
                    "FROM bundle_domains "
                    "GROUP BY domain ORDER BY cnt DESC",
                ).fetchall()
        return [{"domain": row[0], "count": row[1]} for row in rows]

    def count_bundles_by_topic(self, kb: str | None = None) -> list[dict]:
        """Count bundles per topic, ordered by count descending."""
        with self._get_conn() as conn:
            if kb:
                rows = conn.execute(
                    "SELECT bt.topic, COUNT(*) AS cnt "
                    "FROM bundle_topics bt "
                    "JOIN bundles b ON b.id = bt.bundle_id "
                    "WHERE b.kb = %s "
                    "GROUP BY bt.topic ORDER BY cnt DESC",
                    (kb,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT topic, COUNT(*) AS cnt "
                    "FROM bundle_topics "
                    "GROUP BY topic ORDER BY cnt DESC",
                ).fetchall()
        return [{"topic": row[0], "count": row[1]} for row in rows]

    def count_bundles_by_month(
        self, kb: str | None = None, months: int = 6,
    ) -> list[dict]:
        """Count bundles per month for the last N months."""
        with self._get_conn() as conn:
            if kb:
                rows = conn.execute(
                    "SELECT TO_CHAR(created_at, 'YYYY-MM') AS month, "
                    "COUNT(*) AS cnt FROM bundles "
                    "WHERE created_at >= NOW() - make_interval(months => %s) "
                    "AND kb = %s "
                    "GROUP BY month ORDER BY month",
                    (months, kb),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT TO_CHAR(created_at, 'YYYY-MM') AS month, "
                    "COUNT(*) AS cnt FROM bundles "
                    "WHERE created_at >= NOW() - make_interval(months => %s) "
                    "GROUP BY month ORDER BY month",
                    (months,),
                ).fetchall()
        return [{"month": row[0], "count": row[1]} for row in rows]

    def count_responses_by_platform(self, kb: str | None = None) -> list[dict]:
        """Count responses per platform, ordered by count descending."""
        with self._get_conn() as conn:
            if kb:
                rows = conn.execute(
                    "SELECT br.platform, COUNT(*) AS cnt "
                    "FROM bundle_responses br "
                    "JOIN bundles b ON b.id = br.bundle_id "
                    "WHERE b.kb = %s "
                    "GROUP BY br.platform ORDER BY cnt DESC",
                    (kb,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT platform, COUNT(*) AS cnt "
                    "FROM bundle_responses "
                    "GROUP BY platform ORDER BY cnt DESC",
                ).fetchall()
        return [{"platform": row[0], "count": row[1]} for row in rows]

    def list_bundles_since(
        self, since: datetime, kb: str | None = None,
    ) -> list[dict]:
        """List bundles created since the given datetime."""
        with self._get_conn() as conn:
            if kb:
                rows = conn.execute(
                    "SELECT id, kb, question, summary, created_at "
                    "FROM bundles WHERE created_at >= %s AND kb = %s "
                    "ORDER BY created_at DESC",
                    (since, kb),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, kb, question, summary, created_at "
                    "FROM bundles WHERE created_at >= %s "
                    "ORDER BY created_at DESC",
                    (since,),
                ).fetchall()
        return [
            {
                "bundle_id": row[0],
                "kb": row[1],
                "question": row[2],
                "summary": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    def close(self) -> None:
        """Close the database connection or pool."""
        if self._pool is not None:
            self._pool.close()
        elif self._conn is not None:
            self._conn.close()
