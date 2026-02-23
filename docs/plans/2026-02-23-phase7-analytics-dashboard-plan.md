# Phase 7: Analytics Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a knowledge analytics system that visualizes the user's "knowledge portfolio" — domain/topic distribution, temporal trends, knowledge gaps, and periodic reports — via CLI, Web UI, and generated markdown.

**Architecture:** A new `AnalyticsEngine` aggregates bundle metadata from PostgreSQL for statistical analysis. CLI commands (`pkb stats`, `pkb report`) output summaries and markdown reports. The Web UI dashboard is extended with Chart.js charts. No new DB tables needed — all analytics are computed from existing `bundles`, `bundle_domains`, `bundle_topics`, and `bundle_relations` tables.

**Tech Stack:** PostgreSQL (aggregate queries), Chart.js (web charts), Click (CLI), FastAPI + Jinja2 (web routes), pytest (TDD)

**Depends on:** Phase 5 (bundle_relations for gap/cluster analysis), Phase 6 (DigestEngine for periodic reports)

---

### Task 1: Create AnalyticsEngine module

**Files:**
- Create: `src/pkb/analytics.py`
- Test: `tests/test_analytics.py`

**Step 1: Write the failing tests**

Create `tests/test_analytics.py`:

```python
"""Tests for AnalyticsEngine — knowledge portfolio statistics."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def engine(mock_repo):
    from pkb.analytics import AnalyticsEngine
    return AnalyticsEngine(repo=mock_repo)


class TestDomainDistribution:
    def test_returns_domain_counts(self, engine, mock_repo):
        """domain_distribution should return counts per domain."""
        mock_repo.count_bundles_by_domain.return_value = [
            {"domain": "dev", "count": 500},
            {"domain": "투자", "count": 300},
            {"domain": "학습", "count": 200},
        ]
        result = engine.domain_distribution()
        assert len(result) == 3
        assert result[0]["domain"] == "dev"
        assert result[0]["count"] == 500

    def test_with_kb_filter(self, engine, mock_repo):
        """Should pass KB filter to repository."""
        mock_repo.count_bundles_by_domain.return_value = []
        engine.domain_distribution(kb="personal")
        mock_repo.count_bundles_by_domain.assert_called_once_with(kb="personal")


class TestTopicHeatmap:
    def test_returns_top_topics(self, engine, mock_repo):
        """topic_heatmap should return top N topics by count."""
        mock_repo.count_bundles_by_topic.return_value = [
            {"topic": "python", "count": 100},
            {"topic": "blockchain", "count": 80},
            {"topic": "ai", "count": 60},
        ]
        result = engine.topic_heatmap(top_n=2)
        assert len(result) == 2
        assert result[0]["topic"] == "python"


class TestTemporalTrend:
    def test_returns_monthly_counts(self, engine, mock_repo):
        """temporal_trend should return monthly bundle counts."""
        mock_repo.count_bundles_by_month.return_value = [
            {"month": "2026-01", "count": 50},
            {"month": "2026-02", "count": 30},
        ]
        result = engine.temporal_trend(months=2)
        assert len(result) == 2
        assert result[0]["month"] == "2026-01"


class TestPlatformDistribution:
    def test_returns_platform_counts(self, engine, mock_repo):
        """platform_distribution should return counts per platform."""
        mock_repo.count_responses_by_platform.return_value = [
            {"platform": "claude", "count": 800},
            {"platform": "chatgpt", "count": 600},
        ]
        result = engine.platform_distribution()
        assert len(result) == 2


class TestKnowledgeGaps:
    def test_finds_single_mention_topics(self, engine, mock_repo):
        """knowledge_gaps should find topics with very few bundles."""
        mock_repo.count_bundles_by_topic.return_value = [
            {"topic": "python", "count": 100},
            {"topic": "rust", "count": 1},
            {"topic": "haskell", "count": 2},
        ]
        gaps = engine.knowledge_gaps(threshold=3)
        assert len(gaps) == 2
        assert gaps[0]["topic"] == "rust"

    def test_no_gaps(self, engine, mock_repo):
        mock_repo.count_bundles_by_topic.return_value = [
            {"topic": "python", "count": 100},
        ]
        gaps = engine.knowledge_gaps(threshold=3)
        assert len(gaps) == 0


class TestOverviewStats:
    def test_returns_summary(self, engine, mock_repo):
        """overview should return aggregate stats."""
        mock_repo.list_all_bundle_ids.return_value = ["b1", "b2", "b3"]
        mock_repo.count_relations.return_value = 10
        mock_repo.count_bundles_by_domain.return_value = [
            {"domain": "dev", "count": 2},
            {"domain": "투자", "count": 1},
        ]
        mock_repo.count_bundles_by_topic.return_value = [
            {"topic": "python", "count": 2},
        ]

        result = engine.overview(kb="personal")
        assert result["total_bundles"] == 3
        assert result["total_relations"] == 10
        assert result["domain_count"] == 2
        assert result["topic_count"] == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_analytics.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write implementation**

Create `src/pkb/analytics.py`:

```python
"""Analytics engine — knowledge portfolio statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pkb.db.postgres import BundleRepository


class AnalyticsEngine:
    """Computes knowledge portfolio statistics from bundle metadata."""

    def __init__(self, *, repo: BundleRepository) -> None:
        self._repo = repo

    def domain_distribution(self, kb: str | None = None) -> list[dict]:
        """Count bundles per domain, sorted descending."""
        return self._repo.count_bundles_by_domain(kb=kb)

    def topic_heatmap(
        self, top_n: int = 20, kb: str | None = None,
    ) -> list[dict]:
        """Top N topics by bundle count."""
        all_topics = self._repo.count_bundles_by_topic(kb=kb)
        return all_topics[:top_n]

    def temporal_trend(
        self, months: int = 6, kb: str | None = None,
    ) -> list[dict]:
        """Monthly bundle counts for the last N months."""
        return self._repo.count_bundles_by_month(kb=kb, months=months)

    def platform_distribution(self, kb: str | None = None) -> list[dict]:
        """Count responses per platform."""
        return self._repo.count_responses_by_platform(kb=kb)

    def knowledge_gaps(
        self, threshold: int = 3, kb: str | None = None,
    ) -> list[dict]:
        """Find topics with fewer than `threshold` bundles.

        These represent areas the user asked about but didn't explore deeply.
        """
        all_topics = self._repo.count_bundles_by_topic(kb=kb)
        return [t for t in all_topics if t["count"] < threshold]

    def overview(self, kb: str | None = None) -> dict:
        """Aggregate overview statistics."""
        bundle_ids = self._repo.list_all_bundle_ids(kb=kb)
        relations = self._repo.count_relations()
        domains = self._repo.count_bundles_by_domain(kb=kb)
        topics = self._repo.count_bundles_by_topic(kb=kb)

        return {
            "total_bundles": len(bundle_ids),
            "total_relations": relations,
            "domain_count": len(domains),
            "topic_count": len(topics),
            "top_domains": domains[:5],
            "top_topics": topics[:5],
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_analytics.py -v`
Expected: PASS

**Step 5: Commit**

```
feat: add AnalyticsEngine for knowledge portfolio statistics
```

---

### Task 2: Add aggregate query methods to BundleRepository

**Files:**
- Modify: `src/pkb/db/postgres.py`
- Test: `tests/test_db_analytics.py`
- Test: `tests/integration/db/test_analytics_integration.py`

**Step 1: Write the failing tests**

Create `tests/test_db_analytics.py`:

```python
"""Tests for BundleRepository analytics query methods."""


class TestAnalyticsMethodsExist:
    def test_count_bundles_by_domain(self):
        from pkb.db.postgres import BundleRepository
        assert hasattr(BundleRepository, "count_bundles_by_domain")

    def test_count_bundles_by_topic(self):
        from pkb.db.postgres import BundleRepository
        assert hasattr(BundleRepository, "count_bundles_by_topic")

    def test_count_bundles_by_month(self):
        from pkb.db.postgres import BundleRepository
        assert hasattr(BundleRepository, "count_bundles_by_month")

    def test_count_responses_by_platform(self):
        from pkb.db.postgres import BundleRepository
        assert hasattr(BundleRepository, "count_responses_by_platform")
```

Create `tests/integration/db/test_analytics_integration.py`:

```python
"""Integration tests for analytics aggregate queries."""

from datetime import datetime

import pytest


@pytest.fixture
def _seed_analytics_bundles(repo):
    """Insert test bundles for analytics tests."""
    for i, (domain, topic, platform) in enumerate([
        ("dev", "python", "claude"),
        ("dev", "python", "chatgpt"),
        ("dev", "rust", "claude"),
        ("투자", "blockchain", "chatgpt"),
        ("투자", "blockchain", "gemini"),
    ]):
        bid = f"20260{i+1:02d}01-test-{i:04d}"
        repo.upsert_bundle(
            bundle_id=bid, kb="test",
            question=f"Q{i}", summary=f"S{i}",
            created_at=datetime(2026, i + 1, 1),
            response_count=1, path=f"/b/{bid}", question_hash=f"h{i}",
            domains=[domain], topics=[topic],
            responses=[{"platform": platform, "model": "m", "turn_count": 1}],
        )


class TestCountBundlesByDomain:
    def test_counts(self, repo, _seed_analytics_bundles):
        result = repo.count_bundles_by_domain()
        assert len(result) == 2
        # dev has 3, 투자 has 2
        dev = next(r for r in result if r["domain"] == "dev")
        assert dev["count"] == 3

    def test_kb_filter(self, repo, _seed_analytics_bundles):
        result = repo.count_bundles_by_domain(kb="nonexistent")
        assert len(result) == 0


class TestCountBundlesByTopic:
    def test_counts(self, repo, _seed_analytics_bundles):
        result = repo.count_bundles_by_topic()
        assert len(result) >= 2
        python = next(r for r in result if r["topic"] == "python")
        assert python["count"] == 2


class TestCountBundlesByMonth:
    def test_counts(self, repo, _seed_analytics_bundles):
        result = repo.count_bundles_by_month(months=12)
        assert len(result) >= 1


class TestCountResponsesByPlatform:
    def test_counts(self, repo, _seed_analytics_bundles):
        result = repo.count_responses_by_platform()
        assert len(result) >= 2
        claude = next(r for r in result if r["platform"] == "claude")
        assert claude["count"] == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_db_analytics.py -v`
Expected: FAIL

**Step 3: Write implementation**

Add to `src/pkb/db/postgres.py`:

```python
    def count_bundles_by_domain(self, kb: str | None = None) -> list[dict]:
        """Count bundles per domain, sorted descending."""
        with self._get_conn() as conn:
            if kb:
                rows = conn.execute(
                    "SELECT bd.domain, COUNT(DISTINCT bd.bundle_id) AS cnt "
                    "FROM bundle_domains bd "
                    "JOIN bundles b ON b.id = bd.bundle_id "
                    "WHERE b.kb = %s "
                    "GROUP BY bd.domain ORDER BY cnt DESC",
                    (kb,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT domain, COUNT(DISTINCT bundle_id) AS cnt "
                    "FROM bundle_domains "
                    "GROUP BY domain ORDER BY cnt DESC"
                ).fetchall()
        return [{"domain": row[0], "count": row[1]} for row in rows]

    def count_bundles_by_topic(self, kb: str | None = None) -> list[dict]:
        """Count bundles per topic, sorted descending."""
        with self._get_conn() as conn:
            if kb:
                rows = conn.execute(
                    "SELECT bt.topic, COUNT(DISTINCT bt.bundle_id) AS cnt "
                    "FROM bundle_topics bt "
                    "JOIN bundles b ON b.id = bt.bundle_id "
                    "WHERE b.kb = %s "
                    "GROUP BY bt.topic ORDER BY cnt DESC",
                    (kb,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT topic, COUNT(DISTINCT bundle_id) AS cnt "
                    "FROM bundle_topics "
                    "GROUP BY topic ORDER BY cnt DESC"
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
                    "COUNT(*) AS cnt "
                    "FROM bundles "
                    "WHERE kb = %s "
                    "AND created_at >= NOW() - INTERVAL '%s months' "
                    "GROUP BY month ORDER BY month",
                    (kb, months),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT TO_CHAR(created_at, 'YYYY-MM') AS month, "
                    "COUNT(*) AS cnt "
                    "FROM bundles "
                    "WHERE created_at >= NOW() - INTERVAL '%s months' "
                    "GROUP BY month ORDER BY month",
                    (months,),
                ).fetchall()
        return [{"month": row[0], "count": row[1]} for row in rows]

    def count_responses_by_platform(self, kb: str | None = None) -> list[dict]:
        """Count responses per platform, sorted descending."""
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
                    "GROUP BY platform ORDER BY cnt DESC"
                ).fetchall()
        return [{"platform": row[0], "count": row[1]} for row in rows]
```

**Step 4: Run tests**

Run: `pytest tests/test_db_analytics.py -v`
Expected: PASS

Run: `PKB_DB_INTEGRATION=1 pytest tests/integration/db/test_analytics_integration.py -v`
Expected: PASS

**Step 5: Commit**

```
feat(db): add aggregate query methods for analytics
```

---

### Task 3: Create ReportGenerator module

**Files:**
- Create: `src/pkb/report.py`
- Test: `tests/test_report.py`

**Step 1: Write the failing tests**

Create `tests/test_report.py`:

```python
"""Tests for ReportGenerator — periodic knowledge reports."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def mock_analytics():
    from pkb.analytics import AnalyticsEngine
    return MagicMock(spec=AnalyticsEngine)


@pytest.fixture
def generator(mock_repo, mock_analytics):
    from pkb.report import ReportGenerator
    return ReportGenerator(repo=mock_repo, analytics=mock_analytics)


class TestWeeklyReport:
    def test_generates_markdown(self, generator, mock_repo, mock_analytics):
        """weekly() should produce a markdown report."""
        mock_repo.list_bundles_since.return_value = [
            {"bundle_id": "20260220-test-abc1", "question": "Q1"},
        ]
        mock_analytics.domain_distribution.return_value = [
            {"domain": "dev", "count": 5},
        ]
        mock_analytics.topic_heatmap.return_value = [
            {"topic": "python", "count": 3},
        ]

        report = generator.weekly(kb="personal")
        assert "# " in report  # Has markdown heading
        assert "20260220-test-abc1" in report

    def test_empty_week(self, generator, mock_repo, mock_analytics):
        mock_repo.list_bundles_since.return_value = []
        mock_analytics.domain_distribution.return_value = []
        mock_analytics.topic_heatmap.return_value = []

        report = generator.weekly()
        assert "없" in report or "활동" in report


class TestMonthlyReport:
    def test_generates_markdown(self, generator, mock_repo, mock_analytics):
        """monthly() should produce a markdown report."""
        mock_repo.list_bundles_since.return_value = [
            {"bundle_id": "20260201-test-abc1", "question": "Q1"},
            {"bundle_id": "20260215-test-def2", "question": "Q2"},
        ]
        mock_analytics.domain_distribution.return_value = [
            {"domain": "dev", "count": 10},
        ]
        mock_analytics.topic_heatmap.return_value = [
            {"topic": "python", "count": 5},
        ]
        mock_analytics.knowledge_gaps.return_value = [
            {"topic": "rust", "count": 1},
        ]

        report = generator.monthly(kb="personal")
        assert "# " in report
        assert "dev" in report
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_report.py -v`
Expected: FAIL

**Step 3: Write implementation**

Create `src/pkb/report.py`:

```python
"""Report generator — periodic knowledge activity reports."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pkb.analytics import AnalyticsEngine
    from pkb.db.postgres import BundleRepository


class ReportGenerator:
    """Generates periodic knowledge activity reports in Markdown."""

    def __init__(
        self,
        *,
        repo: BundleRepository,
        analytics: AnalyticsEngine,
    ) -> None:
        self._repo = repo
        self._analytics = analytics

    def weekly(self, kb: str | None = None) -> str:
        """Generate a weekly activity report."""
        since = datetime.now() - timedelta(days=7)
        return self._generate_report(
            title="주간 지식 활동 리포트",
            since=since,
            kb=kb,
        )

    def monthly(self, kb: str | None = None, month: str | None = None) -> str:
        """Generate a monthly activity report."""
        since = datetime.now() - timedelta(days=30)
        return self._generate_report(
            title="월간 지식 활동 리포트",
            since=since,
            kb=kb,
            include_gaps=True,
        )

    def _generate_report(
        self,
        *,
        title: str,
        since: datetime,
        kb: str | None = None,
        include_gaps: bool = False,
    ) -> str:
        """Generate a markdown report."""
        bundles = self._repo.list_bundles_since(since, kb=kb)
        domains = self._analytics.domain_distribution(kb=kb)
        topics = self._analytics.topic_heatmap(top_n=10, kb=kb)

        lines = [
            f"# {title}",
            f"",
            f"기간: {since.strftime('%Y-%m-%d')} ~ {datetime.now().strftime('%Y-%m-%d')}",
            f"",
        ]

        # New bundles section
        lines.append(f"## 새로 추가된 번들 ({len(bundles)}건)")
        lines.append("")
        if bundles:
            for b in bundles[:20]:
                lines.append(f"- `{b['bundle_id']}`: {b.get('question', '(제목 없음)')}")
        else:
            lines.append("이 기간에 새로 추가된 번들이 없습니다.")
        lines.append("")

        # Domain distribution
        lines.append("## 도메인 분포")
        lines.append("")
        if domains:
            for d in domains:
                lines.append(f"- **{d['domain']}**: {d['count']}건")
        lines.append("")

        # Top topics
        lines.append("## 활발한 토픽 Top 10")
        lines.append("")
        if topics:
            for i, t in enumerate(topics, 1):
                lines.append(f"{i}. {t['topic']} ({t['count']}건)")
        lines.append("")

        # Knowledge gaps (monthly only)
        if include_gaps:
            gaps = self._analytics.knowledge_gaps(kb=kb)
            if gaps:
                lines.append("## 지식 공백 (탐구 추천)")
                lines.append("")
                for g in gaps[:10]:
                    lines.append(f"- **{g['topic']}**: {g['count']}건만 — 더 깊이 파보면 좋겠습니다")
                lines.append("")

        return "\n".join(lines)
```

Add `list_bundles_since` to `BundleRepository`:

```python
    def list_bundles_since(
        self, since: datetime, kb: str | None = None,
    ) -> list[dict]:
        """List bundles created since a given datetime."""
        with self._get_conn() as conn:
            if kb:
                rows = conn.execute(
                    "SELECT id AS bundle_id, question, summary, created_at "
                    "FROM bundles WHERE created_at >= %s AND kb = %s "
                    "ORDER BY created_at DESC",
                    (since, kb),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id AS bundle_id, question, summary, created_at "
                    "FROM bundles WHERE created_at >= %s "
                    "ORDER BY created_at DESC",
                    (since,),
                ).fetchall()
        return [
            {
                "bundle_id": row[0],
                "question": row[1],
                "summary": row[2],
                "created_at": row[3],
            }
            for row in rows
        ]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_report.py -v`
Expected: PASS

**Step 5: Commit**

```
feat: add ReportGenerator for periodic knowledge reports
```

---

### Task 4: Add `pkb stats` and `pkb report` CLI commands

**Files:**
- Modify: `src/pkb/cli.py`
- Test: `tests/test_cli_commands.py`

**Step 1: Write the failing test**

```python
class TestStatsCommand:
    def test_stats_command_exists(self):
        from click.testing import CliRunner
        from pkb.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", "--help"])
        assert result.exit_code == 0

class TestReportCommand:
    def test_report_command_exists(self):
        from click.testing import CliRunner
        from pkb.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--help"])
        assert result.exit_code == 0
```

**Step 2: Write implementation**

```python
@cli.command()
@click.option("--kb", default=None, help="Knowledge base name filter.")
@click.option("--domain", default=None, help="Show detail for a domain.")
@click.option("--json", "as_json", is_flag=True, help="JSON output.")
def stats(kb: str | None, domain: str | None, as_json: bool) -> None:
    """Show knowledge base statistics."""
    import json as json_mod

    from pkb.analytics import AnalyticsEngine
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    analytics = AnalyticsEngine(repo=repo)

    if domain:
        bundles = repo.list_bundles_by_domain(domain, kb=kb)
        click.echo(f"\nDomain '{domain}': {len(bundles)} bundles")
        for b in bundles[:20]:
            click.echo(f"  {b['bundle_id']}: {b['question']}")
    else:
        overview = analytics.overview(kb=kb)
        if as_json:
            click.echo(json_mod.dumps(overview, default=str, indent=2, ensure_ascii=False))
        else:
            click.echo(f"\n  Bundles:    {overview['total_bundles']}")
            click.echo(f"  Relations:  {overview['total_relations']}")
            click.echo(f"  Domains:    {overview['domain_count']}")
            click.echo(f"  Topics:     {overview['topic_count']}")
            click.echo("\n  Top Domains:")
            for d in overview.get("top_domains", []):
                click.echo(f"    {d['domain']}: {d['count']}")
            click.echo("\n  Top Topics:")
            for t in overview.get("top_topics", []):
                click.echo(f"    {t['topic']}: {t['count']}")

    repo.close()


@cli.command()
@click.option(
    "--period",
    type=click.Choice(["weekly", "monthly"]),
    default="weekly",
    help="Report period.",
)
@click.option("--kb", default=None, help="Knowledge base name filter.")
@click.option("--output", "-o", default=None, type=click.Path(), help="Save to file.")
def report(period: str, kb: str | None, output: str | None) -> None:
    """Generate a periodic knowledge activity report."""
    from pkb.analytics import AnalyticsEngine
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.report import ReportGenerator

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    analytics = AnalyticsEngine(repo=repo)
    generator = ReportGenerator(repo=repo, analytics=analytics)

    if period == "weekly":
        content = generator.weekly(kb=kb)
    else:
        content = generator.monthly(kb=kb)

    if output:
        from pathlib import Path
        Path(output).write_text(content, encoding="utf-8")
        click.echo(f"Saved to {output}")
    else:
        click.echo(content)

    repo.close()
```

**Step 3: Run test**

Run: `pytest tests/test_cli_commands.py::TestStatsCommand tests/test_cli_commands.py::TestReportCommand -v`
Expected: PASS

**Step 4: Commit**

```
feat(cli): add 'pkb stats' and 'pkb report' commands
```

---

### Task 5: Extend Web UI dashboard with analytics

**Files:**
- Create: `src/pkb/web/routes/analytics.py`
- Create: `src/pkb/web/templates/analytics/dashboard.html`
- Modify: `src/pkb/web/app.py` (register router)
- Test: `tests/test_web_analytics.py`

**Step 1: Write the failing test**

Create `tests/test_web_analytics.py`:

```python
"""Tests for analytics web routes."""


class TestAnalyticsRoutes:
    def test_dashboard_route_exists(self):
        from pkb.web.routes.analytics import router
        paths = [r.path for r in router.routes]
        assert "" in paths or "/" in paths

    def test_api_domains_route_exists(self):
        from pkb.web.routes.analytics import router
        paths = [r.path for r in router.routes]
        assert "/api/domains" in paths

    def test_api_topics_route_exists(self):
        from pkb.web.routes.analytics import router
        paths = [r.path for r in router.routes]
        assert "/api/topics" in paths

    def test_api_trend_route_exists(self):
        from pkb.web.routes.analytics import router
        paths = [r.path for r in router.routes]
        assert "/api/trend" in paths
```

**Step 2: Write implementation**

Create `src/pkb/web/routes/analytics.py`:

```python
"""Web routes for knowledge analytics dashboard."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("")
def analytics_dashboard(request: Request):
    """Render analytics dashboard with Chart.js charts."""
    from pkb.analytics import AnalyticsEngine

    pkb = request.app.state.pkb
    templates = request.app.state.templates
    analytics = AnalyticsEngine(repo=pkb.repo)

    overview = analytics.overview()
    return templates.TemplateResponse(request, "analytics/dashboard.html", {
        "overview": overview,
    })


@router.get("/api/domains")
def api_domains(request: Request, kb: str | None = None):
    """Return domain distribution as JSON for Chart.js."""
    from pkb.analytics import AnalyticsEngine

    pkb = request.app.state.pkb
    analytics = AnalyticsEngine(repo=pkb.repo)
    return analytics.domain_distribution(kb=kb)


@router.get("/api/topics")
def api_topics(request: Request, kb: str | None = None, top_n: int = 20):
    """Return top topics as JSON for Chart.js."""
    from pkb.analytics import AnalyticsEngine

    pkb = request.app.state.pkb
    analytics = AnalyticsEngine(repo=pkb.repo)
    return analytics.topic_heatmap(top_n=top_n, kb=kb)


@router.get("/api/trend")
def api_trend(request: Request, kb: str | None = None, months: int = 6):
    """Return monthly trend as JSON for Chart.js."""
    from pkb.analytics import AnalyticsEngine

    pkb = request.app.state.pkb
    analytics = AnalyticsEngine(repo=pkb.repo)
    return analytics.temporal_trend(months=months, kb=kb)


@router.get("/api/platforms")
def api_platforms(request: Request, kb: str | None = None):
    """Return platform distribution as JSON for Chart.js."""
    from pkb.analytics import AnalyticsEngine

    pkb = request.app.state.pkb
    analytics = AnalyticsEngine(repo=pkb.repo)
    return analytics.platform_distribution(kb=kb)


@router.get("/api/gaps")
def api_gaps(request: Request, kb: str | None = None, threshold: int = 3):
    """Return knowledge gaps as JSON."""
    from pkb.analytics import AnalyticsEngine

    pkb = request.app.state.pkb
    analytics = AnalyticsEngine(repo=pkb.repo)
    return analytics.knowledge_gaps(threshold=threshold, kb=kb)
```

Create `src/pkb/web/templates/analytics/dashboard.html`:

```html
{% extends "base.html" %}
{% block title %}Analytics — PKB{% endblock %}
{% block head %}
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
{% endblock %}
{% block content %}
<h1>Knowledge Portfolio</h1>

<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem;">
  <div style="padding: 1rem; border: 1px solid #ddd; border-radius: 4px; text-align: center;">
    <div style="font-size: 2rem; font-weight: bold;">{{ overview.total_bundles }}</div>
    <div>Total Bundles</div>
  </div>
  <div style="padding: 1rem; border: 1px solid #ddd; border-radius: 4px; text-align: center;">
    <div style="font-size: 2rem; font-weight: bold;">{{ overview.total_relations }}</div>
    <div>Relations</div>
  </div>
  <div style="padding: 1rem; border: 1px solid #ddd; border-radius: 4px; text-align: center;">
    <div style="font-size: 2rem; font-weight: bold;">{{ overview.domain_count }}</div>
    <div>Domains</div>
  </div>
  <div style="padding: 1rem; border: 1px solid #ddd; border-radius: 4px; text-align: center;">
    <div style="font-size: 2rem; font-weight: bold;">{{ overview.topic_count }}</div>
    <div>Topics</div>
  </div>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
  <div>
    <h2>Domain Distribution</h2>
    <canvas id="domainChart"></canvas>
  </div>
  <div>
    <h2>Platform Usage</h2>
    <canvas id="platformChart"></canvas>
  </div>
  <div>
    <h2>Monthly Trend</h2>
    <canvas id="trendChart"></canvas>
  </div>
  <div>
    <h2>Top Topics</h2>
    <canvas id="topicChart"></canvas>
  </div>
</div>

<script>
async function loadCharts() {
  // Domain distribution (donut)
  const domains = await fetch('/analytics/api/domains').then(r => r.json());
  new Chart(document.getElementById('domainChart'), {
    type: 'doughnut',
    data: {
      labels: domains.map(d => d.domain),
      datasets: [{
        data: domains.map(d => d.count),
        backgroundColor: [
          '#4CAF50', '#2196F3', '#FF9800', '#9C27B0',
          '#F44336', '#00BCD4', '#795548', '#607D8B',
        ],
      }],
    },
  });

  // Platform distribution (bar)
  const platforms = await fetch('/analytics/api/platforms').then(r => r.json());
  new Chart(document.getElementById('platformChart'), {
    type: 'bar',
    data: {
      labels: platforms.map(p => p.platform),
      datasets: [{
        label: 'Responses',
        data: platforms.map(p => p.count),
        backgroundColor: '#2196F3',
      }],
    },
  });

  // Monthly trend (line)
  const trend = await fetch('/analytics/api/trend?months=12').then(r => r.json());
  new Chart(document.getElementById('trendChart'), {
    type: 'line',
    data: {
      labels: trend.map(t => t.month),
      datasets: [{
        label: 'Bundles',
        data: trend.map(t => t.count),
        borderColor: '#4CAF50',
        fill: false,
      }],
    },
  });

  // Top topics (horizontal bar)
  const topics = await fetch('/analytics/api/topics?top_n=10').then(r => r.json());
  new Chart(document.getElementById('topicChart'), {
    type: 'bar',
    data: {
      labels: topics.map(t => t.topic),
      datasets: [{
        label: 'Bundles',
        data: topics.map(t => t.count),
        backgroundColor: '#FF9800',
      }],
    },
    options: { indexAxis: 'y' },
  });
}
loadCharts();
</script>
{% endblock %}
```

Register in `app.py`:

```python
from pkb.web.routes.analytics import router as analytics_router
app.include_router(analytics_router)
```

**Step 3: Run test**

Run: `pytest tests/test_web_analytics.py -v`
Expected: PASS

**Step 4: Commit**

```
feat(web): add analytics dashboard with Chart.js charts
```

---

### Task 6: Ruff/lint + full test suite

**Step 1:** Run `ruff check src/ tests/`
**Step 2:** Run `pytest tests/ -v --ignore=tests/integration`
**Step 3:** Fix any issues
**Step 4:** Commit

```
chore: lint and test cleanup for Phase 7
```

---

### Task 7: Update documentation

Update `CLAUDE.md`, `docs/design-v1.md` with Phase 7 features:
- `pkb stats` / `pkb stats --json`
- `pkb report --weekly` / `pkb report --monthly`
- Analytics web dashboard (`/analytics`)
- Knowledge gap detection
- Updated phase status

```
docs: update documentation for Phase 7
```

---

## Summary

| Task | Description | Key Files |
|------|-------------|-----------|
| 1 | AnalyticsEngine | `src/pkb/analytics.py` |
| 2 | Aggregate query methods | `db/postgres.py` (count_by_domain/topic/month/platform) |
| 3 | ReportGenerator | `src/pkb/report.py`, `db/postgres.py` (list_bundles_since) |
| 4 | CLI stats + report | `cli.py` |
| 5 | Web analytics dashboard | `web/routes/analytics.py`, Chart.js template |
| 6 | Lint + tests | various |
| 7 | Documentation | `CLAUDE.md`, `docs/design-v1.md` |
