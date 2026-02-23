"""Report generator for PKB analytics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pkb.analytics import AnalyticsEngine
    from pkb.db.postgres import BundleRepository


class ReportGenerator:
    """Generates markdown activity reports from analytics data."""

    def __init__(
        self,
        *,
        repo: BundleRepository,
        analytics: AnalyticsEngine,
    ) -> None:
        self._repo = repo
        self._analytics = analytics

    def weekly(self, kb: str | None = None) -> str:
        """Generate a 7-day activity report in markdown."""
        since = datetime.now(timezone.utc) - timedelta(days=7)
        bundles = self._repo.list_bundles_since(since, kb=kb)
        domains = self._analytics.domain_distribution(kb=kb)
        topics = self._analytics.topic_heatmap(top_n=10, kb=kb)

        lines = ["# 주간 지식 활동 리포트", ""]
        lines.append(f"**기간**: 최근 7일 | **새 번들**: {len(bundles)}개")
        lines.append("")

        # Bundle list
        if bundles:
            lines.append("## 새 번들 목록")
            lines.append("")
            for b in bundles:
                lines.append(f"- `{b['bundle_id']}` — {b.get('question', '')}")
            lines.append("")

        # Domain distribution
        if domains:
            lines.append("## 도메인 분포")
            lines.append("")
            for d in domains:
                lines.append(f"- **{d['domain']}**: {d['count']}개")
            lines.append("")

        # Top topics
        if topics:
            lines.append("## 토픽 Top 10")
            lines.append("")
            for t in topics:
                lines.append(f"- {t['topic']}: {t['count']}개")
            lines.append("")

        return "\n".join(lines)

    def monthly(self, kb: str | None = None) -> str:
        """Generate a 30-day activity report with knowledge gaps."""
        since = datetime.now(timezone.utc) - timedelta(days=30)
        bundles = self._repo.list_bundles_since(since, kb=kb)
        domains = self._analytics.domain_distribution(kb=kb)
        topics = self._analytics.topic_heatmap(top_n=10, kb=kb)
        gaps = self._analytics.knowledge_gaps(kb=kb)

        lines = ["# 월간 지식 활동 리포트", ""]
        lines.append(f"**기간**: 최근 30일 | **새 번들**: {len(bundles)}개")
        lines.append("")

        # Bundle list
        if bundles:
            lines.append("## 새 번들 목록")
            lines.append("")
            for b in bundles:
                lines.append(f"- `{b['bundle_id']}` — {b.get('question', '')}")
            lines.append("")

        # Domain distribution
        if domains:
            lines.append("## 도메인 분포")
            lines.append("")
            for d in domains:
                lines.append(f"- **{d['domain']}**: {d['count']}개")
            lines.append("")

        # Top topics
        if topics:
            lines.append("## 토픽 Top 10")
            lines.append("")
            for t in topics:
                lines.append(f"- {t['topic']}: {t['count']}개")
            lines.append("")

        # Knowledge gaps
        if gaps:
            lines.append("## 지식 공백")
            lines.append("")
            for g in gaps:
                lines.append(f"- **{g['topic']}**: {g['count']}개 (보강 필요)")
            lines.append("")

        return "\n".join(lines)
