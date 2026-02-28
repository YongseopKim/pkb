"""Tests for Scheduler — periodic task scheduling using JSON state file."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pkb.models.config import SchedulerConfig
from pkb.scheduler import MONTHLY_INTERVAL, WEEKLY_INTERVAL, Scheduler


@pytest.fixture
def state_file(tmp_path):
    """Return a path to a non-existent state file in a temp directory."""
    return tmp_path / "scheduler_state.json"


def _make_scheduler(
    state_file: Path,
    *,
    weekly_digest: bool = True,
    monthly_report: bool = True,
    gap_threshold: int = 3,
) -> Scheduler:
    config = SchedulerConfig(
        weekly_digest=weekly_digest,
        monthly_report=monthly_report,
        gap_threshold=gap_threshold,
    )
    return Scheduler(config=config, state_path=state_file)


def _write_state(state_file: Path, state: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state))


# ── Weekly Digest Tests ──────────────────────────────────────────


class TestSchedulerWeeklyDigest:
    def test_due_when_no_state_file(self, state_file):
        """No state file exists -> weekly digest is due."""
        scheduler = _make_scheduler(state_file)
        assert scheduler.is_weekly_digest_due() is True

    def test_not_due_when_recently_run(self, state_file):
        """Last run 1 day ago -> not due yet."""
        one_day_ago = datetime.now(timezone.utc) - timedelta(days=1)
        _write_state(state_file, {"last_weekly_digest": one_day_ago.isoformat()})
        scheduler = _make_scheduler(state_file)
        assert scheduler.is_weekly_digest_due() is False

    def test_due_after_7_days(self, state_file):
        """Last run 8 days ago -> due."""
        eight_days_ago = datetime.now(timezone.utc) - timedelta(days=8)
        _write_state(state_file, {"last_weekly_digest": eight_days_ago.isoformat()})
        scheduler = _make_scheduler(state_file)
        assert scheduler.is_weekly_digest_due() is True

    def test_not_due_at_exactly_7_days(self, state_file):
        """Last run exactly 6 days ago -> not yet due (< 7 days)."""
        six_days_ago = datetime.now(timezone.utc) - timedelta(days=6)
        _write_state(state_file, {"last_weekly_digest": six_days_ago.isoformat()})
        scheduler = _make_scheduler(state_file)
        assert scheduler.is_weekly_digest_due() is False

    def test_disabled_returns_false(self, state_file):
        """weekly_digest=False -> always returns False even with no state."""
        scheduler = _make_scheduler(state_file, weekly_digest=False)
        assert scheduler.is_weekly_digest_due() is False

    def test_disabled_returns_false_even_when_overdue(self, state_file):
        """weekly_digest=False -> returns False even when overdue."""
        old_date = datetime.now(timezone.utc) - timedelta(days=30)
        _write_state(state_file, {"last_weekly_digest": old_date.isoformat()})
        scheduler = _make_scheduler(state_file, weekly_digest=False)
        assert scheduler.is_weekly_digest_due() is False


# ── Monthly Report Tests ─────────────────────────────────────────


class TestSchedulerMonthlyReport:
    def test_due_when_no_state_file(self, state_file):
        """No state file exists -> monthly report is due."""
        scheduler = _make_scheduler(state_file)
        assert scheduler.is_monthly_report_due() is True

    def test_not_due_when_recently_run(self, state_file):
        """Last run 5 days ago -> not due yet."""
        five_days_ago = datetime.now(timezone.utc) - timedelta(days=5)
        _write_state(state_file, {"last_monthly_report": five_days_ago.isoformat()})
        scheduler = _make_scheduler(state_file)
        assert scheduler.is_monthly_report_due() is False

    def test_due_after_30_days(self, state_file):
        """Last run 31 days ago -> due."""
        thirty_one_days_ago = datetime.now(timezone.utc) - timedelta(days=31)
        _write_state(state_file, {"last_monthly_report": thirty_one_days_ago.isoformat()})
        scheduler = _make_scheduler(state_file)
        assert scheduler.is_monthly_report_due() is True

    def test_not_due_at_29_days(self, state_file):
        """Last run 29 days ago -> not yet due."""
        twenty_nine_days_ago = datetime.now(timezone.utc) - timedelta(days=29)
        _write_state(state_file, {"last_monthly_report": twenty_nine_days_ago.isoformat()})
        scheduler = _make_scheduler(state_file)
        assert scheduler.is_monthly_report_due() is False

    def test_disabled_returns_false(self, state_file):
        """monthly_report=False -> always returns False."""
        scheduler = _make_scheduler(state_file, monthly_report=False)
        assert scheduler.is_monthly_report_due() is False

    def test_disabled_returns_false_even_when_overdue(self, state_file):
        """monthly_report=False -> returns False even when overdue."""
        old_date = datetime.now(timezone.utc) - timedelta(days=60)
        _write_state(state_file, {"last_monthly_report": old_date.isoformat()})
        scheduler = _make_scheduler(state_file, monthly_report=False)
        assert scheduler.is_monthly_report_due() is False


# ── Mark Done Tests ──────────────────────────────────────────────


class TestSchedulerMarkDone:
    def test_mark_weekly_digest_done_updates_state(self, state_file):
        """After marking weekly digest done, is_weekly_digest_due() returns False."""
        scheduler = _make_scheduler(state_file)
        assert scheduler.is_weekly_digest_due() is True
        scheduler.mark_weekly_digest_done()
        assert scheduler.is_weekly_digest_due() is False

    def test_mark_monthly_report_done_updates_state(self, state_file):
        """After marking monthly report done, is_monthly_report_due() returns False."""
        scheduler = _make_scheduler(state_file)
        assert scheduler.is_monthly_report_due() is True
        scheduler.mark_monthly_report_done()
        assert scheduler.is_monthly_report_due() is False

    def test_state_persists_to_file(self, state_file):
        """State file is written after mark_*_done()."""
        scheduler = _make_scheduler(state_file)
        scheduler.mark_weekly_digest_done()

        assert state_file.exists()
        saved = json.loads(state_file.read_text())
        assert "last_weekly_digest" in saved
        # Verify the saved timestamp is parseable and recent
        dt = datetime.fromisoformat(saved["last_weekly_digest"])
        assert datetime.now(timezone.utc) - dt < timedelta(seconds=10)

    def test_state_persists_monthly_to_file(self, state_file):
        """State file contains monthly report timestamp after marking done."""
        scheduler = _make_scheduler(state_file)
        scheduler.mark_monthly_report_done()

        assert state_file.exists()
        saved = json.loads(state_file.read_text())
        assert "last_monthly_report" in saved
        dt = datetime.fromisoformat(saved["last_monthly_report"])
        assert datetime.now(timezone.utc) - dt < timedelta(seconds=10)

    def test_state_survives_reload(self, state_file):
        """State persists across Scheduler instances."""
        scheduler1 = _make_scheduler(state_file)
        scheduler1.mark_weekly_digest_done()
        scheduler1.mark_monthly_report_done()

        # Create a new Scheduler instance that loads from the same state file
        scheduler2 = _make_scheduler(state_file)
        assert scheduler2.is_weekly_digest_due() is False
        assert scheduler2.is_monthly_report_due() is False

    def test_mark_one_does_not_affect_other(self, state_file):
        """Marking weekly done should not affect monthly status."""
        scheduler = _make_scheduler(state_file)
        scheduler.mark_weekly_digest_done()
        # Monthly should still be due (no state file entry for it)
        assert scheduler.is_monthly_report_due() is True


# ── State Recovery Tests ─────────────────────────────────────────


class TestSchedulerStateRecovery:
    def test_corrupt_state_file_starts_fresh(self, state_file):
        """Corrupt (non-JSON) state file -> loads empty state, tasks are due."""
        state_file.write_text("not valid json {{{")
        scheduler = _make_scheduler(state_file)
        assert scheduler.is_weekly_digest_due() is True
        assert scheduler.is_monthly_report_due() is True

    def test_invalid_date_in_state_returns_due(self, state_file):
        """Invalid ISO date string in state -> treated as due."""
        _write_state(state_file, {"last_weekly_digest": "not-a-date"})
        scheduler = _make_scheduler(state_file)
        assert scheduler.is_weekly_digest_due() is True

    def test_invalid_date_monthly_returns_due(self, state_file):
        """Invalid ISO date for monthly -> treated as due."""
        _write_state(state_file, {"last_monthly_report": "garbage"})
        scheduler = _make_scheduler(state_file)
        assert scheduler.is_monthly_report_due() is True

    def test_empty_state_file(self, state_file):
        """Empty file -> corrupt JSON -> starts fresh."""
        state_file.write_text("")
        scheduler = _make_scheduler(state_file)
        assert scheduler.is_weekly_digest_due() is True

    def test_state_file_in_nested_dir(self, tmp_path):
        """State file in a non-existent nested directory -> created on mark."""
        nested = tmp_path / "deep" / "nested" / "scheduler_state.json"
        scheduler = _make_scheduler(nested)
        scheduler.mark_weekly_digest_done()
        assert nested.exists()


# ── Interval Constants Tests ─────────────────────────────────────


class TestSchedulerIntervals:
    def test_weekly_interval_is_7_days(self):
        assert WEEKLY_INTERVAL == timedelta(days=7)

    def test_monthly_interval_is_30_days(self):
        assert MONTHLY_INTERVAL == timedelta(days=30)
