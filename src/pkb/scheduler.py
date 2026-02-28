"""Periodic task scheduler for PKB watch daemon."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pkb.models.config import SchedulerConfig

logger = logging.getLogger(__name__)

WEEKLY_INTERVAL = timedelta(days=7)
MONTHLY_INTERVAL = timedelta(days=30)


class Scheduler:
    """Tracks and checks periodic task schedules using a JSON state file."""

    def __init__(self, *, config: SchedulerConfig, state_path: Path) -> None:
        self._config = config
        self._state_path = state_path
        self._state = self._load_state()

    def _load_state(self) -> dict:
        if self._state_path.exists():
            try:
                return json.loads(self._state_path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load scheduler state, starting fresh")
        return {}

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(self._state, indent=2))

    def is_weekly_digest_due(self) -> bool:
        """Check if weekly digest should run."""
        if not self._config.weekly_digest:
            return False
        return self._is_due("last_weekly_digest", WEEKLY_INTERVAL)

    def is_monthly_report_due(self) -> bool:
        """Check if monthly report should run."""
        if not self._config.monthly_report:
            return False
        return self._is_due("last_monthly_report", MONTHLY_INTERVAL)

    def mark_weekly_digest_done(self) -> None:
        """Record that weekly digest was just completed."""
        self._state["last_weekly_digest"] = datetime.now(timezone.utc).isoformat()
        self._save_state()

    def mark_monthly_report_done(self) -> None:
        """Record that monthly report was just completed."""
        self._state["last_monthly_report"] = datetime.now(timezone.utc).isoformat()
        self._save_state()

    def _is_due(self, state_key: str, interval: timedelta) -> bool:
        """Check if enough time has elapsed since last run."""
        last_run = self._state.get(state_key)
        if not last_run:
            return True
        try:
            last_dt = datetime.fromisoformat(last_run)
            return datetime.now(timezone.utc) - last_dt >= interval
        except (ValueError, TypeError):
            return True
