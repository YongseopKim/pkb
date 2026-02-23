"""PKB logging configuration."""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_LOG_RETENTION_DAYS = 30
_LOG_DIR_NAME = "logs"


def _get_log_dir() -> Path:
    """Get log directory path (~/.pkb/logs/)."""
    from pkb.constants import DEFAULT_PKB_HOME
    return DEFAULT_PKB_HOME / _LOG_DIR_NAME


def _cleanup_old_logs(log_dir: Path) -> None:
    """Delete log files older than _LOG_RETENTION_DAYS."""
    if not log_dir.exists():
        return
    cutoff = time.time() - _LOG_RETENTION_DAYS * 86400
    for f in log_dir.glob("pkb-*.log"):
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)


def setup_logging(verbosity: int = 0) -> None:
    """Configure logging with console + file handlers.

    Args:
        verbosity: 0=WARNING, 1=INFO, 2+=DEBUG
    """
    level_map = {0: logging.WARNING, 1: logging.INFO}
    level = level_map.get(verbosity, logging.DEBUG)

    root = logging.getLogger()
    # Remove existing handlers to avoid duplicates on repeated calls
    root.handlers.clear()
    root.setLevel(logging.DEBUG)  # Root captures all; handlers filter

    # Console handler (stderr)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console)

    # File handler (always)
    log_dir = _get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_old_logs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"pkb-{timestamp}-{os.getpid()}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s")
    )
    root.addHandler(file_handler)
