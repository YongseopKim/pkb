"""Tests for logging configuration."""

import logging

from pkb.logging_config import setup_logging


class TestSetupLogging:
    def test_default_verbosity_sets_warning_level(self, tmp_path, monkeypatch):
        """verbosity=0이면 root logger가 WARNING."""
        monkeypatch.setattr("pkb.logging_config._get_log_dir", lambda: tmp_path)
        setup_logging(verbosity=0)
        root = logging.getLogger()
        # Console handler should be WARNING
        stream_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert stream_handlers[0].level == logging.WARNING

    def test_verbose_sets_info_level(self, tmp_path, monkeypatch):
        """verbosity=1이면 INFO."""
        monkeypatch.setattr("pkb.logging_config._get_log_dir", lambda: tmp_path)
        setup_logging(verbosity=1)
        root = logging.getLogger()
        stream_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert stream_handlers[0].level == logging.INFO

    def test_debug_sets_debug_level(self, tmp_path, monkeypatch):
        """verbosity=2이면 DEBUG."""
        monkeypatch.setattr("pkb.logging_config._get_log_dir", lambda: tmp_path)
        setup_logging(verbosity=2)
        root = logging.getLogger()
        stream_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert stream_handlers[0].level == logging.DEBUG

    def test_creates_log_file(self, tmp_path, monkeypatch):
        """로그 파일이 생성됨."""
        monkeypatch.setattr("pkb.logging_config._get_log_dir", lambda: tmp_path)
        setup_logging(verbosity=0)
        log_files = list(tmp_path.glob("pkb-*.log"))
        assert len(log_files) >= 1

    def test_log_file_receives_messages(self, tmp_path, monkeypatch):
        """WARNING 메시지가 로그 파일에 기록됨."""
        monkeypatch.setattr("pkb.logging_config._get_log_dir", lambda: tmp_path)
        setup_logging(verbosity=0)
        test_logger = logging.getLogger("pkb.test")
        test_logger.warning("test warning message")
        # Flush handlers
        for h in logging.getLogger().handlers:
            h.flush()
        log_files = list(tmp_path.glob("pkb-*.log"))
        content = log_files[0].read_text()
        assert "test warning message" in content

    def test_cleanup_old_logs(self, tmp_path, monkeypatch):
        """30일 이상 된 로그 파일 삭제."""
        import os
        import time

        monkeypatch.setattr("pkb.logging_config._get_log_dir", lambda: tmp_path)
        # Create an old log file
        old_log = tmp_path / "pkb-20250101-120000.log"
        old_log.write_text("old")
        # Set mtime to 40 days ago
        old_mtime = time.time() - 40 * 86400
        os.utime(old_log, (old_mtime, old_mtime))

        setup_logging(verbosity=0)
        assert not old_log.exists()

    def test_keeps_recent_logs(self, tmp_path, monkeypatch):
        """최근 로그 파일은 유지."""
        monkeypatch.setattr("pkb.logging_config._get_log_dir", lambda: tmp_path)
        recent_log = tmp_path / "pkb-20260222-120000.log"
        recent_log.write_text("recent")

        setup_logging(verbosity=0)
        assert recent_log.exists()
