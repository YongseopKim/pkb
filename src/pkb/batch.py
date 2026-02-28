"""Batch processor for bulk file ingestion (JSONL and MD)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from pkb.ingest import IngestPipeline, move_to_done
from pkb.parser.directory import find_input_files_recursive

if TYPE_CHECKING:
    from pkb.engine import IngestEngine


class BatchProcessor:
    """Processes multiple JSONL files with checkpoint-based resume.

    Supports two processing modes:
    - Sequential: default, uses pipeline.ingest_file() in a loop
    - Concurrent: when engine is provided, uses IngestEngine.ingest_batch()
    """

    def __init__(
        self,
        *,
        pipeline: IngestPipeline,
        checkpoint_path: Path,
        max_files: int = 0,
        resume: bool = True,
        engine: IngestEngine | None = None,
        watch_dir: Path | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._checkpoint_path = checkpoint_path
        self._max_files = max_files
        self._resume = resume
        self._engine = engine
        self._watch_dir = watch_dir

    def discover_files(self, source_dir: Path) -> list[Path]:
        """Find all input files (JSONL and MD) in source directory (recursive).

        Files inside .done/ directories are excluded.
        """
        return find_input_files_recursive(source_dir)

    def _load_checkpoint(self) -> dict:
        """Load checkpoint data if resuming."""
        if not self._resume or not self._checkpoint_path.exists():
            return {"completed": [], "failed": []}
        return yaml.safe_load(self._checkpoint_path.read_text(encoding="utf-8"))

    def _save_checkpoint(self, data: dict) -> None:
        """Save checkpoint data."""
        self._checkpoint_path.write_text(
            yaml.dump(data, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )

    def process(self, source_dir: Path) -> dict:
        """Process all input files in source_dir.

        Returns:
            Stats dict with success, skipped, errors counts.
        """
        if self._engine is not None:
            return self._process_concurrent(source_dir)
        return self._process_sequential(source_dir)

    def _process_sequential(self, source_dir: Path) -> dict:
        """Original sequential processing path."""
        files = self.discover_files(source_dir)
        checkpoint = self._load_checkpoint()
        completed_set = set(checkpoint["completed"])

        success = 0
        skipped = 0
        errors = 0
        processed = 0

        for f in files:
            if self._max_files > 0 and processed >= self._max_files:
                break

            f_str = str(f)
            if f_str in completed_set:
                skipped += 1
                continue

            try:
                result = self._pipeline.ingest_file(f)
                if result is None:
                    skipped += 1
                elif result.get("status", "").startswith("skip_"):
                    skipped += 1
                else:
                    success += 1
                if self._watch_dir is not None:
                    if result is None or result.get("status") != "skip_file_not_found":
                        move_to_done(f, self._watch_dir)
                checkpoint["completed"].append(f_str)
            except Exception:
                errors += 1
                if f_str not in checkpoint.get("failed", []):
                    checkpoint.setdefault("failed", []).append(f_str)

            processed += 1
            self._save_checkpoint(checkpoint)

        return {"success": success, "skipped": skipped, "errors": errors}

    def _process_concurrent(self, source_dir: Path) -> dict:
        """Concurrent processing path using IngestEngine."""
        files = self.discover_files(source_dir)
        checkpoint = self._load_checkpoint()
        completed_set = set(checkpoint["completed"])

        # Filter out already-completed files
        pending = []
        skipped = 0
        for f in files:
            if str(f) in completed_set:
                skipped += 1
                continue
            pending.append(f)

        # Apply max_files limit
        if self._max_files > 0:
            pending = pending[:self._max_files]

        if not pending:
            return {"success": 0, "skipped": skipped, "errors": 0}

        # Run concurrent ingest
        stats = asyncio.run(self._engine.ingest_batch(pending))

        # Update checkpoint: only mark non-error files as completed
        error_paths = {str(r.path) for r in stats.results if r.status == "error"}
        for f in pending:
            f_str = str(f)
            if f_str not in error_paths and f_str not in checkpoint["completed"]:
                checkpoint["completed"].append(f_str)
            elif f_str in error_paths:
                if f_str not in checkpoint.get("failed", []):
                    checkpoint.setdefault("failed", []).append(f_str)
        self._save_checkpoint(checkpoint)

        return {
            "success": stats.success,
            "skipped": skipped + stats.skipped,
            "errors": stats.errors,
        }
