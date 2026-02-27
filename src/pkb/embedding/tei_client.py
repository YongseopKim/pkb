"""HTTP client for HuggingFace Text Embeddings Inference (TEI) server."""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request

logger = logging.getLogger(__name__)


class TEIClient:
    """Low-level HTTP client for TEI /embed endpoint.

    Uses a threading.Semaphore to limit concurrent HTTP requests to the TEI server.
    This prevents overloading TEI when multiple ingest workers call embed() simultaneously
    (each running in its own thread via asyncio.to_thread).
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        max_retries: int = 3,
        max_concurrent: int = 2,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Send texts to TEI and return embedding vectors.

        Acquires the concurrency semaphore before making the HTTP request.
        Retries on TimeoutError up to max_retries times with exponential backoff.

        API: POST {base_url}/embed
        Body: {"inputs": ["text1", ...]}
        Response: [[vec1], [vec2], ...]
        """
        if not texts:
            return []

        payload = json.dumps({"inputs": texts}).encode("utf-8")

        last_error: TimeoutError | None = None
        for attempt in range(self._max_retries):
            req = urllib.request.Request(
                f"{self._base_url}/embed",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            self._semaphore.acquire()
            try:
                resp = urllib.request.urlopen(req, timeout=self._timeout)
                if resp.status != 200:
                    raise RuntimeError(
                        f"TEI server error {resp.status}: "
                        f"{resp.read().decode(errors='replace')}"
                    )
                return json.loads(resp.read().decode("utf-8"))
            except TimeoutError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "TEI embed timeout (attempt %d/%d), retrying in %ds...",
                        attempt + 1, self._max_retries, wait,
                    )
                    time.sleep(wait)
            finally:
                self._semaphore.release()

        raise TimeoutError(
            f"TEI embedding timed out after {self._max_retries} attempts "
            f"(timeout={self._timeout}s, texts={len(texts)})"
        ) from last_error

    def health_check(self) -> bool:
        """Check TEI server health."""
        try:
            req = urllib.request.Request(
                f"{self._base_url}/health",
                method="GET",
            )
            resp = urllib.request.urlopen(req, timeout=self._timeout)
            return resp.status == 200
        except Exception:
            return False
