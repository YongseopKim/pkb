"""HTTP client for HuggingFace Text Embeddings Inference (TEI) server."""

from __future__ import annotations

import json
import urllib.request


class TEIClient:
    """Low-level HTTP client for TEI /embed endpoint."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Send texts to TEI and return embedding vectors.

        API: POST {base_url}/embed
        Body: {"inputs": ["text1", ...]}
        Response: [[vec1], [vec2], ...]
        """
        if not texts:
            return []

        payload = json.dumps({"inputs": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=self._timeout)
        if resp.status != 200:
            raise RuntimeError(
                f"TEI server error {resp.status}: {resp.read().decode(errors='replace')}"
            )
        return json.loads(resp.read().decode("utf-8"))

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
