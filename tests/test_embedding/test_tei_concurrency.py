"""Tests for TEI client concurrency limiting via threading.Semaphore."""

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from pkb.embedding.tei_client import TEIClient


class TestTEIClientConcurrency:
    """TEI 동시 요청 제한 테스트."""

    def test_max_concurrent_parameter(self):
        """max_concurrent 파라미터가 설정되는지 확인."""
        client = TEIClient(base_url="http://localhost:8090", max_concurrent=3)
        assert client._max_concurrent == 3

    def test_default_max_concurrent_is_2(self):
        """기본 max_concurrent가 2인지 확인."""
        client = TEIClient(base_url="http://localhost:8090")
        assert client._max_concurrent == 2

    def test_has_semaphore(self):
        """세마포어가 생성되는지 확인."""
        client = TEIClient(base_url="http://localhost:8090", max_concurrent=2)
        assert isinstance(client._semaphore, type(threading.Semaphore()))

    def test_semaphore_limits_concurrent_calls(self):
        """세마포어가 동시 TEI 호출 수를 제한하는지 확인."""
        client = TEIClient(
            base_url="http://localhost:8090",
            timeout=10.0,
            max_concurrent=2,
        )

        active_count = 0
        max_active = 0
        lock = threading.Lock()

        def mock_urlopen(req, timeout=None):
            nonlocal active_count, max_active
            with lock:
                active_count += 1
                max_active = max(max_active, active_count)

            # Wait long enough for all threads to attempt entry
            time.sleep(0.1)

            with lock:
                active_count -= 1

            resp = MagicMock()
            resp.status = 200
            resp.read.return_value = json.dumps([[0.1, 0.2]]).encode()
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            threads = []
            for _ in range(3):
                t = threading.Thread(target=client.embed, args=(["text"],))
                threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5.0)

        # max_concurrent=2이므로 동시에 최대 2개만 활성
        assert max_active <= 2

    def test_semaphore_releases_after_error(self):
        """에러 발생 시에도 세마포어가 해제되는지 확인."""
        client = TEIClient(
            base_url="http://localhost:8090",
            timeout=10.0,
            max_concurrent=1,
            max_retries=1,
        )

        call_count = 0

        def mock_urlopen(req, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("refused")
            resp = MagicMock()
            resp.status = 200
            resp.read.return_value = json.dumps([[0.1]]).encode()
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            # First call errors
            with pytest.raises(ConnectionError):
                client.embed(["text1"])

            # Second call should NOT deadlock (semaphore released)
            result = client.embed(["text2"])
            assert result == [[0.1]]

    def test_semaphore_releases_after_timeout(self):
        """타임아웃 발생 시에도 세마포어가 해제되는지 확인."""
        client = TEIClient(
            base_url="http://localhost:8090",
            timeout=10.0,
            max_concurrent=1,
            max_retries=1,
        )

        call_count = 0

        def mock_urlopen(req, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("timed out")
            resp = MagicMock()
            resp.status = 200
            resp.read.return_value = json.dumps([[0.1]]).encode()
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with pytest.raises(TimeoutError):
                client.embed(["text1"])

            # Should not deadlock
            result = client.embed(["text2"])
            assert result == [[0.1]]


class TestTEIClientConcurrencyConfig:
    """Config에서 tei_max_concurrent가 전달되는지 테스트."""

    def test_embedding_config_has_tei_max_concurrent(self):
        """EmbeddingConfig에 tei_max_concurrent 필드가 있는지 확인."""
        from pkb.models.config import EmbeddingConfig

        config = EmbeddingConfig()
        assert config.tei_max_concurrent == 2

    def test_embedding_config_custom_tei_max_concurrent(self):
        """tei_max_concurrent를 커스텀 값으로 설정할 수 있는지 확인."""
        from pkb.models.config import EmbeddingConfig

        config = EmbeddingConfig(tei_max_concurrent=4)
        assert config.tei_max_concurrent == 4

    def test_factory_passes_max_concurrent(self):
        """create_embedder가 max_concurrent를 TEIClient에 전달하는지 확인."""
        from pkb.models.config import EmbeddingConfig

        config = EmbeddingConfig(
            mode="tei",
            tei_url="http://localhost:8090",
            tei_max_concurrent=3,
        )

        with patch("pkb.embedding.tei_client.TEIClient.__init__", return_value=None) as mock_init:
            from pkb.embedding.factory import create_embedder
            create_embedder(config)
            mock_init.assert_called_once_with(
                base_url="http://localhost:8090",
                timeout=config.tei_timeout,
                max_concurrent=3,
            )
