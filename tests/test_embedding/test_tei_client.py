"""Tests for TEIClient HTTP wrapper."""

import json
from unittest.mock import MagicMock, patch

import pytest

from pkb.embedding.tei_client import TEIClient


@pytest.fixture
def client():
    return TEIClient(base_url="http://localhost:8090", timeout=10.0)


class TestTEIClient:
    """TEI HTTP 호출 mock 테스트."""

    def test_embed_single_text(self, client):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps([[0.1, 0.2, 0.3]]).encode()

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_open:
            result = client.embed(["hello"])
            assert len(result) == 1
            assert result[0] == [0.1, 0.2, 0.3]
            mock_open.assert_called_once()

    def test_embed_multiple_texts(self, client):
        vecs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(vecs).encode()

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = client.embed(["a", "b", "c"])
            assert len(result) == 3

    def test_embed_empty_list(self, client):
        result = client.embed([])
        assert result == []

    def test_server_error_raises(self, client):
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.read.return_value = b"Internal Server Error"

        with patch("urllib.request.urlopen", return_value=mock_response):
            with pytest.raises(RuntimeError, match="TEI server error"):
                client.embed(["hello"])

    def test_connection_error_raises(self, client):
        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            with pytest.raises(ConnectionError):
                client.embed(["hello"])

    def test_health_check_ok(self, client):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b"{}"

        with patch("urllib.request.urlopen", return_value=mock_response):
            assert client.health_check() is True

    def test_health_check_fail(self, client):
        with patch("urllib.request.urlopen", side_effect=ConnectionError):
            assert client.health_check() is False


class TestTEIClientRetry:
    """TEI timeout retry 테스트."""

    def test_timeout_retries_then_succeeds(self):
        """TimeoutError 발생 시 retry하여 성공하는 케이스."""
        client = TEIClient(base_url="http://localhost:8090", timeout=10.0, max_retries=3)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps([[0.1, 0.2]]).encode()

        with patch(
            "urllib.request.urlopen",
            side_effect=[TimeoutError("timed out"), mock_response],
        ) as mock_open:
            result = client.embed(["hello"])
            assert result == [[0.1, 0.2]]
            assert mock_open.call_count == 2

    def test_timeout_exhausts_retries(self):
        """모든 retry 소진 후 TimeoutError가 TEI 컨텍스트와 함께 전파되는 케이스."""
        client = TEIClient(base_url="http://localhost:8090", timeout=10.0, max_retries=2)

        with patch(
            "urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            with pytest.raises(TimeoutError, match="TEI embedding timed out"):
                client.embed(["hello"])

    def test_timeout_default_retries_is_3(self):
        """Default max_retries가 3인지 확인."""
        client = TEIClient(base_url="http://localhost:8090")
        assert client._max_retries == 3

    def test_non_timeout_error_not_retried(self):
        """TimeoutError가 아닌 에러는 retry하지 않고 즉시 전파."""
        client = TEIClient(base_url="http://localhost:8090", max_retries=3)

        with patch(
            "urllib.request.urlopen",
            side_effect=ConnectionError("refused"),
        ) as mock_open:
            with pytest.raises(ConnectionError):
                client.embed(["hello"])
            assert mock_open.call_count == 1
