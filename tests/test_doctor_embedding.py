"""Tests for Doctor embedding-related checks (TEI, model consistency)."""

from unittest.mock import MagicMock, patch

import pytest

from pkb.doctor import DoctorRunner
from pkb.models.config import (
    ChromaDBConfig,
    DatabaseConfig,
    EmbeddingConfig,
    PKBConfig,
)


@pytest.fixture
def sample_tei_config():
    return PKBConfig(
        embedding=EmbeddingConfig(
            mode="tei",
            model_name="BAAI/bge-m3",
            dimensions=1024,
            tei_url="http://localhost:8090",
        ),
        database=DatabaseConfig(
            chromadb=ChromaDBConfig(host="localhost", port=9000),
        ),
    )


@pytest.fixture
def sample_server_config():
    return PKBConfig(
        embedding=EmbeddingConfig(mode="server"),
    )


class TestCheckTEI:
    def test_tei_ok(self, tmp_path, sample_tei_config):
        doctor = DoctorRunner(pkb_home=tmp_path)
        with patch("pkb.embedding.tei_client.TEIClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.health_check.return_value = True
            mock_cls.return_value = mock_client
            result = doctor.check_tei(sample_tei_config)
        assert result.ok is True
        assert "8090" in result.detail

    def test_tei_fail(self, tmp_path, sample_tei_config):
        doctor = DoctorRunner(pkb_home=tmp_path)
        with patch("pkb.embedding.tei_client.TEIClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.health_check.return_value = False
            mock_cls.return_value = mock_client
            result = doctor.check_tei(sample_tei_config)
        assert result.ok is False

    def test_tei_skipped_in_server_mode(self, tmp_path, sample_server_config):
        doctor = DoctorRunner(pkb_home=tmp_path)
        result = doctor.check_tei(sample_server_config)
        assert result is None


class TestCheckEmbeddingConsistency:
    def test_matching_model(self, tmp_path, sample_tei_config):
        doctor = DoctorRunner(pkb_home=tmp_path)
        mock_collection = MagicMock()
        mock_collection.metadata = {
            "embedding_model": "BAAI/bge-m3",
            "embedding_dimensions": 1024,
        }
        with patch("pkb.doctor.chromadb") as mock_chromadb:
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.HttpClient.return_value = mock_client
            result = doctor.check_embedding_consistency(sample_tei_config)
        assert result.ok is True

    def test_mismatched_model(self, tmp_path, sample_tei_config):
        doctor = DoctorRunner(pkb_home=tmp_path)
        mock_collection = MagicMock()
        mock_collection.metadata = {
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimensions": 384,
        }
        with patch("pkb.doctor.chromadb") as mock_chromadb:
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.HttpClient.return_value = mock_client
            result = doctor.check_embedding_consistency(sample_tei_config)
        assert result.ok is False
        assert "mismatch" in result.detail.lower()

    def test_legacy_no_metadata(self, tmp_path, sample_tei_config):
        doctor = DoctorRunner(pkb_home=tmp_path)
        mock_collection = MagicMock()
        mock_collection.metadata = {}
        with patch("pkb.doctor.chromadb") as mock_chromadb:
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.HttpClient.return_value = mock_client
            result = doctor.check_embedding_consistency(sample_tei_config)
        assert result.ok is False
        assert "legacy" in result.detail.lower() or "no model" in result.detail.lower()

    def test_skipped_in_server_mode(self, tmp_path, sample_server_config):
        doctor = DoctorRunner(pkb_home=tmp_path)
        result = doctor.check_embedding_consistency(sample_server_config)
        assert result is None

    def test_chromadb_connection_error(self, tmp_path, sample_tei_config):
        doctor = DoctorRunner(pkb_home=tmp_path)
        with patch("pkb.doctor.chromadb") as mock_chromadb:
            mock_chromadb.HttpClient.side_effect = Exception("Connection refused")
            result = doctor.check_embedding_consistency(sample_tei_config)
        assert result.ok is False
        assert "connection" in result.detail.lower() or "refused" in result.detail.lower()
