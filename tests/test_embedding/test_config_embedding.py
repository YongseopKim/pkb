"""Tests for EmbeddingConfig model extensions."""

import yaml

from pkb.models.config import EmbeddingConfig, PKBConfig


class TestEmbeddingConfigDefaults:
    """기본값으로 하위 호환성 보장."""

    def test_default_mode_is_tei(self):
        config = EmbeddingConfig()
        assert config.mode == "tei"

    def test_default_model_name(self):
        config = EmbeddingConfig()
        assert config.model_name == "BAAI/bge-m3"

    def test_default_dimensions(self):
        config = EmbeddingConfig()
        assert config.dimensions == 1024

    def test_default_tei_url(self):
        config = EmbeddingConfig()
        assert config.tei_url == "http://localhost:8090"

    def test_default_tei_batch_size(self):
        config = EmbeddingConfig()
        assert config.tei_batch_size == 32

    def test_default_tei_timeout(self):
        config = EmbeddingConfig()
        assert config.tei_timeout == 30.0

    def test_chunk_size_and_overlap_defaults(self):
        config = EmbeddingConfig()
        assert config.chunk_size == 1500
        assert config.chunk_overlap == 200


class TestEmbeddingConfigTEI:
    """TEI 모드 설정."""

    def test_tei_mode(self):
        config = EmbeddingConfig(
            mode="tei",
            model_name="BAAI/bge-m3",
            dimensions=1024,
            tei_url="http://localhost:8090",
        )
        assert config.mode == "tei"
        assert config.model_name == "BAAI/bge-m3"
        assert config.dimensions == 1024
        assert config.tei_url == "http://localhost:8090"

    def test_custom_batch_size_and_timeout(self):
        config = EmbeddingConfig(
            tei_batch_size=64,
            tei_timeout=60.0,
        )
        assert config.tei_batch_size == 64
        assert config.tei_timeout == 60.0


class TestEmbeddingConfigInPKBConfig:
    """PKBConfig 통합 검증."""

    def test_pkbconfig_embedding_defaults(self):
        config = PKBConfig()
        assert config.embedding.mode == "tei"
        assert config.embedding.model_name == "BAAI/bge-m3"

    def test_pkbconfig_with_tei_embedding(self):
        config = PKBConfig(embedding=EmbeddingConfig(
            mode="tei",
            model_name="BAAI/bge-m3",
            dimensions=1024,
            tei_url="http://localhost:8090",
        ))
        assert config.embedding.mode == "tei"
        assert config.embedding.dimensions == 1024


class TestEmbeddingConfigYAMLRoundtrip:
    """YAML 직렬화/역직렬화 호환성."""

    def test_legacy_yaml_without_new_fields(self, tmp_path):
        """기존 config.yaml에 새 필드가 없어도 로드 가능 (기본값 적용)."""
        data = {
            "embedding": {
                "chunk_size": 512,
                "chunk_overlap": 50,
            },
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(data))
        raw = yaml.safe_load(p.read_text())
        config = PKBConfig(**raw)
        assert config.embedding.chunk_size == 512
        assert config.embedding.chunk_overlap == 50
        assert config.embedding.mode == "tei"
        assert config.embedding.model_name == "BAAI/bge-m3"

    def test_full_tei_yaml(self, tmp_path):
        """TEI 전체 설정이 포함된 YAML 로드."""
        data = {
            "embedding": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "mode": "tei",
                "model_name": "BAAI/bge-m3",
                "dimensions": 1024,
                "tei_url": "http://localhost:8090",
                "tei_batch_size": 64,
                "tei_timeout": 45.0,
            },
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(data))
        raw = yaml.safe_load(p.read_text())
        config = PKBConfig(**raw)
        assert config.embedding.mode == "tei"
        assert config.embedding.model_name == "BAAI/bge-m3"
        assert config.embedding.dimensions == 1024
        assert config.embedding.tei_url == "http://localhost:8090"
        assert config.embedding.tei_batch_size == 64
        assert config.embedding.tei_timeout == 45.0
