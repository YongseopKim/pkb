"""Tests for pkb doctor module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from pkb.cli import cli
from pkb.models.config import (
    ChromaDBConfig,
    DatabaseConfig,
    KBEntry,
    LLMConfig,
    LLMModelEntry,
    LLMProviderConfig,
    LLMRoutingConfig,
    PKBConfig,
    PostgresConfig,
)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_config():
    return PKBConfig(
        knowledge_bases=[
            KBEntry(name="test-kb", path=Path("/tmp/test-kb")),
        ],
        database=DatabaseConfig(
            postgres=PostgresConfig(host="192.168.0.2", port=5432, database="pkb"),
            chromadb=ChromaDBConfig(host="192.168.0.2", port=9000),
        ),
        llm=LLMConfig(
            default_provider="anthropic",
            providers={
                "anthropic": LLMProviderConfig(
                    api_key_env="ANTHROPIC_API_KEY",
                    models=[LLMModelEntry(name="claude-haiku-4-5-20251001", tier=1)],
                ),
            },
            routing=LLMRoutingConfig(meta_extraction=1, chat=1),
        ),
    )


# --- DoctorRunner unit tests ---


class TestCheckConfig:
    def test_config_ok(self, tmp_path):
        from pkb.doctor import DoctorRunner

        config_path = tmp_path / "config.yaml"
        config_data = {
            "knowledge_bases": [],
            "meta_llm": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        }
        config_path.write_text(yaml.dump(config_data))

        doctor = DoctorRunner(pkb_home=tmp_path, config_filename="config.yaml")
        result = doctor.check_config()
        assert result.ok is True
        assert "config.yaml" in result.detail

    def test_config_missing(self, tmp_path):
        from pkb.doctor import DoctorRunner

        doctor = DoctorRunner(pkb_home=tmp_path, config_filename="config.yaml")
        result = doctor.check_config()
        assert result.ok is False
        assert "not found" in result.detail.lower()

    def test_config_invalid_yaml(self, tmp_path):
        from pkb.doctor import DoctorRunner

        config_path = tmp_path / "config.yaml"
        config_path.write_text("invalid: [yaml: broken")

        doctor = DoctorRunner(pkb_home=tmp_path, config_filename="config.yaml")
        result = doctor.check_config()
        assert result.ok is False

    def test_config_invalid_schema(self, tmp_path):
        from pkb.doctor import DoctorRunner

        config_path = tmp_path / "config.yaml"
        # knowledge_bases expects a list, not a string
        config_path.write_text(yaml.dump({"knowledge_bases": "not-a-list"}))

        doctor = DoctorRunner(pkb_home=tmp_path, config_filename="config.yaml")
        result = doctor.check_config()
        assert result.ok is False


class TestCheckKnowledgeBases:
    def test_kb_path_exists(self, tmp_path, sample_config):
        from pkb.doctor import DoctorRunner

        kb_path = tmp_path / "test-kb"
        kb_path.mkdir()
        inbox = kb_path / "inbox"
        inbox.mkdir()

        sample_config.knowledge_bases = [KBEntry(name="test-kb", path=kb_path)]

        doctor = DoctorRunner(pkb_home=tmp_path)
        results = doctor.check_knowledge_bases(sample_config)
        # 2 results: KB path + inbox
        assert len(results) == 2
        assert all(r.ok for r in results)

    def test_kb_path_missing(self, tmp_path, sample_config):
        from pkb.doctor import DoctorRunner

        sample_config.knowledge_bases = [
            KBEntry(name="missing", path=tmp_path / "nonexistent"),
        ]

        doctor = DoctorRunner(pkb_home=tmp_path)
        results = doctor.check_knowledge_bases(sample_config)
        assert len(results) >= 1
        assert results[0].ok is False

    def test_kb_inbox_missing(self, tmp_path, sample_config):
        from pkb.doctor import DoctorRunner

        kb_path = tmp_path / "test-kb"
        kb_path.mkdir()
        # No inbox created

        sample_config.knowledge_bases = [KBEntry(name="test-kb", path=kb_path)]

        doctor = DoctorRunner(pkb_home=tmp_path)
        results = doctor.check_knowledge_bases(sample_config)
        # KB path OK, inbox FAIL
        assert results[0].ok is True
        assert results[1].ok is False

    def test_no_knowledge_bases(self, tmp_path, sample_config):
        from pkb.doctor import DoctorRunner

        sample_config.knowledge_bases = []

        doctor = DoctorRunner(pkb_home=tmp_path)
        results = doctor.check_knowledge_bases(sample_config)
        assert len(results) == 1
        assert results[0].ok is False
        assert "none configured" in results[0].detail.lower()


class TestCheckPostgres:
    @patch("pkb.doctor.psycopg")
    def test_postgres_ok(self, mock_psycopg, tmp_path, sample_config):
        from pkb.doctor import DoctorRunner

        mock_conn = MagicMock()
        mock_psycopg.connect.return_value = mock_conn
        mock_conn.execute.return_value = None

        doctor = DoctorRunner(pkb_home=tmp_path)
        result = doctor.check_postgres(sample_config)
        assert result.ok is True

        mock_conn.close.assert_called_once()

    @patch("pkb.doctor.psycopg")
    def test_postgres_fail(self, mock_psycopg, tmp_path, sample_config):
        from pkb.doctor import DoctorRunner

        mock_psycopg.connect.side_effect = Exception("Connection refused")

        doctor = DoctorRunner(pkb_home=tmp_path)
        result = doctor.check_postgres(sample_config)
        assert result.ok is False
        assert "Connection refused" in result.detail


class TestCheckChromaDB:
    @patch("pkb.doctor.chromadb")
    def test_chromadb_ok(self, mock_chromadb, tmp_path, sample_config):
        from pkb.doctor import DoctorRunner

        mock_client = MagicMock()
        mock_chromadb.HttpClient.return_value = mock_client
        mock_client.heartbeat.return_value = 123456

        doctor = DoctorRunner(pkb_home=tmp_path)
        result = doctor.check_chromadb(sample_config)
        assert result.ok is True

    @patch("pkb.doctor.chromadb")
    def test_chromadb_fail(self, mock_chromadb, tmp_path, sample_config):
        from pkb.doctor import DoctorRunner

        mock_chromadb.HttpClient.side_effect = Exception("timeout")

        doctor = DoctorRunner(pkb_home=tmp_path)
        result = doctor.check_chromadb(sample_config)
        assert result.ok is False
        assert "timeout" in result.detail


class TestCheckLLMProviders:
    def test_llm_provider_ok(self, tmp_path, sample_config, monkeypatch):
        from pkb.doctor import DoctorRunner

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        mock_provider = MagicMock()
        mock_provider.complete.return_value = "Hello!"
        mock_provider.model_name.return_value = "claude-haiku-4-5-20251001"

        doctor = DoctorRunner(pkb_home=tmp_path)

        with patch.object(
            doctor, "_create_test_provider", return_value=mock_provider,
        ):
            results = doctor.check_llm_providers(sample_config)

        assert len(results) == 1
        assert results[0].ok is True
        assert "env" in results[0].detail.lower()

    def test_llm_provider_fail(self, tmp_path, sample_config, monkeypatch):
        from pkb.doctor import DoctorRunner

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-invalid")

        mock_provider = MagicMock()
        mock_provider.complete.side_effect = Exception("401 Unauthorized")
        mock_provider.model_name.return_value = "claude-haiku-4-5-20251001"

        doctor = DoctorRunner(pkb_home=tmp_path)

        with patch.object(
            doctor, "_create_test_provider", return_value=mock_provider,
        ):
            results = doctor.check_llm_providers(sample_config)

        assert len(results) == 1
        assert results[0].ok is False
        assert "401" in results[0].detail

    def test_llm_no_key(self, tmp_path, sample_config, monkeypatch):
        from pkb.doctor import DoctorRunner

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        sample_config.llm.providers["anthropic"].api_key = None

        doctor = DoctorRunner(pkb_home=tmp_path)
        results = doctor.check_llm_providers(sample_config)

        assert len(results) == 1
        # Should still attempt (SDK default), report key source as "SDK default"
        # The provider creation itself would handle no-key scenarios

    def test_llm_config_key_source(self, tmp_path, monkeypatch):
        from pkb.doctor import DoctorRunner

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = PKBConfig(
            llm=LLMConfig(
                providers={
                    "anthropic": LLMProviderConfig(
                        api_key_env="ANTHROPIC_API_KEY",
                        api_key="sk-from-config",
                        models=[LLMModelEntry(name="claude-haiku-4-5-20251001", tier=1)],
                    ),
                },
            ),
        )

        mock_provider = MagicMock()
        mock_provider.complete.return_value = "Hello!"
        mock_provider.model_name.return_value = "claude-haiku-4-5-20251001"

        doctor = DoctorRunner(pkb_home=tmp_path)

        with patch.object(
            doctor, "_create_test_provider", return_value=mock_provider,
        ):
            results = doctor.check_llm_providers(config)

        assert len(results) == 1
        assert results[0].ok is True
        assert "config.yaml" in results[0].detail.lower()

    def test_llm_multiple_providers(self, tmp_path, monkeypatch):
        from pkb.doctor import DoctorRunner

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")

        config = PKBConfig(
            llm=LLMConfig(
                providers={
                    "anthropic": LLMProviderConfig(
                        api_key_env="ANTHROPIC_API_KEY",
                        models=[LLMModelEntry(name="claude-haiku-4-5-20251001", tier=1)],
                    ),
                    "openai": LLMProviderConfig(
                        api_key_env="OPENAI_API_KEY",
                        models=[LLMModelEntry(name="gpt-4o-mini", tier=1)],
                    ),
                },
            ),
        )

        mock_provider = MagicMock()
        mock_provider.complete.return_value = "Hello!"

        doctor = DoctorRunner(pkb_home=tmp_path)

        with patch.object(
            doctor, "_create_test_provider", return_value=mock_provider,
        ):
            results = doctor.check_llm_providers(config)

        assert len(results) == 2

    def test_no_llm_config(self, tmp_path):
        from pkb.doctor import DoctorRunner

        config = PKBConfig(llm=None)
        mock_provider = MagicMock()
        mock_provider.complete.return_value = "Hello!"
        mock_provider.model_name.return_value = "claude-haiku-4-5-20251001"

        doctor = DoctorRunner(pkb_home=tmp_path)

        with patch.object(
            doctor, "_create_test_provider", return_value=mock_provider,
        ):
            results = doctor.check_llm_providers(config)

        # Falls back to meta_llm — should produce 1 result
        assert len(results) == 1


class TestRunAll:
    def test_run_all_collects_results(self, tmp_path):
        from pkb.doctor import DoctorRunner

        # Create a valid config
        config_path = tmp_path / "config.yaml"
        config_data = {
            "knowledge_bases": [],
            "meta_llm": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        }
        config_path.write_text(yaml.dump(config_data))

        doctor = DoctorRunner(pkb_home=tmp_path, config_filename="config.yaml")
        results = doctor.run_all(skip_db=True, skip_llm=True)

        # Config check + KB check (none configured)
        assert len(results) >= 2

    def test_summary_counts(self, tmp_path):
        from pkb.doctor import CheckResult, DoctorRunner

        doctor = DoctorRunner(pkb_home=tmp_path)
        results = [
            CheckResult(label="Test 1", ok=True, detail=""),
            CheckResult(label="Test 2", ok=False, detail="failed"),
            CheckResult(label="Test 3", ok=True, detail=""),
        ]
        passed, failed = doctor.summary(results)
        assert passed == 2
        assert failed == 1


# --- CLI integration tests ---


class TestDoctorCLI:
    def test_doctor_help(self, runner):
        result = runner.invoke(cli, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "Diagnose" in result.output
        assert "--skip-llm" in result.output
        assert "--skip-db" in result.output

    def test_doctor_config_missing(self, runner, tmp_path, monkeypatch):
        monkeypatch.setenv("PKB_HOME", str(tmp_path))
        result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0
        assert "FAIL" in result.output

    def test_doctor_skip_flags(self, runner, tmp_path, monkeypatch):
        config_path = tmp_path / "config.yaml"
        config_data = {
            "knowledge_bases": [],
            "meta_llm": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        }
        config_path.write_text(yaml.dump(config_data))
        monkeypatch.setenv("PKB_HOME", str(tmp_path))

        result = runner.invoke(cli, ["doctor", "--skip-llm", "--skip-db"])
        assert result.exit_code == 0
        assert "PKB Doctor" in result.output
        # Should not show LLM/DB sections
        assert "PostgreSQL" not in result.output
        assert "LLM Providers" not in result.output

    def test_doctor_full_run_with_mocks(self, runner, tmp_path, monkeypatch):
        # Create config + KB directory
        kb_path = tmp_path / "my-kb"
        kb_path.mkdir()
        (kb_path / "inbox").mkdir()

        config_data = {
            "knowledge_bases": [{"name": "my-kb", "path": str(kb_path)}],
            "meta_llm": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
            "database": {
                "postgres": {"host": "db-host", "port": 5432, "database": "pkb"},
                "chromadb": {"host": "db-host", "port": 9000},
            },
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config_data))
        monkeypatch.setenv("PKB_HOME", str(tmp_path))

        with (
            patch("pkb.doctor.psycopg") as mock_pg,
            patch("pkb.doctor.chromadb") as mock_chroma,
        ):
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_client = MagicMock()
            mock_chroma.HttpClient.return_value = mock_client

            result = runner.invoke(cli, ["doctor", "--skip-llm"])

        assert result.exit_code == 0
        assert "Config" in result.output
        assert "my-kb" in result.output
        assert "PostgreSQL" in result.output
        assert "ChromaDB" in result.output
        assert "Summary" in result.output

    def test_doctor_exit_code_0_even_on_failures(self, runner, tmp_path, monkeypatch):
        """Doctor always exits 0 — failures are informational, not errors."""
        monkeypatch.setenv("PKB_HOME", str(tmp_path))
        result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0
