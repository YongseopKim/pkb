"""PKB Doctor — diagnose configuration, database, and LLM connectivity."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import psycopg
import yaml

from pkb.models.config import PKBConfig


@dataclass
class CheckResult:
    """Result of a single diagnostic check."""

    label: str
    ok: bool
    detail: str


@dataclass
class SectionResult:
    """A section header plus its check results."""

    header: str
    checks: list[CheckResult] = field(default_factory=list)


class DoctorRunner:
    """Runs diagnostic checks against PKB configuration and services."""

    def __init__(
        self,
        pkb_home: Path,
        config_filename: str = "config.yaml",
    ) -> None:
        self._pkb_home = pkb_home
        self._config_filename = config_filename

    # --- Config ---

    def check_config(self) -> CheckResult:
        """Check config.yaml exists, parses, and validates."""
        config_path = self._pkb_home / self._config_filename
        if not config_path.exists():
            return CheckResult(
                label="Config",
                ok=False,
                detail=f"Not found: {config_path}",
            )
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            PKBConfig(**raw)
        except Exception as e:
            return CheckResult(label="Config", ok=False, detail=str(e))

        return CheckResult(
            label="Config",
            ok=True,
            detail=f"{config_path}",
        )

    # --- Knowledge Bases ---

    def check_knowledge_bases(self, config: PKBConfig) -> list[CheckResult]:
        """Check each KB path and inbox directory."""
        results: list[CheckResult] = []

        if not config.knowledge_bases:
            results.append(CheckResult(
                label="Knowledge Bases",
                ok=False,
                detail="None configured",
            ))
            return results

        for kb in config.knowledge_bases:
            # KB path
            if kb.path.exists():
                results.append(CheckResult(
                    label=f"  {kb.name}",
                    ok=True,
                    detail=f"{kb.path}",
                ))
            else:
                results.append(CheckResult(
                    label=f"  {kb.name}",
                    ok=False,
                    detail=f"Path not found: {kb.path}",
                ))
                continue  # skip inbox check if path missing

            # Inbox
            inbox = kb.get_watch_dir()
            if inbox.exists():
                results.append(CheckResult(
                    label="    inbox",
                    ok=True,
                    detail=f"{inbox}",
                ))
            else:
                results.append(CheckResult(
                    label="    inbox",
                    ok=False,
                    detail=f"Not found: {inbox}",
                ))

        return results

    # --- PostgreSQL ---

    def check_postgres(self, config: PKBConfig) -> CheckResult:
        """Check PostgreSQL connectivity."""
        pg = config.database.postgres
        label = "PostgreSQL"
        try:
            conn = psycopg.connect(pg.get_dsn())
            conn.execute("SELECT 1")
            conn.close()
            return CheckResult(
                label=label,
                ok=True,
                detail=f"{pg.host}:{pg.port}/{pg.database}",
            )
        except Exception as e:
            return CheckResult(label=label, ok=False, detail=str(e))

    # --- ChromaDB ---

    def check_chromadb(self, config: PKBConfig) -> CheckResult:
        """Check ChromaDB connectivity."""
        ch = config.database.chromadb
        label = "ChromaDB"
        try:
            client = chromadb.HttpClient(host=ch.host, port=ch.port)
            client.heartbeat()
            return CheckResult(
                label=label,
                ok=True,
                detail=f"{ch.host}:{ch.port}",
            )
        except Exception as e:
            return CheckResult(label=label, ok=False, detail=str(e))

    # --- LLM Providers ---

    def _resolve_api_key(
        self, provider_name: str, api_key_env: str | None, api_key: str | None,
    ) -> tuple[str | None, str]:
        """Resolve API key and return (key, source_description)."""
        if api_key_env:
            env_val = os.environ.get(api_key_env)
            if env_val:
                return env_val, f"env {api_key_env}"
        if api_key:
            return api_key, "config.yaml"
        return None, "SDK default"

    def _create_test_provider(self, provider_name: str, model: str, api_key: str | None):
        """Create a provider instance for testing. Factored out for mocking."""
        from pkb.llm.router import LLMRouter
        return LLMRouter._create_provider(provider_name, model, api_key)

    def check_llm_providers(self, config: PKBConfig) -> list[CheckResult]:
        """Check each LLM provider with a minimal API call."""
        results: list[CheckResult] = []

        if config.llm is not None:
            providers_config = config.llm.providers
        else:
            # Fall back to meta_llm
            providers_config = {
                config.meta_llm.provider: type(
                    "FakeProviderConfig", (), {
                        "api_key_env": None,
                        "api_key": None,
                        "models": [type("M", (), {"name": config.meta_llm.model, "tier": 1})()],
                    },
                )(),
            }

        for provider_name, prov_config in providers_config.items():
            for model_entry in prov_config.models:
                api_key, key_source = self._resolve_api_key(
                    provider_name,
                    getattr(prov_config, "api_key_env", None),
                    getattr(prov_config, "api_key", None),
                )
                label = f"  {provider_name} ({model_entry.name})"
                try:
                    provider = self._create_test_provider(
                        provider_name, model_entry.name, api_key,
                    )
                    provider.complete("Say hello", max_tokens=10)
                    results.append(CheckResult(
                        label=label,
                        ok=True,
                        detail=f"key: {key_source}",
                    ))
                except Exception as e:
                    results.append(CheckResult(
                        label=label,
                        ok=False,
                        detail=str(e),
                    ))

        return results

    # --- Run all ---

    def run_all(
        self,
        *,
        skip_db: bool = False,
        skip_llm: bool = False,
    ) -> list[CheckResult]:
        """Run all checks and return flat results list."""
        results: list[CheckResult] = []

        # 1. Config
        config_result = self.check_config()
        results.append(config_result)

        if not config_result.ok:
            return results

        # Load config for remaining checks
        config_path = self._pkb_home / self._config_filename
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = PKBConfig(**raw)

        # 2. Knowledge Bases
        results.extend(self.check_knowledge_bases(config))

        # 3. Database
        if not skip_db:
            results.append(self.check_postgres(config))
            results.append(self.check_chromadb(config))

        # 4. LLM Providers
        if not skip_llm:
            results.extend(self.check_llm_providers(config))

        return results

    def run_all_sectioned(
        self,
        *,
        skip_db: bool = False,
        skip_llm: bool = False,
    ) -> list[SectionResult]:
        """Run all checks and return results grouped by section."""
        sections: list[SectionResult] = []

        # 1. Config
        config_result = self.check_config()
        sections.append(SectionResult(header="", checks=[config_result]))

        if not config_result.ok:
            return sections

        # Load config for remaining checks
        config_path = self._pkb_home / self._config_filename
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = PKBConfig(**raw)

        # 2. Knowledge Bases
        kb_checks = self.check_knowledge_bases(config)
        sections.append(SectionResult(header="Knowledge Bases", checks=kb_checks))

        # 3. Database
        if not skip_db:
            pg = self.check_postgres(config)
            ch = self.check_chromadb(config)
            # Indent DB check labels under section header
            pg.label = f"  {pg.label}"
            ch.label = f"  {ch.label}"
            sections.append(SectionResult(header="Database", checks=[pg, ch]))

        # 4. LLM Providers
        if not skip_llm:
            llm_checks = self.check_llm_providers(config)
            sections.append(SectionResult(header="LLM Providers", checks=llm_checks))

        return sections

    @staticmethod
    def summary(results: list[CheckResult]) -> tuple[int, int]:
        """Return (passed, failed) counts."""
        passed = sum(1 for r in results if r.ok)
        failed = sum(1 for r in results if not r.ok)
        return passed, failed


def _format_line(label: str, ok: bool, detail: str) -> str:
    """Format a single check line with aligned dots."""
    status = "OK" if ok else "FAIL"
    pad = max(1, 30 - len(label))
    dots = "." * pad
    detail_str = f"  ({detail})" if detail else ""
    return f"{label} {dots} {status}{detail_str}"


def format_results(sections: list[SectionResult]) -> str:
    """Format sectioned check results for CLI output."""
    lines: list[str] = ["PKB Doctor", "=" * 40, ""]

    all_checks: list[CheckResult] = []
    for section in sections:
        if section.header:
            lines.append(section.header)
        for check in section.checks:
            all_checks.append(check)
            lines.append(_format_line(check.label, check.ok, check.detail))

    lines.append("")
    passed, failed = DoctorRunner.summary(all_checks)
    total = passed + failed
    lines.append(f"Summary: {passed}/{total} checks passed, {failed} failed")

    return "\n".join(lines)
