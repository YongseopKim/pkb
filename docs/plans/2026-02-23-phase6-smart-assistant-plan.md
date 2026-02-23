# Phase 6: Smart Assistant Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enhance the existing RAG chatbot into a "knowledge assistant" with advanced queries (digest, comparison, analysis), conversation modes, and expose PKB as an MCP server for Claude Code integration.

**Architecture:** Extends `ChatEngine` with a `DigestEngine` for topic/domain summaries, adds conversation modes via specialized system prompts, and creates a standalone MCP server (`pkb mcp-serve`) using the Python `mcp` package. All features build on Phase 5's `bundle_relations` for richer context assembly.

**Tech Stack:** Python `mcp` package (MCP server), Click (CLI), FastAPI (digest web routes), existing LLMRouter + SearchEngine, pytest (TDD)

**Depends on:** Phase 5 (Knowledge Graph) — uses `bundle_relations` and `find_bundles_sharing_topics`

---

### Task 1: Add DigestConfig to config models

**Files:**
- Modify: `src/pkb/models/config.py` (after RelationConfig)
- Modify: `src/pkb/models/config.py` (PKBConfig — add digest field)
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_config.py

class TestDigestConfig:
    def test_defaults(self):
        from pkb.models.config import DigestConfig
        config = DigestConfig()
        assert config.max_bundles == 20
        assert config.max_tokens == 4096

    def test_pkbconfig_includes_digest(self):
        from pkb.models.config import PKBConfig
        config = PKBConfig()
        assert hasattr(config, "digest")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::TestDigestConfig -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In `src/pkb/models/config.py` (after `RelationConfig`):

```python
class DigestConfig(BaseModel):
    """Configuration for Smart Digest generation."""

    max_bundles: int = 20
    max_tokens: int = 4096
```

In `PKBConfig`, add field:

```python
    digest: DigestConfig = DigestConfig()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::TestDigestConfig -v`
Expected: PASS

**Step 5: Commit**

```
feat(config): add DigestConfig for smart assistant settings
```

---

### Task 2: Create DigestEngine module

**Files:**
- Create: `src/pkb/digest.py`
- Test: `tests/test_digest.py`

**Step 1: Write the failing tests**

Create `tests/test_digest.py`:

```python
"""Tests for DigestEngine — topic/domain knowledge summaries."""

from unittest.mock import MagicMock

import pytest

from pkb.models.config import DigestConfig


@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def mock_search_engine():
    return MagicMock()


@pytest.fixture
def mock_router():
    return MagicMock()


@pytest.fixture
def engine(mock_repo, mock_search_engine, mock_router):
    config = DigestConfig()
    from pkb.digest import DigestEngine
    return DigestEngine(
        repo=mock_repo,
        search_engine=mock_search_engine,
        router=mock_router,
        config=config,
    )


class TestDigestByTopic:
    def test_generates_digest_for_topic(self, engine, mock_repo, mock_search_engine, mock_router):
        """digest_topic should gather bundles and produce an LLM summary."""
        from pkb.search.models import BundleSearchResult
        from datetime import datetime

        mock_search_engine.search.return_value = [
            BundleSearchResult(
                bundle_id="20260101-test-abc1",
                question="Python async는?",
                summary="async/await 패턴 설명",
                domains=["dev"],
                topics=["python"],
                score=0.9,
                created_at=datetime(2026, 1, 1),
                source="hybrid",
            ),
        ]
        mock_router.complete.return_value = "Python async에 대한 종합 요약입니다."

        result = engine.digest_topic("python", kb="personal")
        assert result.content is not None
        assert len(result.sources) >= 1
        mock_router.complete.assert_called_once()

    def test_empty_topic_returns_no_data(self, engine, mock_search_engine):
        """Topic with no bundles should return appropriate message."""
        mock_search_engine.search.return_value = []

        result = engine.digest_topic("nonexistent")
        assert "없" in result.content or "찾을 수 없" in result.content


class TestDigestByDomain:
    def test_generates_digest_for_domain(self, engine, mock_repo, mock_router):
        """digest_domain should gather domain bundles and summarize."""
        mock_repo.list_bundles_by_domain.return_value = [
            {
                "bundle_id": "20260101-test-abc1",
                "question": "투자 전략은?",
                "summary": "투자 전략 설명",
                "kb": "personal",
            },
        ]
        mock_router.complete.return_value = "투자 도메인 종합 요약입니다."

        result = engine.digest_domain("투자", kb="personal")
        assert result.content is not None

    def test_empty_domain_returns_no_data(self, engine, mock_repo):
        """Domain with no bundles should return appropriate message."""
        mock_repo.list_bundles_by_domain.return_value = []

        result = engine.digest_domain("nonexistent")
        assert "없" in result.content or "찾을 수 없" in result.content


class TestDigestReport:
    def test_report_includes_bundle_count(self, engine, mock_repo, mock_search_engine, mock_router):
        """Report should include metadata about bundles used."""
        from pkb.search.models import BundleSearchResult
        from datetime import datetime

        mock_search_engine.search.return_value = [
            BundleSearchResult(
                bundle_id=f"20260101-test-{i:04d}",
                question=f"Q{i}",
                summary=f"S{i}",
                domains=["dev"],
                topics=["python"],
                score=0.8,
                created_at=datetime(2026, 1, 1),
                source="hybrid",
            )
            for i in range(3)
        ]
        mock_router.complete.return_value = "종합 요약"

        result = engine.digest_topic("python")
        assert result.bundle_count == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_digest.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write implementation**

Create `src/pkb/digest.py`:

```python
"""Smart Digest — topic/domain knowledge summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pkb.generator.prompts import load_prompt
from pkb.models.config import DigestConfig
from pkb.search.models import SearchMode, SearchQuery

if TYPE_CHECKING:
    from pkb.db.postgres import BundleRepository
    from pkb.llm.router import LLMRouter
    from pkb.search.engine import SearchEngine


@dataclass
class DigestResult:
    """Result of a digest operation."""

    content: str
    sources: list[dict] = field(default_factory=list)
    bundle_count: int = 0
    topic: str | None = None
    domain: str | None = None


class DigestEngine:
    """Generates topic/domain knowledge summaries from the knowledge base."""

    def __init__(
        self,
        *,
        repo: BundleRepository,
        search_engine: SearchEngine,
        router: LLMRouter,
        config: DigestConfig,
    ) -> None:
        self._repo = repo
        self._search = search_engine
        self._router = router
        self._max_bundles = config.max_bundles
        self._max_tokens = config.max_tokens

    def digest_topic(
        self, topic: str, *, kb: str | None = None,
    ) -> DigestResult:
        """Generate a comprehensive digest for a topic.

        Searches for bundles related to the topic and produces
        an LLM-generated summary of accumulated knowledge.
        """
        query = SearchQuery(
            query=topic,
            mode=SearchMode.HYBRID,
            kb=kb,
            topics=[topic],
            limit=self._max_bundles,
        )
        results = self._search.search(query)

        if not results:
            return DigestResult(
                content=f"'{topic}' 토픽에 관련된 번들을 찾을 수 없습니다.",
                topic=topic,
            )

        context = self._build_digest_context(results)
        prompt = self._build_digest_prompt(
            subject=f"토픽: {topic}",
            context=context,
        )

        summary = self._router.complete(
            prompt, task="chat", max_tokens=self._max_tokens, temperature=0.3,
        )

        sources = [
            {"bundle_id": r.bundle_id, "summary": r.summary}
            for r in results
        ]

        return DigestResult(
            content=summary,
            sources=sources,
            bundle_count=len(results),
            topic=topic,
        )

    def digest_domain(
        self, domain: str, *, kb: str | None = None,
    ) -> DigestResult:
        """Generate a comprehensive digest for a domain."""
        bundles = self._repo.list_bundles_by_domain(domain, kb=kb)

        if not bundles:
            return DigestResult(
                content=f"'{domain}' 도메인에 관련된 번들을 찾을 수 없습니다.",
                domain=domain,
            )

        # Build context from bundle metadata
        context_parts = []
        for i, b in enumerate(bundles[:self._max_bundles], 1):
            context_parts.append(
                f"[{i}] {b['bundle_id']}: {b['question']}\n"
                f"    요약: {b.get('summary', '(없음)')}"
            )
        context = "\n\n".join(context_parts)

        prompt = self._build_digest_prompt(
            subject=f"도메인: {domain}",
            context=context,
        )

        summary = self._router.complete(
            prompt, task="chat", max_tokens=self._max_tokens, temperature=0.3,
        )

        sources = [
            {"bundle_id": b["bundle_id"], "summary": b.get("summary")}
            for b in bundles[:self._max_bundles]
        ]

        return DigestResult(
            content=summary,
            sources=sources,
            bundle_count=len(bundles),
            domain=domain,
        )

    def _build_digest_context(self, results: list) -> str:
        """Build context string from search results."""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] {r.bundle_id}: {r.question}\n"
                f"    요약: {r.summary or '(없음)'}\n"
                f"    도메인: {', '.join(r.domains)}\n"
                f"    토픽: {', '.join(r.topics)}"
            )
        return "\n\n".join(parts)

    def _build_digest_prompt(self, *, subject: str, context: str) -> str:
        """Build the LLM prompt for digest generation."""
        return (
            "당신은 PKB(Private Knowledge Base)의 지식 분석가입니다.\n\n"
            f"## 분석 대상: {subject}\n\n"
            f"## 관련 번들 목록\n\n{context}\n\n"
            "## 지시사항\n\n"
            "위 번들들을 종합하여 다음을 포함하는 한국어 리포트를 작성하세요:\n"
            "1. **핵심 요약**: 이 주제에 대해 축적된 지식의 전체 요약\n"
            "2. **주요 인사이트**: 반복적으로 등장하는 핵심 개념이나 결론\n"
            "3. **관점 변화**: 시간에 따른 관점이나 이해의 변화 (있다면)\n"
            "4. **출처**: 참조한 번들 ID 목록\n"
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_digest.py -v`
Expected: PASS

**Step 5: Add list_bundles_by_domain to BundleRepository**

Add to `tests/test_db_relations.py`:

```python
class TestListBundlesByDomain:
    def test_method_exists(self):
        from pkb.db.postgres import BundleRepository
        assert hasattr(BundleRepository, "list_bundles_by_domain")
```

Add to `src/pkb/db/postgres.py`:

```python
    def list_bundles_by_domain(
        self, domain: str, kb: str | None = None,
    ) -> list[dict]:
        """List bundles belonging to a domain, optionally filtered by KB."""
        with self._get_conn() as conn:
            if kb:
                rows = conn.execute(
                    "SELECT b.id AS bundle_id, b.kb, b.question, b.summary, "
                    "b.created_at "
                    "FROM bundles b "
                    "JOIN bundle_domains bd ON bd.bundle_id = b.id "
                    "WHERE bd.domain = %s AND b.kb = %s "
                    "ORDER BY b.created_at DESC",
                    (domain, kb),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT b.id AS bundle_id, b.kb, b.question, b.summary, "
                    "b.created_at "
                    "FROM bundles b "
                    "JOIN bundle_domains bd ON bd.bundle_id = b.id "
                    "WHERE bd.domain = %s "
                    "ORDER BY b.created_at DESC",
                    (domain,),
                ).fetchall()
        return [
            {
                "bundle_id": row[0],
                "kb": row[1],
                "question": row[2],
                "summary": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]
```

**Step 6: Run all tests**

Run: `pytest tests/test_digest.py tests/test_db_relations.py -v`
Expected: PASS

**Step 7: Commit**

```
feat: add DigestEngine for topic/domain knowledge summaries
```

---

### Task 3: Add conversation modes to ChatEngine

**Files:**
- Create: `prompts/chat_analyst.txt`
- Create: `prompts/chat_writer.txt`
- Modify: `src/pkb/chat/engine.py` (add mode parameter)
- Modify: `src/pkb/chat/context.py` (mode-aware prompt loading)
- Test: `tests/test_chat_modes.py`

**Step 1: Write the failing tests**

Create `tests/test_chat_modes.py`:

```python
"""Tests for ChatEngine conversation modes."""

from unittest.mock import MagicMock

import pytest


class TestChatModes:
    def test_explorer_mode_is_default(self):
        from pkb.chat.engine import ChatEngine

        engine = ChatEngine(
            search_engine=MagicMock(),
            router=MagicMock(),
        )
        assert engine._mode == "explorer"

    def test_analyst_mode_accepted(self):
        from pkb.chat.engine import ChatEngine

        engine = ChatEngine(
            search_engine=MagicMock(),
            router=MagicMock(),
            mode="analyst",
        )
        assert engine._mode == "analyst"

    def test_writer_mode_accepted(self):
        from pkb.chat.engine import ChatEngine

        engine = ChatEngine(
            search_engine=MagicMock(),
            router=MagicMock(),
            mode="writer",
        )
        assert engine._mode == "writer"

    def test_invalid_mode_raises(self):
        from pkb.chat.engine import ChatEngine

        with pytest.raises(ValueError):
            ChatEngine(
                search_engine=MagicMock(),
                router=MagicMock(),
                mode="invalid",
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_chat_modes.py -v`
Expected: FAIL — ChatEngine doesn't accept `mode` param

**Step 3: Write implementation**

Modify `src/pkb/chat/engine.py` — add `mode` parameter:

```python
VALID_MODES = ("explorer", "analyst", "writer")

class ChatEngine:
    def __init__(
        self,
        *,
        search_engine: SearchEngine,
        router: LLMRouter,
        kb: str | None = None,
        max_results: int = 5,
        max_history: int = 10,
        mode: str = "explorer",
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {VALID_MODES}")
        self._search = search_engine
        self._router = router
        self._kb = kb
        self._max_results = max_results
        self._max_history = max_history
        self._mode = mode
```

Update `assemble_context` in `src/pkb/chat/context.py` to accept `mode`:

```python
def assemble_context(
    *,
    question: str,
    search_results: list[BundleSearchResult],
    history: list[ChatMessage],
    max_results: int = 5,
    mode: str = "explorer",
) -> str:
    system_prompt = _load_system_prompt(mode)
    # ... rest stays the same

def _load_system_prompt(mode: str = "explorer") -> str:
    prompt_name = f"chat_{mode}" if mode != "explorer" else "chat_system"
    try:
        return load_prompt(prompt_name)
    except FileNotFoundError:
        return (
            "당신은 PKB(Private Knowledge Base)의 RAG 챗봇입니다. "
            "사용자의 지식베이스에 저장된 대화 내용을 바탕으로 질문에 답변합니다. "
            "답변은 한국어로 제공하고, 출처를 명확히 밝혀주세요."
        )
```

Update `ChatEngine.ask()` to pass `mode`:

```python
        prompt = assemble_context(
            question=question,
            search_results=results,
            history=history,
            max_results=self._max_results,
            mode=self._mode,
        )
```

Create `prompts/chat_analyst.txt`:

```
당신은 PKB(Private Knowledge Base)의 지식 분석가입니다.

역할:
- 사용자의 지식베이스에 축적된 데이터를 분석하고 인사이트를 도출합니다.
- 시간에 따른 관점 변화, 패턴, 트렌드를 식별합니다.
- 여러 LLM 응답 간의 비교 분석을 수행합니다.

답변 규칙:
1. 한국어로 답변합니다.
2. 데이터 기반의 분석적 답변을 제공합니다.
3. 관점 변화나 모순이 발견되면 명시합니다.
4. 출처 형식: [번호] bundle_id
```

Create `prompts/chat_writer.txt`:

```
당신은 PKB(Private Knowledge Base)의 콘텐츠 작성 도우미입니다.

역할:
- 사용자의 지식베이스를 기반으로 글, 보고서, 노트 초안을 작성합니다.
- 축적된 지식을 구조화하고 재구성합니다.
- 아웃라인, 요약, 전체 초안을 생성합니다.

답변 규칙:
1. 한국어로 작성합니다.
2. 논리적 구조와 흐름을 갖춘 콘텐츠를 생성합니다.
3. 지식베이스의 원본 인사이트를 충실히 반영합니다.
4. 출처 형식: [번호] bundle_id
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_chat_modes.py -v`
Expected: PASS

**Step 5: Update CLI chat command**

In `src/pkb/cli.py`, add `--mode` option to `chat`:

```python
@cli.command()
@click.option("--kb", default=None, help="Knowledge base name filter.")
@click.option(
    "--mode",
    type=click.Choice(["explorer", "analyst", "writer"]),
    default="explorer",
    help="Conversation mode.",
)
def chat(kb: str | None, mode: str) -> None:
    # ... existing setup ...
    engine = ChatEngine(
        search_engine=search_engine,
        router=router,
        kb=kb,
        mode=mode,
    )
```

**Step 6: Commit**

```
feat(chat): add conversation modes (explorer/analyst/writer)
```

---

### Task 4: Add `pkb digest` CLI command

**Files:**
- Modify: `src/pkb/cli.py` (add `digest` command)
- Test: `tests/test_cli_commands.py`

**Step 1: Write the failing test**

Add to `tests/test_cli_commands.py`:

```python
class TestDigestCommand:
    def test_digest_topic_command_exists(self):
        from click.testing import CliRunner
        from pkb.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["digest", "--help"])
        assert result.exit_code == 0
        assert "topic" in result.output.lower() or "domain" in result.output.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_commands.py::TestDigestCommand -v`
Expected: FAIL — no such command

**Step 3: Write implementation**

Add to `src/pkb/cli.py`:

```python
@cli.command()
@click.option("--topic", default=None, help="Topic to digest.")
@click.option("--domain", default=None, help="Domain to digest.")
@click.option("--kb", default=None, help="Knowledge base name filter.")
@click.option("--output", "-o", default=None, type=click.Path(), help="Save to file.")
def digest(topic: str | None, domain: str | None, kb: str | None, output: str | None) -> None:
    """Generate a knowledge digest for a topic or domain.

    Summarizes all accumulated knowledge on the specified subject.
    """
    if not topic and not domain:
        raise click.ClickException("Specify --topic or --domain")

    from pkb.config import build_llm_router, get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.chromadb_client import ChunkStore
    from pkb.db.postgres import BundleRepository
    from pkb.digest import DigestEngine
    from pkb.search.engine import SearchEngine

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
        chunk_store = ChunkStore(config.database.chromadb)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    search_engine = SearchEngine(repo=repo, chunk_store=chunk_store)
    router = build_llm_router(config)

    engine = DigestEngine(
        repo=repo,
        search_engine=search_engine,
        router=router,
        config=config.digest,
    )

    click.echo("Generating digest...")

    if topic:
        result = engine.digest_topic(topic, kb=kb)
    else:
        result = engine.digest_domain(domain, kb=kb)

    if output:
        from pathlib import Path
        Path(output).write_text(result.content, encoding="utf-8")
        click.echo(f"Saved to {output}")
    else:
        click.echo(f"\n{result.content}")
        click.echo(f"\n--- {result.bundle_count} bundles referenced ---")

    repo.close()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_commands.py::TestDigestCommand -v`
Expected: PASS

**Step 5: Commit**

```
feat(cli): add 'pkb digest' command for knowledge summaries
```

---

### Task 5: Create MCP server

**Files:**
- Create: `src/pkb/mcp_server.py`
- Modify: `src/pkb/cli.py` (add `mcp-serve` command)
- Modify: `pyproject.toml` (add `mcp` dependency)
- Test: `tests/test_mcp_server.py`

**Step 1: Add `mcp` to dependencies**

In `pyproject.toml`, add to `dependencies`:
```
"mcp>=1.0",
```

Run: `pip install -e ".[dev]"`

**Step 2: Write the failing tests**

Create `tests/test_mcp_server.py`:

```python
"""Tests for PKB MCP server tools."""

from unittest.mock import MagicMock

import pytest


class TestMCPTools:
    def test_pkb_search_tool_defined(self):
        from pkb.mcp_server import create_mcp_server
        server = create_mcp_server.__wrapped__  # Check function exists
        # Or just verify import works
        from pkb.mcp_server import create_mcp_server
        assert callable(create_mcp_server)

    def test_tool_names(self):
        """MCP server should expose expected tools."""
        from pkb.mcp_server import TOOL_NAMES
        assert "pkb_search" in TOOL_NAMES
        assert "pkb_digest" in TOOL_NAMES
        assert "pkb_related" in TOOL_NAMES
        assert "pkb_stats" in TOOL_NAMES
```

**Step 3: Write implementation**

Create `src/pkb/mcp_server.py`:

```python
"""PKB MCP Server — expose knowledge base to Claude Code."""

from __future__ import annotations

import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

TOOL_NAMES = {"pkb_search", "pkb_digest", "pkb_related", "pkb_stats"}


def create_mcp_server() -> Server:
    """Create and configure the PKB MCP server."""
    server = Server("pkb")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="pkb_search",
                description="Search the PKB knowledge base (hybrid FTS + semantic)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "mode": {
                            "type": "string",
                            "enum": ["hybrid", "keyword", "semantic"],
                            "default": "hybrid",
                        },
                        "domain": {"type": "string", "description": "Domain filter"},
                        "kb": {"type": "string", "description": "KB name filter"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="pkb_digest",
                description="Generate a knowledge digest for a topic or domain",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "domain": {"type": "string"},
                        "kb": {"type": "string"},
                    },
                },
            ),
            Tool(
                name="pkb_related",
                description="Find bundles related to a given bundle",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bundle_id": {"type": "string"},
                        "relation_type": {
                            "type": "string",
                            "enum": ["similar", "related", "all"],
                            "default": "all",
                        },
                    },
                    "required": ["bundle_id"],
                },
            ),
            Tool(
                name="pkb_stats",
                description="Get PKB statistics (bundle counts, domain/topic distribution)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "kb": {"type": "string"},
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        from pkb.config import build_llm_router, get_pkb_home, load_config
        from pkb.constants import CONFIG_FILENAME
        from pkb.db.chromadb_client import ChunkStore
        from pkb.db.postgres import BundleRepository
        from pkb.search.engine import SearchEngine

        pkb_home = get_pkb_home()
        config = load_config(pkb_home / CONFIG_FILENAME)

        repo = BundleRepository(config.database.postgres)
        chunk_store = ChunkStore(config.database.chromadb)
        search_engine = SearchEngine(repo=repo, chunk_store=chunk_store)

        try:
            if name == "pkb_search":
                result = _handle_search(search_engine, arguments)
            elif name == "pkb_digest":
                router = build_llm_router(config)
                result = _handle_digest(repo, search_engine, router, config, arguments)
            elif name == "pkb_related":
                result = _handle_related(repo, arguments)
            elif name == "pkb_stats":
                result = _handle_stats(repo, arguments)
            else:
                result = f"Unknown tool: {name}"
        finally:
            repo.close()

        return [TextContent(type="text", text=result)]

    return server


def _handle_search(search_engine, args: dict) -> str:
    from pkb.search.models import SearchMode, SearchQuery
    query = SearchQuery(
        query=args["query"],
        mode=SearchMode(args.get("mode", "hybrid")),
        kb=args.get("kb"),
        domains=[args["domain"]] if args.get("domain") else [],
        limit=args.get("limit", 5),
    )
    results = search_engine.search(query)
    output = []
    for r in results:
        output.append({
            "bundle_id": r.bundle_id,
            "question": r.question,
            "summary": r.summary,
            "domains": r.domains,
            "topics": r.topics,
            "score": round(r.score, 3),
        })
    return json.dumps(output, ensure_ascii=False, indent=2)


def _handle_digest(repo, search_engine, router, config, args: dict) -> str:
    from pkb.digest import DigestEngine
    engine = DigestEngine(
        repo=repo, search_engine=search_engine,
        router=router, config=config.digest,
    )
    if args.get("topic"):
        result = engine.digest_topic(args["topic"], kb=args.get("kb"))
    elif args.get("domain"):
        result = engine.digest_domain(args["domain"], kb=args.get("kb"))
    else:
        return "Error: specify 'topic' or 'domain'"
    return result.content


def _handle_related(repo, args: dict) -> str:
    bundle_id = args["bundle_id"]
    rel_type = args.get("relation_type", "all")
    filter_type = None if rel_type == "all" else rel_type
    relations = repo.list_relations(bundle_id, relation_type=filter_type)
    output = []
    for r in relations:
        other = (
            r["target_bundle_id"]
            if r["source_bundle_id"] == bundle_id
            else r["source_bundle_id"]
        )
        output.append({
            "related_bundle": other,
            "type": r["relation_type"],
            "score": r["score"],
        })
    return json.dumps(output, ensure_ascii=False, indent=2)


def _handle_stats(repo, args: dict) -> str:
    kb = args.get("kb")
    bundle_ids = repo.list_all_bundle_ids(kb=kb)
    relation_count = repo.count_relations()
    return json.dumps({
        "total_bundles": len(bundle_ids),
        "total_relations": relation_count,
    }, indent=2)


async def run_server() -> None:
    """Run the MCP server via stdio."""
    server = create_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
```

Add to `src/pkb/cli.py`:

```python
@cli.command("mcp-serve")
def mcp_serve() -> None:
    """Start PKB as an MCP server (stdio transport)."""
    import asyncio

    from pkb.mcp_server import run_server

    asyncio.run(run_server())
```

**Step 4: Run tests**

Run: `pytest tests/test_mcp_server.py -v`
Expected: PASS

**Step 5: Commit**

```
feat: add MCP server for Claude Code integration
```

---

### Task 6: Add digest web routes

**Files:**
- Create: `src/pkb/web/routes/digest.py`
- Create: `src/pkb/web/templates/digest/form.html`
- Create: `src/pkb/web/templates/digest/result.html`
- Modify: `src/pkb/web/app.py` (register router)
- Test: `tests/test_web_digest.py`

**Step 1: Write the failing test**

Create `tests/test_web_digest.py`:

```python
"""Tests for digest web routes."""


class TestDigestRoutes:
    def test_digest_form_route_exists(self):
        from pkb.web.routes.digest import router
        paths = [r.path for r in router.routes]
        assert "" in paths or "/" in paths

    def test_digest_generate_route_exists(self):
        from pkb.web.routes.digest import router
        methods = []
        for r in router.routes:
            if hasattr(r, "methods"):
                methods.extend(r.methods)
        assert "POST" in methods
```

**Step 2: Write implementation**

Create `src/pkb/web/routes/digest.py`:

```python
"""Web routes for knowledge digest."""

from fastapi import APIRouter, Form, Request

router = APIRouter(prefix="/digest", tags=["digest"])


@router.get("")
def digest_form(request: Request):
    """Show digest generation form."""
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "digest/form.html", {})


@router.post("")
def digest_generate(
    request: Request,
    topic: str = Form(default=None),
    domain: str = Form(default=None),
    kb: str = Form(default=None),
):
    """Generate a digest and show results."""
    from pkb.config import build_llm_router, get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.digest import DigestEngine

    pkb = request.app.state.pkb
    templates = request.app.state.templates

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)
    router = build_llm_router(config)

    engine = DigestEngine(
        repo=pkb.repo,
        search_engine=pkb.search_engine,
        router=router,
        config=config.digest,
    )

    if topic:
        result = engine.digest_topic(topic, kb=kb or None)
    elif domain:
        result = engine.digest_domain(domain, kb=kb or None)
    else:
        return templates.TemplateResponse(request, "digest/form.html", {
            "error": "토픽 또는 도메인을 지정하세요.",
        })

    return templates.TemplateResponse(request, "digest/result.html", {
        "result": result,
    })
```

Create templates and register router in `app.py`.

**Step 3: Commit**

```
feat(web): add digest routes and templates
```

---

### Task 7: Ruff/lint + full test suite

**Step 1:** Run `ruff check src/ tests/`
**Step 2:** Run `pytest tests/ -v --ignore=tests/integration`
**Step 3:** Fix any issues
**Step 4:** Commit

```
chore: lint and test cleanup for Phase 6
```

---

### Task 8: Update documentation

Update `CLAUDE.md`, `docs/design-v1.md` with Phase 6 features:
- `pkb digest --topic/--domain`
- `pkb chat --mode analyst/writer`
- `pkb mcp-serve`
- MCP server configuration example

```
docs: update documentation for Phase 6
```

---

## Summary

| Task | Description | Key Files |
|------|-------------|-----------|
| 1 | DigestConfig | `models/config.py` |
| 2 | DigestEngine | `src/pkb/digest.py`, `db/postgres.py` (list_bundles_by_domain) |
| 3 | Conversation modes | `chat/engine.py`, `chat/context.py`, `prompts/chat_*.txt` |
| 4 | CLI digest command | `cli.py` |
| 5 | MCP server | `src/pkb/mcp_server.py`, `pyproject.toml` |
| 6 | Web digest routes | `web/routes/digest.py`, templates |
| 7 | Lint + tests | various |
| 8 | Documentation | `CLAUDE.md`, `docs/design-v1.md` |
