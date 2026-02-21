# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PKB (Private Knowledge Base) is a Python CLI tool that processes multi-LLM conversation exports (JSONL and MD from llm-chat-exporter Chrome extension) into organized, searchable "bundles" with auto-generated metadata. The primary language of the project documents and user conversations is Korean.

**Current status**: Phase 0 through Phase 4 complete. `docs/design-v1.md` is the authoritative design document.

## Architecture

PKB separates **tools** (this repo) from **data** (separate KB repos):

```
pkb/                          <- This repo (tool)
├── src/pkb/                  <- Python package
│   ├── cli.py                <- Click CLI (init, parse, ingest, batch, topics, search, reindex, regenerate, watch, dedup, web, chat, kb, db)
│   ├── config.py             <- Config loading
│   ├── init.py               <- pkb init logic
│   ├── ingest.py             <- IngestPipeline orchestrator
│   ├── engine.py             <- IngestEngine (async concurrent ingest)
│   ├── batch.py              <- BatchProcessor for bulk ingestion (sequential + concurrent)
│   ├── reindex.py            <- Reindexer (sync _bundle.md edits to DB)
│   ├── regenerate.py         <- Regenerator (re-run LLM meta extraction)
│   ├── watcher.py            <- KBWatcher + AsyncFileEventHandler (watchdog auto-ingest)
│   ├── dedup.py              <- DuplicateDetector (embedding similarity)
│   ├── constants.py          <- Platform names, path constants
│   ├── models/               <- Pydantic models (jsonl, config, vocab, meta)
│   ├── parser/               <- Input parsing (JSONL + MD, file + directory)
│   ├── db/                   <- Database layer (PostgreSQL + ChromaDB + Alembic migrations)
│   ├── search/               <- Search engine (FTS + semantic + hybrid)
│   ├── generator/            <- MD, meta, prompts, chunker, frontmatter_parser
│   ├── vocab/                <- Vocabulary loading + TopicManager + TopicSyncer
│   ├── llm/                  <- LLM routing (Anthropic + OpenAI + Google + Grok providers)
│   ├── chat/                 <- RAG chatbot engine (ChatEngine, context assembly)
│   ├── web/                  <- FastAPI web UI (htmx, Jinja2 templates)
│   └── data/                 <- Bundled seed data (domains, topics)
├── tests/                    <- 778 tests (733 mock + 45 integration)
├── prompts/                  <- LLM prompt templates (response_meta, bundle_meta, chat_system)
└── pyproject.toml

~/.pkb/                       <- Global config (auto-generated)
├── config.yaml               <- KB registry + DB settings + LLM config
└── vocab/                    <- Shared vocabulary (domains.yaml, topics.yaml)

~/kb-personal/                <- KB repo (data, separate git repo)
├── inbox/                    <- Watch directory for auto-ingest (recursive)
│   ├── PKB/                  <- Subdirectories from exporter (auto-detected)
│   │   ├── chatgpt.md
│   │   └── claude.md
│   └── .done/                <- Successfully ingested files (structure preserved)
└── bundles/{bundle_id}/
    ├── _raw/*.jsonl           <- Immutable JSONL from exporter
    ├── {platform}.md          <- Generated MD with frontmatter (derived)
    └── _bundle.md             <- Aggregate metadata (derived)
```

**Core principle**: Raw data (JSONL) is immutable. All generated files (MD, frontmatter, _bundle.md) are "derived" and can be regenerated.

## Key Design Decisions

- **Input**: JSONL and MD formats from llm-chat-exporter
- **Meta LLM**: Multi-provider via LLMRouter (Anthropic, OpenAI, Google, Grok) with tier-based routing
- **Structured DB**: PostgreSQL (remote, meta + FTS via tsvector/GIN)
- **VectorDB**: ChromaDB (remote, server-side embedding)
- **Viewer**: Obsidian with Dataview plugin
- **Bundle ID**: `{YYYYMMDD}-{slug}-{hash4}` (e.g., `20260221-pkb-system-design-a3f2`)
- **Tag system**: 2-tier — Domain (L1, 8 fixed) + Topic (L2, controlled vocab with pending workflow)
- **API keys**: Priority env var > config.yaml `api_key` > SDK default. `PKB_DB_PASSWORD` for DB.

## Development Commands

```bash
pip install -e ".[dev]"          # Install in dev mode (includes all deps)
pytest                            # Run 733 mock tests (+ 45 integration with PKB_DB_INTEGRATION=1)
ruff check src/ tests/            # Lint (line-length=100)
pkb --version                     # CLI version check
pkb init                          # Initialize ~/.pkb/
pkb parse <file_or_dir>           # Parse JSONL and show summary
pkb ingest <path> --kb <name>     # Ingest into KB (success → file moves to inbox/.done/)
pkb batch <dir> --kb <name>       # Bulk ingest with checkpoint/resume (sequential)
pkb batch <dir> --kb <name> --workers 4  # Concurrent ingest (success → .done/)
pkb topics                        # List all topics (backward compatible)
pkb topics list --status pending  # List pending topics only
pkb topics approve <name>         # Approve a pending topic
pkb topics merge <name> --into <target>  # Merge topic into another
pkb topics reject <name>          # Reject/remove a topic
pkb search "query" --mode hybrid  # Search bundles (hybrid/keyword/semantic)
pkb search "query" --domain dev   # Search with domain filter
pkb search "query" --json         # JSON output
pkb reindex <bundle_id> --kb <n>  # Sync _bundle.md frontmatter edits to DB
pkb reindex --full --kb <name>    # Full reindex + orphan cleanup
pkb regenerate <id> --kb <name>   # Re-run LLM meta extraction from raw JSONL
pkb regenerate --all --kb <name>  # Regenerate all bundles
pkb regenerate <id> --dry-run     # Preview without DB writes
pkb watch                         # Watch all KB inboxes (recursive, success → .done/)
pkb watch --kb <name>             # Watch specific KB inbox (subdirectory support)
pkb dedup scan --kb <name>        # Scan for duplicate bundles
pkb dedup list                    # List duplicate pairs
pkb dedup dismiss <pair_id>       # Mark pair as non-duplicate
pkb dedup confirm <pair_id>       # Confirm duplicate pair
pkb web --port 8080               # Start local web UI (FastAPI + htmx)
pkb chat --kb <name>              # Interactive RAG chatbot REPL
pkb db upgrade                    # Upgrade DB schema to latest (Alembic)
pkb db upgrade --revision 0001    # Upgrade to specific revision
pkb db downgrade 0001             # Downgrade to specific revision
pkb db current                    # Show current DB revision
pkb db history                    # Show migration history
pkb db stamp head                 # Stamp revision without running SQL
pkb db migrate-domain <old> <new> # Rename domain in bundle_domains
pkb kb list                       # List configured KBs with bundle counts
pkb db reset --kb <name>          # Delete all data for a KB (requires confirmation)
```

## Integration Testing (Local DB)

Requires Docker Desktop.

```bash
# Start test containers (once)
docker compose -f docker/docker-compose.test.yml up -d

# Run integration tests only (35 DB tests)
PKB_DB_INTEGRATION=1 pytest tests/integration/db/ -v

# Run all tests (mock + integration)
PKB_DB_INTEGRATION=1 pytest -v

# Stop containers
docker compose -f docker/docker-compose.test.yml down
```

Ports: PostgreSQL 5433, ChromaDB 8001 (separated from production).

## Config Schema (config.yaml)

```yaml
knowledge_bases:
  - name: personal
    path: ~/kb-personal
    watch_dir: ~/kb-personal/inbox  # Optional, defaults to {path}/inbox
meta_llm:                          # Legacy (still works, fallback)
  provider: anthropic
  model: claude-haiku-4-5-20251001
  max_retries: 3
  temperature: 0
llm:                               # New multi-provider config (takes priority over meta_llm)
  default_provider: anthropic
  providers:
    anthropic:
      api_key_env: ANTHROPIC_API_KEY
      api_key: ""                        # Or set key directly (priority: env > config > SDK)
      models:
        - name: claude-haiku-4-5-20251001
          tier: 1
    openai:
      api_key_env: OPENAI_API_KEY
      api_key: ""
      models:
        - name: gpt-4o-mini
          tier: 1
    google:
      api_key_env: GOOGLE_API_KEY
      api_key: ""
      models:
        - name: gemini-2.0-flash
          tier: 1
    grok:
      api_key_env: XAI_API_KEY
      api_key: ""
      models:
        - name: grok-3-mini-fast
          tier: 1
  routing:
    meta_extraction: 1
    chat: 1
    escalation: true
embedding:
  chunk_size: 512
  chunk_overlap: 50
database:
  postgres:
    host: "192.168.1.100"
    port: 5432
    database: pkb_db
    username: pkb_user
    password: ""  # Use PKB_DB_PASSWORD env var
  chromadb:
    host: "192.168.1.100"
    port: 8000
    collection: pkb_chunks
concurrency:                      # Optional, all fields have defaults
  max_concurrent_files: 4         # Concurrent file processing workers
  max_concurrent_llm: 4           # LLM API concurrent call limit
  max_queue_size: 10000           # Bounded event queue size
  batch_window: 5.0               # Event collection window (seconds)
  max_batch_size: 50              # Max files per drain batch
  chunk_buffer_size: 0            # ChromaDB batch flush (0=disabled, Phase 2)
  db_pool_min: 2                  # Connection pool min size
  db_pool_max: 8                  # Connection pool max size
```

## Repository Contents

- `src/pkb/` — Python package (Phase 0 through 4 implemented)
- `tests/` — 778 tests (733 mock + 45 integration) covering models, parser (JSONL + MD), vocab, config, CLI, DB, migrations, generator, ingest, batch, engine, search, reindex, regenerate, watcher, dedup, LLM routing, web, chat, kb
- `docker/` — Docker Compose for local test DB (PostgreSQL + ChromaDB)
- `prompts/` — LLM prompt templates (response_meta, bundle_meta, chat_system)
- `docs/design-v1.md` — **Unified design document**. Single source of truth.
- `exporter-examples/` — Sample JSONL/MD files for testing

## JSONL Format (Input)

```jsonl
{"_meta":true,"platform":"claude","url":"...","exported_at":"2026-02-21T06:02:42.230Z","title":"..."}
{"role":"user","content":"...","timestamp":"..."}
{"role":"assistant","content":"...","timestamp":"..."}
```

Line 1 is always a `_meta` object. Subsequent lines are `role`/`content`/`timestamp` turns.

**Important**: Claude/Perplexity exports have consecutive assistant turns (thinking + response, multi-answer blocks). Parser does NOT assume strict user/assistant alternation.

## MD Format (Input)

```markdown
# [Claude](https://claude.ai/chat/abc...)   ← header with platform + URL (may be absent)

---
---                                          ← double separator (may be absent)

## LLM 응답 1                                ← response section (may be absent)

Content with arbitrary markdown...
```

MD parser uses **graceful degradation**:
- **Level 1**: `# [Platform](URL)` header + `## LLM 응답 N` sections → structured assistant turns
- **Level 2**: Any `## ` headings → split into sections as assistant turns
- **Level 3**: No structure → entire content as single assistant turn

All MD turns are `role="assistant"`. Platform detected from header URL domain, filename stem, or explicit parameter.

## Phased Implementation Plan

- **Phase 0** ✓: Repo scaffolding, vocab seed, JSONL parser prototype
- **Phase 1** ✓: Ingest pipeline — `pkb ingest` produces MD + frontmatter + _bundle.md via Haiku API, stores meta in PostgreSQL, chunks in ChromaDB
- **Phase 2** ✓: Search layer — `pkb search` with hybrid FTS + semantic search, weighted scoring (0.4 FTS + 0.6 semantic)
- **Phase 3** ✓: Automation — `pkb reindex` (frontmatter sync), `pkb regenerate` (LLM re-extraction), `pkb watch` (watchdog auto-ingest)
- **Phase 4** ✓: Topic CLI (approve/merge/reject), duplicate detection, LLM routing (multi-provider), web UI (FastAPI + htmx), RAG chatbot

## Database Migration Workflow (Alembic)

Schema is managed by Alembic migrations in `src/pkb/db/migrations/versions/`. Raw SQL via `op.execute()`.

**Migrations**:
- `0001_initial_schema` — All 6 core tables + tsvector trigger
- `0002_add_source_path` — `bundles.source_path` column + index
- `0003_add_source_path_to_responses` — `bundle_responses.source_path` column + index (per-platform file tracking for merged bundles)

**Source path tracking**: `bundles.source_path` stores the first-ingested file path. `bundle_responses.source_path` stores per-platform file paths (important for merged bundles). `find_by_source_path()` checks `bundle_responses` first, then falls back to `bundles`.

**Existing DB auto-stamp**: If `bundles` table exists but `alembic_version` doesn't, `pkb db upgrade` auto-stamps at head (no SQL re-run).

**Adding a new migration**:
1. Create `src/pkb/db/migrations/versions/NNNN_descriptive_name.py`
2. Write `upgrade()` and `downgrade()` with raw SQL
3. Update `CREATE_TABLES_SQL` in `schema.py` to reflect final state
4. Run `pkb db upgrade`

## Documentation Maintenance

When updating documentation at the end of a development cycle, always update these files:
- `README.md` — User-facing project overview, usage, roadmap status
- `CLAUDE.md` — Developer/AI guidance, architecture, commands, phase status
- `docs/design-v1.md` — If design decisions change
