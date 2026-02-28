# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PKB (Private Knowledge Base) is a Python CLI tool that processes multi-LLM conversation exports (JSONL and MD from llm-chat-exporter Chrome extension) into organized, searchable "bundles" with auto-generated metadata. The primary language of the project documents and user conversations is Korean.

**Current status**: Phase 0 through Phase 10 complete. `docs/design-v1.md` is the authoritative design document. `docs/plans/2026-02-28-second-brain-evolution-design.md` defines the Phase 8-10 roadmap.

## Architecture

PKB separates **tools** (this repo) from **data** (separate KB repos):

```
pkb/                          <- This repo (tool)
├── src/pkb/                  <- Python package
│   ├── cli.py                <- Click CLI (init, parse, ingest, batch, topics, search, reindex, regenerate, reembed, watch, dedup, relate, digest, stats, report, web, chat, mcp-serve, kb, db, doctor)
│   ├── config.py             <- Config loading
│   ├── init.py               <- pkb init logic
│   ├── ingest.py             <- IngestPipeline orchestrator
│   ├── engine.py             <- IngestEngine (async concurrent ingest)
│   ├── batch.py              <- BatchProcessor for bulk ingestion (sequential + concurrent)
│   ├── reindex.py            <- Reindexer (sync _bundle.md edits to DB)
│   ├── regenerate.py         <- Regenerator (re-run LLM meta extraction)
│   ├── reembed.py            <- ReembedEngine (re-embed with new embedding model)
│   ├── watcher.py            <- KBWatcher + AsyncFileEventHandler (watchdog auto-ingest)
│   ├── dedup.py              <- DuplicateDetector (embedding similarity)
│   ├── relations.py          <- RelationBuilder (knowledge graph edges)
│   ├── digest.py             <- DigestEngine (topic/domain knowledge summaries)
│   ├── analytics.py          <- AnalyticsEngine (bundle statistics aggregation)
│   ├── report.py             <- ReportGenerator (weekly/monthly markdown reports)
│   ├── post_ingest.py        <- PostIngestProcessor (auto-relate, auto-dedup, gap-update)
│   ├── scheduler.py          <- Scheduler (periodic tasks: weekly digest, monthly report)
│   ├── doctor.py             <- System diagnostics (DB, ChromaDB, LLM API health checks)
│   ├── mcp_server.py         <- MCP server (FastMCP, 14 tools for Claude Code)
│   ├── logging_config.py     <- Logging setup (console + file handlers)
│   ├── constants.py          <- Platform names, path constants, skip filenames
│   ├── models/               <- Pydantic models (jsonl, config, vocab, meta)
│   ├── parser/               <- Input parsing (JSONL + MD, file + directory)
│   ├── embedding/            <- Embedding abstraction (Embedder ABC, TEIEmbedder, factory)
│   ├── db/                   <- Database layer (PostgreSQL + ChromaDB + Alembic migrations)
│   ├── search/               <- Search engine (FTS + semantic + hybrid)
│   ├── generator/            <- MD, meta, prompts, chunker, frontmatter_parser
│   ├── vocab/                <- Vocabulary loading + TopicManager + TopicSyncer
│   ├── llm/                  <- LLM routing (Anthropic + OpenAI + Google + Grok providers)
│   ├── chat/                 <- RAG chatbot engine (ChatEngine, context assembly)
│   ├── web/                  <- FastAPI web UI (htmx, Jinja2, D3.js graph, Chart.js, compare view)
│   └── data/                 <- Bundled seed data (domains, topics)
├── tests/                    <- 1342 tests (1286 mock + 56 integration)
├── scripts/                  <- Build + release scripts
│   └── hooks/                <- Git hooks (core.hooksPath target)
├── prompts/                  <- LLM prompt templates (response_meta, bundle_meta, chat_system, chat_analyst, chat_writer)
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
- **VectorDB**: ChromaDB (remote). Embedding: server-side (default) or client-side via TEI (bge-m3, 1024d)
- **Viewer**: Obsidian with Dataview plugin
- **Bundle ID**: `{YYYYMMDD}-{slug}-{hash4}` (e.g., `20260221-pkb-system-design-a3f2`)
- **Stable ID**: SHA-256 of normalized URL (primary) or initial turns fingerprint (fallback). Enables file relocation tracking, content update detection, and dedup. Same stable_id + same platform → UPDATE (re-run LLM, refresh DB/ChromaDB). Same stable_id + different platform → MERGE.
- **Tag system**: 2-tier — Domain (L1, 8 fixed) + Topic (L2, controlled vocab with pending workflow)
- **API keys**: Priority env var > config.yaml `api_key` > SDK default. `PKB_DB_PASSWORD` for DB.

## Git Hooks

Git hooks are stored in `scripts/hooks/` and activated via `core.hooksPath`:

```bash
git config core.hooksPath scripts/hooks   # Required after fresh clone
```

**Available hooks**:
- `post-commit` — Auto patch bump + build after each commit (`release.sh patch`). Skips during rebase/merge/cherry-pick. Set `PKB_SKIP_POST_COMMIT=1` to disable.

## Development Commands

```bash
pip install -e ".[dev]"          # Install in dev mode (includes all deps)
pytest                            # Run 1342 tests (1286 mock + 56 integration with PKB_DB_INTEGRATION=1)
ruff check src/ tests/            # Lint (line-length=100)
pkb --version                     # CLI version check
pkb -v <command>                  # Verbose mode (INFO logging to console + file)
pkb -vv <command>                 # Debug mode (DEBUG logging to console + file)
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
pkb reembed <bundle_id> --kb <n>  # Re-embed single bundle with current model
pkb reembed --all --kb <name>     # Re-embed all bundles
pkb reembed --all --fresh --kb <name>  # Drop collection + re-embed all (model change)
pkb watch                         # Watch all KB inboxes (recursive, success → .done/)
pkb watch --kb <name>             # Watch specific KB inbox (subdirectory support)
pkb dedup scan --kb <name>        # Scan for duplicate bundles
pkb dedup list                    # List duplicate pairs
pkb dedup dismiss <pair_id>       # Mark pair as non-duplicate
pkb dedup confirm <pair_id>       # Confirm duplicate pair
pkb relate scan --kb <name>       # Scan for bundle relations (knowledge graph)
pkb relate scan --type similar    # Scan specific relation type
pkb relate list                   # List all relations
pkb relate show <bundle_id>       # Show relations for a bundle
pkb digest --topic python         # Generate knowledge digest for a topic
pkb digest --domain dev --kb n    # Generate digest for a domain
pkb digest --topic ai -o out.md   # Save digest to file
pkb stats                         # Show KB overview stats
pkb stats --domain                # Show domain distribution detail
pkb stats --json                  # JSON output mode
pkb stats --kb <name>             # Filter by KB
pkb report                        # Generate weekly activity report
pkb report --period monthly       # Generate monthly report (+ knowledge gaps)
pkb report --kb <name> -o out.md  # Filter by KB, save to file
pkb web --port 8080               # Start local web UI (FastAPI + htmx)
pkb chat --kb <name>              # Interactive RAG chatbot REPL
pkb chat --mode analyst           # Chat in analyst mode (explorer/analyst/writer)
pkb mcp-serve                     # Start PKB as MCP server (stdio transport)
pkb db upgrade                    # Upgrade DB schema to latest (Alembic)
pkb db upgrade --revision 0001    # Upgrade to specific revision
pkb db downgrade 0001             # Downgrade to specific revision
pkb db current                    # Show current DB revision
pkb db history                    # Show migration history
pkb db stamp head                 # Stamp revision without running SQL
pkb db migrate-domain <old> <new> # Rename domain in bundle_domains
pkb db migrate-stable-id         # Recompute stable_id from raw files
pkb db migrate-stable-id --dry-run  # Preview without writing
pkb kb list                       # List configured KBs with bundle counts
pkb doctor                        # System diagnostics (DB, ChromaDB, LLM API health)
pkb doctor --skip-llm             # Skip LLM API checks
pkb doctor --skip-db              # Skip DB checks
pkb db reset --kb <name>          # Delete all data for a KB (requires confirmation)
```

## Integration Testing (Local DB)

Requires Docker Desktop.

```bash
# Start test containers (once)
docker compose -f docker/docker-compose.test.yml up -d

# Run integration tests only (46 DB tests)
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
  chunk_size: 1500
  chunk_overlap: 200
  mode: tei                       # "server" (ChromaDB default) | "tei" (client-side via TEI)
  model_name: BAAI/bge-m3         # Model identifier (stored in collection metadata)
  dimensions: 1024                # Vector dimensions (0=server auto)
  tei_url: http://localhost:8090  # TEI server URL (mode=tei only)
  tei_batch_size: 32              # TEI batch size
  tei_timeout: 30.0               # TEI request timeout (seconds)
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
digest:                           # Optional, all fields have defaults
  max_bundles: 20                 # Max bundles to include in digest
  max_tokens: 4096                # Max LLM response tokens
post_ingest:                      # Optional, all fields have defaults
  auto_relate: true               # Auto-scan bundle relations after ingest
  auto_dedup: true                # Auto-scan duplicate detection after ingest
  gap_update: true                # Check if bundle topics are knowledge gaps
scheduler:                        # Optional, all fields have defaults
  weekly_digest: true             # Enable weekly digest generation
  monthly_report: true            # Enable monthly report generation
  gap_threshold: 3                # Topic bundle count below this = knowledge gap
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

- `src/pkb/` — Python package (Phase 0 through 10 implemented)
- `tests/` — 1342 tests (1286 mock + 56 integration) covering models, parser (JSONL + MD), vocab, config, CLI, DB, migrations, generator, ingest, batch, engine, search, reindex, regenerate, reembed, watcher, dedup, LLM routing, embedding, web (app, analytics, relations, compare), chat, kb, relations, digest, MCP server (14 tools), analytics, doctor, stable_id, post-ingest, scheduler
- `docker/` — Docker Compose for local test DB (PostgreSQL + ChromaDB)
- `prompts/` — LLM prompt templates (response_meta, bundle_meta, chat_system, chat_analyst, chat_writer)
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
- **Phase 5** ✓: Knowledge graph — `bundle_relations` table, `RelationBuilder` (similar/related/sequel edges), `pkb relate` CLI, relations web UI
- **Phase 6** ✓: Smart Assistant — `DigestEngine` (topic/domain summaries), conversation modes (explorer/analyst/writer), MCP server (`pkb mcp-serve`), digest web UI
- **Phase 7** ✓: Analytics Dashboard — `AnalyticsEngine` (statistics aggregation), `ReportGenerator` (weekly/monthly reports), `pkb stats`/`pkb report` CLI, web dashboard (Chart.js)
- **Phase 8** ✓: Automation Pipeline — `PostIngestProcessor` (auto-relate, auto-dedup, gap-update), metadata utilization (consensus/divergence/key_claims/stance stored + searchable), `Scheduler` (periodic weekly digest/monthly report), DB migration 0006
- **Phase 9** ✓: MCP Extension — 4→14 tools (`pkb_ingest`, `pkb_browse`, `pkb_detail`, `pkb_graph`, `pkb_gaps`, `pkb_claims`, `pkb_timeline`, `pkb_recent`, `pkb_compare`, `pkb_suggest`), `get_responses_for_bundle()`, `list_bundles_by_topic()` DB methods
- **Phase 10** ✓: Web UI Enhancement — D3.js 지식 그래프 시각화, 인사이트 대시보드 (AnalyticsEngine 통합), LLM 비교 뷰 (consensus/divergence highlighting), Chat 3-panel layout (context sidebar + htmx OOB swap), compare 라우트

## Database Migration Workflow (Alembic)

Schema is managed by Alembic migrations in `src/pkb/db/migrations/versions/`. Raw SQL via `op.execute()`.

**Migrations**:
- `0001_initial_schema` — All 6 core tables + tsvector trigger
- `0002_add_source_path` — `bundles.source_path` column + index
- `0003_add_source_path_to_responses` — `bundle_responses.source_path` column + index (per-platform file tracking for merged bundles)
- `0004_bundle_relations` — `bundle_relations` table + indexes (knowledge graph edges)
- `0005_add_stable_id` — `bundles.stable_id` column (NOT NULL, UNIQUE) + backfill from question_hash
- `0006_add_metadata_columns` — `bundles.consensus`, `bundles.divergence` TEXT columns + `bundle_responses.key_claims` JSONB, `bundle_responses.stance` TEXT + GIN index

**Source path tracking**: `bundles.source_path` stores the first-ingested file path. `bundle_responses.source_path` stores per-platform file paths (important for merged bundles). `find_by_source_path()` checks `bundle_responses` first, then falls back to `bundles`.

**Stable ID**: `bundles.stable_id` is the primary dedup key. Computed as SHA-256 of normalized URL (strip query/fragment, lowercase hostname) or first-5-turns fingerprint (fallback for URL-less exports). `find_bundle_by_stable_id()` replaces `find_bundle_by_question_hash()` for dedup. `question_hash` column is kept for backward compatibility but deprecated.

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
