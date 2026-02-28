"""PKB MCP Server — expose knowledge base to Claude Code."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from pkb.digest import DigestEngine

TOOL_NAMES = {"pkb_search", "pkb_digest", "pkb_related", "pkb_stats", "pkb_ingest", "pkb_browse"}

# Lazy-init state for DB connections (shared across tool calls)
_state: dict[str, Any] = {}


def _get_state() -> dict[str, Any]:
    """Lazily initialise shared PKB state (config, repo, search, router)."""
    if "repo" not in _state:
        from pkb.config import build_chunk_store, build_llm_router, get_pkb_home, load_config
        from pkb.constants import CONFIG_FILENAME
        from pkb.db.postgres import BundleRepository
        from pkb.search.engine import SearchEngine

        pkb_home = get_pkb_home()
        config = load_config(pkb_home / CONFIG_FILENAME)

        repo = BundleRepository(config.database.postgres)
        chunk_store = build_chunk_store(config)
        search_engine = SearchEngine(repo=repo, chunk_store=chunk_store)
        router = build_llm_router(config)

        _state["config"] = config
        _state["repo"] = repo
        _state["search_engine"] = search_engine
        _state["router"] = router
    return _state


def _build_pipeline(state: dict, kb: str | None = None) -> Any:
    """Build an IngestPipeline for the first (or specified) KB."""
    from pathlib import Path

    from pkb.config import get_pkb_home
    from pkb.generator.meta import MetaGenerator
    from pkb.ingest import IngestPipeline
    from pkb.vocab import load_domains, load_topics

    config = state["config"]
    kbs = config.knowledge_bases
    if not kbs:
        return None
    kb_config = next((k for k in kbs if k.name == kb), kbs[0]) if kb else kbs[0]

    pkb_home = get_pkb_home()
    domains = load_domains(pkb_home)
    topics = load_topics(pkb_home)
    meta_gen = MetaGenerator(router=state["router"])

    if "chunk_store" not in state:
        from pkb.config import build_chunk_store

        state["chunk_store"] = build_chunk_store(config)

    return IngestPipeline(
        repo=state["repo"],
        chunk_store=state["chunk_store"],
        meta_gen=meta_gen,
        kb_path=Path(kb_config.path).expanduser(),
        kb_name=kb_config.name,
        domains=domains,
        topics=topics,
    )


def create_mcp_server() -> FastMCP:
    """Create and configure the PKB MCP server."""
    mcp = FastMCP("pkb")

    @mcp.tool()
    def pkb_search(
        query: str,
        mode: str = "hybrid",
        domain: str | None = None,
        kb: str | None = None,
        limit: int = 5,
    ) -> str:
        """Search the PKB knowledge base (hybrid FTS + semantic)."""
        state = _get_state()
        return _handle_search(
            state["search_engine"],
            {"query": query, "mode": mode, "domain": domain, "kb": kb, "limit": limit},
        )

    @mcp.tool()
    def pkb_digest(
        topic: str | None = None,
        domain: str | None = None,
        kb: str | None = None,
    ) -> str:
        """Generate a knowledge digest for a topic or domain."""
        state = _get_state()
        return _handle_digest(
            state["repo"],
            state["search_engine"],
            state["router"],
            state["config"],
            {"topic": topic, "domain": domain, "kb": kb},
        )

    @mcp.tool()
    def pkb_related(
        bundle_id: str,
        relation_type: str = "all",
    ) -> str:
        """Find bundles related to a given bundle."""
        state = _get_state()
        return _handle_related(
            state["repo"],
            {"bundle_id": bundle_id, "relation_type": relation_type},
        )

    @mcp.tool()
    def pkb_stats(kb: str | None = None) -> str:
        """Get PKB statistics (bundle counts, relation counts)."""
        state = _get_state()
        return _handle_stats(state["repo"], {"kb": kb})

    @mcp.tool()
    def pkb_ingest(file_path: str, kb: str | None = None) -> str:
        """Ingest a file (JSONL or MD) into the knowledge base."""
        state = _get_state()
        pipeline = _build_pipeline(state, kb=kb)
        if pipeline is None:
            return json.dumps({"error": "No KB configured"})
        return _handle_ingest(pipeline, {"file_path": file_path})

    @mcp.tool()
    def pkb_browse(
        domain: str | None = None,
        topic: str | None = None,
        days: int | None = None,
        kb: str | None = None,
        limit: int = 20,
    ) -> str:
        """Browse bundles by domain, topic, or recent days."""
        state = _get_state()
        return _handle_browse(
            state["repo"],
            {"domain": domain, "topic": topic, "days": days, "kb": kb, "limit": limit},
        )

    return mcp


# --- Handler functions (testable without MCP runtime) ---


def _handle_search(search_engine: Any, args: dict) -> str:
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


def _handle_digest(repo: Any, search_engine: Any, router: Any, config: Any, args: dict) -> str:
    engine = DigestEngine(
        repo=repo,
        search_engine=search_engine,
        router=router,
        config=config.digest,
    )
    if args.get("topic"):
        result = engine.digest_topic(args["topic"], kb=args.get("kb"))
    elif args.get("domain"):
        result = engine.digest_domain(args["domain"], kb=args.get("kb"))
    else:
        return "Error: specify 'topic' or 'domain'"
    return result.content


def _handle_related(repo: Any, args: dict) -> str:
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


def _handle_stats(repo: Any, args: dict) -> str:
    kb = args.get("kb")
    bundle_ids = repo.list_all_bundle_ids(kb=kb)
    relation_count = repo.count_relations()
    return json.dumps({
        "total_bundles": len(bundle_ids),
        "total_relations": relation_count,
    }, indent=2)


def _handle_ingest(pipeline: Any, args: dict) -> str:
    """Ingest a file into the knowledge base."""
    from pathlib import Path

    file_path = Path(args["file_path"])
    if not file_path.exists():
        return json.dumps({"error": f"File not found: {file_path}"})

    result = pipeline.ingest_file(file_path)
    if result is None:
        return json.dumps({"error": "Ingest returned None"})
    return json.dumps(result, ensure_ascii=False, default=str, indent=2)


def _handle_browse(repo: Any, args: dict) -> str:
    """Browse bundles by domain, topic, or recent days."""
    from datetime import datetime, timedelta, timezone

    domain = args.get("domain")
    topic = args.get("topic")
    days = args.get("days")
    kb = args.get("kb")
    limit = args.get("limit", 20)

    if domain:
        bundles = repo.list_bundles_by_domain(domain, kb=kb)
    elif topic:
        bundles = repo.list_bundles_by_topic(topic, kb=kb)
    elif days:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        bundles = repo.list_bundles_since(since, kb=kb)
    else:
        return json.dumps({"error": "Specify domain, topic, or days"})

    bundles = bundles[:limit]
    return json.dumps(bundles, ensure_ascii=False, default=str, indent=2)


def main() -> None:
    """Run the MCP server via stdio."""
    mcp = create_mcp_server()
    mcp.run(transport="stdio")
