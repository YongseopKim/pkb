"""PKB MCP Server — expose knowledge base to Claude Code."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from pkb.digest import DigestEngine

TOOL_NAMES = {"pkb_search", "pkb_digest", "pkb_related", "pkb_stats"}

# Lazy-init state for DB connections (shared across tool calls)
_state: dict[str, Any] = {}


def _get_state() -> dict[str, Any]:
    """Lazily initialise shared PKB state (config, repo, search, router)."""
    if "repo" not in _state:
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
        router = build_llm_router(config)

        _state["config"] = config
        _state["repo"] = repo
        _state["search_engine"] = search_engine
        _state["router"] = router
    return _state


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


def main() -> None:
    """Run the MCP server via stdio."""
    mcp = create_mcp_server()
    mcp.run(transport="stdio")
