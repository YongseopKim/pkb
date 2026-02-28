"""PKB MCP Server — expose knowledge base to Claude Code."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from pkb.digest import DigestEngine

TOOL_NAMES = {
    "pkb_search", "pkb_digest", "pkb_related", "pkb_stats",
    "pkb_ingest", "pkb_browse", "pkb_detail", "pkb_graph",
    "pkb_gaps", "pkb_claims", "pkb_timeline", "pkb_recent",
    "pkb_compare", "pkb_suggest",
}

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


def _get_analytics(state: dict) -> Any:
    """Get or create AnalyticsEngine from state."""
    if "analytics" not in state:
        from pkb.analytics import AnalyticsEngine

        state["analytics"] = AnalyticsEngine(repo=state["repo"])
    return state["analytics"]


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

    @mcp.tool()
    def pkb_detail(bundle_id: str) -> str:
        """Get detailed bundle information including responses, claims, and relations."""
        state = _get_state()
        return _handle_detail(state["repo"], {"bundle_id": bundle_id})

    @mcp.tool()
    def pkb_graph(bundle_id: str, depth: int = 1) -> str:
        """Explore the knowledge graph around a bundle (BFS traversal)."""
        state = _get_state()
        return _handle_graph(state["repo"], {"bundle_id": bundle_id, "depth": depth})

    @mcp.tool()
    def pkb_gaps(threshold: int = 3, kb: str | None = None) -> str:
        """Find knowledge gap topics (fewer bundles than threshold)."""
        state = _get_state()
        analytics = _get_analytics(state)
        return _handle_gaps(analytics, {"threshold": threshold, "kb": kb})

    @mcp.tool()
    def pkb_claims(query: str, kb: str | None = None, limit: int = 10) -> str:
        """Search key claims across all bundle responses."""
        state = _get_state()
        return _handle_claims(state["repo"], {"query": query, "kb": kb, "limit": limit})

    @mcp.tool()
    def pkb_timeline(topic: str, kb: str | None = None) -> str:
        """Get chronological bundle list for a topic (oldest first)."""
        state = _get_state()
        return _handle_timeline(state["repo"], {"topic": topic, "kb": kb})

    @mcp.tool()
    def pkb_recent(days: int = 7, kb: str | None = None) -> str:
        """Get recently ingested bundles."""
        state = _get_state()
        return _handle_recent(state["repo"], {"days": days, "kb": kb})

    @mcp.tool()
    def pkb_compare(bundle_id: str) -> str:
        """Compare LLM responses within a bundle (consensus, divergence, per-platform claims)."""
        state = _get_state()
        return _handle_compare(state["repo"], {"bundle_id": bundle_id})

    @mcp.tool()
    def pkb_suggest(topic: str | None = None, kb: str | None = None) -> str:
        """Suggest exploration directions based on knowledge gaps and thin topics."""
        state = _get_state()
        analytics = _get_analytics(state)
        return _handle_suggest(
            state["repo"], analytics, {"topic": topic, "kb": kb},
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


def _handle_detail(repo: Any, args: dict) -> str:
    """Get full bundle metadata including per-platform responses and relations."""
    bundle_id = args["bundle_id"]
    bundle = repo.get_bundle_by_id(bundle_id)
    if bundle is None:
        return json.dumps({"error": f"Bundle not found: {bundle_id}"})

    responses = repo.get_responses_for_bundle(bundle_id)
    relations = repo.list_relations(bundle_id)

    output = {**bundle, "responses": responses, "relations": relations}
    return json.dumps(output, ensure_ascii=False, default=str, indent=2)


def _handle_graph(repo: Any, args: dict) -> str:
    """BFS traversal of the knowledge graph around a bundle_id."""
    root = args["bundle_id"]
    depth = min(args.get("depth", 1), 3)  # cap at 3

    bundle = repo.get_bundle_by_id(root)
    if bundle is None:
        return json.dumps({"error": f"Bundle not found: {root}"})

    visited: set[str] = set()
    nodes: list[dict] = []
    edges: list[dict] = []
    queue = [root]

    for _ in range(depth):
        next_queue: list[str] = []
        for bid in queue:
            if bid in visited:
                continue
            visited.add(bid)
            info = repo.get_bundle_by_id(bid)
            if info:
                nodes.append({
                    "bundle_id": bid,
                    "question": info.get("question", ""),
                    "domains": info.get("domains", []),
                    "topics": info.get("topics", []),
                })
            for rel in repo.list_relations(bid):
                other = (
                    rel["target_bundle_id"]
                    if rel["source_bundle_id"] == bid
                    else rel["source_bundle_id"]
                )
                edges.append({
                    "source": rel["source_bundle_id"],
                    "target": rel["target_bundle_id"],
                    "type": rel["relation_type"],
                    "score": rel["score"],
                })
                if other not in visited:
                    next_queue.append(other)
        queue = next_queue

    # deduplicate edges
    seen_edges: set[tuple] = set()
    unique_edges = []
    for e in edges:
        key = (e["source"], e["target"], e["type"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(e)

    return json.dumps(
        {"nodes": nodes, "edges": unique_edges},
        ensure_ascii=False,
        default=str,
        indent=2,
    )


def _handle_gaps(analytics: Any, args: dict) -> str:
    """Return knowledge gap topics as JSON."""
    threshold = args.get("threshold", 3)
    kb = args.get("kb")
    gaps = analytics.knowledge_gaps(threshold=threshold, kb=kb)
    return json.dumps(gaps, ensure_ascii=False, indent=2)


def _handle_claims(repo: Any, args: dict) -> str:
    """Search key claims across all responses and return JSON."""
    results = repo.search_claims(
        args["query"],
        kb=args.get("kb"),
        limit=args.get("limit", 10),
    )
    return json.dumps(results, ensure_ascii=False, default=str, indent=2)


def _handle_timeline(repo: Any, args: dict) -> str:
    """Return bundles for a topic in chronological order (oldest first)."""
    topic = args.get("topic")
    if not topic:
        return json.dumps({"error": "Specify a topic"})
    kb = args.get("kb")
    bundles = repo.list_bundles_by_topic(topic, kb=kb)
    # Reverse to chronological order (oldest first)
    bundles.sort(key=lambda b: str(b.get("created_at", "")))
    return json.dumps(bundles, ensure_ascii=False, default=str, indent=2)


def _handle_recent(repo: Any, args: dict) -> str:
    """Return recently ingested bundles."""
    from datetime import datetime, timedelta, timezone

    days = args.get("days", 7)
    kb = args.get("kb")
    since = datetime.now(timezone.utc) - timedelta(days=days)
    bundles = repo.list_bundles_since(since, kb=kb)
    return json.dumps(bundles, ensure_ascii=False, default=str, indent=2)


def _handle_compare(repo: Any, args: dict) -> str:
    """Compare LLM responses within a bundle."""
    bundle_id = args["bundle_id"]
    bundle = repo.get_bundle_by_id(bundle_id)
    if bundle is None:
        return json.dumps({"error": f"Bundle not found: {bundle_id}"})

    responses = repo.get_responses_for_bundle(bundle_id)
    output = {
        "bundle_id": bundle_id,
        "question": bundle.get("question", ""),
        "consensus": bundle.get("consensus"),
        "divergence": bundle.get("divergence"),
        "has_synthesis": bundle.get("has_synthesis", False),
        "responses": responses,
    }
    return json.dumps(output, ensure_ascii=False, default=str, indent=2)


def _handle_suggest(repo: Any, analytics: Any, args: dict) -> str:
    """Suggest exploration directions based on knowledge gaps and thin topics."""
    topic = args.get("topic")
    kb = args.get("kb")

    gaps = analytics.knowledge_gaps(threshold=3, kb=kb)
    related_bundles = []
    if topic:
        related_bundles = repo.list_bundles_by_topic(topic, kb=kb)[:5]

    output = {
        "gaps": gaps[:10],
        "related_bundles": [
            {"bundle_id": b["bundle_id"], "question": b.get("question", "")}
            for b in related_bundles
        ],
    }
    return json.dumps(output, ensure_ascii=False, default=str, indent=2)


def main() -> None:
    """Run the MCP server via stdio."""
    mcp = create_mcp_server()
    mcp.run(transport="stdio")
