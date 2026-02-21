"""Search routes."""

from fastapi import APIRouter, Request

from pkb.search.models import SearchMode, SearchQuery

router = APIRouter(prefix="/search", tags=["search"])


@router.get("")
def search_page(
    request: Request,
    q: str = "",
    mode: str = "hybrid",
    domain: str = "",
    topic: str = "",
    kb: str = "",
    limit: int = 10,
):
    """Search bundles."""
    pkb = request.app.state.pkb
    templates = request.app.state.templates

    results = []
    if q:
        query = SearchQuery(
            query=q,
            mode=SearchMode(mode),
            domains=[domain] if domain else [],
            topics=[topic] if topic else [],
            kb=kb or None,
            limit=limit,
        )
        results = pkb.search_engine.search(query)

    return templates.TemplateResponse(request, "search.html", {
        "query": q,
        "mode": mode,
        "domain": domain,
        "topic": topic,
        "kb": kb,
        "results": results,
    })
