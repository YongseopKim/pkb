"""Web routes for bundle relations (knowledge graph)."""

from fastapi import APIRouter, Request

from pkb.web.deps import AppState

router = APIRouter(prefix="/relations", tags=["relations"])


@router.get("")
def relations_list(request: Request, relation_type: str = "all"):
    """List all bundle relations."""
    pkb: AppState = request.app.state.pkb
    templates = request.app.state.templates

    filter_type = None if relation_type == "all" else relation_type
    relations = pkb.repo.list_all_relations(relation_type=filter_type)
    count = pkb.repo.count_relations()

    return templates.TemplateResponse(request, "relations/list.html", {
        "relations": relations,
        "total": count,
        "current_type": relation_type,
    })


@router.get("/api/graph")
def relations_graph_json(request: Request, kb: str | None = None):
    """Return relation graph as JSON for D3.js visualization."""
    pkb: AppState = request.app.state.pkb

    relations = pkb.repo.list_all_relations(kb=kb)

    nodes = set()
    edges = []
    for r in relations:
        nodes.add(r["source_bundle_id"])
        nodes.add(r["target_bundle_id"])
        edges.append({
            "source": r["source_bundle_id"],
            "target": r["target_bundle_id"],
            "type": r["relation_type"],
            "score": r["score"],
        })

    return {
        "nodes": [{"id": n} for n in sorted(nodes)],
        "edges": edges,
    }


@router.get("/{bundle_id}")
def relations_detail(request: Request, bundle_id: str):
    """Show relations for a specific bundle."""
    pkb: AppState = request.app.state.pkb
    templates = request.app.state.templates

    relations = pkb.repo.list_relations(bundle_id)
    bundle = pkb.repo.get_bundle_by_id(bundle_id)

    return templates.TemplateResponse(request, "relations/detail.html", {
        "bundle_id": bundle_id,
        "bundle": bundle,
        "relations": relations,
    })
