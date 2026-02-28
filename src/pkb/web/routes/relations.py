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


@router.get("/graph")
def relations_graph_page(request: Request):
    """Render interactive knowledge graph visualization."""
    templates = request.app.state.templates
    pkb: AppState = request.app.state.pkb
    count = pkb.repo.count_relations()
    return templates.TemplateResponse(request, "relations/graph.html", {
        "total": count,
    })


@router.get("/api/graph")
def relations_graph_json(request: Request, kb: str | None = None):
    """Return relation graph as JSON for D3.js visualization."""
    pkb: AppState = request.app.state.pkb

    relations = pkb.repo.list_all_relations(kb=kb)

    node_ids = set()
    edges = []
    for r in relations:
        node_ids.add(r["source_bundle_id"])
        node_ids.add(r["target_bundle_id"])
        edges.append({
            "source": r["source_bundle_id"],
            "target": r["target_bundle_id"],
            "type": r["relation_type"],
            "score": r["score"],
        })

    nodes = []
    for nid in sorted(node_ids):
        bundle = pkb.repo.get_bundle_by_id(nid)
        if bundle:
            nodes.append({
                "id": nid,
                "question": bundle.get("question", ""),
                "domains": bundle.get("domains", ""),
                "topics": bundle.get("topics", ""),
            })
        else:
            nodes.append({"id": nid, "question": "", "domains": "", "topics": ""})

    return {"nodes": nodes, "edges": edges}


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
