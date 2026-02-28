"""Web routes for LLM response comparison view."""

from collections import Counter

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from pkb.web.deps import AppState

router = APIRouter(prefix="/compare", tags=["compare"])


@router.get("/api/{bundle_id}")
def compare_api(request: Request, bundle_id: str):
    """Return comparison data for a bundle's multi-platform responses."""
    pkb: AppState = request.app.state.pkb
    bundle = pkb.repo.get_bundle_by_id(bundle_id)
    if not bundle:
        return JSONResponse({"error": "Bundle not found"}, status_code=404)

    responses = pkb.repo.get_responses_for_bundle(bundle_id)
    platforms = []
    all_claims = []
    for r in responses:
        claims = r.get("key_claims") or []
        platforms.append({
            "platform": r["platform"],
            "model": r.get("model", ""),
            "turn_count": r.get("turn_count", 0),
            "stance": r.get("stance", ""),
            "key_claims": claims,
        })
        all_claims.extend(claims)

    claim_counts = Counter(all_claims)
    consensus = [c for c, n in claim_counts.items() if n >= 2]

    return {
        "bundle_id": bundle_id,
        "question": bundle.get("question", ""),
        "platforms": platforms,
        "consensus": consensus,
    }


@router.get("/{bundle_id}")
def compare_page(request: Request, bundle_id: str):
    """Render side-by-side comparison view."""
    pkb: AppState = request.app.state.pkb
    templates = request.app.state.templates
    bundle = pkb.repo.get_bundle_by_id(bundle_id)
    if not bundle:
        return templates.TemplateResponse(request, "error.html", {
            "message": f"Bundle {bundle_id} not found",
        }, status_code=404)

    responses = pkb.repo.get_responses_for_bundle(bundle_id)
    return templates.TemplateResponse(request, "compare/detail.html", {
        "bundle": bundle,
        "responses": responses,
    })
