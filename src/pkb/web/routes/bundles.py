"""Bundle routes — list, detail, delete."""

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

router = APIRouter(prefix="/bundles", tags=["bundles"])


@router.get("")
def bundle_list(
    request: Request,
    kb: str | None = None,
    page: int = 1,
    per_page: int = 20,
):
    """List bundles with pagination."""
    pkb = request.app.state.pkb
    templates = request.app.state.templates
    offset = (page - 1) * per_page

    all_ids = pkb.repo.list_all_bundle_ids(kb=kb)
    total = len(all_ids)
    page_ids = all_ids[offset:offset + per_page]

    bundles = []
    for bid in page_ids:
        bundle = pkb.repo.get_bundle_by_id(bid)
        if bundle:
            bundles.append(bundle)

    total_pages = (total + per_page - 1) // per_page if total > 0 else 1

    return templates.TemplateResponse(request, "bundles/list.html", {
        "bundles": bundles,
        "page": page,
        "total_pages": total_pages,
        "total": total,
        "kb": kb,
    })


@router.get("/{bundle_id}")
def bundle_detail(request: Request, bundle_id: str):
    """Show bundle detail."""
    pkb = request.app.state.pkb
    templates = request.app.state.templates

    bundle = pkb.repo.get_bundle_by_id(bundle_id)
    if bundle is None:
        return templates.TemplateResponse(request, "error.html", {
            "message": f"Bundle '{bundle_id}' not found.",
        }, status_code=404)

    responses = pkb.repo.get_responses_for_bundle(bundle_id)

    return templates.TemplateResponse(request, "bundles/detail.html", {
        "bundle": bundle,
        "responses": responses,
    })


@router.post("/{bundle_id}/delete")
def bundle_delete(request: Request, bundle_id: str):
    """Delete a bundle."""
    pkb = request.app.state.pkb
    pkb.repo.delete_bundle(bundle_id)
    return RedirectResponse(url="/bundles", status_code=303)
