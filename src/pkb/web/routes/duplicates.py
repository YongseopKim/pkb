"""Duplicate management routes."""

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

router = APIRouter(prefix="/duplicates", tags=["duplicates"])


@router.get("")
def duplicates_list(request: Request, status: str = "pending"):
    """List duplicate pairs."""
    pkb = request.app.state.pkb
    templates = request.app.state.templates

    filter_status = None if status == "all" else status
    pairs = pkb.repo.list_duplicate_pairs(status=filter_status)

    return templates.TemplateResponse(request, "duplicates/list.html", {
        "pairs": pairs,
        "status": status,
    })


@router.post("/{pair_id}/dismiss")
def duplicate_dismiss(request: Request, pair_id: int):
    """Dismiss a duplicate pair."""
    pkb = request.app.state.pkb
    pkb.repo.update_duplicate_status(pair_id, "dismissed")
    return RedirectResponse(url="/duplicates", status_code=303)


@router.post("/{pair_id}/confirm")
def duplicate_confirm(request: Request, pair_id: int):
    """Confirm a duplicate pair."""
    pkb = request.app.state.pkb
    pkb.repo.update_duplicate_status(pair_id, "confirmed")
    return RedirectResponse(url="/duplicates", status_code=303)
