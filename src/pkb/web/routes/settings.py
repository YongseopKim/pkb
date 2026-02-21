"""Settings route (read-only)."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("")
def settings_page(request: Request):
    """Show current configuration."""
    templates = request.app.state.templates

    from pkb.config import get_pkb_home
    from pkb.constants import CONFIG_FILENAME

    pkb_home = get_pkb_home()
    config_path = pkb_home / CONFIG_FILENAME

    config_text = ""
    if config_path.exists():
        config_text = config_path.read_text(encoding="utf-8")

    return templates.TemplateResponse(request, "settings.html", {
        "config_text": config_text,
    })
