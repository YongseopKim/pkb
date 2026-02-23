"""Web routes for knowledge digest."""

from fastapi import APIRouter, Form, Request

from pkb.web.deps import AppState

router = APIRouter(prefix="/digest", tags=["digest"])


@router.get("")
def digest_form(request: Request):
    """Show digest generation form."""
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "digest/form.html", {})


@router.post("")
def digest_generate(
    request: Request,
    topic: str = Form(default=None),
    domain: str = Form(default=None),
    kb: str = Form(default=None),
):
    """Generate a digest and show results."""
    from pkb.config import build_llm_router, get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.digest import DigestEngine

    pkb: AppState = request.app.state.pkb
    templates = request.app.state.templates

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)
    llm_router = build_llm_router(config)

    engine = DigestEngine(
        repo=pkb.repo,
        search_engine=pkb.search_engine,
        router=llm_router,
        config=config.digest,
    )

    if topic:
        result = engine.digest_topic(topic, kb=kb or None)
    elif domain:
        result = engine.digest_domain(domain, kb=kb or None)
    else:
        return templates.TemplateResponse(request, "digest/form.html", {
            "error": "토픽 또는 도메인을 지정하세요.",
        })

    return templates.TemplateResponse(request, "digest/result.html", {
        "result": result,
    })
