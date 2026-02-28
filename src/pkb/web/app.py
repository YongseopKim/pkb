"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pkb.web.deps import AppState

if TYPE_CHECKING:
    pass

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def create_app(state: AppState) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        state.close()

    app = FastAPI(title="PKB Web UI", version="0.4.0", lifespan=lifespan)
    app.state.pkb = state

    # Static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Templates
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.templates = templates

    # Register routes
    from pkb.web.routes.analytics import router as analytics_router
    from pkb.web.routes.bundles import router as bundles_router
    from pkb.web.routes.chat import router as chat_router
    from pkb.web.routes.compare import router as compare_router
    from pkb.web.routes.digest import router as digest_router
    from pkb.web.routes.duplicates import router as duplicates_router
    from pkb.web.routes.relations import router as relations_router
    from pkb.web.routes.search import router as search_router
    from pkb.web.routes.settings import router as settings_router
    from pkb.web.routes.topics import router as topics_router

    app.include_router(bundles_router)
    app.include_router(search_router)
    app.include_router(topics_router)
    app.include_router(duplicates_router)
    app.include_router(relations_router)
    app.include_router(digest_router)
    app.include_router(analytics_router)
    app.include_router(settings_router)
    app.include_router(chat_router)
    app.include_router(compare_router)

    @app.get("/")
    def dashboard(request: Request):
        from datetime import datetime, timedelta, timezone

        from pkb.analytics import AnalyticsEngine

        pkb: AppState = request.app.state.pkb
        engine = AnalyticsEngine(repo=pkb.repo)

        bundle_ids = pkb.repo.list_all_bundle_ids()
        since = datetime.now(timezone.utc) - timedelta(days=7)
        recent = pkb.repo.list_bundles_since(since)
        overview = engine.overview()
        gaps = engine.knowledge_gaps(threshold=3)

        return templates.TemplateResponse(request, "dashboard.html", {
            "total_bundles": len(bundle_ids),
            "recent_bundles": recent[:10],
            "overview": overview,
            "gaps": gaps[:5],
        })

    return app
