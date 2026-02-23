"""Web routes for analytics dashboard."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from pkb.analytics import AnalyticsEngine
from pkb.web.deps import AppState

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _engine(request: Request) -> AnalyticsEngine:
    pkb: AppState = request.app.state.pkb
    return AnalyticsEngine(repo=pkb.repo)


@router.get("")
def analytics_dashboard(request: Request):
    """Render the analytics dashboard page."""
    templates = request.app.state.templates
    engine = _engine(request)
    overview = engine.overview()
    return templates.TemplateResponse(request, "analytics/dashboard.html", {
        "overview": overview,
    })


@router.get("/api/domains")
def api_domains(request: Request, kb: str | None = None):
    """Domain distribution data."""
    engine = _engine(request)
    return JSONResponse(engine.domain_distribution(kb=kb))


@router.get("/api/topics")
def api_topics(request: Request, kb: str | None = None):
    """Topic heatmap data (top 20)."""
    engine = _engine(request)
    return JSONResponse(engine.topic_heatmap(kb=kb))


@router.get("/api/trend")
def api_trend(request: Request, kb: str | None = None, months: int = 6):
    """Monthly trend data."""
    engine = _engine(request)
    return JSONResponse(engine.temporal_trend(months=months, kb=kb))


@router.get("/api/platforms")
def api_platforms(request: Request, kb: str | None = None):
    """Platform distribution data."""
    engine = _engine(request)
    return JSONResponse(engine.platform_distribution(kb=kb))


@router.get("/api/gaps")
def api_gaps(request: Request, kb: str | None = None, threshold: int = 3):
    """Knowledge gaps data."""
    engine = _engine(request)
    return JSONResponse(engine.knowledge_gaps(threshold=threshold, kb=kb))
