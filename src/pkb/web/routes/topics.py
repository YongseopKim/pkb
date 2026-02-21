"""Topic management routes."""

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

router = APIRouter(prefix="/topics", tags=["topics"])


@router.get("")
def topics_list(request: Request, status: str = "all"):
    """List topics."""
    templates = request.app.state.templates

    from pkb.config import get_pkb_home
    from pkb.vocab.manager import TopicManager

    pkb_home = get_pkb_home()
    topics_path = pkb_home / "vocab" / "topics.yaml"
    if not topics_path.exists():
        return templates.TemplateResponse(request, "topics/list.html", {
            "topics": [],
            "status": status,
        })

    mgr = TopicManager(topics_path)
    filter_status = None if status == "all" else status
    topic_list = mgr.list_topics(status=filter_status)

    return templates.TemplateResponse(request, "topics/list.html", {
        "topics": sorted(topic_list, key=lambda t: t.canonical),
        "status": status,
    })


@router.post("/{name}/approve")
def topic_approve(request: Request, name: str):
    """Approve a pending topic."""
    from pkb.config import get_pkb_home
    from pkb.vocab.manager import TopicManager
    from pkb.vocab.syncer import TopicSyncer

    pkb_home = get_pkb_home()
    topics_path = pkb_home / "vocab" / "topics.yaml"
    mgr = TopicManager(topics_path)
    mgr.approve(name)

    pkb = request.app.state.pkb
    syncer = TopicSyncer(repo=pkb.repo)
    syncer.sync_approve(name)

    return RedirectResponse(url="/topics", status_code=303)


@router.post("/{name}/reject")
def topic_reject(request: Request, name: str):
    """Reject a topic."""
    from pkb.config import get_pkb_home
    from pkb.vocab.manager import TopicManager
    from pkb.vocab.syncer import TopicSyncer

    pkb_home = get_pkb_home()
    topics_path = pkb_home / "vocab" / "topics.yaml"
    mgr = TopicManager(topics_path)
    mgr.reject(name)

    pkb = request.app.state.pkb
    syncer = TopicSyncer(repo=pkb.repo)
    syncer.sync_reject(name)

    return RedirectResponse(url="/topics", status_code=303)


@router.post("/{name}/merge")
def topic_merge(request: Request, name: str, into: str = Form(...)):
    """Merge a topic into another."""
    from pkb.config import get_pkb_home
    from pkb.vocab.manager import TopicManager
    from pkb.vocab.syncer import TopicSyncer

    pkb_home = get_pkb_home()
    topics_path = pkb_home / "vocab" / "topics.yaml"
    mgr = TopicManager(topics_path)
    mgr.merge(name, into=into)

    pkb = request.app.state.pkb
    syncer = TopicSyncer(repo=pkb.repo)
    syncer.sync_merge(name, into=into)

    return RedirectResponse(url="/topics", status_code=303)
