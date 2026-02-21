"""Chat route — RAG chatbot web UI."""

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("")
def chat_page(request: Request):
    """Chat page."""
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "chat.html", {
        "messages": [],
    })


@router.post("/send")
def chat_send(request: Request, message: str = Form(...)):
    """Handle chat message (htmx partial response)."""
    pkb = request.app.state.pkb

    # Lazy init chat engine
    chat_engine = getattr(pkb, "chat_engine", None)
    if chat_engine is None:
        return HTMLResponse(
            '<div class="chat-msg assistant">'
            "<strong>assistant:</strong> Chat engine not configured. "
            "Start the server with LLM routing enabled.</div>"
        )

    session = getattr(pkb, "chat_session", None)
    if session is None:
        from pkb.chat.models import ChatSession
        pkb.chat_session = ChatSession()
        session = pkb.chat_session

    try:
        response = chat_engine.ask(message, session=session)
        user_html = (
            f'<div class="chat-msg user">'
            f"<strong>user:</strong> {message}</div>"
        )
        assistant_html = (
            f'<div class="chat-msg assistant">'
            f"<strong>assistant:</strong> {response.content}</div>"
        )
        return HTMLResponse(user_html + assistant_html)
    except Exception as e:
        return HTMLResponse(
            f'<div class="chat-msg user"><strong>user:</strong> {message}</div>'
            f'<div class="chat-msg assistant"><strong>assistant:</strong> Error: {e}</div>'
        )
