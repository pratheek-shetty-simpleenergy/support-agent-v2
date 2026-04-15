from __future__ import annotations

from functools import lru_cache

from support_agent.config import Settings, get_settings
from support_agent.llm.client import LlamaClient
from support_agent.schemas.session import (
    SupportAiSessionCreatedResponse,
    SupportAiSessionCreateRequest,
    SupportAiSessionMessageRequest,
    SupportAiSessionResponse,
    SupportAiSessionStatusResponse,
)
from support_agent.services.bootstrap import build_application
from support_agent.services.session_store import RedisSessionStore
from support_agent.services.support_ai_sessions import SupportAiSessionService


@lru_cache
def get_session_service(settings: Settings | None = None) -> SupportAiSessionService:
    app_settings = settings or get_settings()
    graph = build_application(app_settings)
    llm_client = LlamaClient(app_settings)
    session_store = RedisSessionStore(app_settings)
    return SupportAiSessionService(graph=graph, llm_client=llm_client, session_store=session_store)


def create_api_app():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
    except ModuleNotFoundError as exc:
        raise RuntimeError("FastAPI is not installed. Install project dependencies before running the API.") from exc

    app = FastAPI(title="Support Agent API", version="0.1.0")
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    session_service = get_session_service()

    @app.post("/api/support-ai/sessions", response_model=SupportAiSessionCreatedResponse)
    def create_support_ai_session(request: SupportAiSessionCreateRequest) -> SupportAiSessionCreatedResponse:
        try:
            return session_service.create_session(request)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/support-ai/sessions/{session_id}/messages", response_model=SupportAiSessionCreatedResponse)
    def send_support_ai_message(session_id: str, request: SupportAiSessionMessageRequest) -> SupportAiSessionCreatedResponse:
        try:
            return session_service.send_message(session_id, request)
        except Exception as exc:
            detail = str(exc)
            status_code = 404 if "not found" in detail.lower() else 500
            raise HTTPException(status_code=status_code, detail=detail) from exc

    @app.get("/api/support-ai/sessions/{session_id}", response_model=SupportAiSessionStatusResponse)
    def get_support_ai_session(session_id: str) -> SupportAiSessionStatusResponse:
        try:
            return session_service.get_session_status(session_id)
        except Exception as exc:
            detail = str(exc)
            status_code = 404 if "not found" in detail.lower() else 500
            raise HTTPException(status_code=status_code, detail=detail) from exc

    @app.get("/api/support-ai/sessions/{session_id}/stream")
    def stream_support_ai_session(session_id: str):
        if session_service.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} was not found.")
        return StreamingResponse(
            session_service.stream_session(session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    return app


app = create_api_app()
