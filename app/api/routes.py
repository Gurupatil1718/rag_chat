"""
API routes for the Mitratech AI chatbot.

Endpoints
---------
GET  /health   — liveness / readiness check
POST /chat     — main chat endpoint
POST /ingest   — rebuild FAISS index (admin use)
"""

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
)
from app.services.gemini_service import gemini_service
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["System"],
    status_code=status.HTTP_200_OK,
)
def health_check() -> HealthResponse:
    """
    Returns API liveness and RAG readiness status.
    Use this to verify the service is running before sending chat requests.
    """
    from app.core.config import get_settings
    s = get_settings()
    return HealthResponse(
        status="ok",
        rag_ready=rag_service.is_ready,
        model=s.GEMINI_MODEL,
    )


# ─── Chat ─────────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with Mitra AI",
    tags=["Chat"],
    status_code=status.HTTP_200_OK,
)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a question and receive an AI-generated answer grounded in
    MitraTech's knowledge base.

    - Pass the **session_id** returned in a previous response to maintain
      conversation context (last N turns remembered).
    - Repeated identical questions are served from cache for speed.
    - Empty queries are rejected with HTTP 400.
    """

    # ✅ FIX: strip first, then validate
    query = request.query.strip() if request.query else ""
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    # ✅ RAG readiness check with proper 503 + helpful message
    if not rag_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The knowledge base is still loading. Please retry in a moment.",
        )

    try:
        answer, session_id, num_chunks = gemini_service.chat(
            query=query,
            session_id=request.session_id,
        )
    except Exception as exc:
        logger.error("Chat error: %s", exc, exc_info=True)   
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request.",
        ) from exc                                            

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        sources_used=num_chunks,
    )


# ─── Ingest (Admin) ───────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Rebuild knowledge-base index",
    tags=["Admin"],
    status_code=status.HTTP_202_ACCEPTED,              
)
def ingest(
    background_tasks: BackgroundTasks,
    scrape: bool = False,
) -> IngestResponse:
    """
    Trigger a rebuild of the FAISS index.

    - `scrape=false` (default) — re-index the local data file only (fast).
    - `scrape=true` — crawl the live website first, then index (slow, ~30 s).

    The rebuild runs in the background; the endpoint returns immediately.
    Use GET /health to check rag_ready status after triggering.
    """

    def _rebuild() -> None:
        try:
            n = rag_service.build_index(force_scrape=scrape)  
            logger.info("Re-index complete: %d chunks.", n)
        except Exception as exc:
            logger.error("Re-index failed: %s", exc, exc_info=True)

    background_tasks.add_task(_rebuild)

    return IngestResponse(
        status="accepted",                             
        chunks_indexed=len(rag_service._chunks),       
        message=(
            "Index rebuild started in background. "
            "Use GET /health to check rag_ready status."
        ),
    )