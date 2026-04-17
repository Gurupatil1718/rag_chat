"""
Application entry point.

Run with:
    uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings, setup_logging
from app.services.rag_service import rag_service

# ── Logging ──────────────────────────────────────────────────────────────────
settings = get_settings()
setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: configure Gemini API and build (or load) the FAISS index.
    Shutdown: nothing special needed.
    """
    logger.info("=== Mitra AI Backend starting up ===")
    logger.info("Using Gemini model: %s", settings.GEMINI_MODEL)

    # Build / load the FAISS knowledge-base index
    try:
        n_chunks = rag_service.build_index(force_scrape=False)
        logger.info("RAG index ready with %d chunks.", n_chunks)
    except Exception as exc:
        # Non-fatal: service still starts; /chat returns 503 until ready
        logger.error("RAG index build failed: %s", exc, exc_info=True)

    yield  # ← application runs here

    logger.info("=== Mitra AI Backend shutting down ===")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Mitra AI Chatbot API",
    description=(
        "Production-ready RAG chatbot backend for MY MiTRAA Technology Private Limited. "
        "Powered by Google Gemini + FAISS."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Adjust origins in .env / config when you connect a real frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(router)


# ── Root ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"], summary="API root")
def root():
    return {
        "service": "Mitra AI Chatbot API",
        "company": "MY MiTRAA Technology Private Limited",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )
