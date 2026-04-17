"""
RAG Service — Retrieval-Augmented Generation over MitraTech knowledge base.

Uses the new google-genai SDK for embeddings.

Responsibilities
-----------------
1. Build / load a FAISS index from the knowledge base text.
2. Embed user queries and retrieve the top-K most relevant chunks.
3. Expose retrieve() for use by the Gemini service.
"""

import logging
import pickle
import threading
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from google import genai
from google.genai import types

from app.core.config import get_settings
from app.services.scraper import load_local_data, scrape_website

logger = logging.getLogger(__name__)
settings = get_settings()

# ─── Text chunking helper ────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks. Optimized for readability."""
    lines: List[str] = [line.strip() for line in text.split("\n") if line.strip()]
    
    chunks: List[str] = []
    current = ""

    for line in lines:
        if len(current) + len(line) + 1 <= chunk_size:
            current = f"{current}\n{line}" if current else line
        else:
            if current:
                chunks.append(current.strip())
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = f"{overlap_text}\n{line}" if overlap_text else line

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 30]


# ─── RAGService ──────────────────────────────────────────────────────────────

class RAGService:
    """Singleton wrapper around the FAISS index + chunk store."""

    def __init__(self) -> None:
        self._index: Optional[faiss.Index] = None
        self._chunks: List[str] = []
        self._ready: bool = False
        self._lock = threading.Lock()
        # Initialize client once to reuse connections
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Embedding internal methods ───────────────────────────────────────────

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Calls Gemini embedding API. Returns (N, D) float32 array."""
        vectors = []
        for i, text in enumerate(texts):
            try:
                response = self._client.models.embed_content(
                    model=settings.GEMINI_EMBEDDING_MODEL,
                    contents=text,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                vectors.append(response.embeddings[0].values)
            except Exception as e:
                logger.error("Embedding failed for chunk %d: %s", i, e)
                raise RuntimeError(f"Failed to embed chunk {i}") from e
        return np.array(vectors, dtype=np.float32)

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string for retrieval."""
        response = self._client.models.embed_content(
            model=settings.GEMINI_EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return np.array([response.embeddings[0].values], dtype=np.float32)

    # ── Public API ──────────────────────────────────────────────────────────

    def build_index(self, force_scrape: bool = False) -> int:
        """Build or reload the FAISS index with thread safety."""
        with self._lock:
            self._ready = False  # Reset state during build
            index_path = Path(settings.FAISS_INDEX_PATH)
            chunks_path = index_path.parent / "chunks.pkl"

            if not force_scrape and self._try_load_from_disk(index_path, chunks_path):
                self._ready = True
                return len(self._chunks)

            try:
                # 1. Gather text
                text_parts: List[str] = []
                local_text = load_local_data(settings.DATA_FILE_PATH)
                if local_text:
                    text_parts.append(local_text)

                if force_scrape:
                    scraped = scrape_website(settings.WEBSITE_URL)
                    if scraped:
                        text_parts.append(scraped)

                if not text_parts:
                    raise RuntimeError("No content available for indexing.")

                full_text = "\n\n".join(text_parts)

                # 2. Chunk
                self._chunks = _chunk_text(
                    full_text,
                    chunk_size=settings.CHUNK_SIZE,
                    overlap=settings.CHUNK_OVERLAP,
                )

                # 3. Embed & Index
                embeddings = self._embed_texts(self._chunks)
                dim = embeddings.shape[1]
                
                new_index = faiss.IndexFlatIP(dim)
                faiss.normalize_L2(embeddings)
                new_index.add(embeddings)

                # 4. Update State
                self._index = new_index
                self._save_to_disk(index_path, chunks_path)
                self._ready = True
                
                logger.info("FAISS index built successfully: %d chunks.", len(self._chunks))
                return len(self._chunks)

            except Exception as e:
                logger.error("Failed to build index: %s", e, exc_info=True)
                raise

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Return the most relevant chunks for *query*."""
        if not self._ready or self._index is None:
            logger.warning("Retrieval attempted but RAG is not ready.")
            return []

        k = top_k or settings.TOP_K_RESULTS

        try:
            q_vec = self._embed_query(query)
            faiss.normalize_L2(q_vec)

            scores, indices = self._index.search(q_vec, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score < 0.25:
                    continue

                chunk = self._chunks[idx]
                if len(chunk.strip()) >= 50:
                    results.append(chunk)

            if not results:
                logger.info("No relevant chunks found for query (Top score: %s)", 
                            scores[0][0] if len(scores[0]) > 0 else "N/A")
            else:
                logger.info("Retrieved %d chunks", len(results))

            return results

        except Exception as e:
            logger.error("Retrieval error: %s", e, exc_info=True)
            return []

    # ── Persistence helpers ───────────────────────────────────────────────

    def _save_to_disk(self, index_path: Path, chunks_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        with open(chunks_path, "wb") as fh:
            pickle.dump(self._chunks, fh)

    def _try_load_from_disk(self, index_path: Path, chunks_path: Path) -> bool:
        if not index_path.exists() or not chunks_path.exists():
            return False
        try:
            self._index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as fh:
                self._chunks = pickle.load(fh)
            logger.info("Loaded index from disk (%d chunks).", len(self._chunks))
            return True
        except Exception as exc:
            logger.warning("Disk load failed: %s", exc)
            return False


# Module singleton
rag_service = RAGService()