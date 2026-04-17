"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class ChatRequest(BaseModel):
    """Incoming chat message from the user."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's question or message",
        examples=["What services does Mitratech offer?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation memory. "
                    "Leave empty to start a new session.",
    )


class ChatResponse(BaseModel):
    """Answer returned to the user."""

    answer: str = Field(..., description="AI-generated answer")
    session_id: str = Field(..., description="Session ID for follow-up messages")
    sources_used: int = Field(
        ..., description="Number of knowledge-base chunks used to build the answer"
    )


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: str = "ok"
    rag_ready: bool = Field(..., description="Whether the FAISS index is loaded")
    model: str = Field(..., description="Gemini model in use")


class IngestResponse(BaseModel):
    """Returned after re-ingesting website content."""

    status: str
    chunks_indexed: int
    message: str
