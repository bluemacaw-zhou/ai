"""HTTP endpoints for natural-language chat requests."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from graph.main_graph import MainGraph

router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(description="A natural-language question.", min_length=1)


class ChatResponse(BaseModel):
    answer: str


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be blank")
    graph: MainGraph = request.app.state.main_graph
    return ChatResponse(answer=await graph.run(question))
