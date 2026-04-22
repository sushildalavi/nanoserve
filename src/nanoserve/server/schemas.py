"""openai-compatible request/response schemas (subset).

we don't try to match the full ChatCompletion spec — only the fields a
common client (curl, openai-python, langchain) actually sends and reads.
strict pydantic validation with extra='ignore' so caller-side optional
fields don't break us.
"""
from __future__ import annotations

import time
import uuid

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.0  # we only do greedy for now; non-zero ignored
    stream: bool = False


class Delta(BaseModel):
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: str | None = None


class StreamChunk(BaseModel):
    """one SSE data: line in a streaming response."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


class CompletionMessage(BaseModel):
    role: str = "assistant"
    content: str


class CompletionChoice(BaseModel):
    index: int = 0
    message: CompletionMessage
    finish_reason: str | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletion(BaseModel):
    """non-streaming chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


def new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def now_ts() -> int:
    return int(time.time())
