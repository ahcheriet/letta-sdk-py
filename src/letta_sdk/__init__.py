from __future__ import annotations

from .client import LettaAgentClient
from .images import image_from_base64, image_from_file
from .query import QueryStream
from .session import LettaSession
from .transcript import TranscriptAccumulator
from .types import (
    AssistantMessage,
    Backend,
    CreateAgentOptions,
    CreateSessionOptions,
    ErrorMessage,
    MessageContentPart,
    PingMessage,
    QueryOptions,
    ReasoningMessage,
    ResultMessage,
    SDKMessage,
    SendMessage,
    ToolCallMessage,
    ToolResultMessage,
    UsageMessage,
    text_content,
)

__all__ = [
    "AssistantMessage",
    "Backend",
    "CreateAgentOptions",
    "CreateSessionOptions",
    "ErrorMessage",
    "LettaAgentClient",
    "LettaSession",
    "MessageContentPart",
    "PingMessage",
    "QueryOptions",
    "QueryStream",
    "ReasoningMessage",
    "ResultMessage",
    "SDKMessage",
    "SendMessage",
    "ToolCallMessage",
    "ToolResultMessage",
    "TranscriptAccumulator",
    "UsageMessage",
    "create_agent",
    "create_session",
    "image_from_base64",
    "image_from_file",
    "query",
    "resume_session",
    "text_content",
]


async def create_agent(
    options: CreateAgentOptions | None = None,
    *,
    client: LettaAgentClient | None = None,
    **client_kwargs: object,
) -> str:
    managed_client = client or LettaAgentClient(**client_kwargs)
    try:
        return await managed_client.create_agent(options)
    finally:
        if client is None:
            await managed_client.close()


async def create_session(
    agent_id: str,
    options: CreateSessionOptions | None = None,
    *,
    client: LettaAgentClient | None = None,
    **client_kwargs: object,
) -> LettaSession:
    managed_client = client or LettaAgentClient(**client_kwargs)
    session = await managed_client.create_session(agent_id, options)
    if client is None:
        session._owns_client = True
    return session


def resume_session(
    identifier: str,
    options: CreateSessionOptions | None = None,
    *,
    client: LettaAgentClient | None = None,
    **client_kwargs: object,
) -> LettaSession:
    managed_client = client or LettaAgentClient(**client_kwargs)
    session = managed_client.resume_session(identifier, options)
    if client is None:
        session._owns_client = True
    return session


def query(
    prompt: SendMessage,
    options: QueryOptions | None = None,
    *,
    client: LettaAgentClient | None = None,
    **client_kwargs: object,
) -> QueryStream:
    managed_client = client or LettaAgentClient(**client_kwargs)
    stream = managed_client.query(prompt, options)
    if client is None:
        stream.on_close = managed_client.close
    return stream
