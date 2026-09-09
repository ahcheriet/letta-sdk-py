"""Letta Agent SDK (Python) — a pythonic SDK for Letta agents.

Built on the Letta Code **app server** websocket protocol (agent SDK v2),
the same protocol the TypeScript ``@letta-ai/letta-agent-sdk`` speaks.
"""

from __future__ import annotations

from typing import Any

from .app_server import (
    AppServerClosedError,
    AppServerConnection,
    AppServerError,
    AppServerRequestError,
    AppServerTimeoutError,
)
from .client import LettaAgentClient
from .images import image_from_base64, image_from_file, image_from_url
from .mcp import McpToolBridge, connect_mcp_servers, expand_mcp_tool_wildcards
from .query import QueryStream
from .session import LettaSession
from .skills import (
    AgentSkill,
    load_skill_directory,
    parse_skill_markdown,
    resolve_skill_items,
    skill_memory_blocks,
    skills_have_support_files,
)
from .stream_events import StreamTextDelta, extract_stream_text_delta
from .tool_helpers import (
    json_result,
    read_boolean_param,
    read_number_param,
    read_string_array_param,
    read_string_param,
)
from .transcript import (
    TranscriptAccumulator,
    TranscriptRow,
    TranscriptToolResult,
)
from .types import (
    AssistantMessage,
    Backend,
    CanUseToolDecision,
    CLIENT_TOOLSET_BASES,
    CreateAgentOptions,
    CreateSessionOptions,
    DEFAULT_APP_SERVER_URL,
    DREAMING_BEHAVIORS,
    DREAMING_TRIGGERS,
    DreamingOptions,
    ErrorMessage,
    LoopStatusMessage,
    MessageContentPart,
    PingMessage,
    QueryOptions,
    ReasoningMessage,
    ResultMessage,
    RetryMessage,
    SDKInitMessage,
    SDKMessage,
    SendMessage,
    StreamEventMessage,
    ToolCallMessage,
    ToolResult,
    ToolResultMessage,
    ToolsetConfig,
    ToolSpec,
    UsageMessage,
    text_content,
)

__all__ = [
    "AgentSkill",
    "AppServerClosedError",
    "AppServerConnection",
    "AppServerError",
    "AppServerRequestError",
    "AppServerTimeoutError",
    "AssistantMessage",
    "Backend",
    "CanUseToolDecision",
    "CLIENT_TOOLSET_BASES",
    "CreateAgentOptions",
    "CreateSessionOptions",
    "DEFAULT_APP_SERVER_URL",
    "DREAMING_BEHAVIORS",
    "DREAMING_TRIGGERS",
    "DreamingOptions",
    "ErrorMessage",
    "LettaAgentClient",
    "LettaSession",
    "McpToolBridge",
    "LoopStatusMessage",
    "MessageContentPart",
    "PingMessage",
    "QueryOptions",
    "QueryStream",
    "ReasoningMessage",
    "ResultMessage",
    "RetryMessage",
    "SDKInitMessage",
    "SDKMessage",
    "SendMessage",
    "StreamEventMessage",
    "StreamTextDelta",
    "ToolCallMessage",
    "ToolResult",
    "ToolResultMessage",
    "ToolsetConfig",
    "ToolSpec",
    "TranscriptAccumulator",
    "TranscriptRow",
    "TranscriptToolResult",
    "UsageMessage",
    "connect_mcp_servers",
    "create_agent",
    "create_session",
    "expand_mcp_tool_wildcards",
    "extract_stream_text_delta",
    "image_from_base64",
    "image_from_file",
    "image_from_url",
    "json_result",
    "load_skill_directory",
    "parse_skill_markdown",
    "read_boolean_param",
    "read_number_param",
    "read_string_array_param",
    "read_string_param",
    "resolve_skill_items",
    "skill_memory_blocks",
    "skills_have_support_files",
    "prompt",
    "query",
    "resume_session",
    "text_content",
]


def _managed(
    client: LettaAgentClient | None,
    **client_kwargs: Any,
) -> tuple[LettaAgentClient, bool]:
    if client is not None:
        return client, False
    return LettaAgentClient(**client_kwargs), True


async def create_agent(
    options: CreateAgentOptions | None = None,
    *,
    client: LettaAgentClient | None = None,
    **client_kwargs: Any,
) -> str:
    """Create an agent and return its id (closes a self-managed client)."""
    managed_client, managed = _managed(client, **client_kwargs)
    try:
        return await managed_client.create_agent(options)
    finally:
        if managed:
            await managed_client.close()


def create_session(
    agent_id: str,
    options: CreateSessionOptions | None = None,
    *,
    client: LettaAgentClient | None = None,
    **client_kwargs: Any,
) -> LettaSession:
    """Open a session on a new conversation for ``agent_id``.

    The returned session is lazy: the runtime starts on first use. When the
    client is self-managed, closing the session closes the client too.
    """
    managed_client, managed = _managed(client, **client_kwargs)
    session = managed_client.create_session(agent_id, options)
    if managed:
        session.attach_owner(managed_client, owns=True)
    return session


def resume_session(
    identifier: str,
    options: CreateSessionOptions | None = None,
    *,
    client: LettaAgentClient | None = None,
    **client_kwargs: Any,
) -> LettaSession:
    """Resume an agent (bare id, default conversation) or a conversation
    (``conv-...`` id)."""
    managed_client, managed = _managed(client, **client_kwargs)
    session = managed_client.resume_session(identifier, options)
    if managed:
        session.attach_owner(managed_client, owns=True)
    return session


def query(
    prompt: SendMessage,
    options: QueryOptions | None = None,
    *,
    client: LettaAgentClient | None = None,
    **client_kwargs: Any,
) -> QueryStream:
    """Run one agent-free query in a new ephemeral conversation."""
    managed_client, managed = _managed(client, **client_kwargs)
    stream = managed_client.query(prompt, options)
    if managed:
        stream.on_close = managed_client.close
    return stream


async def prompt(
    agent_id: str,
    message: SendMessage,
    options: CreateSessionOptions | None = None,
    *,
    client: LettaAgentClient | None = None,
    **client_kwargs: Any,
) -> ResultMessage:
    """One-shot turn against an agent; returns the terminal result."""
    managed_client, managed = _managed(client, **client_kwargs)
    try:
        return await managed_client.prompt(agent_id, message, options)
    finally:
        if managed:
            await managed_client.close()
