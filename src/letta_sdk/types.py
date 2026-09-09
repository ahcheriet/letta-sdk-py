"""Public types for the Letta Agent SDK (Python).

These types mirror the TypeScript ``@letta-ai/letta-agent-sdk`` (v2) public
surface, projected onto the Letta Code **app server** websocket protocol.

The SDK speaks the app-server JSON-over-WebSocket protocol used by Letta Code:

* ``runtime_start`` / ``runtime_start_response``  -- open a runtime
* ``input`` (``create_message``)                  -- send a user turn
* ``stream_delta``                                -- streamed turn events
* ``conversation_messages_list`` / ``list_models`` / ``agent_*`` / ...
* ``ack``                                         -- sequence acknowledgement

Every wire message that carries a ``seq`` field must be acknowledged; the
transport layer in :mod:`letta_sdk.app_server` does that automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, TypedDict, TypeAlias


# ═══════════════════════════════════════════════════════════════
# BACKEND / TRANSPORT
# ═══════════════════════════════════════════════════════════════


class Backend(StrEnum):
    """How the SDK reaches the Letta Code harness.

    ``local``  -- connect to a local Letta Code app server over loopback
                  websockets (default endpoint ``ws://127.0.0.1:4500/ws``).
    ``remote`` -- connect to a user-managed app server websocket endpoint.
    ``app-server`` -- explicit websocket app-server transport (default).
    ``cloud``  -- accepted for forward compatibility; not implemented.
    """

    LOCAL = "local"
    REMOTE = "remote"
    APP_SERVER = "app-server"
    CLOUD = "cloud"


#: Default loopback websocket endpoint for a locally running app server.
DEFAULT_APP_SERVER_URL = "ws://127.0.0.1:4500/ws"


# ═══════════════════════════════════════════════════════════════
# MESSAGE CONTENT (multimodal)
# ═══════════════════════════════════════════════════════════════


class TextContentPart(TypedDict):
    type: Literal["text"]
    text: str


class ImageSource(TypedDict, total=False):
    type: Literal["base64", "url"]
    media_type: str
    data: str
    url: str


class ImageContentPart(TypedDict):
    type: Literal["image"]
    source: ImageSource


MessageContentPart: TypeAlias = TextContentPart | ImageContentPart
#: What :meth:`LettaSession.send` accepts: a plain string or a content list.
SendMessage: TypeAlias = str | list[MessageContentPart]


def text_content(text: str) -> TextContentPart:
    """Build a text content part (multimodal message helper)."""
    return {"type": "text", "text": text}


ALLOWED_IMAGE_SUFFIXES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def infer_media_type(path: str | Path, fallback: str = "image/png") -> str:
    suffix = Path(path).suffix.lower()
    return ALLOWED_IMAGE_SUFFIXES.get(suffix, fallback)


# ═══════════════════════════════════════════════════════════════
# SDK MESSAGE TYPES (stream projections of the wire protocol)
# ═══════════════════════════════════════════════════════════════


@dataclass(slots=True, kw_only=True)
class SDKMessage:
    """Base class for all SDK stream messages."""

    type: str = "unknown"
    #: The raw wire payload (a ``dict``) for this event, when available.
    raw: Any = None


@dataclass(slots=True, kw_only=True)
class SDKInitMessage(SDKMessage):
    agent_id: str | None = None
    session_id: str | None = None
    conversation_id: str | None = None
    model: str = ""
    tools: list[str] | None = None
    memfs_enabled: bool | None = None


@dataclass(slots=True, kw_only=True)
class AssistantMessage(SDKMessage):
    content: str = ""
    uuid: str | None = None
    otid: str | None = None
    seq_id: int | None = None
    run_id: str | None = None


@dataclass(slots=True, kw_only=True)
class ReasoningMessage(SDKMessage):
    content: str = ""
    uuid: str | None = None
    otid: str | None = None
    seq_id: int | None = None
    run_id: str | None = None


@dataclass(slots=True, kw_only=True)
class ToolCallMessage(SDKMessage):
    tool_call_id: str = ""
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str | None = None
    uuid: str | None = None
    run_id: str | None = None


@dataclass(slots=True, kw_only=True)
class ToolResultMessage(SDKMessage):
    tool_call_id: str = ""
    content: str = ""
    is_error: bool = False
    uuid: str | None = None
    run_id: str | None = None


@dataclass(slots=True, kw_only=True)
class UsageMessage(SDKMessage):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    step_count: int | None = None
    run_ids: list[str] = field(default_factory=list)


@dataclass(slots=True, kw_only=True)
class PingMessage(SDKMessage):
    uuid: str | None = None


@dataclass(slots=True, kw_only=True)
class RetryMessage(SDKMessage):
    reason: str = "error"
    attempt: int = 0
    max_attempts: int = 0
    delay_ms: int = 0
    run_id: str | None = None


@dataclass(slots=True, kw_only=True)
class LoopStatusMessage(SDKMessage):
    status: str = ""
    active_run_ids: list[str] = field(default_factory=list)


@dataclass(slots=True, kw_only=True)
class QueueUpdateMessage(SDKMessage):
    queue: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True, kw_only=True)
class StreamEventMessage(SDKMessage):
    event: dict[str, Any] = field(default_factory=dict)
    uuid: str | None = None


@dataclass(slots=True, kw_only=True)
class ErrorMessage(SDKMessage):
    message: str = ""
    error_code: str | None = None
    stop_reason: str | None = None
    error_detail: str | None = None
    recoverable: bool = False
    run_id: str | None = None


@dataclass(slots=True, kw_only=True)
class ResultMessage(SDKMessage):
    success: bool = False
    result: str | None = None
    error: str | None = None
    error_code: str | None = None
    stop_reason: str | None = None
    duration_ms: float | None = None
    conversation_id: str | None = None
    run_ids: list[str] | None = None
    error_detail: str | None = None
    recoverable: bool | None = None


@dataclass(slots=True, kw_only=True)
class UnknownMessage(SDKMessage):
    message_type: str | None = None


#: Union of all SDK message types.
SDKMessageUnion = (
    SDKInitMessage
    | AssistantMessage
    | ReasoningMessage
    | ToolCallMessage
    | ToolResultMessage
    | UsageMessage
    | PingMessage
    | RetryMessage
    | LoopStatusMessage
    | QueueUpdateMessage
    | StreamEventMessage
    | ErrorMessage
    | ResultMessage
    | UnknownMessage
)


# ═══════════════════════════════════════════════════════════════
# OPTIONS
# ═══════════════════════════════════════════════════════════════


@dataclass(slots=True)
class CreateAgentOptions:
    """Options for :meth:`LettaAgentClient.create_agent`."""

    name: str | None = None
    description: str | None = None
    #: Model handle, e.g. ``"openai/gpt-5.5"`` or ``"openai-compatible/Qwen3.8-27B"``.
    model: str | None = None
    #: System prompt (a full string, or a preset name handled server-side).
    system_prompt: str | None = None
    #: Embedding model handle.
    embedding: str | None = None
    #: Legacy memory blocks: list of ``{label, value, description?}``.
    memory_blocks: list[dict[str, Any]] = field(default_factory=list)
    #: Persona / human convenience fields (mapped to memory blocks).
    persona: str | None = None
    human: str | None = None
    #: Hide the agent from default listings (worker semantics).
    hidden: bool | None = None
    #: Whether to enable the git-backed memory filesystem (default: server).
    memfs: bool | None = None
    #: Server-side tools to attach at creation (``[]`` for none).
    base_tools: list[str] | None = None
    tags: list[str] = field(default_factory=list)
    #: Raw overrides merged into the ``create_agent`` body.
    extra_body: dict[str, Any] = field(default_factory=dict)

    def to_body(self) -> dict[str, Any]:
        body: dict[str, Any] = dict(self.extra_body)
        if self.name is not None:
            body["name"] = self.name
        if self.description is not None:
            body["description"] = self.description
        if self.model is not None:
            body["model"] = self.model
        if self.system_prompt is not None:
            body["system"] = self.system_prompt
        if self.embedding is not None:
            body["embedding"] = self.embedding
        memory: list[dict[str, Any]] = list(self.memory_blocks)
        if self.persona is not None:
            memory.append({"label": "persona", "value": self.persona})
        if self.human is not None:
            memory.append({"label": "human", "value": self.human})
        if memory:
            body["memory_blocks"] = memory
        if self.hidden is not None:
            body["hidden"] = self.hidden
        if self.memfs is not None:
            body["memfs"] = self.memfs
        if self.base_tools is not None:
            body["base_tools"] = list(self.base_tools)
        if self.tags:
            body["tags"] = list(self.tags)
        return body


@dataclass(slots=True)
class CreateSessionOptions:
    """Options for :meth:`LettaAgentClient.create_session` / ``resume_session``."""

    #: Model override applied to the session target (conversation/agent).
    model: str | None = None
    cwd: str | None = None
    permission_mode: str | None = None
    #: Run without loading/changing the agent's MemFS.
    stateless: bool | None = None
    #: Restrict available skills by source (``[]`` disables all).
    skill_sources: list[str] | None = None
    #: Custom tools executed locally in the SDK process (see ``ToolSpec``).
    tools: list["ToolSpec"] = field(default_factory=list)
    #: Callback deciding server tool approvals (``control_request`` with
    #: subtype ``can_use_tool``). When absent, the SDK assumes the server
    #: auto-handles approvals and keeps the turn open across
    #: ``requires_approval`` stops.
    can_use_tool: "CanUseToolCallback | None" = None
    #: Raw overrides merged into the runtime_start command.
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class CanUseToolDecision:
    """Decision returned by a ``can_use_tool`` callback."""

    #: ``"allow"`` or ``"deny"``.
    behavior: str
    #: Optional reason / message (required on the wire for ``deny``).
    message: str | None = None
    #: Optional replacement tool input (allow only).
    updated_input: dict[str, Any] | None = None

    def to_wire(self) -> dict[str, Any]:
        if self.behavior == "deny":
            return {
                "behavior": "deny",
                "message": self.message or "Denied by can_use_tool callback",
            }
        wire: dict[str, Any] = {
            "behavior": "allow",
            "updated_input": self.updated_input,
            "selected_permission_suggestion_ids": [],
        }
        if self.message is not None:
            wire["message"] = self.message
        return wire


#: Signature of a ``can_use_tool`` callback: (tool_name, tool_input,
#: context) -> decision (sync or awaitable). ``context`` carries the raw
#: request fields (request_id, tool_call_id, permission_suggestions, ...).
CanUseToolCallback: TypeAlias = Callable[
    [str, dict[str, Any], dict[str, Any]],
    "CanUseToolDecision | Awaitable[CanUseToolDecision]",
]


@dataclass(slots=True)
class QueryOptions:
    """Options for :meth:`LettaAgentClient.query` (agent-free ephemeral turn)."""

    #: Model handle for the ephemeral conversation (required).
    model: str | None = None
    #: System prompt for the ephemeral conversation (required).
    system: str | None = None
    model_settings: dict[str, Any] | None = None
    context_window_limit: int | None = None
    #: Session options forwarded to the ephemeral session.
    stateless: bool | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    def conversation_body(self) -> dict[str, Any]:
        body: dict[str, Any] = dict(self.extra_body)
        if self.model is not None:
            body["model"] = self.model
        if self.system is not None:
            body["system"] = self.system
        if self.model_settings is not None:
            body["model_settings"] = self.model_settings
        if self.context_window_limit is not None:
            body["context_window_limit"] = self.context_window_limit
        return body


@dataclass(slots=True)
class ListMessagesOptions:
    conversation_id: str | None = None
    before: str | None = None
    after: str | None = None
    order: str | None = None
    limit: int | None = None

    def to_query(self) -> dict[str, Any]:
        query: dict[str, Any] = {}
        if self.before is not None:
            query["before"] = self.before
        if self.after is not None:
            query["after"] = self.after
        if self.order is not None:
            query["order"] = self.order
        if self.limit is not None:
            query["limit"] = self.limit
        return query


@dataclass(slots=True)
class ListMessagesResult:
    messages: list[Any] = field(default_factory=list)
    next_before: str | None = None
    has_more: bool | None = None


# ═══════════════════════════════════════════════════════════════
# CLIENT-SIDE TOOLS
# ═══════════════════════════════════════════════════════════════


@dataclass(slots=True)
class ToolResult:
    """Result returned by a :class:`ToolSpec` executor."""

    content: list[dict[str, Any]]
    is_error: bool = False


class ToolSpec:
    """A client-side tool the agent can call; executed in the SDK process.

    Parameters
    ----------
    name, label, description:
        Tool identity shown to the model.
    parameters:
        A JSON Schema (plain dict) describing the tool's arguments.
    execute:
        An ``async def execute(tool_call_id: str, args: dict) -> ToolResult``
        coroutine. It may also return a ``list`` of content blocks directly.
    """

    def __init__(
        self,
        *,
        name: str,
        label: str | None = None,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        execute: Any = None,
    ) -> None:
        self.name = name
        self.label = label or name
        self.description = description
        self.parameters = parameters or {"type": "object", "properties": {}}
        self.execute = execute

    def to_wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "parameters": self.parameters,
        }
