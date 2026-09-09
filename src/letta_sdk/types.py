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
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, TypedDict, TypeAlias

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from .skills import AgentSkill


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

#: Valid ``toolset.base`` values (TS ``ClientToolsetBase``).
CLIENT_TOOLSET_BASES: frozenset[str] = frozenset(
    {"auto", "codex", "codex_snake", "default", "gemini", "gemini_snake", "none"}
)

#: Valid ``dreaming.trigger`` values (TS ``DreamingTrigger``).
DREAMING_TRIGGERS: frozenset[str] = frozenset(
    {"off", "step-count", "compaction-event"}
)

#: Valid ``dreaming.behavior`` values (TS ``DreamingBehavior``).
DREAMING_BEHAVIORS: frozenset[str] = frozenset({"reminder", "auto-launch"})


@dataclass(slots=True)
class ToolsetConfig:
    """Request-scoped client toolset (TS ``ClientToolsetConfig``).

    Sent as ``client_toolset`` in every ``create_message`` payload. Omitted
    fields preserve the harness preference.
    """

    #: ``auto``, ``codex``, ``codex_snake``, ``default``, ``gemini``,
    #: ``gemini_snake``, or ``none``.
    base: str | None = None
    #: Additional bundled client tools to load before applying allowedTools.
    include: list[str] | None = None

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {}
        if self.base is not None:
            wire["base"] = self.base
        if self.include is not None:
            wire["include"] = list(dict.fromkeys(self.include))
        return wire


@dataclass(slots=True)
class DreamingOptions:
    """Reflection ("dreaming") settings.

    The wire settings always carry both keys: ``trigger`` (default
    ``"step-count"``) and ``step_count`` (default ``5``).
    """

    #: ``off``, ``step-count``, or ``compaction-event``.
    trigger: str | None = None
    #: ``reminder`` or ``auto-launch``. Not supported by the app server and
    #: rejected by the normalizers (TS parity).
    behavior: str | None = None
    #: Positive integer.
    step_count: int | None = None

    def to_settings(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger if self.trigger is not None else "step-count",
            "step_count": self.step_count if self.step_count is not None else 5,
        }


def _dreaming_from_dict(value: dict[str, Any]) -> DreamingOptions:
    unknown = set(value) - {"trigger", "behavior", "step_count"}
    if unknown:
        raise ValueError(
            f"Unknown dreaming option(s): {', '.join(sorted(unknown))}"
        )
    return DreamingOptions(
        trigger=value.get("trigger"),
        behavior=value.get("behavior"),
        step_count=value.get("step_count"),
    )


def normalize_dreaming(
    value: DreamingOptions | dict[str, Any] | None,
    *,
    allow_behavior: bool,
) -> DreamingOptions | None:
    """Validate/normalize dreaming options (port of TS
    ``validateDreamingOptions`` + the behavior-rejection rules)."""
    if value is None:
        return None
    options = _dreaming_from_dict(value) if isinstance(value, dict) else value
    if options.behavior is not None and not allow_behavior:
        raise ValueError(
            "dreaming.behavior is not supported when opening an existing "
            "agent session."
        )
    if (
        options.trigger is not None
        and options.trigger not in DREAMING_TRIGGERS
    ):
        raise ValueError(
            f"Invalid dreaming.trigger '{options.trigger}'. "
            "Valid values: off, step-count, compaction-event"
        )
    if options.behavior is not None and options.behavior not in DREAMING_BEHAVIORS:
        raise ValueError(
            f"Invalid dreaming.behavior '{options.behavior}'. "
            "Valid values: reminder, auto-launch"
        )
    if options.step_count is not None and (
        isinstance(options.step_count, bool)
        or not isinstance(options.step_count, int)
        or options.step_count <= 0
    ):
        raise ValueError("Invalid dreaming.step_count. Expected a positive integer.")
    return options


def normalize_toolset(
    value: ToolsetConfig | dict[str, Any] | None,
) -> ToolsetConfig | None:
    """Validate/normalize a toolset option (port of TS validation)."""
    if value is None:
        return None
    config = (
        _toolset_from_dict(value)
        if isinstance(value, dict)
        else value
    )
    if config.base is not None and config.base not in CLIENT_TOOLSET_BASES:
        raise ValueError(
            f"Invalid toolset.base '{config.base}'. Valid values: auto, "
            "codex, codex_snake, default, gemini, gemini_snake, none"
        )
    if config.include is not None and not all(
        isinstance(name, str) and name for name in config.include
    ):
        raise ValueError("toolset.include must be a list of non-empty strings.")
    return config


def _toolset_from_dict(value: dict[str, Any]) -> ToolsetConfig:
    unknown = set(value) - {"base", "include"}
    if unknown:
        raise ValueError(f"Unknown toolset option(s): {', '.join(sorted(unknown))}")
    return ToolsetConfig(base=value.get("base"), include=value.get("include"))


@dataclass(slots=True)
class CreateAgentOptions:
    """Options for :meth:`LettaAgentClient.create_agent`."""

    name: str | None = None
    description: str | None = None
    #: Model handle, e.g. ``"openai/gpt-5.5"``.
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
    #: Letta Code personality preset (``memo``, ``blank``, ``tutorial``,
    #: ``linus``, ``kawaii``). Resolved server-side via the app server's
    #: native ``create_agent`` command; cannot be combined with custom
    #: memory blocks, persona, human, or system_prompt.
    personality: str | None = None
    #: Skill seeding: skill directory paths (must contain ``SKILL.md``) or
    #: inline ``AgentSkill``/dict items. Seeded as ``skills/{name}`` memory
    #: blocks; requires memfs (the default).
    skills: "list[str | AgentSkill | dict[str, Any]] | None" = None
    #: Reflection ("dreaming") settings applied after the runtime starts.
    #: ``behavior`` is not supported by the app server and is rejected.
    dreaming: "DreamingOptions | dict[str, Any] | None" = None
    #: Pin the new agent globally (default: pinned unless ``hidden``).
    pin_global: bool | None = None
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
    #: MCP servers keyed by name (stdio configs: ``{"command", "args"?,
    #: "env"?, "cwd"?}``); their tools are exposed as
    #: ``mcp__<server>__<tool>`` external tools. ``http``/``sse`` transports
    #: are accepted but reported as unavailable (stdio only).
    mcp_servers: "dict[str, dict[str, Any]] | None" = None
    #: Request-scoped client toolset (``ToolsetConfig`` or dict); sent as
    #: ``client_toolset`` on every turn.
    toolset: "ToolsetConfig | dict[str, Any] | None" = None
    #: Reflection ("dreaming") settings applied after the runtime starts
    #: (``set_reflection_settings``). ``behavior`` is not supported when
    #: opening an existing agent session and is rejected.
    dreaming: "DreamingOptions | dict[str, Any] | None" = None
    #: Callback deciding server tool approvals (``control_request`` with
    #: subtype ``can_use_tool``). When absent, the SDK assumes the server
    #: auto-handles approvals and keeps the turn open across
    #: ``requires_approval`` stops.
    can_use_tool: "CanUseToolCallback | None" = None
    #: Raw overrides merged into the runtime_start command.
    extra_body: dict[str, Any] = field(default_factory=dict)


def permission_suggestion_id(value: Any) -> str | None:
    """Extract a stable id from a permission suggestion (TS parity).

    Accepts a bare string or an object with ``id`` / ``suggestion_id`` /
    ``permission_suggestion_id``; anything else yields ``None``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("id", "suggestion_id", "permission_suggestion_id"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return None


@dataclass(slots=True, kw_only=True)
class CanUseToolDecision:
    """Decision returned by a ``can_use_tool`` callback.

    Mirrors the TypeScript SDK's ``CanUseToolResponse``; :meth:`to_wire`
    mirrors the TS ``toAppServerApprovalDecision`` wire mapping.
    """

    #: ``"allow"`` or ``"deny"``.
    behavior: str
    #: Optional reason / message (required on the wire for ``deny``).
    message: str | None = None
    #: Optional replacement tool input (allow only).
    updated_input: dict[str, Any] | None = None
    #: Permission suggestions granted with an allow (TS ``updatedPermissions``);
    #: mapped to ``selected_permission_suggestion_ids`` on the wire.
    updated_permissions: list[Any] | None = None
    #: TS parity field — accepted from callbacks but never sent on the wire
    #: (the TS wire mapper drops it).
    interrupt: bool | None = None

    def to_wire(self) -> dict[str, Any]:
        if self.behavior == "deny":
            return {
                "behavior": "deny",
                "message": self.message or "Denied by canUseTool callback",
            }
        wire: dict[str, Any] = {
            "behavior": "allow",
            "updated_input": self.updated_input,
            "selected_permission_suggestion_ids": [
                suggestion_id
                for suggestion_id in (
                    permission_suggestion_id(item)
                    for item in (self.updated_permissions or [])
                )
                if suggestion_id is not None
            ],
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
    details: Any = None


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
