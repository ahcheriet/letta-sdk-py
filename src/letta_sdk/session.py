"""App-server session: runtime lifecycle, turns, and stream conversion.

:class:`LettaSession` mirrors the TypeScript ``AppServerSession`` /
``RemoteClientSessionCore`` flow:

* ``ready()``   -- single-flight ``runtime_start`` (create agent, open a new
  conversation, or resume an agent/conversation);
* ``send()``    -- fire an ``input`` ``create_message`` turn and track it;
* ``stream()``  -- yield SDK messages for the tracked turn, ending with a
  ``ResultMessage``;
* ``list_messages()`` / ``list_models()`` / ``abort()`` / ``close()``.

Turn completion mirrors the TS turn coordinator: a ``stop_reason`` delta arms
a short trailing-usage grace window; a ``usage_statistics`` delta (or the
grace expiring, a ``turn_finished`` event, or ``update_loop_status``
``WAITING_ON_INPUT``) completes the turn and emits the result.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from importlib.metadata import version as _pkg_version
from typing import Any, AsyncIterator

from .app_server import (
    AppServerClosedError,
    AppServerConnection,
    AppServerConnectionLike,
    AppServerRequestError,
    AppServerTimeoutError,
)
from .mcp import McpToolBridge, connect_mcp_servers
from .types import (
    AssistantMessage,
    CanUseToolDecision,
    CreateSessionOptions,
    DreamingOptions,
    ErrorMessage,
    ListMessagesOptions,
    ListMessagesResult,
    LoopStatusMessage,
    PingMessage,
    QueueUpdateMessage,
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
    normalize_dreaming,
    normalize_toolset,
)

try:
    _SDK_VERSION = _pkg_version("letta-sdk-py")
except Exception:  # pragma: no cover
    _SDK_VERSION = "0.0.0"

logger = logging.getLogger("letta_sdk")

#: Trailing grace after stop_reason to let final usage accounting arrive.
TRAILING_USAGE_GRACE_SECONDS = 0.15

#: Tools auto-approved when no ``can_use_tool`` callback is registered
#: (mirrors the TypeScript SDK's headless policy).
HEADLESS_AUTO_ALLOW_TOOLS = {"EnterPlanMode"}

#: Tools that require real user input — never auto-allowed, even in
#: unrestricted permission mode (TS ``interactiveToolPolicy`` parity).
RUNTIME_USER_INPUT_TOOLS = {"AskUserQuestion", "ExitPlanMode"}

#: Interactive approval tools (TS ``interactiveToolPolicy`` parity).
INTERACTIVE_APPROVAL_TOOLS = {"AskUserQuestion", "EnterPlanMode", "ExitPlanMode"}

#: Permission-mode values that normalize to ``unrestricted`` — the current
#: name plus the legacy aliases (TS ``normalizePermissionMode`` parity).
_UNRESTRICTED_PERMISSION_MODES = {"unrestricted", "bypassPermissions", "fullAccess"}


def _is_unrestricted_permission_mode(mode: str | None) -> bool:
    """True when the session permission mode normalizes to ``unrestricted``.

    Port of the TS ``normalizePermissionMode`` + ``isUnrestrictedPermissionMode``
    pair: ``None``/``"default"`` → standard; ``"bypassPermissions"`` /
    ``"fullAccess"`` → unrestricted; ``standard``/``acceptEdits`` /
    ``unrestricted``/``strict`` pass through; anything else is unknown
    (never unrestricted).
    """
    return isinstance(mode, str) and mode in _UNRESTRICTED_PERMISSION_MODES

_FAILURE_STOP_REASONS = {
    "error",
    "llm_api_error",
    "max_steps",
    "loop_error",
    "interrupted",
    "stream_closed",
    "cancelled",
}


# ═══════════════════════════════════════════════════════════════
# wire helpers
# ═══════════════════════════════════════════════════════════════


def _content_to_text(content: Any) -> str:
    """Flatten wire ``content`` (str | list[part] | part) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text is None:
                    text = block.get("content")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        if text is None:
            text = content.get("content")
        if isinstance(text, str):
            return text
    return str(content)


def _tool_input_from_arguments(
    arguments: Any,
) -> tuple[dict[str, Any], str | None]:
    """Parse tool call arguments (dict or JSON string) -> (input, raw)."""
    if isinstance(arguments, dict):
        return arguments, None
    if isinstance(arguments, str):
        raw = arguments
        if not raw.strip():
            return {}, raw
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}, raw
        if isinstance(parsed, dict):
            return parsed, raw
        return {}, raw
    return {}, None


def _first_tool_call(delta: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("tool_call", "tool_calls"):
        value = delta.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value[0]
    return None


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _agent_tool_names(agent: Any) -> list[str] | None:
    if not isinstance(agent, dict):
        return None
    tools = agent.get("tools")
    if not isinstance(tools, list):
        return None
    names: list[str] = []
    for tool in tools:
        if isinstance(tool, str) and tool:
            names.append(tool)
        elif isinstance(tool, dict) and isinstance(tool.get("name"), str):
            names.append(tool["name"])
    return names


def _same_runtime(message: dict[str, Any], runtime: dict[str, str]) -> bool:
    scope = message.get("runtime")
    if not isinstance(scope, dict):
        # Messages without an explicit runtime (e.g. stream_delta on a
        # single-runtime local backend) are treated as matching.
        return True
    return (
        scope.get("agent_id") == runtime.get("agent_id")
        and scope.get("conversation_id") == runtime.get("conversation_id")
    )


# ═══════════════════════════════════════════════════════════════
# turn tracking
# ═══════════════════════════════════════════════════════════════


@dataclass(slots=True)
class _Turn:
    runtime: dict[str, str]
    client_message_id: str
    started_at: float
    assistant_text: str = ""
    run_ids: set[str] = field(default_factory=set)
    stop_reason: str | None = None
    usage: UsageMessage | None = None
    error: ErrorMessage | None = None
    observed_evidence: bool = False
    observed_requires_approval: bool = False
    pending_terminal: bool = False
    terminal_timeout: asyncio.TimerHandle | None = None


# ═══════════════════════════════════════════════════════════════
# session
# ═══════════════════════════════════════════════════════════════


class LettaSession:
    """A runtime-scoped session on a Letta Code app server.

    Constructed by :class:`~letta_sdk.client.LettaAgentClient`; normally not
    instantiated directly.
    """

    def __init__(
        self,
        *,
        connection: AppServerConnectionLike,
        owns_connection: bool = True,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        create_agent_body: dict[str, Any] | None = None,
        create_conversation_body: dict[str, Any] | None = None,
        new_conversation: bool = False,
        default_conversation: bool = False,
        options: CreateSessionOptions | None = None,
    ) -> None:
        self._connection = connection
        self._owns_connection = owns_connection
        self._mode_agent_id = agent_id
        self._mode_conversation_id = conversation_id
        self._create_agent_body = create_agent_body
        self._create_conversation_body = create_conversation_body
        self._new_conversation = new_conversation
        self._default_conversation = default_conversation
        self._options = options or CreateSessionOptions()
        self._client_tools = {
            t.name: t for t in (self._options.tools or [])
        }
        self._mcp_bridge: McpToolBridge | None = None
        self._toolset: ToolsetConfig | None = normalize_toolset(
            self._options.toolset
        )
        self._dreaming: DreamingOptions | None = normalize_dreaming(
            self._options.dreaming, allow_behavior=False
        )

        self._runtime: dict[str, str] | None = None
        self._agent_id: str | None = None
        self._conversation_id: str | None = None
        self._model = ""
        self._tools: list[str] | None = None
        self._initialized = False
        self._initialize_task: asyncio.Task[SDKInitMessage] | None = None
        self._last_init: SDKInitMessage | None = None
        self._closed = False
        self._owner: Any = None
        self._owns_owner = False
        self._remove_message_handler: Any = None

        self._turn_queue: list[_Turn] = []
        self._stream_queue: asyncio.Queue[SDKMessage] = asyncio.Queue()

    # ── properties ───────────────────────────────────────────────

    @property
    def agent_id(self) -> str | None:
        return self._agent_id

    @property
    def conversation_id(self) -> str | None:
        return self._conversation_id

    @property
    def model(self) -> str:
        return self._model

    @property
    def tools(self) -> list[str] | None:
        return list(self._tools) if self._tools is not None else None

    # ── lifecycle ────────────────────────────────────────────────

    async def __aenter__(self) -> LettaSession:
        await self.ready()
        return self

    async def ready(self) -> SDKInitMessage:
        """Initialize the runtime (idempotent, single-flight).

        Concurrent callers await the same initialization; a failed start is
        discarded and the next call retries.
        """
        if self._closed:
            raise RuntimeError("Session is closed")
        if self._initialized and self._last_init is not None:
            return self._last_init
        if self._initialize_task is None:
            self._initialize_task = asyncio.get_running_loop().create_task(
                self._perform_initialize(), name="letta-sdk-session-init"
            )
            self._initialize_task.add_done_callback(self._on_init_settled)
        return await self._initialize_task

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def attach_owner(self, owner: Any, *, owns: bool) -> None:
        """Tie this session's lifetime to a parent (e.g. a self-managed
        client). When ``owns`` is true, :meth:`close` also closes the owner.
        """
        self._owner = owner
        self._owns_owner = owns

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._remove_message_handler is not None:
            self._remove_message_handler()
            self._remove_message_handler = None
        for turn in list(self._turn_queue):
            if turn.terminal_timeout is not None:
                turn.terminal_timeout.cancel()
        self._turn_queue.clear()
        if self._owns_connection:
            await self._connection.close()
        await self._close_mcp_bridge()
        if self._owns_owner and self._owner is not None:
            owner = self._owner
            self._owner = None
            try:
                await owner.close()
            except Exception:  # pragma: no cover - best effort
                pass



    def _on_init_settled(self, task: "asyncio.Task[SDKInitMessage]") -> None:
        if task.cancelled():
            self._initialize_task = None
            return
        if task.exception() is not None:
            # Failed init is retryable: release partial state.
            self._initialize_task = None
            self._cleanup_failed_initialize()

    async def _ensure_initialized(self) -> None:
        await self.ready()

    # ── turns ────────────────────────────────────────────────────

    async def send(self, message: SendMessage, *, otid: str | None = None) -> None:
        """Send a user message, starting a new tracked turn."""
        if self._closed:
            raise RuntimeError("Session is closed")
        await self._ensure_initialized()
        assert self._runtime is not None

        if isinstance(message, str):
            if not message.strip():
                raise ValueError("A non-empty user message is required")
            content: Any = message
        else:
            content = [dict(part) for part in message]

        client_message_id = otid or f"sdk-message-{uuid.uuid4()}"
        self._turn_queue.append(
            _Turn(
                runtime=dict(self._runtime),
                client_message_id=client_message_id,
                started_at=time.monotonic(),
            )
        )
        payload: dict[str, Any] = {
            "kind": "create_message",
            "messages": [
                {
                    "role": "user",
                    "content": content,
                    "client_message_id": client_message_id,
                }
            ],
            "exclude_interactive_tools": True,
        }
        if self._toolset is not None:
            payload["client_toolset"] = self._toolset.to_wire()
        await self._connection.send(
            {
                "type": "input",
                "runtime": dict(self._runtime),
                "payload": payload,
            }
        )

    async def stream(self) -> AsyncIterator[SDKMessage]:
        """Yield SDK messages for the current turn, ending with a result.

        Must be called after :meth:`send`. Yields until the terminal
        ``ResultMessage`` for the sent turn, then stops.
        """
        if self._closed:
            raise RuntimeError("Session is closed")
        while True:
            message = await self._stream_queue.get()
            yield message
            if isinstance(message, ResultMessage):
                return

    async def send_and_wait(self, message: SendMessage) -> ResultMessage:
        """Send and consume the stream until the terminal result."""
        await self.send(message)
        async for sdk_message in self.stream():
            if isinstance(sdk_message, ResultMessage):
                return sdk_message
        raise RuntimeError("Stream ended before a result message.")

    async def prompt(self, message: SendMessage) -> ResultMessage:
        """One-shot: send a message and return the terminal result."""
        return await self.send_and_wait(message)

    # ── session management ───────────────────────────────────────

    async def list_messages(
        self, options: ListMessagesOptions | None = None, **kwargs: Any
    ) -> ListMessagesResult:
        if self._closed:
            raise RuntimeError("Session is closed")
        await self._ensure_initialized()
        opts = options or ListMessagesOptions(**kwargs)
        conversation_id = opts.conversation_id or self._conversation_id
        if not conversation_id:
            raise RuntimeError("No conversation id available for list_messages()")
        body: dict[str, Any] = {"conversation_id": conversation_id}
        query = opts.to_query()
        if query:
            body["query"] = query
        response = await self._connection.request(
            "conversation_messages_list",
            body,
            response_type="conversation_messages_list_response",
        )
        if not response.get("success", True):
            raise AppServerRequestError(
                str(response.get("error") or "conversation_messages_list failed")
            )
        result = ListMessagesResult(messages=list(response.get("messages") or []))
        for key in ("nextBefore", "next_before"):
            value = response.get(key)
            if isinstance(value, str) or value is None:
                result.next_before = value
                break
        for key in ("hasMore", "has_more"):
            if isinstance(response.get(key), bool):
                result.has_more = response[key]
                break
        return result

    async def list_models(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Session is closed")
        await self._ensure_initialized()
        response = await self._connection.request(
            "list_models", {}, response_type="list_models_response"
        )
        if not response.get("success", True):
            raise AppServerRequestError(
                str(response.get("error") or "list_models failed")
            )
        return {
            "entries": list(response.get("entries") or []),
            "available_handles": response.get("available_handles"),
            "byok_provider_aliases": response.get("byok_provider_aliases"),
        }

    async def abort(self) -> None:
        if not self._initialized or self._runtime is None:
            return
        try:
            await self._connection.request(
                "abort",
                {"runtime": dict(self._runtime)},
                response_type="abort_response",
            )
        except (AppServerRequestError, AppServerTimeoutError, AppServerClosedError):
            # Some harnesses do not acknowledge abort; the loop status will
            # reflect the interruption on the stream instead.
            pass

    # ── initialization internals ─────────────────────────────────

    def _build_runtime_start_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "client_info": {
                "name": "letta-sdk-py",
                "title": "Letta Agent SDK (Python)",
                "version": _SDK_VERSION,
            },
            "recover_approvals": False,
            "force_device_status": True,
        }
        options = self._options
        if options.permission_mode is not None:
            body["mode"] = options.permission_mode
        if options.cwd is not None:
            body["cwd"] = options.cwd
        if options.stateless is not None:
            body["stateless"] = options.stateless
        if options.skill_sources is not None:
            body["skill_sources"] = list(dict.fromkeys(options.skill_sources))
        if self._client_tools:
            body["external_tools"] = [
                {"tools": [t.to_wire() for t in self._client_tools.values()]}
            ]
        if options.extra_body:
            body.update(options.extra_body)

        if self._create_agent_body is not None:
            body["create_agent"] = {
                "body": self._create_agent_body,
                # Hidden (worker-style) agents default to unpinned.
                "pin_global": not self._create_agent_body.get("hidden"),
            }
            return body

        if self._create_conversation_body is not None:
            body["create_conversation"] = {"body": self._create_conversation_body}
            return body

        if self._mode_agent_id is not None:
            body["agent_id"] = self._mode_agent_id
            if self._new_conversation:
                body["create_conversation"] = {"body": {}}
            elif self._default_conversation:
                body["conversation_id"] = "default"
            elif self._mode_conversation_id is not None:
                body["conversation_id"] = self._mode_conversation_id
            return body

        if self._mode_conversation_id is not None:
            # Bare conversation id: agent is resolved in _perform_initialize.
            body["conversation_id"] = self._mode_conversation_id
            return body

        raise RuntimeError(
            "Session requires an agent id, a conversation id, or a "
            "create_agent/create_conversation body."
        )

    # ── MCP bridge ─────────────────────────────────────────────

    async def _connect_mcp_bridge(self) -> None:
        servers = self._options.mcp_servers
        if not servers:
            return
        if self._mcp_bridge is not None:
            await self._close_mcp_bridge()
        bridge = await connect_mcp_servers(
            servers,
            cwd=self._options.cwd,
            reserved_tool_names=set(self._client_tools),
            log=lambda message: logger.warning(message),
        )
        for tool in bridge.tools:
            self._client_tools[tool.name] = tool
        self._mcp_bridge = bridge

    async def _close_mcp_bridge(self) -> None:
        bridge = self._mcp_bridge
        if bridge is None:
            return
        self._mcp_bridge = None
        try:
            await bridge.close()
        except Exception:  # pragma: no cover - best effort
            pass

    async def _perform_initialize(self) -> SDKInitMessage:
        await self._connect_mcp_bridge()
        try:
            return await self._perform_initialize_inner()
        except BaseException:
            await self._close_mcp_bridge()
            raise

    async def _perform_initialize_inner(self) -> SDKInitMessage:
        body = self._build_runtime_start_body()
        if (
            self._create_agent_body is None
            and self._create_conversation_body is None
            and self._mode_agent_id is None
            and self._mode_conversation_id is not None
        ):
            agent_id = await self._resolve_conversation_agent_id(
                self._mode_conversation_id
            )
            body["agent_id"] = agent_id

        response = await self._connection.request(
            "runtime_start",
            body,
            response_type="runtime_start_response",
        )
        if not response.get("success", True) or not isinstance(
            response.get("runtime"), dict
        ):
            raise AppServerRequestError(
                str(
                    response.get("error")
                    or "runtime_start failed (no runtime in response)"
                )
            )

        runtime = response["runtime"]
        self._runtime = {
            "agent_id": runtime["agent_id"],
            "conversation_id": runtime["conversation_id"],
        }
        self._agent_id = runtime["agent_id"]
        self._conversation_id = runtime["conversation_id"]

        if self._dreaming is not None and not self._options.stateless:
            # TS parity: after the runtime is up (and after the memfs step,
            # which the server handles from the create body), apply the
            # reflection settings; skipped for stateless sessions.
            settings_response = await self._connection.request(
                "set_reflection_settings",
                {
                    "runtime": dict(self._runtime),
                    "settings": self._dreaming.to_settings(),
                    "scope": "both",
                },
                response_type="set_reflection_settings_response",
            )
            if not settings_response.get("success", True):
                raise AppServerRequestError(
                    str(
                        settings_response.get("error")
                        or "set_reflection_settings failed"
                    )
                )

        agent_raw = response.get("agent")
        agent: dict[str, Any] = agent_raw if isinstance(agent_raw, dict) else {}
        conversation_raw = response.get("conversation")
        conversation: dict[str, Any] = (
            conversation_raw if isinstance(conversation_raw, dict) else {}
        )
        agent_model = agent.get("model")
        if isinstance(agent_model, str):
            self._model = agent_model
        else:
            conversation_model = conversation.get("model")
            if isinstance(conversation_model, str):
                self._model = conversation_model
        agent_tools = _agent_tool_names(agent)
        mcp_tool_names = (
            [t.name for t in self._mcp_bridge.tools]
            if self._mcp_bridge is not None
            else []
        )
        if agent_tools is not None or mcp_tool_names:
            self._tools = list(agent_tools or []) + mcp_tool_names

        self._remove_message_handler = self._connection.on_message(
            self._on_protocol_message
        )
        self._initialized = True
        init = SDKInitMessage(
            type="init",
            raw=response,
            agent_id=self._agent_id,
            session_id=(
                f"{self._agent_id}:{self._conversation_id}"
                if self._agent_id
                else self._conversation_id
            ),
            conversation_id=self._conversation_id,
            model=self._model,
            tools=self._tools,
        )
        self._last_init = init
        return init

    async def _resolve_conversation_agent_id(self, conversation_id: str) -> str:
        response = await self._connection.request(
            "conversation_retrieve",
            {"conversation_id": conversation_id},
            response_type="conversation_retrieve_response",
        )
        conversation = response.get("conversation")
        if (
            not response.get("success", True)
            or not isinstance(conversation, dict)
            or not isinstance(conversation.get("agent_id"), str)
        ):
            raise AppServerRequestError(
                str(response.get("error"))
                or f"Failed to retrieve conversation {conversation_id}"
            )
        return conversation["agent_id"]

    def _cleanup_failed_initialize(self) -> None:
        if self._remove_message_handler is not None:
            self._remove_message_handler()
            self._remove_message_handler = None
        self._runtime = None
        self._agent_id = None
        self._conversation_id = None
        self._model = ""
        self._tools = None
        self._initialized = False
        self._last_init = None
        self._turn_queue.clear()

    # ── protocol message handling ────────────────────────────────

    def _on_protocol_message(self, message: dict[str, Any]) -> None:
        if not isinstance(message, dict):
            return
        if message.get("type") == "_transport_closed" and not self._closed:
            self._fail_turns(str(message.get("error") or "connection closed"))
            return
        if self._runtime is None or not _same_runtime(message, self._runtime):
            return
        if message.get("type") == "external_tool_call_request":
            self._on_external_tool_request(message)
            return
        if message.get("type") == "control_request":
            asyncio.get_running_loop().create_task(
                self._handle_control_request(message)
            )
            return
        self._handle_status_message(message)
        if message.get("type") == "turn_finished":
            self._handle_turn_finished(message)
            return
        if message.get("type") == "stream_delta":
            delta = message.get("delta")
            if isinstance(delta, dict):
                self._handle_stream_delta(delta)

    def _handle_status_message(self, message: dict[str, Any]) -> None:
        msg_type = message.get("type")
        if msg_type == "update_loop_status":
            self._handle_loop_status(message)
        elif msg_type == "update_queue":
            self._stream_queue.put_nowait(
                QueueUpdateMessage(
                    type="queue_update",
                    raw=message,
                    queue=list(message.get("queue") or []),
                )
            )
        # update_device_status / update_subagent_state: ignored in v1.

    def _handle_loop_status(self, message: dict[str, Any]) -> None:
        loop_status = message.get("loop_status")
        status = loop_status.get("status") if isinstance(loop_status, dict) else None
        if not isinstance(status, str):
            return
        active_run_ids = (
            [r for r in loop_status.get("active_run_ids") or [] if isinstance(r, str)]
            if isinstance(loop_status, dict)
            else []
        )
        self._stream_queue.put_nowait(
            LoopStatusMessage(
                type="loop_status",
                raw=message,
                status=status,
                active_run_ids=active_run_ids,
            )
        )
        turn = self._current_turn()
        if turn is None:
            return
        turn.run_ids.update(active_run_ids)
        if not turn.observed_evidence:
            return
        if status == "WAITING_ON_INPUT" and not turn.pending_terminal:
            self._complete_turn(
                turn,
                stop_reason=turn.stop_reason,
                success=turn.error is None,
                detail=turn.error.message if turn.error else None,
                error_code=turn.error.error_code if turn.error else None,
            )
        elif status == "WAITING_ON_APPROVAL":
            turn.observed_requires_approval = True
            if self._options.can_use_tool is None:
                # Server-side auto-approval (no callback registered): the
                # turn continues on a follow-up run — keep it open.
                return
            self._complete_turn(
                turn,
                stop_reason="requires_approval",
                success=True,
                detail=None,
                error_code=None,
            )

    async def _handle_control_request(self, message: dict[str, Any]) -> None:
        """Answer a ``can_use_tool`` control request with an approval decision."""
        if message.get("subtype") != "can_use_tool":
            return
        request_id = message.get("request_id")
        tool_name = message.get("tool_name")
        if not isinstance(request_id, str) or not isinstance(tool_name, str):
            return
        tool_input = message.get("input")
        tool_input = tool_input if isinstance(tool_input, dict) else {}
        runtime = message.get("runtime")
        runtime = runtime if isinstance(runtime, dict) else self._runtime
        if not isinstance(runtime, dict):
            return
        context: dict[str, Any] = {"request_id": request_id}
        for key in ("tool_call_id", "permission_suggestions", "blocked_path", "diffs"):
            if key in message:
                context[key] = message[key]
        decision = await self._resolve_tool_approval(tool_name, tool_input, context)
        try:
            await self._connection.send(
                {
                    "type": "input",
                    "runtime": runtime,
                    "payload": {
                        "kind": "approval_response",
                        "request_id": request_id,
                        "decision": decision,
                    },
                }
            )
        except Exception as exc:  # pragma: no cover - transport failure
            logger.warning("failed to send approval response: %s", exc)

    async def _resolve_tool_approval(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve a tool-approval request.

        Port of the TS ``resolveAppServerToolApproval`` ordering:
        1. runtime-user-input tool without a callback → deny;
        2. unrestricted permission mode (non-user-input tool) → allow,
           callback not consulted;
        3. callback → its decision (error → deny);
        4. headless auto-allow tools (``EnterPlanMode``) → allow;
        5. otherwise → deny.
        """
        callback = self._options.can_use_tool
        has_callback = callback is not None
        tool_needs_runtime_user_input = tool_name in RUNTIME_USER_INPUT_TOOLS

        if tool_needs_runtime_user_input and not has_callback:
            return {
                "behavior": "deny",
                "message": "No canUseTool callback registered",
            }

        if (
            _is_unrestricted_permission_mode(self._options.permission_mode)
            and not tool_needs_runtime_user_input
        ):
            return {
                "behavior": "allow",
                "updated_input": None,
                "selected_permission_suggestion_ids": [],
            }

        if has_callback:
            try:
                result = callback(tool_name, tool_input, context)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:
                return {"behavior": "deny", "message": str(exc) or "Callback error"}
            if isinstance(result, CanUseToolDecision):
                return result.to_wire()
            if isinstance(result, dict):
                return result
            return {
                "behavior": "deny",
                "message": f"can_use_tool callback returned {type(result).__name__}",
            }

        if tool_name in HEADLESS_AUTO_ALLOW_TOOLS:
            return {
                "behavior": "allow",
                "updated_input": None,
                "selected_permission_suggestion_ids": [],
            }
        return {"behavior": "deny", "message": "No canUseTool callback registered"}

    def _handle_turn_finished(self, message: dict[str, Any]) -> None:
        run_id = message.get("run_id") or message.get("runId")
        if not isinstance(run_id, str):
            return
        stop_reason = message.get("stop_reason") or message.get("stopReason")
        turn = self._current_turn()
        if turn is None or turn.pending_terminal:
            return
        if turn.run_ids and run_id not in turn.run_ids:
            return
        turn.run_ids.add(run_id)
        if stop_reason == "requires_approval":
            turn.observed_requires_approval = True
            if self._options.can_use_tool is None:
                # Server-side auto-approval: wait for the follow-up run's
                # terminal stop.
                return
            self._complete_turn(
                turn,
                stop_reason="requires_approval",
                success=True,
                detail=None,
                error_code=None,
            )
            return
        success = (
            stop_reason not in _FAILURE_STOP_REASONS
            if isinstance(stop_reason, str)
            else True
        )
        self._complete_turn(
            turn,
            stop_reason=stop_reason if isinstance(stop_reason, str) else None,
            success=success,
            detail=None if success else str(stop_reason),
            error_code=None if success else (stop_reason or "error"),
        )

    def _handle_stream_delta(self, delta: dict[str, Any]) -> None:
        message_type = delta.get("message_type")
        if not isinstance(message_type, str):
            return
        run_id = delta.get("run_id") if isinstance(delta.get("run_id"), str) else None

        turn = self._current_turn()
        # Terminal metadata can trail the turn that produced it; never let it
        # route to a queued (next) turn.
        if turn is None and message_type not in {"stop_reason", "usage_statistics"}:
            return
        if turn is not None:
            turn.observed_evidence = True
            if run_id:
                turn.run_ids.add(run_id)

        sdk_message = self._transform_stream_delta(delta)
        if sdk_message is not None:
            self._stream_queue.put_nowait(sdk_message)

        if turn is None:
            return
        if isinstance(sdk_message, UsageMessage):
            turn.usage = sdk_message
            if turn.pending_terminal:
                self._complete_pending_terminal(turn)
            return
        if isinstance(sdk_message, ErrorMessage):
            self._complete_turn(
                turn,
                stop_reason=sdk_message.stop_reason or "error",
                success=False,
                detail=sdk_message.error_detail or sdk_message.message,
                error_code=sdk_message.error_code or "error",
            )
            return
        if message_type == "stop_reason":
            stop_reason = delta.get("stop_reason") or delta.get("reason")
            reason = stop_reason if isinstance(stop_reason, str) else "end_turn"
            turn.stop_reason = reason
            if reason == "requires_approval":
                # The approval is resolved server-side (or by the registered
                # can_use_tool callback); the turn continues on a follow-up
                # run, so keep it open instead of arming a terminal.
                turn.observed_requires_approval = True
                return
            turn.pending_terminal = True
            if turn.terminal_timeout is None:
                asyncio.get_running_loop().call_later(
                    TRAILING_USAGE_GRACE_SECONDS,
                    self._complete_pending_terminal,
                    turn,
                )

    def _current_turn(self) -> _Turn | None:
        return self._turn_queue[0] if self._turn_queue else None

    def _complete_pending_terminal(self, turn: _Turn) -> None:
        if not turn.pending_terminal:
            return
        if self._turn_queue[:1] != [turn]:
            return
        self._complete_turn(
            turn,
            stop_reason=turn.stop_reason,
            success=turn.error is None,
            detail=turn.error.message if turn.error else None,
            error_code=turn.error.error_code if turn.error else None,
        )

    def _complete_turn(
        self,
        turn: _Turn,
        *,
        stop_reason: str | None,
        success: bool,
        detail: str | None,
        error_code: str | None,
    ) -> None:
        if turn.terminal_timeout is not None:
            turn.terminal_timeout.cancel()
            turn.terminal_timeout = None
        if self._turn_queue and self._turn_queue[0] is turn:
            self._turn_queue.pop(0)
        duration_ms = (time.monotonic() - turn.started_at) * 1000.0
        result = ResultMessage(
            type="result",
            raw={"stop_reason": stop_reason, "usage": turn.usage.raw if turn.usage else None},
            success=bool(success),
            result=turn.assistant_text or None if success else None,
            error=None if success else (detail or error_code or "error"),
            error_code=None if success else error_code,
            stop_reason=stop_reason,
            duration_ms=duration_ms,
            conversation_id=turn.runtime.get("conversation_id"),
            run_ids=sorted(turn.run_ids) or None,
            error_detail=detail if not success else None,
            recoverable=None if success else False,
        )
        self._stream_queue.put_nowait(result)

    def _fail_turns(self, detail: str) -> None:
        for turn in list(self._turn_queue):
            if turn.terminal_timeout is not None:
                turn.terminal_timeout.cancel()
        self._turn_queue.clear()
        self._stream_queue.put_nowait(
            ErrorMessage(
                type="error",
                raw={"error": detail},
                message=detail,
                error_code="stream_closed",
                stop_reason="stream_closed",
                error_detail=detail,
                recoverable=True,
            )
        )

    # ── external tools ───────────────────────────────────────────

    def _on_external_tool_request(self, message: dict[str, Any]) -> None:
        if message.get("type") != "external_tool_call_request":
            return
        request_id = message.get("request_id")
        if not isinstance(request_id, str):
            return
        tool_name = message.get("tool_name")
        tool = (
            self._client_tools.get(tool_name)
            if isinstance(tool_name, str)
            else None
        )
        loop = asyncio.get_running_loop()
        if tool is None or tool.execute is None:
            loop.create_task(
                self._connection.send(
                    {
                        "type": "external_tool_call_response",
                        "request_id": request_id,
                        "error": (
                            f"No controller handler is registered for '{tool_name}'"
                        ),
                    }
                )
            )
            return
        loop.create_task(self._run_external_tool(request_id, tool, message))

    async def _run_external_tool(
        self, request_id: str, tool: ToolSpec, message: dict[str, Any]
    ) -> None:
        payload: dict[str, Any] = {
            "type": "external_tool_call_response",
            "request_id": request_id,
        }
        try:
            result = await tool.execute(
                message.get("tool_call_id"), message.get("input") or {}
            )
            if isinstance(result, ToolResult):
                payload["result"] = {
                    "content": [
                        {k: v for k, v in part.items() if v is not None}
                        for part in result.content
                    ],
                    **({"is_error": True} if result.is_error else {}),
                }
            elif isinstance(result, list):
                payload["result"] = {
                    "content": [
                        {k: v for k, v in part.items() if v is not None}
                        for part in result
                    ]
                }
            else:
                payload["result"] = {
                    "content": [{"type": "text", "text": str(result)}]
                }
        except Exception as exc:
            payload["error"] = f"External tool '{tool.name}' failed: {exc}"
        try:
            await self._connection.send(payload)
        except AppServerClosedError:
            pass

    # ── message conversion ───────────────────────────────────────

    def _transform_stream_delta(self, delta: dict[str, Any]) -> SDKMessage | None:
        message_type = delta.get("message_type")
        if not isinstance(message_type, str):
            return StreamEventMessage(
                type="stream_event", raw={"stream_delta": delta}, event=delta
            )
        run_id = delta.get("run_id") if isinstance(delta.get("run_id"), str) else None
        otid = delta.get("otid") if isinstance(delta.get("otid"), str) else None
        seq_id = _as_int(delta.get("seq_id"))
        uuid_value = delta.get("id") if isinstance(delta.get("id"), str) else None
        raw = {"stream_delta": delta}

        if message_type == "assistant_message":
            content = _content_to_text(delta.get("content"))
            if not content:
                return None
            turn = self._current_turn()
            if turn is not None:
                turn.assistant_text += content
            return AssistantMessage(
                type="assistant",
                raw=raw,
                content=content,
                uuid=uuid_value,
                otid=otid,
                seq_id=seq_id,
                run_id=run_id,
            )

        if message_type == "reasoning_message":
            reasoning = delta.get("reasoning")
            content = (
                reasoning
                if isinstance(reasoning, str)
                else _content_to_text(delta.get("content"))
            )
            if not content:
                return None
            return ReasoningMessage(
                type="reasoning",
                raw=raw,
                content=content,
                uuid=uuid_value,
                otid=otid,
                seq_id=seq_id,
                run_id=run_id,
            )

        if message_type in {"tool_call_message", "approval_request_message"}:
            tool_call = _first_tool_call(delta)
            if tool_call is None:
                return None
            function = tool_call.get("function")
            function = function if isinstance(function, dict) else {}
            tool_call_id = delta.get("tool_call_id")
            if not isinstance(tool_call_id, str):
                tool_call_id = tool_call.get("tool_call_id")
            if not isinstance(tool_call_id, str):
                tool_call_id = tool_call.get("id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                tool_call_id = ""  # id is metadata; never fail the turn
            tool_name = tool_call.get("name")
            if not isinstance(tool_name, str):
                tool_name = function.get("name")
            if not isinstance(tool_name, str):
                tool_name = "?"
            arguments = tool_call.get("arguments")
            if arguments is None:
                arguments = function.get("arguments")
            tool_input, raw_arguments = _tool_input_from_arguments(arguments)
            return ToolCallMessage(
                type="tool_call",
                raw=raw,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_input=tool_input,
                raw_arguments=raw_arguments,
                uuid=uuid_value,
                run_id=run_id,
            )

        if message_type == "tool_return_message":
            tool_return = delta.get("tool_return")
            tool_return = tool_return if isinstance(tool_return, dict) else {}
            tool_call_id = delta.get("tool_call_id")
            if not isinstance(tool_call_id, str):
                tool_call_id = tool_return.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                return None
            content_value = delta.get("tool_return")
            if content_value is None:
                content_value = tool_return.get("tool_return")
            if content_value is None:
                content_value = tool_return.get("content")
            status = delta.get("status")
            return ToolResultMessage(
                type="tool_result",
                raw=raw,
                tool_call_id=tool_call_id,
                content=_content_to_text(content_value),
                is_error=status == "error",
                uuid=uuid_value,
                run_id=run_id,
            )

        if message_type == "usage_statistics":
            run_ids = delta.get("run_ids")
            return UsageMessage(
                type="usage",
                raw=raw,
                prompt_tokens=_as_int(delta.get("prompt_tokens")),
                completion_tokens=_as_int(delta.get("completion_tokens")),
                total_tokens=_as_int(delta.get("total_tokens")),
                step_count=_as_int(delta.get("step_count")),
                run_ids=(
                    [r for r in run_ids if isinstance(r, str)]
                    if isinstance(run_ids, list)
                    else []
                ),
            )

        if message_type in {"error_message", "loop_error"}:
            detail_value = delta.get("detail")
            if not isinstance(detail_value, str):
                detail_value = delta.get("message")
            detail = detail_value if isinstance(detail_value, str) else "app-server turn failed"
            stop_reason = delta.get("stop_reason")
            if not isinstance(stop_reason, str):
                stop_reason = delta.get("error_type")
            if not isinstance(stop_reason, str):
                stop_reason = "error"
            return ErrorMessage(
                type="error",
                raw=raw,
                message=detail,
                error_code=(
                    stop_reason if stop_reason in _FAILURE_STOP_REASONS else "error"
                ),
                stop_reason=stop_reason,
                error_detail=detail,
                run_id=run_id,
            )

        if message_type == "retry":
            reason = delta.get("reason")
            return RetryMessage(
                type="retry",
                raw=raw,
                reason=reason if isinstance(reason, str) else "error",
                attempt=_as_int(delta.get("attempt")) or 0,
                max_attempts=_as_int(delta.get("max_attempts")) or 0,
                delay_ms=_as_int(delta.get("delay_ms")) or 0,
                run_id=run_id,
            )

        if message_type == "ping":
            return PingMessage(type="ping", raw=raw, uuid=uuid_value)

        if message_type == "stop_reason":
            return None

        return StreamEventMessage(
            type="stream_event", raw=raw, event=delta, uuid=uuid_value
        )
