from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator

from .types import (
    AssistantMessage,
    CreateSessionOptions,
    ErrorMessage,
    PingMessage,
    ReasoningMessage,
    ResultMessage,
    SDKMessage,
    SendMessage,
    SessionState,
    ToolCallMessage,
    ToolResultMessage,
    UnknownMessage,
    UsageMessage,
)


@dataclass(slots=True)
class LettaSession:
    _client: Any
    state: SessionState
    _options: CreateSessionOptions
    _owns_client: bool = False
    _pending_message: SendMessage | None = None
    _closed: bool = False

    async def __aenter__(self) -> LettaSession:
        await self.ready()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def ready(self) -> LettaSession:
        self._ensure_open()
        return self

    async def send(self, message: SendMessage) -> None:
        self._ensure_open()
        if self._pending_message is not None:
            raise RuntimeError("A message is already pending; stream it before sending another.")
        self._pending_message = message

    async def stream(self) -> AsyncIterator[SDKMessage]:
        self._ensure_open()
        if self._pending_message is None:
            raise RuntimeError("No pending message. Call send() before stream().")

        payload = self._options.message_payload()
        payload["input"] = _serialize_send_message(self._pending_message)
        payload["streaming"] = True
        if self.state.is_default_conversation and self.state.agent_id is not None:
            payload["agent_id"] = self.state.agent_id

        pending = self._pending_message
        self._pending_message = None

        usage: UsageMessage | None = None
        error: ErrorMessage | None = None
        stop_reason: str | None = None

        try:
            stream = await self._client.conversations.messages.create(
                self.state.conversation_id,
                **payload,
            )
            async for raw_message in stream:
                message = _convert_stream_message(raw_message)
                if isinstance(message, UsageMessage):
                    usage = message
                elif isinstance(message, ErrorMessage):
                    error = message
                elif message.type == "stop_reason":
                    stop_reason = getattr(raw_message, "stop_reason", None)
                yield message
        except Exception:
            self._pending_message = pending
            raise

        yield ResultMessage(
            type="result",
            raw={"stop_reason": stop_reason},
            success=error is None,
            stop_reason=stop_reason,
            conversation_id=None if self.state.is_default_conversation else self.state.conversation_id,
            usage=usage,
            error=error,
        )

    async def list_messages(self, **kwargs: Any) -> Any:
        self._ensure_open()
        if self.state.is_default_conversation and self.state.agent_id is not None:
            kwargs.setdefault("agent_id", self.state.agent_id)
        return await self._client.conversations.messages.list(self.state.conversation_id, **kwargs)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            maybe_close = getattr(self._client, "close", None)
            if maybe_close is not None:
                result = maybe_close()
                if hasattr(result, "__await__"):
                    await result

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Session is closed")


def _serialize_send_message(message: SendMessage) -> Any:
    if isinstance(message, str):
        return message
    serialized: list[dict[str, Any]] = []
    for part in message:
        serialized.append(dict(part))
    return serialized


def _convert_stream_message(raw_message: Any) -> SDKMessage:
    message_type = getattr(raw_message, "message_type", None) or getattr(raw_message, "type", None)
    if message_type == "assistant_message":
        return AssistantMessage(
            type="assistant",
            raw=raw_message,
            content=_coerce_message_text(getattr(raw_message, "content", "")),
            message_id=getattr(raw_message, "id", None),
            created_at=getattr(raw_message, "date", None),
        )
    if message_type == "reasoning_message":
        return ReasoningMessage(
            type="reasoning",
            raw=raw_message,
            reasoning=getattr(raw_message, "reasoning", ""),
            message_id=getattr(raw_message, "id", None),
            created_at=getattr(raw_message, "date", None),
        )
    if message_type == "tool_call_message":
        tool_call = getattr(raw_message, "tool_call", None)
        return ToolCallMessage(
            type="tool_call",
            raw=raw_message,
            tool_name=getattr(tool_call, "name", None),
            arguments=getattr(tool_call, "arguments", None),
            tool_call_id=getattr(tool_call, "id", None),
            message_id=getattr(raw_message, "id", None),
        )
    if message_type == "tool_return_message":
        return ToolResultMessage(
            type="tool_result",
            raw=raw_message,
            status=getattr(raw_message, "status", None),
            tool_call_id=getattr(raw_message, "tool_call_id", None),
            value=getattr(raw_message, "tool_return", None),
            stdout=list(getattr(raw_message, "stdout", None) or []),
            stderr=list(getattr(raw_message, "stderr", None) or []),
            message_id=getattr(raw_message, "id", None),
        )
    if message_type == "usage_statistics":
        return UsageMessage(
            type="usage",
            raw=raw_message,
            prompt_tokens=getattr(raw_message, "prompt_tokens", None),
            completion_tokens=getattr(raw_message, "completion_tokens", None),
            total_tokens=getattr(raw_message, "total_tokens", None),
            step_count=getattr(raw_message, "step_count", None),
            run_ids=list(getattr(raw_message, "run_ids", None) or []),
        )
    if message_type == "error_message":
        return ErrorMessage(
            type="error",
            raw=raw_message,
            error_type=getattr(raw_message, "error_type", "error"),
            message=getattr(raw_message, "message", "Unknown error"),
            detail=getattr(raw_message, "detail", None),
            run_id=getattr(raw_message, "run_id", None),
        )
    if message_type == "ping":
        return PingMessage(
            type="ping",
            raw=raw_message,
            message_id=getattr(raw_message, "id", None),
            created_at=getattr(raw_message, "date", None),
        )
    if message_type == "stop_reason":
        return SDKMessage(type="stop_reason", raw=raw_message)
    return UnknownMessage(type="unknown", raw=raw_message, message_type=message_type)


def _coerce_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is None and isinstance(item, dict):
                text = item.get("text")
            if text:
                parts.append(text)
        return "".join(parts)
    return str(content or "")
