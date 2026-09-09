from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypedDict, TypeAlias


class Backend(StrEnum):
    CLOUD = "cloud"
    LOCAL = "local"
    REMOTE = "remote"


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


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
SendMessage: TypeAlias = str | list[MessageContentPart]


@dataclass(slots=True)
class CreateAgentOptions:
    name: str | None = None
    description: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    hidden: bool | None = None
    tags: list[str] = field(default_factory=list)
    extra_body: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = dict(self.extra_body)
        if self.name is not None:
            payload["name"] = self.name
        if self.description is not None:
            payload["description"] = self.description
        if self.model is not None:
            payload["model"] = self.model
        if self.system_prompt is not None:
            payload["system"] = self.system_prompt
        if self.hidden is not None:
            payload["hidden"] = self.hidden
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload


@dataclass(slots=True)
class CreateSessionOptions:
    model: str | None = None
    summary: str | None = None
    description: str | None = None
    hidden: bool | None = None
    max_steps: int | None = None
    stream_tokens: bool | None = None
    include_pings: bool | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    def create_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.model is not None:
            payload["model"] = self.model
        if self.summary is not None:
            payload["summary"] = self.summary
        if self.description is not None:
            payload["description"] = self.description
        if self.hidden is not None:
            payload["hidden"] = self.hidden
        return payload

    def message_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = dict(self.extra_body)
        if self.model is not None:
            payload["override_model"] = self.model
        if self.max_steps is not None:
            payload["max_steps"] = self.max_steps
        if self.stream_tokens is not None:
            payload["stream_tokens"] = self.stream_tokens
        if self.include_pings is not None:
            payload["include_pings"] = self.include_pings
        return payload


@dataclass(slots=True)
class QueryOptions:
    model: str | None = None
    system_prompt: str | None = None
    hidden: bool = True
    tags: list[str] = field(default_factory=lambda: ["ephemeral"])


@dataclass(slots=True)
class SDKMessage:
    type: str
    raw: Any


@dataclass(slots=True)
class AssistantMessage(SDKMessage):
    content: str
    message_id: str | None = None
    created_at: datetime | None = None


@dataclass(slots=True)
class ReasoningMessage(SDKMessage):
    reasoning: str
    message_id: str | None = None
    created_at: datetime | None = None


@dataclass(slots=True)
class ToolCallMessage(SDKMessage):
    tool_name: str | None
    arguments: str | dict[str, Any] | None
    tool_call_id: str | None
    message_id: str | None = None


@dataclass(slots=True)
class ToolResultMessage(SDKMessage):
    status: str | None
    tool_call_id: str | None
    value: str | list[Any] | None
    stdout: list[str] = field(default_factory=list)
    stderr: list[str] = field(default_factory=list)
    message_id: str | None = None


@dataclass(slots=True)
class UsageMessage(SDKMessage):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    step_count: int | None = None
    run_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PingMessage(SDKMessage):
    message_id: str | None = None
    created_at: datetime | None = None


@dataclass(slots=True)
class ErrorMessage(SDKMessage):
    error_type: str
    message: str
    detail: str | None = None
    run_id: str | None = None


@dataclass(slots=True)
class ResultMessage(SDKMessage):
    success: bool
    stop_reason: str | None = None
    conversation_id: str | None = None
    usage: UsageMessage | None = None
    error: ErrorMessage | None = None


@dataclass(slots=True)
class UnknownMessage(SDKMessage):
    message_type: str | None = None


@dataclass(slots=True)
class SessionState:
    conversation_id: str
    agent_id: str | None = None
    is_default_conversation: bool = False


def text_content(text: str) -> TextContentPart:
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
