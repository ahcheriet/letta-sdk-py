"""Stream-event text-delta extraction (port of the TypeScript SDK's
``stream-events.ts``).

Extracts appendable assistant/reasoning text slices from raw
``stream_event`` payloads, which come in two shapes in headless mode:

1. content_block style: ``{"type": ..., "delta": {"text" | "reasoning": ...}}``
2. message chunk style: ``{"message_type": "assistant_message" |
   "reasoning_message", ...}``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(slots=True, frozen=True)
class StreamTextDelta:
    """An appendable assistant or reasoning text slice from a stream event."""

    kind: Literal["assistant", "reasoning"]
    text: str


def _extract_text_from_content(content: Any) -> str | None:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
                continue
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                pieces.append(part["text"])
        joined = "".join(pieces)
        return joined if joined else None

    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text

    return None


def extract_stream_text_delta(event: Any) -> StreamTextDelta | None:
    """Extract appendable assistant/reasoning text from a stream_event payload.

    Returns ``None`` when the payload is not a dict or carries no
    appendable text (TS parity).
    """
    if not isinstance(event, dict):
        return None
    delta = event.get("delta")
    if isinstance(delta, dict):
        reasoning = delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            return StreamTextDelta(kind="reasoning", text=reasoning)
        text = delta.get("text")
        if isinstance(text, str) and text:
            return StreamTextDelta(kind="assistant", text=text)

    message_type = event.get("message_type")
    if message_type == "reasoning_message":
        reasoning = event.get("reasoning")
        if not isinstance(reasoning, str):
            reasoning = _extract_text_from_content(event.get("content"))
        if isinstance(reasoning, str) and reasoning:
            return StreamTextDelta(kind="reasoning", text=reasoning)

    if message_type == "assistant_message":
        text = _extract_text_from_content(event.get("content"))
        if text:
            return StreamTextDelta(kind="assistant", text=text)

    return None
