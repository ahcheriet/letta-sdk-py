from __future__ import annotations

from dataclasses import dataclass, field

from .types import AssistantMessage, SDKMessage


@dataclass(slots=True)
class TranscriptAccumulator:
    messages: list[SDKMessage] = field(default_factory=list)
    assistant_chunks: list[str] = field(default_factory=list)

    def add(self, message: SDKMessage) -> SDKMessage:
        self.messages.append(message)
        if isinstance(message, AssistantMessage) and message.content:
            self.assistant_chunks.append(message.content)
        return message

    @property
    def assistant_text(self) -> str:
        return "".join(self.assistant_chunks)
