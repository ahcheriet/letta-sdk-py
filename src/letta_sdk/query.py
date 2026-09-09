from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .session import LettaSession
from .types import CreateSessionOptions, QueryOptions, SDKMessage


@dataclass(slots=True)
class QueryStream:
    client: Any
    prompt: Any
    options: QueryOptions
    on_close: Any | None = None
    _session: LettaSession | None = None
    _started: bool = False
    _closed: bool = False

    def __aiter__(self) -> AsyncIterator[SDKMessage]:
        return self._run()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._session is not None:
                await self._session.close()
        finally:
            if self.on_close is not None:
                await self.on_close()

    async def _run(self) -> AsyncIterator[SDKMessage]:
        if self._started:
            raise RuntimeError("Query streams can only be consumed once.")
        self._started = True
        try:
            conversation_id = await self.client.create_ephemeral_conversation(self.options)
            session_options = CreateSessionOptions(
                max_steps=self.options.max_steps,
                stream_tokens=self.options.stream_tokens,
                include_pings=self.options.include_pings,
                extra_body=dict(self.options.extra_body),
            )
            self._session = self.client.resume_session(conversation_id, session_options)
            await self._session.send(self.prompt)
            async for message in self._session.stream():
                yield message
        finally:
            await self.close()
