"""Agent-free ephemeral queries.

:meth:`LettaAgentClient.query` returns a :class:`QueryStream`: one-shot,
single-consumption async iteration over the SDK messages of a new
agent-free conversation (``runtime_start`` + ``create_conversation``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .session import LettaSession
from .types import QueryOptions, SDKMessage, SendMessage


@dataclass(slots=True)
class QueryStream:
    """A one-shot agent-free query stream.

    Consume exactly once::

        async for message in client.query("Hello", options):
            ...
    """

    client: Any
    prompt: SendMessage
    options: QueryOptions
    on_close: Any | None = None
    _session: LettaSession | None = field(default=None, repr=False)
    _started: bool = field(default=False, repr=False)
    _closed: bool = field(default=False, repr=False)

    def __aiter__(self) -> AsyncIterator[SDKMessage]:
        return self._run()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._session is not None:
            try:
                await self._session.close()
            finally:
                self._session = None
        if self.on_close is not None:
            await self.on_close()

    async def _run(self) -> AsyncIterator[SDKMessage]:
        if self._started:
            raise RuntimeError("Query streams can only be consumed once.")
        self._started = True
        try:
            options = self.options
            if not options.model:
                raise ValueError("query() requires QueryOptions.model.")
            if not options.system:
                raise ValueError("query() requires QueryOptions.system.")
            session = self.client._new_session(
                create_conversation_body=options.conversation_body(),
                options=None,
            )
            self._session = session
            await session.ready()
            await session.send(self.prompt)
            async for message in session.stream():
                yield message
        finally:
            await self.close()
