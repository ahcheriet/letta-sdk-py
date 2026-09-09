from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .session import LettaSession
from .types import CreateAgentOptions, QueryOptions, SDKMessage


@dataclass(slots=True)
class QueryStream:
    client: Any
    prompt: Any
    options: QueryOptions
    _agent_id: str | None = None
    _session: LettaSession | None = None
    _started: bool = False
    _closed: bool = False

    def __aiter__(self) -> AsyncIterator[SDKMessage]:
        return self._run()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._session is not None:
            await self._session.close()
        if self._agent_id is not None:
            await self.client.agents.delete(self._agent_id)

    async def _run(self) -> AsyncIterator[SDKMessage]:
        if self._started:
            raise RuntimeError("Query streams can only be consumed once.")
        self._started = True
        try:
            self._agent_id = await self.client.create_agent(
                CreateAgentOptions(
                    model=self.options.model,
                    system_prompt=self.options.system_prompt,
                    hidden=self.options.hidden,
                    tags=list(self.options.tags),
                )
            )
            self._session = await self.client.create_session(self._agent_id)
            await self._session.send(self.prompt)
            async for message in self._session.stream():
                yield message
        finally:
            await self.close()
