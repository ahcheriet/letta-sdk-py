from __future__ import annotations

from typing import Any


class AgentsManager:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def create(self, **kwargs: Any) -> Any:
        return await self._client.agents.create(**kwargs)

    async def list(self, **kwargs: Any) -> Any:
        return await self._client.agents.list(**kwargs)

    async def retrieve(self, agent_id: str) -> Any:
        return await self._client.agents.retrieve(agent_id)

    async def update(self, agent_id: str, **kwargs: Any) -> Any:
        return await self._client.agents.update(agent_id, **kwargs)

    async def delete(self, agent_id: str) -> None:
        await self._client.agents.delete(agent_id)
