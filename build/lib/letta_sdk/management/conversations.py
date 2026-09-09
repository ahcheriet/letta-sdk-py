from __future__ import annotations

from typing import Any


class ConversationsManager:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def list(self, **kwargs: Any) -> Any:
        return await self._client.conversations.list(**kwargs)

    async def retrieve(self, conversation_id: str) -> Any:
        return await self._client.conversations.retrieve(conversation_id)

    async def create(self, **kwargs: Any) -> Any:
        return await self._client.conversations.create(**kwargs)

    async def update(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self._client.conversations.update(conversation_id, **kwargs)

    async def delete(self, conversation_id: str) -> None:
        await self._client.conversations.delete(conversation_id)

    async def list_messages(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self._client.conversations.messages.list(conversation_id, **kwargs)
