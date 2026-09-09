from __future__ import annotations

from typing import Any


class ModelsManager:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def list(self, **kwargs: Any) -> Any:
        return await self._client.models.list(**kwargs)
