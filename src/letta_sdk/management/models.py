"""Model management over the app-server control protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..app_server import AppServerConnectionLike, AppServerRequestError


class ModelsManager:
    def __init__(self, connection_provider: Callable[[], AppServerConnectionLike]) -> None:
        self._connections = connection_provider

    async def list(self) -> dict[str, Any]:
        """List available models.

        Returns a dict with:

        * ``entries`` -- the model catalog entries;
        * ``available_handles`` -- resolved model handles, when provided;
        * ``byok_provider_aliases`` -- BYOK provider alias mapping, when provided.
        """
        response = await self._connections().request(
            "list_models", {}, response_type="list_models_response"
        )
        if not response.get("success", True):
            raise AppServerRequestError(
                str(response.get("error") or "Failed to list models.")
            )
        return {
            "entries": list(response.get("entries") or []),
            "available_handles": response.get("available_handles"),
            "byok_provider_aliases": response.get("byok_provider_aliases"),
        }
