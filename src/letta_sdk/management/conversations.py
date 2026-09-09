"""Conversation management over the app-server control protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..app_server import AppServerConnectionLike, AppServerRequestError
from ..types import ListMessagesResult


def _require(value: Any, what: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"A non-empty {what} is required.")


def _expect(response: dict[str, Any], key: str, fallback: str) -> Any:
    if not response.get("success", True) or response.get(key) is None:
        raise AppServerRequestError(str(response.get("error")) or fallback)
    return response[key]


class ConversationsManager:
    def __init__(self, connection_provider: Callable[[], AppServerConnectionLike]) -> None:
        self._connections = connection_provider

    async def list(self, **query: Any) -> list[dict[str, Any]]:
        response = await self._connections().request(
            "conversation_list", {"query": query},
            response_type="conversation_list_response",
        )
        if not response.get("success", True):
            raise AppServerRequestError(
                str(response.get("error") or "Failed to list conversations.")
            )
        conversations = response.get("conversations")
        return list(conversations) if isinstance(conversations, list) else []

    async def retrieve(self, conversation_id: str) -> dict[str, Any]:
        _require(conversation_id, "conversation id")
        response = await self._connections().request(
            "conversation_retrieve",
            {"conversation_id": conversation_id},
            response_type="conversation_retrieve_response",
        )
        conversation = _expect(
            response,
            "conversation",
            f"Failed to retrieve conversation {conversation_id}.",
        )
        return conversation if isinstance(conversation, dict) else {"id": conversation}

    async def create(self, **body: Any) -> dict[str, Any]:
        response = await self._connections().request(
            "conversation_create", {"body": body},
            response_type="conversation_create_response",
        )
        conversation = _expect(response, "conversation", "Failed to create conversation.")
        return conversation if isinstance(conversation, dict) else {"id": conversation}

    async def update(self, conversation_id: str, **body: Any) -> dict[str, Any]:
        _require(conversation_id, "conversation id")
        response = await self._connections().request(
            "conversation_update",
            {"conversation_id": conversation_id, "body": body},
            response_type="conversation_update_response",
        )
        conversation = _expect(
            response,
            "conversation",
            f"Failed to update conversation {conversation_id}.",
        )
        return conversation if isinstance(conversation, dict) else {"id": conversation}

    async def fork(self, conversation_id: str, **body: Any) -> dict[str, Any]:
        _require(conversation_id, "conversation id")
        response = await self._connections().request(
            "conversation_fork",
            {"conversation_id": conversation_id, "body": body},
            response_type="conversation_fork_response",
        )
        conversation = _expect(
            response,
            "conversation",
            f"Failed to fork conversation {conversation_id}.",
        )
        return conversation if isinstance(conversation, dict) else {"id": conversation}

    async def list_messages(
        self, conversation_id: str, **query: Any
    ) -> ListMessagesResult:
        _require(conversation_id, "conversation id")
        response = await self._connections().request(
            "conversation_messages_list",
            {"conversation_id": conversation_id, **({"query": query} if query else {})},
            response_type="conversation_messages_list_response",
        )
        if not response.get("success", True):
            raise AppServerRequestError(
                str(response.get("error") or "Failed to list conversation messages.")
            )
        result = ListMessagesResult(
            messages=list(response.get("messages") or [])
        )
        for key in ("nextBefore", "next_before"):
            value = response.get(key)
            if isinstance(value, str) or value is None:
                result.next_before = value
                break
        for key in ("hasMore", "has_more"):
            if isinstance(response.get(key), bool):
                result.has_more = response[key]
                break
        return result

    async def delete(self, conversation_id: str) -> None:
        # The app-server control protocol (v2) exposes no conversation_delete
        # command; deletion is an agent-lifecycle concern (agent_delete).
        raise AppServerRequestError(
            "conversation delete is not supported by the app-server protocol; "
            "delete the owning agent instead (agents.delete)."
        )
