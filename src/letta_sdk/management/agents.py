"""Agent management over the app-server control protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..app_server import AppServerConnectionLike, AppServerRequestError
from ..types import CreateAgentOptions

ClientInfo = {
    "name": "letta-sdk-py",
    "title": "Letta Agent SDK (Python)",
}


def _require(value: Any, what: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"A non-empty {what} is required.")


def _expect(response: dict[str, Any], key: str, fallback: str) -> Any:
    if not response.get("success", True) or response.get(key) is None:
        raise AppServerRequestError(str(response.get("error")) or fallback)
    return response[key]


class AgentsManager:
    def __init__(self, connection_provider: Callable[[], AppServerConnectionLike]) -> None:
        self._connections = connection_provider

    async def create(
        self,
        options: CreateAgentOptions | None = None,
        **body: Any,
    ) -> dict[str, Any]:
        """Create an agent (``runtime_start`` + ``create_agent``) and return
        the agent record."""
        if body:
            options = CreateAgentOptions(**options.__dict__ if options else {}, **body)  # type: ignore[arg-type]
        agent_body = (options or CreateAgentOptions()).to_body()
        connection = self._connections()
        response = await connection.request(
            "runtime_start",
            {
                "client_info": ClientInfo,
                "recover_approvals": False,
                "force_device_status": True,
                "create_agent": {
                    "body": agent_body,
                    "pin_global": not agent_body.get("hidden"),
                },
            },
            response_type="runtime_start_response",
        )
        if not response.get("success", True) or not isinstance(
            response.get("runtime"), dict
        ):
            raise AppServerRequestError(
                str(response.get("error") or "Failed to create agent.")
            )
        agent = response.get("agent")
        if isinstance(agent, dict):
            return agent
        return {"id": response["runtime"].get("agent_id")}

    async def list(self, **query: Any) -> list[dict[str, Any]]:
        response = await self._connections().request(
            "agent_list", {"query": query}, response_type="agent_list_response"
        )
        if not response.get("success", True):
            raise AppServerRequestError(
                str(response.get("error") or "Failed to list agents.")
            )
        agents = response.get("agents")
        return list(agents) if isinstance(agents, list) else []

    async def retrieve(self, agent_id: str) -> dict[str, Any]:
        _require(agent_id, "agent id")
        response = await self._connections().request(
            "agent_retrieve",
            {"agent_id": agent_id},
            response_type="agent_retrieve_response",
        )
        agent = _expect(
            response, "agent", f"Failed to retrieve agent {agent_id}."
        )
        return agent if isinstance(agent, dict) else {"id": agent}

    async def update(self, agent_id: str, **body: Any) -> dict[str, Any]:
        _require(agent_id, "agent id")
        response = await self._connections().request(
            "agent_update",
            {"agent_id": agent_id, "body": body},
            response_type="agent_update_response",
        )
        agent = _expect(
            response, "agent", f"Failed to update agent {agent_id}."
        )
        return agent if isinstance(agent, dict) else {"id": agent}

    async def delete(self, agent_id: str) -> None:
        _require(agent_id, "agent id")
        response = await self._connections().request(
            "agent_delete",
            {"agent_id": agent_id},
            response_type="agent_delete_response",
        )
        if not response.get("success", True):
            raise AppServerRequestError(
                str(response.get("error") or f"Failed to delete agent {agent_id}.")
            )
