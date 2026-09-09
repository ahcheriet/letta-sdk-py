from __future__ import annotations

from typing import Any

from letta_client import AsyncLetta

from .management import AgentsManager, ConversationsManager, ModelsManager
from .query import QueryStream
from .session import LettaSession
from .types import (
    Backend,
    CreateAgentOptions,
    CreateSessionOptions,
    QueryOptions,
    ResultMessage,
    SessionState,
    SendMessage,
)


class LettaAgentClient:
    def __init__(
        self,
        *,
        backend: Backend | str = Backend.CLOUD,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        client: Any | None = None,
    ) -> None:
        self.backend = Backend(backend)
        self._owns_client = client is None
        self._client = client or AsyncLetta(**self._client_kwargs(api_key=api_key, base_url=base_url, timeout=timeout))
        self.agents = AgentsManager(self._client)
        self.conversations = ConversationsManager(self._client)
        self.models = ModelsManager(self._client)

    async def __aenter__(self) -> LettaAgentClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if not self._owns_client:
            return
        result = self._client.close()
        if hasattr(result, "__await__"):
            await result

    async def create_agent(self, options: CreateAgentOptions | None = None) -> str:
        payload = (options or CreateAgentOptions()).to_payload()
        agent = await self._client.agents.create(**payload)
        return agent.id

    async def create_session(
        self,
        agent_id: str,
        options: CreateSessionOptions | None = None,
    ) -> LettaSession:
        session_options = options or CreateSessionOptions()
        conversation = await self._client.conversations.create(
            agent_id=agent_id,
            **session_options.create_payload(),
        )
        return LettaSession(
            _client=self._client,
            state=SessionState(conversation_id=conversation.id, agent_id=agent_id),
            _options=session_options,
        )

    def resume_session(
        self,
        identifier: str,
        options: CreateSessionOptions | None = None,
    ) -> LettaSession:
        session_options = options or CreateSessionOptions()
        if identifier.startswith("conv-"):
            state = SessionState(conversation_id=identifier)
        else:
            state = SessionState(
                conversation_id="default",
                agent_id=identifier,
                is_default_conversation=True,
            )
        return LettaSession(_client=self._client, state=state, _options=session_options)

    def query(self, prompt: Any, options: QueryOptions | None = None) -> QueryStream:
        return QueryStream(client=self, prompt=prompt, options=options or QueryOptions())

    async def prompt(
        self,
        agent_id: str,
        message: SendMessage,
        options: CreateSessionOptions | None = None,
    ) -> ResultMessage:
        session = await self.create_session(agent_id, options)
        try:
            await session.send(message)
            async for streamed_message in session.stream():
                if isinstance(streamed_message, ResultMessage):
                    return streamed_message
            raise RuntimeError("Session stream ended without a result message.")
        finally:
            await session.close()

    async def create_ephemeral_conversation(self, options: QueryOptions) -> str:
        if not options.model:
            raise ValueError("query() requires QueryOptions.model.")
        if not options.system_prompt:
            raise ValueError("query() requires QueryOptions.system_prompt.")
        body: dict[str, Any] = {"model": options.model, "system": options.system_prompt}
        if options.model_settings is not None:
            body["model_settings"] = options.model_settings
        if options.context_window_limit is not None:
            body["context_window_limit"] = options.context_window_limit
        response = await self._client.post("/v1/conversations/ephemeral", body=body)
        if isinstance(response, dict):
            conversation_id = response.get("id")
        else:
            conversation_id = getattr(response, "id", None)
        if not isinstance(conversation_id, str) or not conversation_id:
            raise RuntimeError("Ephemeral conversation did not return a valid id.")
        return conversation_id

    def _client_kwargs(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        timeout: float | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        if timeout is not None:
            kwargs["timeout"] = timeout
        if self.backend == Backend.LOCAL:
            kwargs["environment"] = "local"
        return kwargs
