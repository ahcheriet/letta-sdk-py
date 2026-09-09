"""High-level client for the Letta Code app server (agent SDK v2 protocol).

:class:`LettaAgentClient` is the Python counterpart of the TypeScript
``@letta-ai/letta-agent-sdk`` client. It speaks the app-server
JSON-over-WebSocket protocol (not the REST API):

* each **session** owns (or borrows) one websocket connection and starts a
  runtime with ``runtime_start``;
* **management** clients (``agents`` / ``conversations`` / ``models``) share
  one lazily created pooled connection.

Typical usage::

    async with LettaAgentClient() as client:      # endpoint/token from env
        agent_id = await client.create_agent(
            CreateAgentOptions(name="demo", model="<provider>/<model>",
                               system_prompt="You are a helpful assistant.")
        )
        session = client.resume_session(agent_id)
        await session.send("Hello!")
        async for message in session.stream():
            print(message)
            if isinstance(message, ResultMessage):
                break
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator

from .app_server import AppServerConnection, AppServerConnectionLike, load_token_file
from .management import AgentsManager, ConversationsManager, ModelsManager
from .query import QueryStream
from .session import LettaSession
from .types import (
    Backend,
    CreateAgentOptions,
    CreateSessionOptions,
    DEFAULT_APP_SERVER_URL,
    QueryOptions,
    ResultMessage,
    SDKMessage,
    SendMessage,
)


class LettaAgentClient:
    """Client for a Letta Code app server (websocket protocol).

    Parameters
    ----------
    backend:
        ``local`` / ``remote`` / ``app-server`` all use the websocket
        transport. ``cloud`` is accepted for forward compatibility and not
        implemented.
    url / app_server_url:
        Websocket endpoint (default ``ws://127.0.0.1:4500/ws``).
    auth_token / api_key:
        Capability token sent as ``Authorization: Bearer <token>``.
    token_file:
        Path to a file containing the capability token (used when no token
        is passed directly).
    request_timeout:
        Default per-request timeout in seconds.
    connect_timeout:
        WebSocket connect timeout in seconds.
    connection:
        Pre-built :class:`AppServerConnection` to reuse (sessions and
        management share it; the client does not close it).
    connect_factory:
        Test seam passed through to :class:`AppServerConnection`.
    """

    def __init__(
        self,
        *,
        backend: Backend | str = Backend.LOCAL,
        url: str | None = None,
        app_server_url: str | None = None,
        auth_token: str | None = None,
        api_key: str | None = None,
        token_file: str | Path | None = None,
        request_timeout: float | None = None,
        connect_timeout: float = 15.0,
        connection: AppServerConnectionLike | None = None,
        connect_factory: Any | None = None,
    ) -> None:
        backend = Backend(backend)
        if backend is Backend.CLOUD:
            raise ValueError(
                "The 'cloud' backend is not supported by this SDK yet; use "
                "'local', 'remote', or 'app-server' with a websocket endpoint."
            )
        self.backend = backend
        self.url = url or app_server_url or DEFAULT_APP_SERVER_URL
        self._request_timeout = request_timeout
        self._connect_timeout = connect_timeout
        self._connect_factory = connect_factory

        token = auth_token if auth_token is not None else api_key
        if token is None and token_file is not None:
            token = load_token_file(token_file)
        self._auth_token = token

        self._injected = connection
        self._owns_injected = False  # user-provided connections are user-owned
        self._management_connection: AppServerConnectionLike | None = None
        self._closed = False

        provider = self._management_connection_provider
        self.agents = AgentsManager(provider)
        self.conversations = ConversationsManager(provider)
        self.models = ModelsManager(provider)

    # ── lifecycle ────────────────────────────────────────────────

    async def __aenter__(self) -> LettaAgentClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        pool = self._management_connection
        self._management_connection = None
        if pool is not None:
            await pool.close()

    @property
    def closed(self) -> bool:
        return self._closed

    # ── connections ──────────────────────────────────────────────

    def _management_connection_provider(self) -> AppServerConnectionLike:
        if self._closed:
            raise RuntimeError("Client is closed")
        if self._injected is not None:
            return self._injected
        if self._management_connection is None:
            self._management_connection = self._new_connection()
        return self._management_connection

    def _new_connection(self) -> AppServerConnectionLike:
        if self._injected is not None:
            return self._injected
        return AppServerConnection(
            self.url,
            auth_token=self._auth_token,
            request_timeout=self._request_timeout,
            connect_timeout=self._connect_timeout,
            connect_factory=self._connect_factory,
        )

    def _new_session(
        self,
        *,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        create_agent_body: dict[str, Any] | None = None,
        create_conversation_body: dict[str, Any] | None = None,
        new_conversation: bool = False,
        default_conversation: bool = False,
        options: CreateSessionOptions | None = None,
    ) -> LettaSession:
        if self._closed:
            raise RuntimeError("Client is closed")
        return LettaSession(
            connection=self._new_connection(),
            owns_connection=self._injected is None,
            agent_id=agent_id,
            conversation_id=conversation_id,
            create_agent_body=create_agent_body,
            create_conversation_body=create_conversation_body,
            new_conversation=new_conversation,
            default_conversation=default_conversation,
            options=options,
        )

    # ── agents & sessions ────────────────────────────────────────

    async def create_agent(self, options: CreateAgentOptions | None = None) -> str:
        """Create an agent (``runtime_start`` + ``create_agent``) and return
        its id."""
        body = (options or CreateAgentOptions()).to_body()
        session = self._new_session(create_agent_body=body)
        try:
            init = await session.ready()
            if not init.agent_id:
                raise RuntimeError(
                    "App server agent creation did not return an agent id."
                )
            return init.agent_id
        finally:
            await session.close()

    def create_session(
        self,
        agent_id: str,
        options: CreateSessionOptions | None = None,
    ) -> LettaSession:
        """Open a session on a new conversation for ``agent_id``.

        Initialization (``runtime_start``) is lazy: it happens on the first
        ``send()`` / ``ready()`` / ``stream()`` call.
        """
        return self._new_session(agent_id=agent_id, options=options)

    def resume_session(
        self,
        identifier: str,
        options: CreateSessionOptions | None = None,
    ) -> LettaSession:
        """Open a session resuming an agent (and its default conversation)
        or a conversation (``conv-...`` id)."""
        if identifier.startswith(("conv-", "conversation-", "local-conv-")):
            return self._new_session(conversation_id=identifier, options=options)
        return self._new_session(
            agent_id=identifier, default_conversation=True, options=options
        )

    # ── one-shot helpers ─────────────────────────────────────────

    def query(self, prompt: SendMessage, options: QueryOptions | None = None) -> QueryStream:
        """Agent-free ephemeral query (new conversation, no persistent
        agent)."""
        return QueryStream(self, prompt, options or QueryOptions())

    async def prompt(
        self,
        agent_id: str,
        message: SendMessage,
        options: CreateSessionOptions | None = None,
    ) -> ResultMessage:
        """One-shot turn: open a session, send, return the terminal result."""
        session = self.create_session(agent_id, options)
        try:
            return await session.prompt(message)
        finally:
            await session.close()

    async def stream_prompt(
        self,
        agent_id: str,
        message: SendMessage,
        options: CreateSessionOptions | None = None,
    ) -> AsyncIterator[SDKMessage]:
        """One-shot turn exposing the full stream (ending with a result)."""
        session = self.create_session(agent_id, options)
        try:
            await session.send(message)
            async for sdk_message in session.stream():
                yield sdk_message
        finally:
            await session.close()
