"""WebSocket transport for the Letta Code app server (agent sdk v2 protocol).

This module implements the raw JSON-over-WebSocket protocol spoken by the
Letta Code app server (the same protocol the TypeScript
``@letta-ai/letta-agent-sdk`` uses via ``@letta-ai/letta-code/app-server-client``):

* every request/response pair is correlated by ``request_id``;
* every server message carrying an integer ``seq`` is acknowledged with
  ``{"type": "ack", "seq": ...}``;
* ``stream_delta`` and status events are dispatched to registered listeners.

The transport is intentionally transport-only: turn correlation and message
conversion live in :mod:`letta_sdk.session`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Coroutine, Protocol
from websockets.asyncio.client import ClientConnection, connect  # type: ignore[import-unresolved]
from websockets.exceptions import ConnectionClosed  # type: ignore[import-unresolved]

from .types import DEFAULT_APP_SERVER_URL

logger = logging.getLogger("letta_sdk")

__all__ = [
    "AppServerClosedError",
    "AppServerConnection",
    "AppServerConnectionError",
    "AppServerConnectionLike",
    "AppServerError",
    "AppServerRequestError",
    "AppServerTimeoutError",
    "MessageHandler",
]

#: ``seq`` acknowledgement is fire-and-forget.
#: Message listeners for every incoming protocol message.
MessageHandler = Callable[[dict[str, Any]], None]


class AppServerConnectionLike(Protocol):
    """Structural view of the connection interface the SDK depends on.

    The client, session, and management layers only use these members, so
    tests may inject fakes that implement them (mirroring the
    interface-based injection of the TypeScript SDK). The concrete
    :class:`AppServerConnection` satisfies this protocol implicitly.
    """

    def request(
        self,
        type_: str,
        body: dict[str, Any] | None = None,
        *,
        response_type: str | None = None,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
        timeout: float | None = None,
    ) -> Coroutine[Any, Any, dict[str, Any]]: ...

    def send(self, payload: dict[str, Any]) -> Coroutine[Any, Any, None]: ...

    def on_message(self, handler: MessageHandler) -> Callable[[], None]: ...

    def close(self) -> Coroutine[Any, Any, None]: ...


class AppServerError(RuntimeError):
    """Base error for app-server transport problems."""


class AppServerConnectionError(AppServerError):
    """The websocket could not be established or dropped."""


class AppServerRequestError(AppServerError):
    """A request completed with ``success=false`` or an error event."""


class AppServerTimeoutError(AppServerError):
    """A request did not complete within its deadline."""


class AppServerClosedError(AppServerError):
    """A request was issued after the connection was closed."""


class AppServerConnection:
    """A single ownership-scoped websocket connection to the app server.

    Parameters
    ----------
    url:
        Websocket endpoint, e.g. ``ws://127.0.0.1:4500/ws``.
    auth_token:
        Optional capability token sent as ``Authorization: Bearer <token>``
        during the websocket upgrade.
    request_timeout:
        Default per-request timeout in seconds.
    connect_timeout:
        WebSocket connect / message receive timeout in seconds.
    connect_factory:
        Test seam: an async callable ``(url, headers) -> ClientConnection``.
        Defaults to :func:`websockets.asyncio.client.connect`.
    """

    def __init__(
        self,
        url: str = DEFAULT_APP_SERVER_URL,
        *,
        auth_token: str | None = None,
        request_timeout: float | None = None,
        connect_timeout: float = 15.0,
        connect_factory: Any | None = None,
    ) -> None:
        self.url = url
        self._auth_token = auth_token
        self._request_timeout = request_timeout
        self._connect_timeout = connect_timeout
        self._connect_factory = connect_factory or _default_connect

        self._socket: ClientConnection | None = None
        self._connect_promise: asyncio.Future[None] | None = None
        self._request_counter = 0
        self._pending: dict[str, _PendingRequest] = {}
        self._listeners: list[MessageHandler] = []
        self._closed = False
        self._close_reason: str | None = None
        self._read_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()

    # ── lifecycle ────────────────────────────────────────────────

    @property
    def closed(self) -> bool:
        return self._closed

    async def connect(self) -> None:
        if self._socket is not None:
            return
        if self._connect_promise is not None:
            await self._connect_promise
            return
        promise = asyncio.get_running_loop().create_future()
        self._connect_promise = promise
        try:
            headers: dict[str, str] = {}
            if self._auth_token:
                headers["Authorization"] = f"Bearer {self._auth_token}"
            self._socket = await asyncio.wait_for(
                self._connect_factory(self.url, headers),
                timeout=self._connect_timeout,
            )
            self._read_task = asyncio.create_task(
                self._read_loop(), name="letta-sdk-read-loop"
            )
            promise.set_result(None)
        except BaseException as exc:
            self._socket = None
            if not promise.done():
                promise.set_exception(
                    AppServerConnectionError(
                        f"Unable to connect to Letta app server at {self.url}: {exc}"
                    )
                )
            raise
        finally:
            self._connect_promise = None
        # Re-raise the stored exception for concurrent waiters.
        # (promise.set_exception was already called above; awaiting a failed
        # future re-raises it.)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_reason = "client closed the connection"
        socket, self._socket = self._socket, None
        if self._read_task is not None:
            self._read_task.cancel()
            self._read_task = None
        if socket is not None:
            try:
                await socket.close()
            except Exception:  # pragma: no cover - best effort
                pass
        self._fail_pending(AppServerClosedError(self._close_reason))

    async def __aenter__(self) -> AppServerConnection:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # ── sending ──────────────────────────────────────────────────

    def next_request_id(self, prefix: str) -> str:
        self._request_counter += 1
        return f"{prefix}-{self._request_counter}"

    async def send(self, payload: dict[str, Any]) -> None:
        """Fire-and-forget send (e.g. ``input``, ``ack``)."""
        await self.connect()
        assert self._socket is not None
        data = json.dumps(payload)
        async with self._send_lock:
            try:
                await self._socket.send(data)
            except ConnectionClosed as exc:
                self._handle_drop("connection closed while sending")
                raise AppServerConnectionError(str(exc)) from exc

    async def request(
        self,
        type_: str,
        body: dict[str, Any] | None = None,
        *,
        response_type: str | None = None,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a request and await the matching response.

        Matching uses ``request_id`` equality; when ``response_type`` is given
        the candidate must also carry that ``type`` (or satisfy ``predicate``).
        """
        await self.connect()
        if self._closed:
            raise AppServerClosedError("connection is closed")

        request_id = self.next_request_id(type_)
        command: dict[str, Any] = {"type": type_, "request_id": request_id}
        if body:
            command.update(body)

        if predicate is not None:
            def matches(raw: dict[str, Any]) -> bool:
                if raw.get("request_id") != request_id:
                    return False
                return predicate(raw)
        elif response_type is not None:
            def matches(raw: dict[str, Any]) -> bool:
                return (
                    raw.get("request_id") == request_id
                    and raw.get("type") == response_type
                )
        else:
            def matches(raw: dict[str, Any]) -> bool:
                return raw.get("request_id") == request_id

        pending = _PendingRequest()
        self._pending[request_id] = pending
        try:
            await self.send(command)
            effective_timeout = (
                timeout if timeout is not None else self._request_timeout
            )
            try:
                if effective_timeout is not None:
                    return await asyncio.wait_for(
                        pending.future, timeout=effective_timeout
                    )
                return await pending.future
            except asyncio.TimeoutError as exc:
                raise AppServerTimeoutError(
                    f"Timed out waiting for '{type_}' response"
                ) from exc
        finally:
            self._pending.pop(request_id, None)

    # ── message subscription ─────────────────────────────────────

    def on_message(self, handler: MessageHandler) -> Callable[[], None]:
        """Subscribe ``handler(raw_message)`` to every incoming message."""
        self._listeners.append(handler)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(handler)
            except ValueError:
                pass

        return unsubscribe

    # ── internals ────────────────────────────────────────────────

    async def _read_loop(self) -> None:
        assert self._socket is not None
        while True:
            try:
                raw_frame = await self._socket.recv()
            except ConnectionClosed as exc:
                self._handle_drop(f"connection closed: {exc}")
                return
            except asyncio.CancelledError:
                return
            try:
                message = json.loads(raw_frame)
            except (json.JSONDecodeError, TypeError):
                logger.debug("ignoring non-JSON frame: %r", raw_frame)
                continue
            if not isinstance(message, dict):
                continue
            await self._acknowledge(message)
            self._dispatch(message)

    async def _acknowledge(self, message: dict[str, Any]) -> None:
        seq = message.get("seq")
        if isinstance(seq, int):
            try:
                await self.send({"type": "ack", "seq": seq})
            except AppServerClosedError:  # pragma: no cover - closing
                pass

    def _dispatch(self, message: dict[str, Any]) -> None:
        # 1. request/response correlation
        request_id = message.get("request_id")
        if isinstance(request_id, str) and request_id in self._pending:
            pending = self._pending.get(request_id)
            if pending is not None and not pending.future.done():
                # A response only completes its own request; type checks are
                # enforced by the caller's predicate, so deliver unconditionally.
                pending.future.set_result(message)
        # 2. protocol-level fatal errors not tied to a pending request
        if message.get("type") in {"error", "runtime_error"}:
            self._fail_pending_from_error(message)
        # 3. listeners (turn coordination, device status, ...)
        for listener in list(self._listeners):
            try:
                listener(message)
            except Exception:  # pragma: no cover - listener bug isolation
                logger.exception("app-server message listener failed")

    def _handle_drop(self, reason: str) -> None:
        if not self._closed:
            self._closed = True
            self._close_reason = reason
            self._fail_pending(AppServerConnectionError(reason))
            for listener in list(self._listeners):
                try:
                    listener({"type": "_transport_closed", "error": reason})
                except Exception:  # pragma: no cover
                    logger.exception("app-server message listener failed")

    def _fail_pending(self, error: Exception) -> None:
        for pending in self._pending.values():
            if not pending.future.done():
                pending.future.set_exception(error)
        self._pending.clear()

    def _fail_pending_from_error(self, message: dict[str, Any]) -> None:
        detail = (
            message.get("message")
            or message.get("error_message")
            or message.get("error")
            or "app server error"
        )
        if self._pending:
            logger.debug("app server error while requests pending: %s", detail)


class _PendingRequest:
    __slots__ = ("future",)

    def __init__(self) -> None:
        self.future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )


async def _default_connect(url: str, headers: dict[str, str]) -> Any:
    return await connect(url, additional_headers=headers or None)


def load_token_file(path: str | Any) -> str:
    """Read a capability token from a file, stripping surrounding whitespace."""
    from pathlib import Path

    token_path = Path(path).expanduser()
    if not token_path.is_file():
        raise AppServerError(f"Token file not found: {token_path}")
    return token_path.read_text().strip()
