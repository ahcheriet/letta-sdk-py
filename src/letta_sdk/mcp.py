"""MCP (Model Context Protocol) stdio bridge.

Python port of the TypeScript SDK's ``mcp.ts`` + ``mcp-runtime.ts``:
connect session-scoped MCP servers and expose their tools through the
external-tool protocol. A broken server is reported (via ``log``) but
cannot prevent healthy servers from loading — TS parity.

Only the **stdio** transport is implemented (standard library only:
JSON-RPC 2.0 over the subprocess's stdin/stdout, newline-delimited).
``http`` / ``sse`` server configs are accepted for type parity but are
reported as unavailable at connect time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable

from letta_sdk.types import ToolResult, ToolSpec

logger = logging.getLogger("letta_sdk")

__all__ = [
    "McpToolBridge",
    "connect_mcp_servers",
    "expand_mcp_tool_wildcards",
]

#: MCP protocol version advertised in the initialize handshake.
_MCP_PROTOCOL_VERSION = "2025-06-18"
#: Handshake / tools-list timeout (seconds).
_CONNECT_TIMEOUT = 30.0
#: tools/call timeout (seconds).
_CALL_TIMEOUT = 60.0


# ── server config types ────────────────────────────────────────────

McpStdioServerConfig = dict[str, Any]
McpHttpServerConfig = dict[str, Any]
McpSseServerConfig = dict[str, Any]
#: MCP servers keyed by the name used in ``mcp__<server>__<tool>``.
McpServers = dict[str, dict[str, Any]]

CLIENT_INFO = {"name": "letta-sdk-py", "version": "1"}


# ── stdio JSON-RPC connection ──────────────────────────────────────

class _McpStdioConnection:
    """One MCP server process speaking newline-delimited JSON-RPC 2.0."""

    def __init__(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None,
        cwd: str | None,
    ) -> None:
        self._command = command
        self._args = list(args)
        self._env = env
        self._cwd = cwd
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[int | str, asyncio.Future[Any]] = {}
        self._next_id = 0
        self._write_lock = asyncio.Lock()
        self._closed = False
        self.tools: list[dict[str, Any]] = []

    @property
    def closed(self) -> bool:
        return self._closed

    async def start(self) -> None:
        self._process = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # TS parity: stderr: "inherit"
            stderr=None,
            env=self._env,
            cwd=self._cwd,
        )
        assert self._process.stdin is not None and self._process.stdout is not None
        self._reader_task = asyncio.create_task(
            self._read_loop(), name="mcp-stdio-reader"
        )
        result = await asyncio.wait_for(
            self._request(
                "initialize",
                {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": CLIENT_INFO,
                },
                timeout=_CONNECT_TIMEOUT,
            ),
            timeout=_CONNECT_TIMEOUT + 5,
        )
        if not isinstance(result, dict):
            raise RuntimeError("MCP initialize returned a non-object result")
        await self._notify_initialized()
        listed = await asyncio.wait_for(
            self._request("tools/list", {}, timeout=_CONNECT_TIMEOUT),
            timeout=_CONNECT_TIMEOUT + 5,
        )
        tools = listed.get("tools") if isinstance(listed, dict) else None
        self.tools = [t for t in (tools or []) if isinstance(t, dict)]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.wait_for(
            self._request(
                "tools/call", {"name": name, "arguments": arguments},
                timeout=_CALL_TIMEOUT,
            ),
            timeout=_CALL_TIMEOUT + 5,
        )
        return result if isinstance(result, dict) else {}

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()
        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None
        process = self._process
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except Exception:  # pragma: no cover - best effort
            pass
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await process.wait()
                except Exception:  # pragma: no cover - best effort
                    pass

    # -- wire plumbing

    async def _read_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, dict):
                    continue
                await self._handle_message(message)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            pass

    async def _handle_message(self, message: dict[str, Any]) -> None:
        if "id" in message and ("result" in message or "error" in message):
            request_id = message.get("id")
            if not isinstance(request_id, (int, str)):
                return
            future = self._pending.pop(request_id, None)
            if future is None or future.done():
                return
            if "error" in message:
                error = message.get("error")
                text = error.get("message") if isinstance(error, dict) else None
                future.set_exception(
                    RuntimeError(f"MCP request failed: {text or error}")
                )
            else:
                future.set_result(message.get("result"))
            return
        if "method" in message:
            method = message.get("method")
            if isinstance(method, str) and method.startswith("notifications/"):
                return  # server notification: nothing to answer
            request_id = message.get("id")
            if request_id is None:
                return
            if method == "ping":
                await self._send_message(
                    {"jsonrpc": "2.0", "id": request_id, "result": {}}
                )
            else:
                await self._send_message(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"method not found: {method}",
                        },
                    }
                )

    async def _notify_initialized(self) -> None:
        await self._send_message(
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        )

    async def _request(
        self, method: str, params: dict[str, Any], timeout: float
    ) -> Any:
        if self._closed:
            raise RuntimeError("MCP connection is closed")
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._send_message(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            raise RuntimeError(f"MCP request '{method}' timed out") from None

    async def _send_message(self, message: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            return
        try:
            async with self._write_lock:
                self._process.stdin.write(
                    (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
                )
                await self._process.stdin.drain()
        except (ConnectionResetError, BrokenPipeError, RuntimeError):
            raise RuntimeError("MCP server process went away") from None


# ── bridge (TS ``connectMcpServers``) ──────────────────────────────

@dataclass
class McpToolBridge:
    """Connected MCP servers and the external tools they expose.

    ``close()`` terminates every server process; it is idempotent.
    """

    tools: list[ToolSpec]
    _connections: list[_McpStdioConnection] = field(default_factory=list, repr=False)
    _closed: bool = field(default=False, repr=False)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.gather(
            *(connection.close() for connection in self._connections),
            return_exceptions=True,
        )

    @property
    def closed(self) -> bool:
        return self._closed


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name) or "tool"


def _unique_name(name: str, taken: set[str]) -> str:
    candidate = name
    suffix = 2
    while candidate in taken:
        candidate = f"{name}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _to_tool_result_content(content: Any) -> list[dict[str, Any]]:
    """Map MCP result content blocks onto external-tool result parts (TS parity)."""
    mapped: list[dict[str, Any]] = []
    if not isinstance(content, list):
        return mapped
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            mapped.append({"type": "text", "text": block["text"]})
            continue
        if (
            block.get("type") == "image"
            and isinstance(block.get("data"), str)
            and isinstance(block.get("mimeType"), str)
        ):
            mapped.append(
                {"type": "image", "data": block["data"], "mimeType": block["mimeType"]}
            )
            continue
        mapped.append({"type": "text", "text": json.dumps(block, ensure_ascii=False)})
    return mapped


def _make_executor(
    connection: _McpStdioConnection, tool_name: str
) -> Callable[..., Awaitable[ToolResult]]:
    async def execute(tool_call_id: str, args: dict[str, Any]) -> ToolResult:
        result = await connection.call_tool(
            tool_name, args if isinstance(args, dict) else {}
        )
        is_error = result.get("isError")
        return ToolResult(
            content=_to_tool_result_content(result.get("content")),
            is_error=isinstance(is_error, bool) and is_error,
        )

    return execute


def _bridge_tool(
    connection: _McpStdioConnection,
    server_name: str,
    tool: dict[str, Any],
    name: str,
) -> ToolSpec:
    tool_name = str(tool.get("name") or "tool")
    title = tool.get("title")
    description = tool.get("description")
    if not (isinstance(description, str) and description.strip()):
        description = f"The {tool_name} tool from the {server_name} MCP server."
    input_schema = tool.get("inputSchema")
    if not isinstance(input_schema, dict):
        input_schema = {"type": "object", "properties": {}}
    return ToolSpec(
        name=name,
        label=str(title) if title else tool_name,
        description=description,
        parameters=input_schema,
        execute=_make_executor(connection, tool_name),
    )


async def _connect_one(
    name: str,
    config: dict[str, Any],
    session_cwd: str | None,
) -> tuple[str, _McpStdioConnection | None, str | None]:
    """Connect one server; returns (name, connection | None, error | None)."""
    config_type = config.get("type")
    if config_type in ("http", "sse"):
        return (
            name,
            None,
            f"{config_type} MCP transports are not supported by the Python SDK "
            "(stdio only)",
        )
    command = config.get("command")
    if config_type not in (None, "stdio") or not isinstance(command, str) or not command:
        return name, None, "invalid stdio config (expected a 'command' string)"
    args = config.get("args")
    args = [str(a) for a in args] if isinstance(args, list) else []
    env = dict(os.environ)
    config_env = config.get("env")
    if isinstance(config_env, dict):
        env.update({str(k): str(v) for k, v in config_env.items()})
    cwd = config.get("cwd")
    cwd = cwd if isinstance(cwd, str) and cwd else session_cwd

    connection = _McpStdioConnection(command, args, env, cwd)
    try:
        await connection.start()
        return name, connection, None
    except Exception as exc:
        await connection.close()
        return name, None, str(exc)


async def connect_mcp_servers(
    servers: McpServers | None,
    *,
    cwd: str | None = None,
    reserved_tool_names: Iterable[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> McpToolBridge:
    """Connect MCP servers and bridge their tools as external ``ToolSpec``s.

    Mirrors the TypeScript SDK's ``connectMcpServers``: servers connect in
    parallel; a failing server is logged and skipped without affecting the
    others. Tool names are ``mcp__<server>__<tool>`` (sanitized, collision
    suffixed). The returned bridge must be closed when the session closes.
    """
    if not servers:
        return McpToolBridge(tools=[])
    log = log or (lambda message: logger.warning(message))
    named = list(dict(servers).items())
    results = await asyncio.gather(
        *(_connect_one(name, config, cwd) for name, config in named)
    )

    connections: list[_McpStdioConnection] = []
    tools: list[ToolSpec] = []
    taken: set[str] = set(reserved_tool_names or set())

    for server_name, connection, error in results:
        if connection is None:
            log(f'MCP server "{server_name}" unavailable: {error}')
            continue
        connections.append(connection)
        for tool in connection.tools:
            tool_name = str(tool.get("name") or "tool")
            bridge_name = _unique_name(
                f"mcp__{_sanitize(server_name)}__{_sanitize(tool_name)}", taken
            )
            tools.append(_bridge_tool(connection, server_name, tool, bridge_name))
        count = len(connection.tools)
        log(
            f'MCP server "{server_name}" connected '
            f'({count} tool{"s" if count != 1 else ""})'
        )

    return McpToolBridge(tools=tools, _connections=connections)


# ── tool allowlist wildcards (TS ``expandMcpToolWildcards``) ──────

def expand_mcp_tool_wildcards(
    allowed_tools: list[str] | None,
    mcp_tools: Iterable[str],
) -> list[str] | None:
    """Expand Claude-style MCP wildcards into the exact tool allowlist.

    ``mcp__<server>*`` entries match every connected MCP tool with that
    prefix; the result is deduped. Returns ``None`` for ``None`` input
    (TS parity).
    """
    if allowed_tools is None:
        return None
    available = list(mcp_tools)
    expanded: list[str] = []
    for entry in allowed_tools:
        if entry.startswith("mcp__") and entry.endswith("*"):
            prefix = entry[:-1]
            expanded.extend(name for name in available if name.startswith(prefix))
        else:
            expanded.append(entry)
    return list(dict.fromkeys(expanded))
