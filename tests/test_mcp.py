"""Tests for the MCP stdio bridge (``mcp.py``) and session wiring.

A tiny fake MCP server (newline-delimited JSON-RPC over stdio, stdlib only)
is spawned as a subprocess so the real transport path is exercised.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from letta_sdk import (
    CreateSessionOptions,
    LettaAgentClient,
    ToolSpec,
    connect_mcp_servers,
    expand_mcp_tool_wildcards,
)
from letta_sdk.app_server import AppServerRequestError
from letta_sdk.mcp import McpToolBridge
from letta_sdk.types import ToolResult
from tests.test_sdk import FakeAppServerConnection, make_client, ready_response

# ── fake MCP server (stdio, JSON-RPC 2.0) ─────────────────────────

FAKE_MCP_SERVER = r"""
import json, sys

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except Exception:
        continue
    if not isinstance(msg, dict) or "method" not in msg:
        continue
    method = msg["method"]
    mid = msg.get("id")
    if method == "initialize":
        if mid is not None:
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "serverInfo": {"name": "fake", "version": "1.0"},
            }})
    elif method == "tools/list":
        if mid is not None:
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [
                {"name": "echo", "description": "Echo the input text.",
                 "inputSchema": {"type": "object",
                                 "properties": {"text": {"type": "string"}}}},
                {"name": "boom", "title": "Boom Tool",
                 "inputSchema": {"type": "object"}},
            ]}})
    elif method == "tools/call":
        if mid is None:
            continue
        params = msg.get("params") or {}
        args = params.get("arguments") or {}
        if params.get("name") == "echo":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text",
                             "text": "echo: %s" % args.get("text", "")}]}})
        elif params.get("name") == "boom":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": "it boomed"}],
                "isError": True}})
        else:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "unknown tool"}})
    elif method == "ping":
        if mid is not None:
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
    # notifications: nothing to answer
"""


def echo_server() -> dict[str, Any]:
    return {"command": sys.executable, "args": ["-c", FAKE_MCP_SERVER]}


# ── connect_mcp_servers ───────────────────────────────────────────

async def test_connect_stdio_bridge_exposes_tools():
    bridge = await connect_mcp_servers({"fake": echo_server()})
    try:
        assert [t.name for t in bridge.tools] == [
            "mcp__fake__echo",
            "mcp__fake__boom",
        ]
        echo = bridge.tools[0]
        assert echo.description == "Echo the input text."
        assert echo.parameters == {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        }
        # the titled tool without a description keeps its title as label
        boom = bridge.tools[1]
        assert boom.label == "Boom Tool"

        result = await echo.execute("tc-1", {"text": "hi"})
        assert isinstance(result, ToolResult)
        assert result.content == [{"type": "text", "text": "echo: hi"}]
        assert result.is_error is False

        boomed = await boom.execute("tc-2", {})
        assert boomed.is_error is True
        assert boomed.content == [{"type": "text", "text": "it boomed"}]
    finally:
        await bridge.close()
    assert bridge.closed is True


async def test_empty_servers_yield_empty_bridge():
    bridge = await connect_mcp_servers(None)
    assert bridge.tools == []
    await bridge.close()  # idempotent no-op


async def test_broken_server_is_skipped_not_fatal():
    logs: list[str] = []
    bridge = await connect_mcp_servers(
        {"good": echo_server(), "bad": {"command": "/nonexistent-mcp"}},
        log=logs.append,
    )
    try:
        assert [t.name for t in bridge.tools] == [
            "mcp__good__echo",
            "mcp__good__boom",
        ]
        assert any('MCP server "bad" unavailable' in m for m in logs)
    finally:
        await bridge.close()


async def test_http_config_reported_unsupported():
    logs: list[str] = []
    bridge = await connect_mcp_servers(
        {
            "web": {"type": "http", "url": "http://example.invalid/sse"},
            "good": echo_server(),
        },
        log=logs.append,
    )
    try:
        assert [t.name for t in bridge.tools] == [
            "mcp__good__echo",
            "mcp__good__boom",
        ]
        assert any(
            'MCP server "web" unavailable' in m
            and "stdio only" in m
            for m in logs
        )
    finally:
        await bridge.close()


async def test_collision_suffix_and_reserved_names():
    bridge = await connect_mcp_servers(
        {"a": echo_server(), "b": echo_server()},
        reserved_tool_names={"mcp__a__echo"},
    )
    try:
        names = [t.name for t in bridge.tools]
        # server "a"'s echo is already reserved -> suffixed
        assert "mcp__a__echo_2" in names
        # server "b"'s echo is the next free name
        assert "mcp__b__echo" in names
        assert len(names) == len(set(names)) == 4
    finally:
        await bridge.close()


async def test_server_name_is_sanitized():
    bridge = await connect_mcp_servers({"my server!": echo_server()})
    try:
        assert bridge.tools[0].name == "mcp__my_server___echo"
    finally:
        await bridge.close()


# ── expand_mcp_tool_wildcards ─────────────────────────────────────

def test_expand_wildcards_none_passthrough():
    assert expand_mcp_tool_wildcards(None, ["mcp__a__x"]) is None


def test_expand_wildcards_matches_prefix_and_dedupes():
    available = ["mcp__fs__read", "mcp__fs__write", "mcp__db__query"]
    assert expand_mcp_tool_wildcards(
        ["mcp__fs*", "mcp__fs__read", "Bash", "mcp__db*"], available
    ) == ["mcp__fs__read", "mcp__fs__write", "Bash", "mcp__db__query"]


def test_expand_wildcards_no_match_drops_entry():
    # TS parity: an unmatched wildcard contributes nothing (it is dropped).
    assert expand_mcp_tool_wildcards(["mcp__none*"], ["mcp__a__x"]) == []


# ── session wiring ────────────────────────────────────────────────

async def test_session_wires_mcp_tools_into_runtime_start():
    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response("agent-9", "conv-9"))
    client = make_client(conn)

    local = ToolSpec(
        name="local_tool",
        description="A local tool",
        execute=lambda tc, args: _noop(tc, args),
    )

    session = client.create_session(
        "agent-9",
        CreateSessionOptions(
            tools=[local],
            mcp_servers={"fake": echo_server()},
        ),
    )
    init = await session.ready()

    command, body = conn.request_log[0]
    assert command == "runtime_start"
    tool_names = [
        t["name"] for t in body["external_tools"][0]["tools"]
    ]
    assert tool_names == [
        "local_tool",
        "mcp__fake__echo",
        "mcp__fake__boom",
    ]
    # init tools = agent tools (none in fake response) + mcp tool names
    assert init.tools == ["mcp__fake__echo", "mcp__fake__boom"]
    await session.close()
    assert session._mcp_bridge is None  # closed on session close


async def test_session_init_failure_closes_bridge():
    conn = FakeAppServerConnection()
    # no runtime_start handler -> AppServerRequestError
    client = make_client(conn)
    session = client.create_session(
        "agent-9",
        CreateSessionOptions(mcp_servers={"fake": echo_server()}),
    )
    with pytest.raises(AppServerRequestError):
        await session.ready()
    # the bridge was torn down with the failed initialization
    assert session._mcp_bridge is None


async def _noop(tool_call_id: str, args: dict[str, Any]) -> ToolResult:
    return ToolResult(content=[{"type": "text", "text": "ok"}])
