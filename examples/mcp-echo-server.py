"""A minimal MCP stdio server (stdlib only) for the ``mcp.py`` example.

Speaks newline-delimited JSON-RPC 2.0 over stdin/stdout and exposes two
tools:

* ``echo(text)`` -- returns the text unchanged
* ``add(a, b)`` -- returns the sum of two numbers
"""

from __future__ import annotations

import json
import math
import sys


def send(obj: object) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(msg, dict) or "method" not in msg:
            continue
        method = msg["method"]
        mid = msg.get("id")
        if method == "initialize":
            if mid is not None:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "serverInfo": {"name": "mcp-echo-server", "version": "1"},
                        },
                    }
                )
        elif method == "tools/list":
            if mid is not None:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {
                            "tools": [
                                {
                                    "name": "echo",
                                    "description": "Echo the given text back unchanged.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "text": {
                                                "type": "string",
                                                "description": "Text to echo",
                                            }
                                        },
                                        "required": ["text"],
                                    },
                                },
                                {
                                    "name": "add",
                                    "description": "Add two numbers together.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "a": {"type": "number"},
                                            "b": {"type": "number"},
                                        },
                                        "required": ["a", "b"],
                                    },
                                },
                            ]
                        },
                    }
                )
        elif method == "tools/call":
            if mid is None:
                continue
            params = msg.get("params") or {}
            args = params.get("arguments") or {}
            name = params.get("name")
            if name == "echo":
                text = str(args.get("text", ""))
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {
                            "content": [
                                {"type": "text", "text": f"echo: {text}"}
                            ]
                        },
                    }
                )
            elif name == "add":
                try:
                    total = float(args.get("a", 0)) + float(args.get("b", 0))
                except (TypeError, ValueError):
                    send(
                        {
                            "jsonrpc": "2.0",
                            "id": mid,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "a and b must be numbers",
                                    }
                                ],
                                "isError": True,
                            },
                        }
                    )
                    continue
                shown = (
                    int(total)
                    if math.isfinite(total) and total == int(total)
                    else total
                )
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {
                            "content": [{"type": "text", "text": str(shown)}]
                        },
                    }
                )
            else:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "error": {"code": -32601, "message": f"unknown tool: {name}"},
                    }
                )
        elif method == "ping":
            if mid is not None:
                send({"jsonrpc": "2.0", "id": mid, "result": {}})
        # notifications (e.g. notifications/initialized): nothing to answer


if __name__ == "__main__":
    main()
