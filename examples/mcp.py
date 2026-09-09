"""MCP servers: expose tools from a Model Context Protocol server.

``mcp_servers`` starts the MCP server as a subprocess of this process and
bridges its tools into the agent. The tools appear with the name
``mcp__<server>__<tool>`` and are executed through the same external-tool
protocol as ``tools``.

This example uses the bundled ``mcp-echo-server.py`` (a stdlib-only MCP
stdio server exposing ``echo`` and ``add``).

Run:
    export LETTA_MODEL=<model handle your app server has>
    python examples/mcp.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from letta_sdk import (
    AssistantMessage,
    CreateAgentOptions,
    CreateSessionOptions,
    LettaAgentClient,
    ToolCallMessage,
)

from _env import client_kwargs, model

MCP_SERVER_SCRIPT = str(Path(__file__).with_name("mcp-echo-server.py"))


async def main() -> None:
    client = LettaAgentClient(**client_kwargs())
    try:
        agent_id = await client.create_agent(
            CreateAgentOptions(
                name="mcp-user",
                model=model(),
                system_prompt=(
                    "You are a concise assistant. You can use the echo and "
                    "add tools from the 'calc' MCP server."
                ),
            )
        )
        session = client.create_session(
            agent_id,
            CreateSessionOptions(
                mcp_servers={
                    "calc": {"command": sys.executable, "args": [MCP_SERVER_SCRIPT]},
                },
            ),
        )
        await session.ready()

        await session.send(
            "Use the add tool to compute 40 + 2, then use the echo tool to "
            "send me the sentence: MCP works."
        )
        async for message in session.stream():
            if isinstance(message, AssistantMessage):
                print(message.content, end="", flush=True)
            elif isinstance(message, ToolCallMessage):
                print(
                    f"\n[calling {message.tool_name}({message.tool_input})]",
                    file=sys.stderr,
                )
        print()

        await session.close()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
