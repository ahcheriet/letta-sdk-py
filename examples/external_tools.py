"""External tools: run your own Python functions as agent tools.

The functions execute in *this* process — the app server calls the SDK,
the SDK calls your function, and the SDK sends the result back.

Run:
    export LETTA_MODEL=<model handle your app server has>
    python examples/external_tools.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone

from letta_sdk import (
    AssistantMessage,
    CreateAgentOptions,
    CreateSessionOptions,
    LettaAgentClient,
    ToolCallMessage,
    ToolSpec,
)

from _env import client_kwargs, model


async def get_utc_now(args: dict) -> dict:
    """A tool handler: async, takes the tool arguments, returns JSON-able."""
    return {"utc_now": datetime.now(timezone.utc).isoformat(timespec="seconds")}


TOOLS = [
    ToolSpec(
        name="get_utc_now",
        description="Return the current date and time in UTC.",
        parameters={"type": "object", "properties": {}},
        execute=get_utc_now,
    ),
]


async def main() -> None:
    client = LettaAgentClient(**client_kwargs())
    try:
        agent_id = await client.create_agent(
            CreateAgentOptions(name="tool-user", model=model())
        )
        session = client.create_session(
            agent_id, CreateSessionOptions(tools=TOOLS)
        )
        await session.ready()

        await session.send("What time is it right now? Use the get_utc_now tool.")
        async for message in session.stream():
            if isinstance(message, AssistantMessage):
                print(message.content, end="", flush=True)
            elif isinstance(message, ToolCallMessage):
                print(f"\n[calling tool: {message.tool_name}]", file=sys.stderr)
        print()

        await session.close()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
