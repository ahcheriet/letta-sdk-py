"""Streaming: watch tokens, tool calls, and usage arrive in real time.

Run:
    export LETTA_MODEL=<model handle your app server has>
    python examples/streaming.py
"""

from __future__ import annotations

import asyncio
import sys

from letta_sdk import (
    AssistantMessage,
    CreateAgentOptions,
    LettaAgentClient,
    ResultMessage,
    ToolCallMessage,
    UsageMessage,
)

from _env import client_kwargs, model


async def main() -> None:
    client = LettaAgentClient(**client_kwargs())
    try:
        agent_id = await client.create_agent(
            CreateAgentOptions(name="streamer", model=model())
        )
        # create_session() opens a real (listable) conversation
        session = client.create_session(agent_id)
        await session.ready()

        await session.send("Count from 1 to 5, one number per line.")
        async for message in session.stream():
            if isinstance(message, AssistantMessage):
                print(message.content, end="", flush=True)
            elif isinstance(message, ToolCallMessage):
                print(f"\n[tool call: {message.tool_name}]", file=sys.stderr)
            elif isinstance(message, UsageMessage):
                print(f"\n[tokens: {message.total_tokens}]", file=sys.stderr)
            elif isinstance(message, ResultMessage) and not message.success:
                print(f"\n[turn failed: {message.error}]", file=sys.stderr)
        print()

        await session.close()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
