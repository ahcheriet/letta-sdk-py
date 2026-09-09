"""Tool approvals: decide, in Python, which server-side tool calls may run.

Some agents carry gated tools (memory writes, file edits, ...). When the
model calls one, the backend asks for approval. With a ``can_use_tool``
callback you make that decision in your own code.

Run:
    export LETTA_MODEL=<model handle your app server has>
    python examples/approvals.py "remember that I prefer dark roast coffee"
"""

from __future__ import annotations

import asyncio
import sys

from letta_sdk import (
    AssistantMessage,
    CanUseToolDecision,
    CreateAgentOptions,
    CreateSessionOptions,
    LettaAgentClient,
)

from _env import client_kwargs, model


def can_use_tool(
    tool_name: str, tool_input: dict, context: dict
) -> CanUseToolDecision:
    print(f"[approval requested] {tool_name} <- {tool_input}", file=sys.stderr)
    if tool_name == "memory":
        return CanUseToolDecision(behavior="allow")
    return CanUseToolDecision(behavior="deny", message="not permitted in this example")


async def main() -> None:
    if not sys.argv[1:]:
        raise SystemExit('usage: python examples/approvals.py "your message"')
    text = sys.argv[1]

    client = LettaAgentClient(**client_kwargs())
    try:
        agent_id = await client.create_agent(
            CreateAgentOptions(name="approver", model=model())
        )
        session = client.create_session(
            agent_id, CreateSessionOptions(can_use_tool=can_use_tool)
        )
        await session.ready()

        await session.send(text)
        async for message in session.stream():
            if isinstance(message, AssistantMessage):
                print(message.content, end="", flush=True)
        print()

        await session.close()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
