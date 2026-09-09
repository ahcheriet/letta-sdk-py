"""History: read back what the server stored for a conversation.

``session.list_messages()`` (or ``client.conversations.list_messages``)
fetches the persisted transcript — the same data that powers
``resume_session()``. This example has a two-message conversation and
then reads the server-side history back.

Run:
    export LETTA_MODEL=<model handle your app server has>
    python examples/history.py
"""

from __future__ import annotations

import asyncio

from letta_sdk import CreateAgentOptions, LettaAgentClient

from _env import client_kwargs, model


def preview(message: dict, width: int = 60) -> str:
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    else:
        text = message.get("reasoning") or ""  # reasoning / control messages
    return (text or message.get("message_type") or "?").replace("\n", " ")[:width]


async def main() -> None:
    client = LettaAgentClient(**client_kwargs())
    agent_id: str | None = None
    try:
        agent_id = await client.create_agent(
            CreateAgentOptions(
                name="history-demo",
                model=model(),
                system_prompt="Answer in one short sentence.",
            )
        )
        session = client.create_session(agent_id)

        await session.prompt("My favorite color is teal.")
        await session.prompt("What is my favorite color?")

        result = await session.list_messages(limit=12)
        print(f"server-side history ({len(result.messages)} messages shown):")
        for message in result.messages:
            role = message.get("role") or message.get("message_type") or "?"
            print(f"  [{role}] {preview(message)}")
        await session.close()
    finally:
        if agent_id:
            await client.agents.delete(agent_id)
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
