"""Management: the resource managers (models, agents, conversations).

The SDK ships management facades over the app server's wire commands.
They operate on resources directly and return raw server records —
note ``agents.create()`` returns the **agent dict** (take ``["id"]``),
while the client-level ``create_agent()`` convenience returns the id
string.

Run:
    export LETTA_MODEL=<model handle your app server has>
    python examples/management.py
"""

from __future__ import annotations

import asyncio

from letta_sdk import CreateAgentOptions, LettaAgentClient

from _env import client_kwargs, model


async def main() -> None:
    client = LettaAgentClient(**client_kwargs())
    demo_agent_id: str | None = None
    conv_agent_id: str | None = None
    try:
        # Models available on this server
        models = await client.models.list()
        handles: list[str] = []
        if isinstance(models, dict):
            raw = models.get("available_handles")
            if isinstance(raw, list):
                handles = [str(h) for h in raw if isinstance(h, str)]
            else:
                entries = models.get("entries")
                if isinstance(entries, list):
                    handles = [
                        str(entry.get("handle") or entry.get("id"))
                        for entry in entries
                        if isinstance(entry, dict)
                    ]
        print(f"models ({len(handles)}): {', '.join(handles[:8])}...")

        # Agents
        agents = await client.agents.list()
        print(f"agents ({len(agents)}):")
        for agent in agents[:5]:
            print(f"  {agent.get('id')}  {agent.get('name')}  {agent.get('model')}")

        # Create + retrieve a scratch agent via the manager (returns the
        # raw record, not the id)
        record = await client.agents.create(
            CreateAgentOptions(name="mgmt-demo", model=model())
        )
        demo_agent_id = str(record.get("id"))
        print(f"created: {demo_agent_id}")
        retrieved = await client.agents.retrieve(demo_agent_id)
        print(f"retrieved: name={retrieved.get('name')!r} model={retrieved.get('model')!r}")

        # Conversations of a live agent
        conv_agent_id = await client.create_agent(
            CreateAgentOptions(name="mgmt-convs", model=model())
        )
        session = client.create_session(conv_agent_id)
        await session.prompt("Hi!")
        await session.close()

        conversations = await client.conversations.list(agent_id=conv_agent_id)
        print(f"conversations for {conv_agent_id} ({len(conversations)}):")
        for conversation in conversations[:3]:
            print(f"  {conversation.get('id')}  {conversation.get('title', '')[:40]}")
        await client.agents.delete(conv_agent_id)
        conv_agent_id = None
    finally:
        if demo_agent_id:
            await client.agents.delete(demo_agent_id)
        if conv_agent_id:
            await client.agents.delete(conv_agent_id)
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
