"""Personality: create an agent from a Letta Code personality preset.

The preset is resolved **server-side** by the app server's native
``create_agent`` command (same catalog the TUI uses), so the SDK only
sends the id. Valid ids: ``memo``, ``blank``, ``tutorial``, ``linus``,
``kawaii``.

Run:
    export LETTA_MODEL=<model handle your app server has>
    python examples/personality.py            # uses "kawaii"
    PERSONALITY=blank python examples/personality.py
"""

from __future__ import annotations

import asyncio
import os

from letta_sdk import CreateAgentOptions, LettaAgentClient

from _env import client_kwargs, model

PERSONALITIES = ("memo", "blank", "tutorial", "linus", "kawaii")


async def main() -> None:
    personality = os.environ.get("PERSONALITY", "kawaii").strip()
    if personality not in PERSONALITIES:
        raise SystemExit(
            f"Unknown personality {personality!r}. Valid values: "
            + ", ".join(PERSONALITIES)
        )

    client = LettaAgentClient(**client_kwargs())
    agent_id: str | None = None
    try:
        # Personality agents use the native create_agent command: the
        # server applies its own preset catalog (name, persona, memory
        # files, model defaults). model= overrides the preset default.
        agent_id = await client.create_agent(
            CreateAgentOptions(personality=personality, model=model())
        )
        agent = await client.agents.retrieve(agent_id)
        print(f"created personality agent: {agent_id}")
        print(f"  name:   {agent.get('name')}")
        print(f"  model:  {agent.get('model')}")
        print(f"  tags:   {agent.get('tags')}")

        result = await client.prompt(agent_id, "Introduce yourself in one sentence.")
        print(result.result)
    finally:
        if agent_id:
            await client.agents.delete(agent_id)
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
