"""Skills: seed an agent with skills at creation time.

A skill is either:
  * a directory containing ``SKILL.md`` (frontmatter: name, description), or
  * an inline ``AgentSkill`` object.

Skills are stored as ``skills/{name}`` memory blocks — the agent sees
``skills/{name}/SKILL.md`` in its memory repo and can edit them like any
other memory. (Support files such as ``scripts/`` require the Cloud
backend and are rejected by this SDK — TS parity.)

Run:
    export LETTA_MODEL=<model handle your app server has>
    python examples/skills.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from letta_sdk import (
    AgentSkill,
    CreateAgentOptions,
    LettaAgentClient,
)

from _env import client_kwargs, model

SKILL_DIR = Path(__file__).parent / "skills" / "hello-skill"


async def main() -> None:
    client = LettaAgentClient(**client_kwargs())
    agent_id: str | None = None
    try:
        agent_id = await client.create_agent(
            CreateAgentOptions(
                name="skill-user",
                model=model(),
                skills=[
                    # 1. a directory skill (loaded from disk)
                    str(SKILL_DIR),
                    # 2. an inline skill
                    AgentSkill(
                        name="math-voice",
                        description="Use when doing arithmetic.",
                        instructions=(
                            "When doing arithmetic, show the intermediate "
                            "step and end with the answer on its own line."
                        ),
                    ),
                ],
            )
        )
        print(f"created agent with 2 skills: {agent_id}")

        result = await client.prompt(
            agent_id,
            "What skill files do you have in your memory? List each skill "
            "name and its description.",
        )
        print(result.result)

        # The agent owns the skills: it can apply them.
        result = await client.prompt(agent_id, "Hello there!")
        print("greeting:", result.result)
    finally:
        if agent_id:
            await client.agents.delete(agent_id)
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
