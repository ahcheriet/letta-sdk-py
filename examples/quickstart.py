"""Quick start: create an agent, ask it one question, read the answer.

Run:
    export LETTA_MODEL=<model handle your app server has>
    python examples/quickstart.py

(Plus LETTA_APP_SERVER_URL / LETTA_TOKEN_FILE when your server is not at
the SDK's default endpoint or uses authentication — see examples/README.md.)
"""

from __future__ import annotations

import asyncio

from letta_sdk import CreateAgentOptions, LettaAgentClient

from _env import client_kwargs, model


async def main() -> None:
    client = LettaAgentClient(**client_kwargs())
    try:
        # 1. Create an agent (one time — its id can be saved and reused)
        agent_id = await client.create_agent(
            CreateAgentOptions(
                name="quickstart",
                model=model(),
                system_prompt="You are a concise assistant. Answer in one or two sentences.",
            )
        )
        print(f"created agent: {agent_id}")

        # 2. One-shot prompt: fresh conversation, waits for the terminal result
        result = await client.prompt(agent_id, "What is the capital of France?")
        print(result.result)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
