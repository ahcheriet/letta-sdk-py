"""Session options: toolset + dreaming (reflection) settings.

- ``toolset`` — request-scoped client toolset, sent as ``client_toolset``
  in every turn's payload. ``base`` picks a bundled toolset (``auto``,
  ``codex``, ``codex_snake``, ``default``, ``gemini``, ``gemini_snake``,
  ``none``); ``include`` adds extra bundled tools.
- ``dreaming`` — reflection settings applied right after the runtime
  starts (``set_reflection_settings`` on the wire). ``trigger`` ∈
  ``off`` / ``step-count`` / ``compaction-event``; ``step_count`` is the
  step-count trigger threshold (default 5).

Run:
    export LETTA_MODEL=<model handle your app server has>
    python examples/session_options.py
"""

from __future__ import annotations

import asyncio

from letta_sdk import (
    CreateAgentOptions,
    CreateSessionOptions,
    DreamingOptions,
    LettaAgentClient,
    ToolsetConfig,
)

from _env import client_kwargs, model


async def main() -> None:
    client = LettaAgentClient(**client_kwargs())
    agent_id: str | None = None
    try:
        agent_id = await client.create_agent(
            CreateAgentOptions(name="session-options", model=model())
        )

        session = client.create_session(
            agent_id,
            CreateSessionOptions(
                model=model(),
                toolset=ToolsetConfig(base="auto"),
                # or: toolset={"base": "codex", "include": ["some_tool"]}
                dreaming=DreamingOptions(trigger="step-count", step_count=5),
                # or: dreaming={"trigger": "off"}
            ),
        )
        init = await session.ready()
        print(f"runtime started on agent {init.agent_id}")
        print("  toolset:  base=auto (sent as client_toolset every turn)")
        print("  dreaming: trigger=step-count, step_count=5")

        result = await session.prompt("In one sentence: what is your role?")
        print(result.result)
        await session.close()
    finally:
        if agent_id:
            await client.agents.delete(agent_id)
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
