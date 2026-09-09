"""Transcript: build a stable, render-ready transcript from a live turn.

``TranscriptAccumulator`` is the Python port of the TypeScript SDK's
transcript reconciliation: it folds streamed assistant/reasoning text
and tool calls/results into stable rows (keyed by message lineage),
suppressing per-run replays — the same rules a chat UI needs.

Run:
    export LETTA_MODEL=<model handle your app server has>
    python examples/transcript.py
"""

from __future__ import annotations

import asyncio
import sys

from letta_sdk import (
    CreateAgentOptions,
    CreateSessionOptions,
    LettaAgentClient,
    ToolSpec,
    TranscriptAccumulator,
    TranscriptRow,
)

from _env import client_kwargs, model


async def get_utc_now(tool_call_id: str, args: dict) -> dict:
    from datetime import datetime, timezone

    return {"utc_now": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def show_row(row: TranscriptRow) -> None:
    if row.kind == "tool_call":
        status = f" ({row.status})" if row.status else ""
        print(f"  [tool {row.tool_name}{status}] {row.result and row.result.content or ''}")
    else:
        preview = (row.text or "").replace("\n", " ")[:70]
        print(f"  [{row.kind}] {preview}")


async def main() -> None:
    client = LettaAgentClient(**client_kwargs())
    agent_id: str | None = None
    try:
        agent_id = await client.create_agent(
            CreateAgentOptions(name="transcript-demo", model=model())
        )
        session = client.create_session(
            agent_id,
            CreateSessionOptions(
                tools=[
                    ToolSpec(
                        name="get_utc_now",
                        description="Return the current UTC time.",
                        parameters={"type": "object", "properties": {}},
                        execute=get_utc_now,
                    )
                ]
            ),
        )

        accumulator = TranscriptAccumulator()
        await session.send("What time is it? Check with the tool, then answer.")
        async for message in session.stream():
            accumulator.add(message)  # every SDK message is safe to feed in
            if message.type == "result":
                break

        print("\nReconciled transcript rows:")
        for row in accumulator.rows:
            show_row(row)
        await session.close()
    finally:
        if agent_id:
            await client.agents.delete(agent_id)
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
