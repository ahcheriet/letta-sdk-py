"""Stream events: read raw provider stream events as they happen.

Besides the high-level message types, the SDK surfaces the provider's
raw ``stream_event`` payloads (``StreamEventMessage.event``). The
``extract_stream_text_delta`` helper projects those onto appendable
assistant/reasoning text — useful for fine-grained UIs that want to
render deltas exactly as the provider emits them.

Note: not every server/model emits raw stream events (some deliver
finished ``assistant`` messages only). When that happens, this example
demonstrates the projection on representative payloads instead.

Run:
    export LETTA_MODEL=<model handle your app server has>
    python examples/stream_events.py
"""

from __future__ import annotations

import asyncio

from letta_sdk import CreateAgentOptions, LettaAgentClient, StreamEventMessage
from letta_sdk.stream_events import extract_stream_text_delta

from _env import client_kwargs, model


def project(event: dict) -> str:
    """Print one projected delta; return '' when the event carries no text."""
    delta = extract_stream_text_delta(event)
    if delta is None:
        return ""
    prefix = "reasoning" if delta.kind == "reasoning" else "assistant"
    print(f"[{prefix}] {delta.text}", end="", flush=True)
    return delta.text


# Representative payloads in the two shapes the helper projects (see
# letta_sdk/stream_events.py): content_block style and message chunk style.
SAMPLE_EVENTS = [
    {
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "The capital of France is "},
    },
    {
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "Paris."},
    },
    {
        "type": "content_block_delta",
        "delta": {"type": "reasoning_delta", "reasoning": "checking the map…"},
    },
    {
        "message_type": "assistant_message",
        "content": [{"type": "text", "text": " (message-chunk style)"}],
    },
]


async def main() -> None:
    client = LettaAgentClient(**client_kwargs())
    agent_id: str | None = None
    try:
        agent_id = await client.create_agent(
            CreateAgentOptions(
                name="stream-events",
                model=model(),
                system_prompt="Answer in exactly one short sentence.",
            )
        )
        session = client.create_session(agent_id)
        await session.ready()

        deltas = 0
        await session.send("Say: the stream works.")
        async for message in session.stream():
            if isinstance(message, StreamEventMessage):
                if project(message.event):
                    deltas += 1
            if message.type == "result":
                break
        if deltas:
            print(f"\n({deltas} text deltas observed live)")
        else:
            print("\n(no raw stream events observed for that turn)")
            print("This server delivers finished assistant messages instead —")
            print("here is the same projection on representative payloads,")
            print("as the helper would render them live:")
            for event in SAMPLE_EVENTS:
                project(event)
            print()
        await session.close()
    finally:
        if agent_id:
            await client.agents.delete(agent_id)
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
