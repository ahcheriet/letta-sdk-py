"""Resume: keep an agent alive across separate process runs.

The agent id and conversation id are saved to ``resume-state.json`` next to
this file. Run it twice:

    python examples/resume.py "I like the color teal, remember that."
    python examples/resume.py "What color do I like?"
    # -> "You like teal."

State is removed after a conversation has gone quiet (or manually: just
delete the JSON file).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from letta_sdk import AssistantMessage, CreateAgentOptions, LettaAgentClient

from _env import client_kwargs, model

STATE_FILE = Path(__file__).resolve().with_name("resume-state.json")


def load_state() -> dict[str, str] | None:
    try:
        data = json.loads(STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_state(agent_id: str, conversation_id: str) -> None:
    STATE_FILE.write_text(json.dumps({"agent_id": agent_id, "conversation_id": conversation_id}) + "\n")


async def main() -> None:
    if not sys.argv[1:]:
        raise SystemExit("usage: python examples/resume.py \"your message\"")
    text = sys.argv[1]

    client = LettaAgentClient(**client_kwargs())
    try:
        state = load_state()
        if state and state.get("conversation_id"):
            # resume_session("<conversation_id>") resolves the owning agent
            session = client.resume_session(state["conversation_id"])
        else:
            agent_id = await client.create_agent(
                CreateAgentOptions(
                    name="resumer",
                    model=model(),
                    persona="You are a helpful assistant with a good memory.",
                )
            )
            session = client.create_session(agent_id)

        init = await session.ready()
        await session.send(text)
        async for message in session.stream():
            if isinstance(message, AssistantMessage):
                print(message.content, end="", flush=True)
            elif message.type == "error":
                print(f"\n[error: {message.raw}]", file=sys.stderr)
        print()
        save_state(init.agent_id or "", init.conversation_id or "")
        await session.close()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
