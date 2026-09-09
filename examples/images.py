"""Multimodal input: send text plus an image in one message.

Run:
    python examples/images.py path/to/image.png
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from letta_sdk import AssistantMessage, CreateAgentOptions, LettaAgentClient, image_from_file

from _env import client_kwargs, model


async def main() -> None:
    if not sys.argv[1:]:
        raise SystemExit("usage: python examples/images.py <image-file>")
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        raise SystemExit(f"no such file: {image_path}")

    client = LettaAgentClient(**client_kwargs())
    try:
        agent_id = await client.create_agent(
            CreateAgentOptions(name="vision", model=model())
        )
        session = client.create_session(agent_id)
        await session.ready()

        # SendMessage is str | list[MessageContentPart]
        await session.send(
            [
                {"type": "text", "text": "Describe this image in one sentence."},
                image_from_file(image_path),
            ]
        )
        async for message in session.stream():
            if isinstance(message, AssistantMessage):
                print(message.content, end="", flush=True)
        print()

        await session.close()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
