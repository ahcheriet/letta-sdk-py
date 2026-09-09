# letta-sdk-py

A pythonic SDK for Letta agents, inspired by the TypeScript `letta-agent-sdk`.

## Status

This repository now contains an MVP package built on top of `letta-client`.

## Highlights

- `LettaAgentClient` as the main entry point
- top-level helpers: `create_agent`, `create_session`, `resume_session`, `query`
- async session API with `async with`, `send()`, and `stream()`
- management accessors for agents, conversations, and models
- image helpers and a simple transcript accumulator

## Install

```bash
pip install -e .
```

## Example

```python
from letta_sdk import LettaAgentClient


async def main() -> None:
    client = LettaAgentClient(backend="local")
    agent_id = await client.create_agent()

    async with client.resume_session(agent_id) as session:
        await session.send("Hello")
        async for message in session.stream():
            if message.type == "assistant":
                print(message.content)
```
