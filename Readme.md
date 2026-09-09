# letta-sdk-py

A pythonic SDK for Letta agents, inspired by the TypeScript `letta-agent-sdk`.

## Status

This repository contains a full Python SDK implementation built on top of `letta-client`, with a Pythonic API that mirrors the core workflows of the TypeScript `letta-agent-sdk`.

## Highlights

- `LettaAgentClient` as the main entry point
- top-level helpers: `create_agent`, `create_session`, `resume_session`, `query`, `prompt`
- async session API with `ready()`, `send()`, `stream()`, `list_messages()`, and result metadata
- management accessors for agents, conversations, and models
- ephemeral query conversations via `query()` (model + system prompt)
- image helpers for file and base64 content plus a transcript accumulator

## Install

```bash
pip install -e .
```

## Example: persistent agent session

```python
from letta_sdk import LettaAgentClient


async def main() -> None:
    client = LettaAgentClient(backend="cloud")
    agent_id = await client.create_agent()

    async with client.resume_session(agent_id) as session:
        await session.ready()
        await session.send("Hello")
        async for message in session.stream():
            if message.type == "assistant":
                print(message.content)
```

## Example: one-shot ephemeral query

```python
from letta_sdk import LettaAgentClient, QueryOptions


async def run_query() -> None:
    client = LettaAgentClient(backend="cloud")
    async for message in client.query(
        "What is the capital of France?",
        QueryOptions(
            model="openai/gpt-5.6-luna",
            system_prompt="Answer directly and concisely.",
        ),
    ):
        if message.type == "assistant":
            print(message.content, end="")
```
