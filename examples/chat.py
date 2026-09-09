#!/usr/bin/env python3
"""A simple chat agent with memory.

The agent is created once and reused across runs (its id is stored in
``agent.json`` next to this file), so its memory — core memory blocks
(persona / human) plus the server-side conversation — persists between
sessions and processes.

Run:
    export LETTA_MODEL=<model handle your app server has>
    python examples/chat.py

Commands inside the chat:
    /memory   show the facts the agent has stored about you
    /reset    delete the agent and start completely fresh
    /exit     quit
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from letta_sdk import (
    AppServerError,
    AssistantMessage,
    CreateAgentOptions,
    LettaAgentClient,
    LettaSession,
    ResultMessage,
)

from _env import client_kwargs, model

BASE = Path(__file__).resolve().parent
STATE_FILE = BASE / "agent.json"

AGENT_NAME = "nova"
PERSONA = (
    "I am Nova, a friendly and curious chat assistant. I keep important "
    "facts about the user in my 'human' memory block so I can remember "
    "them between conversations."
)
HUMAN = "Facts about the user: (none recorded yet)"


# ── agent state ────────────────────────────────────────────────────


def load_agent_id() -> str | None:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            return None  # corrupt or unreadable state -> start fresh
        if isinstance(data, dict):
            agent_id = data.get("agent_id")
            if isinstance(agent_id, str) and agent_id:
                return agent_id
    return None


def save_agent_id(agent_id: str) -> None:
    STATE_FILE.write_text(json.dumps({"agent_id": agent_id}, indent=2) + "\n")


def forget_agent() -> None:
    STATE_FILE.unlink(missing_ok=True)


async def ensure_agent(client: LettaAgentClient) -> str:
    """Return a valid agent id, creating a fresh agent if needed."""
    existing = load_agent_id()
    if existing:
        try:
            agent = await client.agents.retrieve(existing)
            if agent.get("id") == existing:
                return existing
        except AppServerError:
            pass  # agent no longer exists -> fall through and recreate
    agent_id = await client.create_agent(
        CreateAgentOptions(
            name=AGENT_NAME,
            model=model(),
            persona=PERSONA,
            human=HUMAN,
            description="A simple chat agent with persistent memory.",
        )
    )
    save_agent_id(agent_id)
    return agent_id


# ── chat loop ──────────────────────────────────────────────────────


async def turn(session: LettaSession, text: str) -> None:
    """Send one user message and stream the agent's reply."""
    await session.send(text)
    async for message in session.stream():
        if isinstance(message, AssistantMessage):
            print(message.content, end="", flush=True)
        elif isinstance(message, ResultMessage) and not message.success:
            print(f"\n[error: {message.error}]", file=sys.stderr)
    print()


def render_memory_blocks(memory: object) -> list[tuple[str, str]]:
    """Normalize the agent's ``memory`` field (dict or list of blocks)."""
    blocks: list[tuple[str, str]] = []
    if isinstance(memory, dict):
        for label, block in memory.items():
            value = block.get("value", "") if isinstance(block, dict) else str(block)
            blocks.append((label, str(value)))
    elif isinstance(memory, list):
        for block in memory:
            if isinstance(block, dict):
                label = str(block.get("label") or "?")
                value = str(block.get("value") or "")
                blocks.append((label, value))
    return blocks


async def show_memory(client: LettaAgentClient, agent_id: str, session: LettaSession) -> None:
    agent = await client.agents.retrieve(agent_id)
    blocks = render_memory_blocks(agent.get("memory"))
    if blocks:
        print(f"core memory blocks for {agent_id}:")
        for label, value in blocks:
            print(f"  [{label}] {value}")
        print()
        return
    # This backend keeps memory in the agent's (git-backed) MemFS and does
    # not expose the blocks on retrieve — ask the agent to read them.
    print("[asking the agent to read its memory files]\n")
    await turn(
        session,
        "List the facts you have stored in your memory about the user, one per line. "
        "If nothing is stored, say so.",
    )


async def main() -> int:
    client = LettaAgentClient(**client_kwargs())
    try:
        agent_id = await ensure_agent(client)
        session = client.resume_session(agent_id)
        init = await session.ready()
        print(f"chatting with {AGENT_NAME} (agent={agent_id})")
        print(f"  model: {init.model}  conversation: {init.conversation_id}")
        print("  commands: /memory  /reset  /exit\n")

        while True:
            line = await asyncio.to_thread(input, "you> ")
            text = line.strip()
            if not text:
                continue
            if text in {"/exit", "/quit", ":q"}:
                break
            if text == "/reset":
                try:
                    await client.agents.delete(agent_id)
                except AppServerError:
                    pass
                forget_agent()
                await session.close()
                agent_id = await ensure_agent(client)
                session = client.resume_session(agent_id)
                await session.ready()
                print(f"  [fresh agent {agent_id} created — memory wiped]\n")
                continue
            if text == "/memory":
                await show_memory(client, agent_id, session)
                continue
            await turn(session, text)

        await session.close()
    finally:
        await client.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except (KeyboardInterrupt, EOFError):
        raise SystemExit(0)
