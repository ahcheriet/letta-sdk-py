"""Telegram bridge: talk to a Letta agent from Telegram.

Receives Telegram messages by long-polling the Bot API (stdlib ``urllib`` —
no extra dependencies) and replies with the agent's answer. One
``LettaSession`` is kept per Telegram chat, so each chat keeps its own
conversation while all chats share the same agent (and therefore memory).

Chat commands:
    /start   - greet / verify the bridge
    /reset   - drop this chat's conversation (a fresh one opens next message)

Run:
    export LETTA_MODEL=<model handle your app server has>
    export TELEGRAM_BOT_TOKEN=<token from @BotFather>
    export TELEGRAM_ALLOWED_USER=<your numeric Telegram user id>  # strongly
                                                                  # recommended
    python examples/telegram_bridge.py

Optional:
    LETTA_AGENT_ID     reuse an existing (persistent) agent instead of
                       creating a scratch "telegram-bridge" agent
    TELEGRAM_API_URL   override the Bot API base (default
                       https://api.telegram.org; useful behind proxies or
                       for testing against a mock)

Note: replies are sent as plain text (no Markdown parsing) to avoid
Telegram entity errors with model-generated markdown.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Any

from letta_sdk import (
    AssistantMessage,
    CreateAgentOptions,
    CreateSessionOptions,
    LettaAgentClient,
    LettaSession,
    ToolCallMessage,
)

from _env import client_kwargs, model

TELEGRAM_API = os.environ.get("TELEGRAM_API_URL", "").strip() or "https://api.telegram.org"
#: Telegram's sendMessage limit is 4096 characters; leave some headroom.
CHUNK_SIZE = 4000


def _tg_call(token: str, method: str, **params: Any) -> Any:
    """One blocking Bot API call (run via asyncio.to_thread)."""
    url = f"{TELEGRAM_API}/bot{token}/{method}"
    data = urllib.parse.urlencode(params).encode("utf-8")
    try:
        with urllib.request.urlopen(url, data=data, timeout=70) as response:
            body = response.read().decode("utf-8")
        payload = json.loads(body)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Telegram {method} failed: {exc}") from exc
    if not isinstance(payload, dict) or not payload.get("ok"):
        detail = payload.get("description") if isinstance(payload, dict) else payload
        raise RuntimeError(f"Telegram {method} failed: {detail}")
    return payload.get("result")


async def tg(token: str, method: str, **params: Any) -> Any:
    """Async Bot API helper (long-poll safe: 70s HTTP timeout)."""
    return await asyncio.to_thread(_tg_call, token, method, **params)


async def send_chunks(
    token: str, chat_id: int, text: str, note: str = ""
) -> None:
    for start in range(0, max(len(text), 1), CHUNK_SIZE):
        await tg(token, "sendMessage", chat_id=chat_id, text=text[start : start + CHUNK_SIZE])
    if note:
        print(note, file=sys.stderr)


class Bridge:
    def __init__(self, client: LettaAgentClient, agent_id: str, token: str) -> None:
        self._client = client
        self._agent_id = agent_id
        self._token = token
        self._sessions: dict[int, LettaSession] = {}
        self._locks: dict[int, asyncio.Lock] = {}

    async def close(self) -> None:
        for session in self._sessions.values():
            try:
                await session.close()
            except Exception:  # pragma: no cover - best effort on shutdown
                pass
        self._sessions.clear()

    def _lock(self, chat_id: int) -> asyncio.Lock:
        return self._locks.setdefault(chat_id, asyncio.Lock())

    async def _session_for(self, chat_id: int) -> LettaSession:
        session = self._sessions.get(chat_id)
        if session is None:
            session = self._client.create_session(
                self._agent_id, CreateSessionOptions()
            )
            self._sessions[chat_id] = session
        await session.ready()
        return session

    async def reset_chat(self, chat_id: int) -> None:
        session = self._sessions.pop(chat_id, None)
        if session is not None:
            try:
                await session.close()
            except Exception:  # pragma: no cover - best effort
                pass

    async def handle_message(self, chat_id: int, text: str) -> None:
        """Run one turn for a chat (serialized per chat) and reply."""
        async with self._lock(chat_id):
            await tg(self._token, "sendChatAction", chat_id=chat_id, action="typing")
            try:
                session = await self._session_for(chat_id)
                await session.send(text)
                parts: list[str] = []
                async for message in session.stream():
                    if isinstance(message, AssistantMessage) and message.content:
                        parts.append(message.content)
                    elif isinstance(message, ToolCallMessage):
                        print(
                            f"[chat {chat_id}] tool call: {message.tool_name}",
                            file=sys.stderr,
                        )
                reply = "\n".join(parts).strip() or "(the agent did not reply)"
            except Exception as exc:
                # A broken session is dropped; the next message starts fresh.
                await self.reset_chat(chat_id)
                await send_chunks(
                    self._token,
                    chat_id,
                    f"Sorry, the agent run failed: {exc}",
                    note=f"[chat {chat_id}] error: {exc}",
                )
                return
            await send_chunks(self._token, chat_id, reply)


async def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is not set.\n"
            "Create a bot with @BotFather on Telegram and export its token:\n"
            "    export TELEGRAM_BOT_TOKEN=<token>"
        )
    allowed = os.environ.get("TELEGRAM_ALLOWED_USER", "").strip()
    reuse_agent = os.environ.get("LETTA_AGENT_ID", "").strip()

    try:
        me = await tg(token, "getMe")
    except RuntimeError as exc:
        raise SystemExit(
            f"Could not authenticate the Telegram bot: {exc}\n"
            "Check TELEGRAM_BOT_TOKEN (from @BotFather)."
        ) from exc
    username = me.get("username") if isinstance(me, dict) else "?"
    print(
        f"Telegram bridge running as @{username} — Ctrl+C to stop.",
        file=sys.stderr,
    )
    if not allowed:
        print(
            "WARNING: TELEGRAM_ALLOWED_USER is not set — anyone who finds the "
            "bot can chat.",
            file=sys.stderr,
        )

    client = LettaAgentClient(**client_kwargs())
    try:
        agent_id = reuse_agent or await client.create_agent(
            CreateAgentOptions(
                name="telegram-bridge",
                model=model(),
                system_prompt=(
                    "You are a concise assistant reachable over Telegram. "
                    "Keep replies short and use plain text only (no markdown "
                    "symbols, no code fences unless asked)."
                ),
            )
        )
        print(f"Agent: {agent_id}", file=sys.stderr)
        bridge = Bridge(client, agent_id, token)

        offset = 0
        try:
            while True:
                try:
                    updates = await tg(
                        token,
                        "getUpdates",
                        offset=offset,
                        timeout=30,
                        allowed_updates=json.dumps(["message"]),
                    )
                except RuntimeError as exc:
                    # Transient network/API hiccup: log and keep polling.
                    # A 401-style auth failure will keep logging — that is
                    # intentional (a misconfigured bridge should be loud).
                    print(f"[poll] {exc} — retrying in 5s", file=sys.stderr)
                    await asyncio.sleep(5)
                    continue
                if not isinstance(updates, list):
                    continue
                for update in updates:
                    if not isinstance(update, dict):
                        continue
                    update_id = update.get("update_id")
                    if isinstance(update_id, int):
                        offset = max(offset, update_id + 1)
                    message = update.get("message")
                    if not isinstance(message, dict):
                        continue
                    chat = message.get("chat")
                    sender = message.get("from")
                    chat_id = chat.get("id") if isinstance(chat, dict) else None
                    user_id = sender.get("id") if isinstance(sender, dict) else None
                    text = str(message.get("text") or "").strip()
                    if not isinstance(chat_id, int) or not text:
                        continue
                    if allowed and str(user_id) != allowed:
                        await send_chunks(
                            token, chat_id, "Sorry, you are not on the allowlist."
                        )
                        continue
                    if text == "/start":
                        await send_chunks(
                            token,
                            chat_id,
                            "Hi! I'm a Letta agent. Send me a message; "
                            "/reset starts a fresh conversation.",
                        )
                        continue
                    if text == "/reset":
                        await bridge.reset_chat(chat_id)
                        await send_chunks(token, chat_id, "Conversation reset.")
                        continue
                    await bridge.handle_message(chat_id, text)
        finally:
            await bridge.close()
    finally:
        await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
