"""Live smoke test against a running Letta Code app server.

Run directly (not collected by pytest)::

    .venv/bin/python tests/live_smoke.py

Requires:
  * an app server (default: the SDK's built-in localhost endpoint; override
    with ``LETTA_APP_SERVER_URL``),
  * a capability token file (``LETTA_TOKEN_FILE`` — required for
    ``--ws-auth capability-token`` servers),
  * a model handle (``LETTA_MODEL`` — required).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

from letta_sdk import (
    AppServerRequestError,
    AssistantMessage,
    CreateAgentOptions,
    LettaAgentClient,
    QueryOptions,
    ResultMessage,
    UsageMessage,
)

URL = os.environ.get("LETTA_APP_SERVER_URL")
TOKEN_FILE = os.environ.get("LETTA_TOKEN_FILE")
MODEL = os.environ.get("LETTA_MODEL")

if not MODEL:
    raise SystemExit("LETTA_MODEL is required (a model handle your server has)")

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail and not ok else ""))
    (PASS if ok else FAIL).append(name)


async def main() -> int:
    client = LettaAgentClient(url=URL, token_file=TOKEN_FILE, request_timeout=60)

    # 1. management: models
    try:
        models = await client.models.list()
        entries = models.get("entries") or []
        check(
            "models.list",
            isinstance(entries, list) and len(entries) > 0,
            f"got {len(entries)} entries",
        )
        handles = {e.get("handle") for e in entries if isinstance(e, dict) and e.get("handle")}
        print(f"       sample models: {sorted(h for h in handles if h)[:4]}")
    except Exception as exc:
        check("models.list", False, repr(exc))

    # 2. create an agent
    agent_id: str | None = None
    try:
        agent_id = await client.create_agent(
            CreateAgentOptions(
                name=f"sdk-smoke-{int(time.time())}",
                model=MODEL,
                system_prompt="You are a concise test assistant. Answer in at most one sentence.",
            )
        )
        check("create_agent", bool(agent_id), f"agent_id={agent_id!r}")
    except Exception as exc:
        check("create_agent", False, repr(exc))

    # 3. agents management
    try:
        agents = await client.agents.list()
        check("agents.list", any(a.get("id") == agent_id for a in agents))
        agent = await client.agents.retrieve(agent_id)  # type: ignore[arg-type]
        check("agents.retrieve", agent.get("id") == agent_id)
    except Exception as exc:
        check("agents.management", False, repr(exc))

    if agent_id:
        # 4. one-shot prompt (fresh real conversation per call)
        try:
            result = await client.prompt(agent_id, "Reply with exactly one word: PONG")
            check(
                "prompt.one_shot",
                isinstance(result, ResultMessage) and result.success,
                f"success={result.success} error={result.error!r} result={result.result!r}",
            )
            print(f"       result: {result.result!r} stop={result.stop_reason}")
            if isinstance(result, ResultMessage):
                check(
                    "prompt.usage",
                    result.duration_ms is not None,
                    f"duration_ms={result.duration_ms}",
                )
        except Exception as exc:
            check("prompt.one_shot", False, repr(exc))

        # 5. session with a real (created) conversation: streaming + state
        try:
            session = client.create_session(agent_id)
            init = await session.ready()
            check(
                "create_session.ready",
                init.agent_id == agent_id
                and bool(init.conversation_id)
                and init.conversation_id != "default",
                f"agent={init.agent_id} conv={init.conversation_id} model={init.model}",
            )
            await session.send("What is 2+2? Answer with just the number.")
            text_parts: list[str] = []
            terminal: ResultMessage | None = None
            saw_usage: UsageMessage | None = None
            async for message in session.stream():
                if isinstance(message, AssistantMessage):
                    text_parts.append(message.content)
                if isinstance(message, UsageMessage):
                    saw_usage = message
                if isinstance(message, ResultMessage):
                    terminal = message
                    break
            full = "".join(text_parts).strip()
            check(
                "session.stream",
                terminal is not None and terminal.success and "4" in full,
                f"terminal={terminal.success if terminal else None} text={full!r}",
            )
            print(f"       streamed: {full!r}")
            if saw_usage:
                print(f"       usage: {saw_usage.total_tokens} tokens")

            conversation_id = session.conversation_id
            if conversation_id:
                listed = await session.list_messages(limit=20)
                check(
                    "session.list_messages",
                    len(listed.messages) >= 2,
                    f"got {len(listed.messages)} messages",
                )
                conv = await client.conversations.retrieve(conversation_id)
                check(
                    "conversations.retrieve",
                    conv.get("id") == conversation_id
                    and conv.get("agent_id") == agent_id,
                    f"conv={conv.get('id')} agent={conv.get('agent_id')}",
                )
                convs = await client.conversations.list()
                check(
                    "conversations.list",
                    any(c.get("id") == conversation_id for c in convs),
                    f"{len(convs)} conversations",
                )
            await session.close()
        except Exception as exc:
            check("session.stream", False, repr(exc))

        # 5b. resume_session: virtual "default" conversation (turns work,
        # conversation-level management does not exist for it server-side)
        try:
            resumed = client.resume_session(agent_id)
            init2 = await resumed.ready()
            check(
                "resume_session.ready",
                init2.agent_id == agent_id,
                f"agent={init2.agent_id} conv={init2.conversation_id}",
            )
            await resumed.close()
        except Exception as exc:
            check("resume_session.ready", False, repr(exc))

        # 6. agent-free query() — the local backend does not support
        # conversation-only runtimes; the SDK must surface the typed server
        # error rather than hang or crash.
        try:
            prompt_text = "Say exactly: READY"
            query_stream_factory = client.query
            stream = query_stream_factory(
                prompt_text,
                QueryOptions(model=MODEL, system="You are a one-word test bot."),
            )
            async for _ in stream:  # noqa: F841
                pass
            check("query.agent_free", True)  # server supports it
        except AppServerRequestError as exc:
            print(f"[SKIP] query.agent_free — not supported by this backend")
            print(f"       server error: {str(exc)[:100]}...")
        except Exception as exc:
            check("query.agent_free", False, f"unexpected {type(exc).__name__}: {exc!r}")

    # 7. cleanup: delete the smoke agent
    if agent_id:
        try:
            await client.agents.delete(agent_id)
            check("agents.delete", True)
        except Exception as exc:
            check("agents.delete", False, repr(exc))

    await client.close()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
