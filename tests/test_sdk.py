"""Unit tests for the Letta Agent SDK (app-server websocket protocol).

A :class:`FakeAppServerConnection` scripts the wire protocol so the full
session / turn-coordination / management surface is exercised without a
live app server. Live integration coverage lives in ``tests/live_smoke.py``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from letta_sdk import (
    AssistantMessage,
    CreateAgentOptions,
    CreateSessionOptions,
    ErrorMessage,
    LettaAgentClient,
    LettaSession,
    QueryOptions,
    ResultMessage,
    ToolSpec,
    UsageMessage,
    image_from_base64,
    image_from_file,
    text_content,
    TranscriptAccumulator,
)
from letta_sdk.app_server import AppServerRequestError
from letta_sdk.types import ToolResult


# ═══════════════════════════════════════════════════════════════
# fake transport
# ═══════════════════════════════════════════════════════════════


class FakeAppServerConnection:
    """Scriptable stand-in for :class:`AppServerConnection`."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.handlers: list[Any] = []
        self.closed = False
        self._counter = 0
        # command name -> callable(body) -> response dict
        self.requests: dict[str, Any] = {}
        self.request_log: list[tuple[str, dict[str, Any]]] = []

    def set_response(self, command: str, response: dict[str, Any]) -> None:
        self.requests[command] = response

    def next_request_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"

    async def send(self, payload: dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("closed")
        self.sent.append(payload)

    async def request(
        self,
        type_: str,
        body: dict[str, Any] | None = None,
        *,
        response_type: str | None = None,
        predicate: Any = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.request_log.append((type_, body or {}))
        response = self.requests.get(type_)
        if response is None:
            raise AppServerRequestError(f"fake: no handler for '{type_}'")
        return response

    def on_message(self, handler: Any) -> Any:
        self.handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self.handlers:
                self.handlers.remove(handler)

        return unsubscribe

    def push(self, message: dict[str, Any]) -> None:
        for handler in list(self.handlers):
            handler(message)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for handler in list(self.handlers):
            handler({"type": "_transport_closed", "error": "test close"})

    def last_input(self) -> dict[str, Any] | None:
        for payload in reversed(self.sent):
            if payload.get("type") == "input":
                return payload
        return None


def make_client(connection: FakeAppServerConnection) -> LettaAgentClient:
    return LettaAgentClient(connection=connection)


def ready_response(agent_id: str = "agent-1", conversation_id: str = "conv-1") -> dict:
    return {
        "success": True,
        "runtime": {"agent_id": agent_id, "conversation_id": conversation_id},
        "agent": {"id": agent_id, "model": "test/model"},
    }


def delta(message_type: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": "stream_delta",
        "delta": {"message_type": message_type, **extra},
    }


# ═══════════════════════════════════════════════════════════════
# client / agent creation
# ═══════════════════════════════════════════════════════════════


async def test_create_agent_returns_id():
    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response("agent-42", "conv-9"))
    client = make_client(conn)
    agent_id = await client.create_agent(
        CreateAgentOptions(name="demo", model="test/model", system_prompt="Be nice.")
    )
    assert agent_id == "agent-42"
    # the runtime_start body carried create_agent
    command, body = conn.request_log[0]
    assert command == "runtime_start"
    assert body["create_agent"]["body"]["name"] == "demo"
    assert body["create_agent"]["body"]["system"] == "Be nice."
    # an injected connection stays open (the test owns it)
    assert conn.closed is False
    await client.close()


async def test_cloud_backend_rejected():
    with pytest.raises(ValueError, match="cloud"):
        LettaAgentClient(backend="cloud")


# ═══════════════════════════════════════════════════════════════
# turns
# ═══════════════════════════════════════════════════════════════


async def test_full_turn_stream():
    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response())
    client = make_client(conn)
    session = client.resume_session("agent-1")

    await session.ready()
    await session.send("Hello!")

    # server echoes the user turn, then streams an assistant reply
    conn.push(
        delta(
            "assistant_message",
            content=[{"type": "text", "text": "Hi "}],
            id="m-1",
            run_id="run-1",
        )
    )
    conn.push(
        delta(
            "assistant_message",
            content=[{"type": "text", "text": "there"}],
            id="m-2",
            run_id="run-1",
        )
    )
    conn.push(delta("stop_reason", stop_reason="end_turn", run_id="run-1"))
    conn.push(
        delta(
            "usage_statistics",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            run_ids=["run-1"],
        )
    )

    seen: list = []
    async for message in session.stream():
        seen.append(message)
        if isinstance(message, ResultMessage):
            break

    assistants = [m for m in seen if isinstance(m, AssistantMessage)]
    assert "".join(m.content for m in assistants) == "Hi there"
    result = seen[-1]
    assert isinstance(result, ResultMessage)
    assert result.success is True
    assert result.result == "Hi there"
    assert result.stop_reason == "end_turn"
    assert result.conversation_id == "conv-1"
    usage = [m for m in seen if isinstance(m, UsageMessage)]
    assert usage and usage[0].total_tokens == 15

    # the input command is well-formed
    payload = conn.last_input()
    assert payload is not None
    assert payload["runtime"] == {"agent_id": "agent-1", "conversation_id": "conv-1"}
    assert payload["payload"]["kind"] == "create_message"
    user_msg = payload["payload"]["messages"][0]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "Hello!"
    assert user_msg["client_message_id"].startswith("sdk-message-")

    await session.close()
    assert conn.closed is False  # injected connection stays open


async def test_error_delta_yields_failed_result():
    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response())
    client = make_client(conn)
    session = client.resume_session("agent-1")
    await session.ready()
    await session.send("boom")

    conn.push(
        delta(
            "error_message",
            detail="the model exploded",
            stop_reason="llm_api_error",
            run_id="run-9",
        )
    )

    seen: list = []
    async for message in session.stream():
        seen.append(message)
        if isinstance(message, ResultMessage):
            break

    assert isinstance(seen[-2], ErrorMessage)
    result = seen[-1]
    assert isinstance(result, ResultMessage)
    assert result.success is False
    assert result.error_code == "llm_api_error"
    assert result.error_detail == "the model exploded"
    await session.close()


async def test_send_requires_nonempty():
    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response())
    client = make_client(conn)
    session = client.resume_session("agent-1")
    with pytest.raises(ValueError, match="non-empty"):
        await session.send("   ")
    await session.close()


async def test_prompt_one_shot_works():
    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response())
    client = make_client(conn)

    async def drive():
        # let prompt() reach the send, then emit the server's reply
        for _ in range(100):
            if conn.last_input() is not None:
                break
            await asyncio.sleep(0.001)
        conn.push(delta("assistant_message", content="pong", id="m-1"))
        conn.push(delta("stop_reason", stop_reason="end_turn"))
        conn.push(
            delta("usage_statistics", prompt_tokens=1, completion_tokens=1, total_tokens=2)
        )

    driver = asyncio.create_task(drive())
    result = await client.prompt("agent-1", "ping")
    await driver
    assert isinstance(result, ResultMessage)
    assert result.success is True
    assert result.result == "pong"
    # injected connection: closing the session does not close it
    assert conn.closed is False
    await client.close()


async def test_stream_prompt_yields_full_stream():
    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response())
    client = make_client(conn)

    async def drive():
        for _ in range(100):
            if conn.last_input() is not None:
                break
            await asyncio.sleep(0.001)
        conn.push(delta("assistant_message", content="A", id="m-1"))
        conn.push(delta("assistant_message", content="B", id="m-2"))
        conn.push(delta("stop_reason", stop_reason="end_turn"))
        conn.push(delta("usage_statistics", total_tokens=9))

    driver = asyncio.create_task(drive())
    seen: list = []
    async for message in client.stream_prompt("agent-1", "go"):
        seen.append(message)
    await driver
    assert [m.content for m in seen if isinstance(m, AssistantMessage)] == ["A", "B"]
    assert isinstance(seen[-1], ResultMessage)
    assert seen[-1].success


# ═══════════════════════════════════════════════════════════════
# resume / conversation resolution
# ═══════════════════════════════════════════════════════════════


async def test_resume_conversation_resolves_agent():
    conn = FakeAppServerConnection()
    conn.set_response(
        "conversation_retrieve",
        {"success": True, "conversation": {"id": "conv-7", "agent_id": "agent-7"}},
    )
    conn.set_response("runtime_start", ready_response("agent-7", "conv-7"))
    client = make_client(conn)
    session = client.resume_session("conv-7")
    init = await session.ready()
    assert init.agent_id == "agent-7"
    assert init.conversation_id == "conv-7"
    # runtime_start carried both the resolved agent and the conversation
    command, body = conn.request_log[-1]
    assert command == "runtime_start"
    assert body["agent_id"] == "agent-7"
    assert body["conversation_id"] == "conv-7"
    await session.close()


async def test_resume_agent_uses_default_conversation():
    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response("agent-1", "conv-1"))
    client = make_client(conn)
    session = client.resume_session("agent-1")
    await session.ready()
    command, body = conn.request_log[0]
    assert body["agent_id"] == "agent-1"
    assert body["conversation_id"] == "default"
    await session.close()


# ═══════════════════════════════════════════════════════════════
# agent-free query()
# ═══════════════════════════════════════════════════════════════


async def test_query_stream():
    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response("agent-ep", "conv-ep"))
    client = make_client(conn)
    stream = client.query(
        "Say hi",
        QueryOptions(model="test/model", system="You are terse."),
    )

    async def drive():
        for _ in range(100):
            if conn.last_input() is not None:
                break
            await asyncio.sleep(0.001)
        conn.push(delta("assistant_message", content="hi", id="m-1"))
        conn.push(delta("stop_reason", stop_reason="end_turn"))
        conn.push(delta("usage_statistics", total_tokens=3))

    driver = asyncio.create_task(drive())
    seen: list = []
    async for message in stream:
        seen.append(message)
    await driver

    # create_conversation carried the ephemeral conversation body
    _, body = conn.request_log[0]
    assert body["create_conversation"]["body"] == {
        "model": "test/model",
        "system": "You are terse.",
    }
    assert [m.content for m in seen if isinstance(m, AssistantMessage)] == ["hi"]
    assert isinstance(seen[-1], ResultMessage)
    assert conn.closed is False  # injected connection stays open
    await client.close()


async def test_query_requires_model_and_system():
    conn = FakeAppServerConnection()
    client = make_client(conn)
    stream = client.query("hi", QueryOptions(model="m"))
    with pytest.raises(ValueError, match="system"):
        async for _ in stream:  # noqa: F841
            pass
    stream2 = client.query("hi", QueryOptions(system="s"))
    with pytest.raises(ValueError, match="model"):
        async for _ in stream2:  # noqa: F841
            pass


async def test_query_stream_single_use():
    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response())
    client = make_client(conn)
    stream = client.query("hi", QueryOptions(model="m", system="s"))
    stream._started = True  # simulate an in-flight first consumption
    with pytest.raises(RuntimeError, match="once"):
        async for _ in stream:  # noqa: F841
            pass


# ═══════════════════════════════════════════════════════════════
# management
# ═══════════════════════════════════════════════════════════════


async def test_agents_management():
    conn = FakeAppServerConnection()
    conn.set_response(
        "agent_list", {"success": True, "agents": [{"id": "a1"}, {"id": "a2"}]}
    )
    conn.set_response(
        "agent_retrieve", {"success": True, "agent": {"id": "a1", "name": "one"}}
    )
    conn.set_response("agent_delete", {"success": True})
    conn.set_response(
        "agent_update", {"success": True, "agent": {"id": "a1", "name": "renamed"}}
    )
    client = make_client(conn)

    agents = await client.agents.list()
    assert [a["id"] for a in agents] == ["a1", "a2"]

    agent = await client.agents.retrieve("a1")
    assert agent["name"] == "one"

    updated = await client.agents.update("a1", name="renamed")
    assert updated["name"] == "renamed"
    _, body = conn.request_log[-1]
    assert body == {"agent_id": "a1", "body": {"name": "renamed"}}

    await client.agents.delete("a1")

    with pytest.raises(ValueError, match="non-empty"):
        await client.agents.retrieve("")
    await client.close()


async def test_agents_create_via_runtime_start():
    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response("agent-new", "conv-n"))
    client = make_client(conn)
    agent = await client.agents.create(CreateAgentOptions(name="fresh"))
    assert agent.get("id") is not None or "agent_id" in json.dumps(agent)
    _, body = conn.request_log[0]
    assert body["create_agent"]["body"]["name"] == "fresh"
    assert body["create_agent"]["pin_global"] is True
    await client.close()


async def test_conversations_management():
    conn = FakeAppServerConnection()
    conn.set_response(
        "conversation_list", {"success": True, "conversations": [{"id": "c1"}]}
    )
    conn.set_response(
        "conversation_retrieve",
        {"success": True, "conversation": {"id": "c1", "agent_id": "a1"}},
    )
    conn.set_response(
        "conversation_create",
        {"success": True, "conversation": {"id": "c2", "agent_id": "a1"}},
    )
    conn.set_response(
        "conversation_fork",
        {"success": True, "conversation": {"id": "c3", "agent_id": "a1"}},
    )
    conn.set_response(
        "conversation_messages_list",
        {
            "success": True,
            "messages": [{"id": "m1", "role": "user"}],
            "nextBefore": None,
            "hasMore": False,
        },
    )
    client = make_client(conn)

    conversations = await client.conversations.list()
    assert conversations[0]["id"] == "c1"

    conversation = await client.conversations.retrieve("c1")
    assert conversation["agent_id"] == "a1"

    created = await client.conversations.create(agent_id="a1")
    assert created["id"] == "c2"
    _, body = conn.request_log[-1]
    assert body == {"body": {"agent_id": "a1"}}

    forked = await client.conversations.fork("c1")
    assert forked["id"] == "c3"

    result = await client.conversations.list_messages("c1", limit=10)
    assert len(result.messages) == 1
    assert result.has_more is False

    with pytest.raises(AppServerRequestError, match="not supported"):
        await client.conversations.delete("c1")
    await client.close()


async def test_models_management():
    conn = FakeAppServerConnection()
    conn.set_response(
        "list_models",
        {
            "success": True,
            "entries": [{"model_handle": "openai/gpt-x", "context_window": 128000}],
            "available_handles": ["openai/gpt-x"],
            "byok_provider_aliases": None,
        },
    )
    client = make_client(conn)
    models = await client.models.list()
    assert models["entries"][0]["model_handle"] == "openai/gpt-x"
    assert models["available_handles"] == ["openai/gpt-x"]
    await client.close()


# ═══════════════════════════════════════════════════════════════
# session management helpers
# ═══════════════════════════════════════════════════════════════


async def test_session_list_messages():
    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response())
    conn.set_response(
        "conversation_messages_list",
        {"success": True, "messages": [{"id": "m1"}, {"id": "m2"}]},
    )
    client = make_client(conn)
    session = client.resume_session("agent-1")
    result = await session.list_messages(limit=5)
    assert len(result.messages) == 2
    _, body = conn.request_log[-1]
    assert body["conversation_id"] == "conv-1"
    assert body["query"] == {"limit": 5}
    await session.close()


async def test_session_list_models():
    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response())
    conn.set_response(
        "list_models", {"success": True, "entries": [{"model_handle": "m"}]}
    )
    client = make_client(conn)
    session = client.resume_session("agent-1")
    models = await session.list_models()
    assert models["entries"][0]["model_handle"] == "m"
    await session.close()


# ═══════════════════════════════════════════════════════════════
# client-side tools
# ═══════════════════════════════════════════════════════════════


async def test_external_tool_execution():
    calls: list[tuple[str, dict]] = []

    async def run(tool_call_id: str, args: dict) -> ToolResult:
        calls.append((tool_call_id, args))
        return ToolResult(content=[{"type": "text", "text": f"echo:{args['word']}"}])

    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response())
    client = make_client(conn)
    session = client.resume_session(
        "agent-1",
        CreateSessionOptions(tools=[ToolSpec(name="echo", execute=run)]),
    )
    await session.ready()

    # the runtime_start body advertised the external tool
    _, body = conn.request_log[0]
    assert body["external_tools"][0]["tools"][0]["name"] == "echo"

    conn.push(
        {
            "type": "external_tool_call_request",
            "request_id": "req-1",
            "tool_name": "echo",
            "tool_call_id": "tc-1",
            "input": {"word": "hello"},
        }
    )

    # wait for the tool response to be sent
    for _ in range(100):
        responses = [
            p
            for p in conn.sent
            if p.get("type") == "external_tool_call_response"
        ]
        if responses:
            break
        await asyncio.sleep(0.001)
    else:
        raise AssertionError("no external_tool_call_response sent")

    response = responses[0]
    assert response["request_id"] == "req-1"
    assert response["result"]["content"] == [{"type": "text", "text": "echo:hello"}]
    assert response["result"].get("is_error") is None
    assert calls == [("tc-1", {"word": "hello"})]
    await session.close()


async def test_external_tool_failure_reports_error():
    async def failing(tool_call_id: str, args: dict) -> ToolResult:
        raise RuntimeError("kaboom")

    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response())
    client = make_client(conn)
    session = client.resume_session(
        "agent-1",
        CreateSessionOptions(tools=[ToolSpec(name="bad", execute=failing)]),
    )
    await session.ready()
    conn.push(
        {
            "type": "external_tool_call_request",
            "request_id": "req-2",
            "tool_name": "bad",
            "tool_call_id": "tc-2",
            "input": {},
        }
    )
    for _ in range(100):
        responses = [
            p
            for p in conn.sent
            if p.get("type") == "external_tool_call_response"
        ]
        if responses:
            break
        await asyncio.sleep(0.001)
    else:
        raise AssertionError("no external_tool_call_response sent")
    assert "kaboom" in responses[0]["error"]
    await session.close()


# ═══════════════════════════════════════════════════════════════
# helpers (unchanged surface)
# ═══════════════════════════════════════════════════════════════


def test_image_helpers(tmp_path):
    part = image_from_base64("AAAA", "image/png")
    assert part["type"] == "image"
    assert part["source"].get("data") == "AAAA"
    assert part["source"].get("media_type") == "image/png"

    png = tmp_path / "pic.png"
    png.write_bytes(b"\x89PNG\r\n")
    part = image_from_file(png)
    assert part["source"].get("media_type") == "image/png"
    assert part["source"].get("type") == "base64"


def test_text_content():
    part = text_content("hello")
    assert part == {"type": "text", "text": "hello"}


def test_transcript_accumulator():
    acc = TranscriptAccumulator()
    acc.add(AssistantMessage(type="assistant", content="Hel"))
    acc.add(
        ResultMessage(
            type="result", success=True, stop_reason="end_turn"
        )
    )
    acc.add(AssistantMessage(type="assistant", content="lo"))
    assert acc.assistant_text == "Hello"
    assert len(acc.messages) == 3


async def test_send_multimodal_content():
    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response())
    client = make_client(conn)
    session = client.resume_session("agent-1")
    await session.send(
        [
            text_content("What is in this image?"),
            image_from_base64("AAAA", "image/png"),
        ]
    )
    payload = conn.last_input()
    assert payload is not None
    content = payload["payload"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "What is in this image?"}
    assert content[1]["type"] == "image"
    await session.close()


async def test_transport_drop_fails_turns():
    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response())
    client = make_client(conn)
    session = client.resume_session("agent-1")

    async def drive():
        for _ in range(100):
            if conn.last_input() is not None:
                break
            await asyncio.sleep(0.001)
        await conn.close()  # simulates a dropped socket

    driver = asyncio.create_task(drive())
    await session.send("hello")
    seen: list = []
    async for message in session.stream():
        seen.append(message)
        if isinstance(message, ErrorMessage):
            break
    await driver
    errors = [m for m in seen if isinstance(m, ErrorMessage)]
    assert errors and errors[0].error_code == "stream_closed"


async def test_session_owns_connection_closes_it():
    conn = FakeAppServerConnection()
    conn.set_response("runtime_start", ready_response())
    session = LettaSession(connection=conn, owns_connection=True, agent_id="agent-1")
    await session.ready()
    await session.close()
    assert conn.closed is True
