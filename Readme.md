# letta-sdk-py

A pythonic agent SDK for Letta, speaking the **Letta Code app-server
WebSocket protocol** (agent SDK v2) — the same protocol the TypeScript
`@letta-ai/letta-agent-sdk` uses.

This is **not** the REST-based `letta-client`. The SDK talks JSON over
WebSocket to a running Letta Code app server, streaming every token, tool
call, and status update in real time.

## Requirements

- Python **3.11+**
- A running Letta Code app server (the local harness or a remote one),
  e.g. `letta server --listen ws://0.0.0.0:4500`
- A capability token (for `--ws-auth capability-token` servers)

## Install

```bash
pip install -e .
```

## Quick start

```python
from letta_sdk import LettaAgentClient, CreateAgentOptions, QueryOptions

client = LettaAgentClient(
    url="ws://127.0.0.1:4500/ws",
    token_file="/root/.letta/app-server-token",
)

# 1. Create an agent
agent_id = await client.create_agent(
    CreateAgentOptions(
        name="demo",
        model="openai-compatible/Qwen3.8-27B",
        system_prompt="You are a helpful, concise assistant.",
    )
)

# 2. One-shot prompt (fresh conversation, terminal result)
result = await client.prompt(agent_id, "Reply with exactly one word: PONG")
print(result.result, result.stop_reason)

# 3. Streaming session on a new conversation
session = client.create_session(agent_id)
init = await session.ready()        # SDKInitMessage: agent_id, conversation_id, model, tools
await session.send("What is 2+2? Answer with just the number.")
async for message in session.stream():
    if message.type == "assistant":
        print(message.content, end="")
    elif message.type == "usage":
        print(f"\n[tokens: {message.total_tokens}]")
    elif message.type == "result":
        print(f"\n[done in {message.duration_ms} ms]")
await session.close()
```

## Resuming

```python
# Resume an agent (server picks the default conversation)
session = client.resume_session("agent-...")

# Resume a specific conversation
session = client.resume_session("conv-...")
```

Both are lazy: the runtime starts on the first `send()` / `ready()` /
`stream()`.

## Agent-free ephemeral queries

`query()` runs one turn in a throwaway conversation — no agent needed
(requires a backend that supports conversation-only runtimes):

```python
from letta_sdk import QueryOptions

async for message in client.query(
    "What is the capital of France?",
    QueryOptions(model="openai-compatible/Qwen3.8-27B", system="Answer directly."),
):
    if message.type == "assistant":
        print(message.content, end="")
```

## Management

```python
models  = await client.models.list()      # entries + available_handles
agents  = await client.agents.list()
agent   = await client.agents.retrieve(agent_id)
await client.agents.update(agent_id, model="...")
await client.agents.delete(agent_id)

convs   = await client.conversations.list()
conv    = await client.conversations.retrieve(conv_id)
fork    = await client.conversations.fork(conv_id)

messages = await session.list_messages(limit=20)   # or client.conversations.list_messages(conv_id)
```

## Image input

```python
from letta_sdk import image_from_file, image_from_base64

part = image_from_file("chart.png")          # or image_from_base64(data, "image/png")
await session.send([{"type": "text", "text": "What is in this image?"}, part])
```

## Configuration

`LettaAgentClient(...)` keyword arguments:

| argument | meaning |
| --- | --- |
| `url` / `app_server_url` | WebSocket endpoint (default `ws://127.0.0.1:4500/ws`) |
| `backend` | `"local"` / `"remote"` / `"app-server"` (all map to a WS endpoint; `"cloud"` is not supported yet) |
| `auth_token` | bearer token sent as `Authorization` during the WS handshake |
| `api_key` | alias of `auth_token` |
| `token_file` | path to a file containing the token (read + stripped) |
| `request_timeout` | default per-request timeout (seconds) |
| `connect_timeout` | websocket handshake timeout (seconds, default 15) |
| `connection` | inject a pre-built `AppServerConnection` (or any object satisfying the `AppServerConnectionLike` protocol — used by the test suite) |

## API surface

- **`LettaAgentClient`** — `create_agent()`, `create_session()`,
  `resume_session()`, `query()`, `prompt()`, `stream_prompt()`,
  `agents` / `conversations` / `models` managers, `close()`.
- **`LettaSession`** — `ready()`, `send()`, `stream()`, `prompt()`,
  `send_and_wait()`, `list_messages()`, `abort()`, `close()`; context
  manager (`async with`).
- **Message types** (`letta_sdk.types`) — `SDKInitMessage`,
  `AssistantMessage`, `ReasoningMessage`, `ToolCallMessage`,
  `ToolResultMessage`, `UsageMessage`, `ResultMessage`, `ErrorMessage`,
  `LoopStatusMessage`, `RetryMessage`, `PingMessage`,
  `StreamEventMessage`, `UnknownMessage`.
- **Errors** — `AppServerError` → `AppServerRequestError` (server said
  no), `AppServerTimeoutError`, `AppServerClosedError`,
  `AppServerConnectionError`.
- **Options** — `CreateAgentOptions`, `CreateSessionOptions`,
  `QueryOptions`, `ToolSpec`.
- **Helpers** — `TranscriptAccumulator` (fold a message stream into a
  text transcript), `text_content()`, image helpers.

## Protocol notes

- One WebSocket per session; the management surface shares a single
  pooled connection.
- Requests and responses correlate by `request_id`; every server message
  carrying an integer `seq` is acknowledged with `{"type": "ack", "seq"}`.
- Turn completion: a `stop_reason` delta arms a short trailing-usage grace
  window; the turn closes when usage statistics arrive, the grace window
  expires, or a `turn_finished` event lands.
- External tools: register a `ToolSpec` (name + async `execute`) in
  `CreateSessionOptions.tools`; the SDK answers the server's
  `external_tool_call_request` with your result (or a typed error).

## Known limitations

- The local backend treats a bare `resume_session(agent_id)` as the
  *virtual* `"default"` conversation: turns work, but
  `list_messages()` / `conversation_retrieve` do not resolve it
  server-side. Use `create_session()` for a real, listable conversation.
- Agent-free `query()` requires a backend that supports
  conversation-only runtimes; the current local harness answers with a
  cloud-fallback error (surfaced as `AppServerRequestError`).
- `conversation_delete` has no protocol command;
  `ConversationsManager.delete()` raises `AppServerRequestError`.

## Development

```bash
pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q      # unit tests (fake transport)
.venv/bin/pyright src/ tests/             # type check (0 errors)
.venv/bin/python tests/live_smoke.py      # live integration (needs a running app server)
```

`tests/live_smoke.py` is env-configurable: `LETTA_APP_SERVER_URL`,
`LETTA_TOKEN_FILE`, `LETTA_MODEL`.
