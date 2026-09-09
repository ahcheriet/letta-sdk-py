# API reference

Complete reference for `letta_sdk` (v0.2.x). All entry points are exported
from the top level:

```python
import letta_sdk
from letta_sdk import LettaAgentClient, LettaSession, ...
```

Requires Python ≥ 3.11. The SDK is fully async.

---

## `LettaAgentClient`

Main entry point. One client = one websocket endpoint. Sessions each open
their own WebSocket; the management surface shares one pooled connection.

```python
import os
client = LettaAgentClient(
    url=os.environ.get("LETTA_APP_SERVER_URL"),       # optional — SDK default
    token_file=os.environ.get("LETTA_TOKEN_FILE"),    # optional if no auth
    request_timeout=60,
)
```

### Constructor arguments

| argument | type | default | meaning |
| --- | --- | --- | --- |
| `backend` | `Backend \| str` | `Backend.LOCAL` | `"local"` / `"remote"` / `"app-server"` (all map to a WS endpoint). `"cloud"` raises `ValueError` (not supported yet) |
| `url` / `app_server_url` | `str \| None` | `DEFAULT_APP_SERVER_URL` (`ws://127.0.0.1:4500/ws`) | websocket endpoint |
| `auth_token` | `str \| None` | `None` | bearer token sent as `Authorization` during the WS handshake |
| `api_key` | `str \| None` | `None` | alias of `auth_token` |
| `token_file` | `str \| Path \| None` | `None` | file containing the token (read + stripped); wins over `auth_token` |
| `request_timeout` | `float \| None` | `None` (no timeout) | default per-request timeout in seconds |
| `connect_timeout` | `float` | `15.0` | websocket handshake timeout (seconds) |
| `connection` | `AppServerConnectionLike \| None` | `None` | inject a pre-built connection (or a fake satisfying the protocol — used by the test suite). When injected, `close()` does **not** close it |
| `connect_factory` | `Any \| None` | `None` | test seam: replaces the `websockets.connect` factory |

### Methods

| method | signature | returns | notes |
| --- | --- | --- | --- |
| `create_agent` | `async (options: CreateAgentOptions \| None = None) -> str` | agent id | `runtime_start` + `create_agent`; the throwaway session is closed |
| `create_session` | `(agent_id: str, options: CreateSessionOptions \| None = None) -> LettaSession` | session | sync, lazy — opens a **new real conversation** (`create_conversation: {body: {}}`) |
| `resume_session` | `(identifier: str, options: CreateSessionOptions \| None = None) -> LettaSession` | session | bare agent id → virtual `"default"` conversation; `conv-*` id → resolves the owning agent via `conversation_retrieve` |
| `query` | `(prompt: SendMessage, options: QueryOptions \| None = None) -> QueryStream` | stream | agent-free ephemeral conversation; requires `options.model` and `options.system` |
| `prompt` | `async (agent_id: str, message: SendMessage, options: CreateSessionOptions \| None = None) -> ResultMessage` | result | one-shot turn in a fresh conversation; session is closed afterwards |
| `stream_prompt` | `async (agent_id: str, message: SendMessage, options: CreateSessionOptions \| None = None) -> AsyncIterator[SDKMessage]` | async iterator | same as `prompt`, streaming every message |
| `close` | `async () -> None` | — | closes the management connection (no-op for an injected one) |

### Properties / managers

| attribute | type | notes |
| --- | --- | --- |
| `closed` | `bool` | |
| `agents` | `AgentsManager` | see below |
| `conversations` | `ConversationsManager` | see below |
| `models` | `ModelsManager` | see below |

---

## Top-level convenience functions

`letta_sdk.create_agent`, `letta_sdk.create_session`, `letta_sdk.resume_session`,
`letta_sdk.query`, `letta_sdk.prompt` — same arguments as the client methods
plus optional `client=` / `**client_kwargs`. When no `client` is given, a
client is created and **closed automatically** when the operation completes
(sessions returned by `create_session`/`resume_session` close it via
`attach_owner`).

```python
agent_id = await letta_sdk.create_agent(CreateAgentOptions(...), url=URL)
session  = letta_sdk.resume_session(agent_id, url=URL)
result   = await letta_sdk.prompt(agent_id, "hi", url=URL)
```

---

## `LettaSession`

One runtime (agent + conversation) over one WebSocket.

### Properties

| property | type | available |
| --- | --- | --- |
| `agent_id` | `str \| None` | after `ready()` |
| `conversation_id` | `str \| None` | after `ready()` |
| `model` | `str` | after `ready()` |
| `tools` | `list[str] \| None` | after `ready()` (agent's tool names, when the server reports them) |

### Methods

| method | signature | returns | notes |
| --- | --- | --- | --- |
| `ready` | `async () -> SDKInitMessage` | init message | single-flight `runtime_start`; retryable after a failure |
| `send` | `async (message: SendMessage, *, otid: str \| None = None) -> None` | — | sends a user message, starts a turn; `SendMessage` = `str` or a list of content parts (`text` / `image`) |
| `stream` | `async () -> AsyncIterator[SDKMessage]` | async iterator | consumes the current turn: streamed messages then a final `ResultMessage` (or `ErrorMessage`) |
| `send_and_wait` | `async (message: SendMessage) -> ResultMessage` | result | `send` + drain until terminal |
| `prompt` | `async (message: SendMessage) -> ResultMessage` | result | alias of `send_and_wait` |
| `list_messages` | `async (limit: int = 50, before: str \| None = None, after: str \| None = None, order: str \| None = None) -> ListMessagesResult` | result | `conversation_messages_list`; needs a resolvable conversation (see limitations) |
| `list_models` | `async () -> dict[str, Any]` | dict | same shape as `ModelsManager.list()` |
| `abort` | `async () -> None` | — | aborts the current run |
| `close` | `async () -> None` | — | idempotent; closes the WebSocket when the session owns it |
| `attach_owner` | `(owner: Any, *, owns: bool) -> None` | — | used by the top-level helpers for lifetime management (not for end users) |

`LettaSession` is an async context manager (`async with client.resume_session(...) as s: ...`).

---

## `QueryStream`

Returned by `LettaAgentClient.query`. Async-iterable, **single use**.

```python
async for message in client.query("hi", QueryOptions(model=..., system=...)):
    ...
```

| member | notes |
| --- | --- |
| `on_close` | optional callback awaited after the stream finishes (set by `letta_sdk.query` to close a self-managed client) |
| `close()` | `async` — abandon the stream |

Raises `ValueError` if `options.model` / `options.system` are missing, and
`RuntimeError` on a second iteration.

---

## Management managers

All methods are `async` and return plain dicts as the server sends them.

### `AgentsManager` (`client.agents`)

| method | signature | notes |
| --- | --- | --- |
| `create` | `(options: CreateAgentOptions \| None = None, **body: Any) -> dict[str, Any]` | `runtime_start` + `create_agent`; kwargs override option fields; returns the agent record |
| `list` | `(**query: Any) -> list[dict[str, Any]]` | `agent_list` |
| `retrieve` | `(agent_id: str) -> dict[str, Any]` | `agent_retrieve` |
| `update` | `(agent_id: str, **body: Any) -> dict[str, Any]` | `agent_update`, e.g. `model=...`, `system=...`, `name=...` |
| `delete` | `(agent_id: str) -> None` | `agent_delete` |

### `ConversationsManager` (`client.conversations`)

| method | signature | notes |
| --- | --- | --- |
| `list` | `(**query: Any) -> list[dict[str, Any]]` | `conversation_list` |
| `retrieve` | `(conversation_id: str) -> dict[str, Any]` | `conversation_retrieve` |
| `create` | `(**body: Any) -> dict[str, Any]` | `conversation_create` |
| `update` | `(conversation_id: str, **body: Any) -> dict[str, Any]` | `conversation_update` |
| `fork` | `(conversation_id: str, **body: Any) -> dict[str, Any]` | `conversation_fork` |
| `list_messages` | `(conversation_id: str, limit: int = 50, before=None, after=None, order=None) -> ListMessagesResult` | `conversation_messages_list` |
| `delete` | `(conversation_id: str) -> None` | **always raises `AppServerRequestError`** — the protocol has no `conversation_delete` |

### `ModelsManager` (`client.models`)

| method | signature | notes |
| --- | --- | --- |
| `list` | `() -> dict[str, Any]` | `list_models` → `{"entries": [...], "available_handles": [...], ...}`; each entry: `id`, `handle`, `label`, `description`, `updateArgs` |

---

## Options

### `CreateAgentOptions`

| field | type | wire key |
| --- | --- | --- |
| `name` | `str \| None` | `name` |
| `description` | `str \| None` | `description` |
| `model` | `str \| None` | `model` (model handle, e.g. `openai/gpt-5.5`) |
| `system_prompt` | `str \| None` | `system` |
| `embedding` | `str \| None` | `embedding` |
| `memory_blocks` | `list[dict]` | `memory` (blocks: `{"label", "value", "description"?}`) |
| `persona` / `human` | `str \| None` | convenience → appended as `persona` / `human` memory blocks |
| `hidden` | `bool \| None` | `hidden` (hidden agents are created unpinned) |
| `memfs` | `bool \| None` | `memfs` |
| `base_tools` | `list[str] \| None` | `base_tools` |
| `tags` | `list[str]` | `tags` |
| `extra_body` | `dict` | merged verbatim into the `create_agent` body |

`to_body()` produces the wire dict.

### `CreateSessionOptions`

| field | type | meaning |
| --- | --- | --- |
| `model` | `str \| None` | model override applied to the session target |
| `cwd` | `str \| None` | working directory for the runtime |
| `permission_mode` | `str \| None` | server permission mode |
| `stateless` | `bool \| None` | run without loading/changing the agent's MemFS |
| `skill_sources` | `list[str] \| None` | restrict skill sources (`[]` disables all) |
| `tools` | `list[ToolSpec]` | local external tools (executed in this process) |
| `can_use_tool` | `CanUseToolCallback \| None` | approval callback for server-side tool calls (see below) |
| `extra_body` | `dict` | raw overrides merged into `runtime_start` |

### `QueryOptions`

| field | type | required | meaning |
| --- | --- | --- | --- |
| `model` | `str \| None` | **yes** | model handle for the ephemeral conversation |
| `system` | `str \| None` | **yes** | system prompt for the ephemeral conversation |
| `model_settings` | `dict \| None` | | forwarded to the conversation |
| `context_window_limit` | `int \| None` | | |
| `stateless` | `bool \| None` | | |
| `extra_body` | `dict` | | merged into the conversation body |

### `ToolSpec`

```python
ToolSpec(
    name="get_time",
    label="Get Time",            # optional, defaults to name
    description="Current UTC time",
    parameters={"type": "object", "properties": {}},
    execute=async_fn,            # async (tool_call_id: str, args: dict) -> result (JSON-able)
)
```

The SDK answers the server's `external_tool_call_request` with your
`execute` result (or a typed error when `execute` is missing/raises).

### `CanUseToolDecision` / `can_use_tool`

```python
def can_use_tool(tool_name: str, tool_input: dict, context: dict) -> CanUseToolDecision:
    ...

decision = CanUseToolDecision(behavior="allow")          # or
decision = CanUseToolDecision(behavior="deny", message="not allowed")
decision = CanUseToolDecision(behavior="allow", updated_input={...})
```

`context` carries the raw request fields: `request_id`, `tool_call_id`,
`permission_suggestions`, `blocked_path`, `diffs`. The callback may be sync
or async; returning a dict is also accepted. Without a callback the SDK
assumes **server-side auto-approval** (the turn stays open across
`requires_approval` stops); a backend that sends `control_request`
approvals without a callback gets a `deny` decision
(`EnterPlanMode` is auto-allowed headless, mirroring the TypeScript SDK).

---

## Message types

All messages are dataclasses subclassing `SDKMessage`:

```python
@dataclass
class SDKMessage:
    type: str      # wire-facing discriminator
    raw: Any       # original wire payload, when available
```

| class | `type` | fields (beyond `SDKMessage`) | emitted when |
| --- | --- | --- | --- |
| `SDKInitMessage` | `init` | `agent_id`, `session_id`, `conversation_id`, `model`, `tools`, `memfs_enabled` | `ready()` returns one |
| `AssistantMessage` | `assistant` | `content`, `uuid`, `otid`, `seq_id`, `run_id` | assistant text/reasoning deltas (text parts only) |
| `ReasoningMessage` | `reasoning` | `content`, `uuid`, `otid`, `seq_id`, `run_id` | model reasoning deltas |
| `ToolCallMessage` | `tool_call` | `tool_call_id`, `tool_name`, `tool_input`, `raw_arguments`, `uuid`, `run_id` | model calls a tool (incl. `approval_request_message`) |
| `ToolResultMessage` | `tool_result` | `tool_call_id`, `content`, `is_error`, `uuid`, `run_id` | tool returned |
| `UsageMessage` | `usage` | `prompt_tokens`, `completion_tokens`, `total_tokens`, `step_count`, `run_ids` | token accounting |
| `ResultMessage` | `result` | `success`, `result`, `error`, `error_code`, `stop_reason`, `duration_ms`, `conversation_id`, `run_ids`, `error_detail`, `recoverable` | **always the last message of a turn** |
| `ErrorMessage` | `error` | `message`, `error_code`, `stop_reason`, `error_detail`, `recoverable`, `run_id` | turn failure (also delivered in-stream) |
| `LoopStatusMessage` | `loop_status` | `status`, `active_run_ids` | `update_loop_status` (e.g. `WAITING_ON_INPUT`, `WAITING_ON_APPROVAL`) |
| `QueueUpdateMessage` | `queue_update` | `queue` | `update_queue` |
| `RetryMessage` | `retry` | `reason`, `attempt`, `max_attempts`, `delay_ms` | model retry |
| `PingMessage` | `ping` | `uuid` | keep-alive |
| `StreamEventMessage` | `stream_event` | `event` (raw delta) | any unrecognized `stream_delta` |
| `UnknownMessage` | `unknown` | `raw` | fallback |

### Content parts (`SendMessage`)

```python
await session.send("plain text")
await session.send([
    {"type": "text", "text": "What is in this image?"},
    image_from_file("chart.png"),
])
```

`MessageContentPart` (TypedDict): `{"type": "text", "text": str}` or
`{"type": "image", "source": {"type": "base64", "media_type": str, "data": str}}`.

---

## Helpers

| helper | signature | notes |
| --- | --- | --- |
| `image_from_base64` | `(data: str, media_type: str = "image/png") -> ImageContentPart` | |
| `image_from_file` | `(path: str \| Path, media_type: str \| None = None) -> ImageContentPart` | media type inferred from the suffix when omitted |
| `text_content` | `(*parts) -> str` | flatten content parts/messages to plain text |
| `TranscriptAccumulator` | `acc = TranscriptAccumulator(); acc.add(message); acc.assistant_text` | fold a stream into an accumulated assistant transcript |

---

## Transport & errors

### `AppServerConnection`

Raw websocket transport (used directly by advanced users, via
`connection=` injection, or by the test suite):

```python
conn = AppServerConnection(
    url="<your app-server ws endpoint>",  # or the SDK default
    auth_token="...",            # or api_key= / token_file=
    request_timeout=30,          # default per-request timeout
    connect_timeout=15,          # handshake timeout
)
resp = await conn.request("list_models", {}, response_type="list_models_response")
conn.on_message(handler)         # handler(dict) -> None; returns unsubscribe()
await conn.send({"type": "ping", ...})
await conn.close()
```

| member | notes |
| --- | --- |
| `connect()` / `closed` | lazy connect / state |
| `request(type_, body=None, *, response_type=None, predicate=None, timeout=None) -> dict` | correlation by `request_id` (`"<type>-<n>"`); `response_type`/`predicate` filter candidates |
| `send(payload) -> None` | fire-and-forget command |
| `on_message(handler) -> unsubscribe` | every incoming protocol message |
| `next_request_id(prefix)` | id generator (public for introspection) |
| `load_token_file(path) -> str` | module-level helper |

### `AppServerConnectionLike`

Structural `Protocol` the SDK depends on (`request`, `send`, `on_message`,
`close`). The concrete class satisfies it; fakes may implement just these
four members.

### Error hierarchy

```text
RuntimeError
└── AppServerError
    ├── AppServerConnectionError   # handshake / transport failure
    ├── AppServerClosedError       # request after close
    ├── AppServerRequestError      # success=false, or a protocol-level failure
    └── AppServerTimeoutError      # no matching response within timeout
```

All are exported from the top level.
