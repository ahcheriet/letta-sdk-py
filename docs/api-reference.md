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
| `personality` | `str \| None` | `create_agent.personality` — preset id: `memo`, `blank`, `tutorial`, `linus`, `kawaii`. Uses the app server's native `create_agent` command so the **server** resolves the preset catalog; cannot be combined with `memory_blocks`/`persona`/`human`/`system_prompt` (pass `model`/`tags` to customize) |
| `skills` | `list[str \| AgentSkill \| dict] \| None` | skill directory paths (must contain `SKILL.md`) or inline skills; seeded as `skills/{name}` memory blocks — the agent sees them as `skills/{name}/SKILL.md` in its memory. Requires memfs (the default); skills with support files (`scripts/`, …) are rejected on this backend (TS parity) |
| `dreaming` | `DreamingOptions \| dict \| None` | reflection settings, applied after the runtime starts (see below) |
| `pin_global` | `bool \| None` | pin the new agent globally (default: pinned unless `hidden`) |
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
| `mcp_servers` | `dict[str, dict] \| None` | MCP servers keyed by name (stdio config: `{"command", "args"?, "env"?, "cwd"?}`); started as subprocesses and their tools bridged in as `mcp__<server>__<tool>` external tools (see MCP servers below). A broken server is logged and skipped, never fatal. `http`/`sse` configs are accepted but reported as unavailable — stdio only |
| `toolset` | `ToolsetConfig \| dict \| None` | request-scoped client toolset — sent as `client_toolset` in **every** turn's `create_message` payload. `base` ∈ `auto, codex, codex_snake, default, gemini, gemini_snake, none`; `include` adds bundled tools (deduped) |
| `dreaming` | `DreamingOptions \| dict \| None` | reflection ("dreaming") settings — after `runtime_start` the SDK sends `set_reflection_settings {runtime, settings: {trigger, step_count}, scope: "both"}` (defaults: `trigger="step-count"`, `step_count=5`); skipped for `stateless=True` sessions. `behavior` is rejected (app-server limitation, TS parity) |
| `can_use_tool` | `CanUseToolCallback \| None` | approval callback for server-side tool calls (see below) |
| `extra_body` | `dict` | raw overrides merged into `runtime_start` |

### `ToolsetConfig` / `DreamingOptions` / `AgentSkill`

```python
ToolsetConfig(base="codex", include=["my_tool"])   # or plain dicts
DreamingOptions(trigger="step-count", step_count=10)  # or plain dicts
# trigger: off | step-count | compaction-event
# behavior: reminder | auto-launch  (rejected — not supported by app server)

AgentSkill(name="greet", description="Greet users warmly",
           instructions="Always start with a warm greeting.")
# or a directory path containing SKILL.md (frontmatter: name, description)
```

`letta_sdk.skills` also exposes `resolve_skill_items()`,
`load_skill_directory()`, `parse_skill_markdown()`, `skill_memory_blocks()`,
and `skills_have_support_files()` for direct use.

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
decision = CanUseToolDecision(behavior="allow", updated_input={...},
                              updated_permissions=["suggestion-id"])  # or suggestion objects
```

`context` carries the raw request fields: `request_id`, `tool_call_id`,
`permission_suggestions`, `blocked_path`, `diffs`. The callback may be sync
or async; returning a dict is also accepted. Decision fields: `behavior`,
`message`, `updated_input`, `updated_permissions` (mapped to
`selected_permission_suggestion_ids` on the wire via `id` /
`suggestion_id` / `permission_suggestion_id`), `interrupt` (TS parity
field — accepted, never sent on the wire).

**Decision order** (port of the TS `resolveAppServerToolApproval`):

1. Tool requires real user input (`AskUserQuestion`, `ExitPlanMode`) and
   no callback → `deny` (never auto-allowed).
2. Session `permission_mode` normalizes to **unrestricted**
   (`"unrestricted"`, legacy `"bypassPermissions"` / `"fullAccess"`) and
   the tool is not user-input → **allow without consulting the callback**.
3. Callback registered → its decision (a raising callback → `deny`).
4. `EnterPlanMode` → allow (headless default, mirroring the TypeScript
   SDK).
5. Otherwise → `deny` (`"No canUseTool callback registered"`).

Without a callback the SDK assumes **server-side auto-approval** for the
`requires_approval` stop path (the turn stays open across those stops and
completes on the server's follow-up run).

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

### Image / content helpers

| helper | signature | notes |
| --- | --- | --- |
| `image_from_base64` | `(data: str, media_type: str = "image/png") -> ImageContentPart` | |
| `image_from_file` | `(path: str \| Path, media_type: str \| None = None) -> ImageContentPart` | media type inferred from the suffix when omitted |
| `image_from_url` | `(url: str, timeout: float = 30.0) -> MessageContentPart` | stdlib `urllib`; media type from the `Content-Type` header with extension fallback; raises `ValueError` on non-2xx (hardening vs. TS) |
| `text_content` | `(*parts) -> str` | flatten content parts/messages to plain text |

### Stream-event text extraction

| helper | signature | notes |
| --- | --- | --- |
| `extract_stream_text_delta` | `(event: Any) -> StreamTextDelta \| None` | projects a raw `stream_event` payload onto appendable text. Handles two shapes: content_block style (`{"delta": {"text" \| "reasoning": ...}}`) and message chunk style (`{"message_type": "assistant_message" \| "reasoning_message", ...}`). `None` for non-dicts / no text (TS parity) |
| `StreamTextDelta` | dataclass `kind: "assistant" \| "reasoning"`, `text: str` | the projected slice |

### Tool helpers (typed tool arguments + results)

Port of the TypeScript SDK's tool helper module.

| helper | signature | notes |
| --- | --- | --- |
| `json_result` | `(payload: Any) -> ToolResult` | pretty-prints (`indent=2`, UTF-8 kept) as the result text and attaches the payload as `details` |
| `read_string_param` | `(args, name, *, required=True, default=None, label=None) -> str \| None` | `ValueError("{label} required")` when required and absent — the error text is what the model sees |
| `read_number_param` | `(args, name, *, required=True, default=None, integer=False, label=None) -> int \| float \| None` | numeric coercion; `integer=True` additionally validates integrality. Stricter than TS `parseFloat` by design (rejects `"12abc"`) |
| `read_boolean_param` | `(args, name, *, required=True, default=None, label=None) -> bool \| None` | accepts bools, `"true"/"false"` (case-insensitive), `1/0` |
| `read_string_array_param` | `(args, name, *, required=True, default=None, label=None) -> list[str] \| None` | accepts a list of strings or a comma/whitespace-separated string |

### MCP servers (Model Context Protocol)

Port of the TypeScript SDK's `mcp.ts` / `mcp-runtime.ts`. The **stdio**
transport is implemented with the standard library (JSON-RPC 2.0 over the
subprocess's stdin/stdout, newline-delimited); `http`/`sse` server configs
are accepted for type parity but reported as unavailable at connect time.

| helper | signature | notes |
| --- | --- | --- |
| `connect_mcp_servers` | `(servers, *, cwd=None, reserved_tool_names=None, log=None) -> McpToolBridge` | connect in parallel; per-server failures are logged and skipped (TS parity). Tools are named `mcp__<server>__<tool>` (sanitized, collision-suffixed `_2`, `_3`, …) |
| `McpToolBridge` | `tools: list[ToolSpec]`, `close() -> Awaitable[None]` | the connected servers' tools; `close()` terminates every server process (idempotent) — the session does this for you on `close()` and on initialization failure |
| `expand_mcp_tool_wildcards` | `(allowed_tools: list[str] \| None, mcp_tools: Iterable[str]) -> list[str] \| None` | expand Claude-style `mcp__<server>*` wildcards into exact tool names (deduped); unmatched wildcards are dropped; `None` → `None` (TS parity) |

Direct use (without the `mcp_servers` session option):

```python
bridge = await connect_mcp_servers(
    {"calc": {"command": "python", "args": ["my_mcp_server.py"]}}
)
for tool in bridge.tools:            # ToolSpec objects, usable in `tools=[...]`
    ...
await bridge.close()
```

### Skill helpers (`letta_sdk.skills`)

The portable core of the TypeScript skill loading (everything except the
Node-only `skill-node.ts` runtime).

| helper | signature | notes |
| --- | --- | --- |
| `resolve_skill_items` | `(skills: list[str \| AgentSkill \| dict]) -> list[AgentSkill]` | normalizes directory paths / inline skills / dicts; validates names (pattern, duplicates) |
| `load_skill_directory` | `(dir_path: str \| Path) -> AgentSkill` | reads `SKILL.md`; frontmatter `name`/`description` + body; directory name as fallback |
| `parse_skill_markdown` | `(content: str) -> tuple[str \| None, str \| None, str]` | frontmatter parser (YAML folded/literal scalars, quote stripping); no frontmatter → whole content is the body |
| `skill_memory_blocks` | `(skills: list[AgentSkill]) -> list[dict]` | the `skills/{name}` memory blocks used for seeding |
| `skills_have_support_files` | `(skills: list[AgentSkill]) -> bool` | `True` if any skill carries `scripts/`, `references/`, … (unsupported on this backend) |
| `AgentSkill` | dataclass `name`, `description`, `instructions`, `files` | one skill; `files` = support files (rejected by `create_agent` on this backend) |

### Transcript reconciliation

| member | notes |
| --- | --- |
| `TranscriptAccumulator` | fold SDK messages into stable, render-ready rows (port of the TS transcript). `add(message)` per streamed message; idempotent under per-run replays (per-run `seq_id` thresholds, 64-run bound, anonymous-run bucket) |
| `acc.rows` | `list[TranscriptRow]` — text rows (`user`/`assistant`/`reasoning`) carry `text`; tool rows carry `tool_name`, `tool_input`, `result`, `status` |
| `TranscriptRow` | `kind`, stable `key`, `uuid`/`otid`/`run_id`/`seq_id`, `text` or `tool_*` fields |
| `TranscriptToolResult` | `content`, `is_error` — the tool row's result |
| `acc.rebase(page, *, order=...)` | merge a `list_messages()` history page mid-run; `order` = page order (auto-detect when omitted) |
| `acc.assistant_text` | legacy plain-text accumulation (unchanged pre-rows behavior) |
| `acc.reset()` | drop all rows, replay state, and the legacy message list |

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
