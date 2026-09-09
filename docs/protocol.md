# App-server protocol (as spoken by `letta-sdk-py`)

The Letta **app server** (agent SDK v2) speaks JSON-over-WebSocket. This page
documents exactly what the SDK sends and receives, verified against a live
local server (the SDK's default endpoint, `--ws-auth capability-token`).

---

## 1. Transport

- One WebSocket per logical client (`websockets` library, HTTP `Authorization:
  Bearer <token>` header at handshake).
- Every frame is a single JSON object.
- **`request_id` correlation**: client-generated `"<type>-<n>"` (e.g.
  `runtime_start-3`). Responses echo `request_id`; the SDK matches responses
  to requests by it, *not* by arrival order.
- **`seq` acks**: server pushes carry an integer `seq`; the client must ack
  each one with `{"type": "ack", "seq": n}` (the transport does this
  automatically).
- **`success` / `error`**: command responses carry `success: bool` and, on
  failure, a human-readable `error`. `success: false` surfaces as
  `AppServerRequestError`.

## 2. Commands (client → server)

All via `AppServerConnection.request(type, body)`, which also returns the
matching response dict.

| command | body | response type | used by |
| --- | --- | --- | --- |
| `runtime_start` | see §3 | `runtime_start_response` | `LettaSession.ready()`, `client.create_agent` |
| `input` | see §5 | `input_response` | user messages, tool approvals, external tool results |
| `abort` | `{"runtime": {...}}` | `abort_response` | `session.abort()` |
| `agent_list` | query filters | `agent_list_response` | `client.agents.list()` |
| `agent_retrieve` | `{"agent_id": ...}` | `agent_retrieve_response` | `client.agents.retrieve()`, `resume_session` id resolution |
| `agent_update` | `{"agent_id", ...}` | `agent_update_response` | `client.agents.update()` |
| `agent_delete` | `{"agent_id": ...}` | `agent_delete_response` | `client.agents.delete()` |
| `conversation_list` | query filters | `conversation_list_response` | `client.conversations.list()` |
| `conversation_retrieve` | `{"conversation_id": ...}` | `conversation_retrieve_response` | `client.conversations.retrieve()`, `resume_session` |
| `conversation_create` | `{"agent_id": ...}` | `conversation_create_response` | session creation |
| `conversation_update` | `{"conversation_id", ...}` | `conversation_update_response` | `client.conversations.update()` |
| `conversation_fork` | `{"conversation_id", ...}` | `conversation_fork_response` | `client.conversations.fork()` |
| `conversation_messages_list` | `{"conversation_id", "limit", "before", "after", "order"}` | `conversation_messages_list_response` | `session.list_messages()`, `client.conversations.list_messages()` |
| `list_models` | `{}` | `list_models_response` | `client.models.list()` |
| `create_agent` | `{"personality", "model"?, "tags"?, "pin_global"?}` | `create_agent_response` `{success, agent_id, name, model}` | **personality** agent creation (server resolves its preset catalog) |
| `set_reflection_settings` | `{"runtime": {...}, "settings": {"trigger", "step_count"}, "scope": "both"}` | `set_reflection_settings_response` | **dreaming** — sent once, right after a successful `runtime_start` (skipped for `stateless` sessions) |
| `execute_command` | `{"command_id": "reload", "runtime": {...}}` | `execute_command_response` | app-server settings/mods reload (no SDK method — raw `AppServerConnection.request` only) |

There is **no** `conversation_delete` command in the protocol.

## 3. `runtime_start` body modes

The SDK builds the body in three modes:

1. **Agent + new conversation** (`create_session`) —
   `{"agent_id": ..., "create_conversation": {"body": {}}}`
2. **Agent + existing conversation** (`resume_session("conv-...")`) —
   `{"agent_id": ..., "conversation_id": ...}`
3. **Agent + default conversation** (`resume_session("<agent_id>")`) —
   `{"agent_id": ..., "conversation_id": "default"}` (virtual; see §7)
4. **Agent creation** (`create_agent`) —
   `{"create_agent": {<CreateAgentOptions.to_body() fields>}}`. Skills are
   seeded in this body as ordinary `memory` blocks with labels
   `skills/{name}` (body-only value; the server synthesizes the
   `SKILL.md` frontmatter from the block description). When
   `personality` is set, the SDK instead issues the native `create_agent`
   command (§2) on the throwaway session's connection — the server applies
   its own preset catalog (name, persona, memory files, model defaults).
5. **Agent-free** (`query`) —
   `{"create_conversation": {"body": {"model": ..., "system": ...}}}`

Always included: `client_info: {name: "letta-sdk-py", version: <pkg>,
sdk_type: "python"}`.

Response: `{"success": true, "runtime": {"agent_id", "conversation_id",
"model", "tools", ...}, ...}`. A missing/invalid `runtime` (or
`success: false`) raises `AppServerRequestError`.

## 4. Server pushes

### `stream_delta`

The workhorse: one frame per token-level event.

```json
{
  "type": "stream_delta",
  "seq": 41,
  "runtime": {"agent_id": "agent-local-...", "conversation_id": "local-conv-..."},
  "message_type": "assistant_message",
  "assistant_message": {"content": "chunk", "id": "letta-msg-..."},
  "stop_reason": "end_turn",
  "run_id": "run-abc",
  "total_tokens": 1234
}
```

Recognized `message_type` values and how the SDK maps them:

| wire `message_type` | → SDK message | wire fields read |
| --- | --- | --- |
| `assistant_message` | `AssistantMessage` | `assistant_message.content` (text parts), `.id` → uuid |
| `reasoning_message` | `ReasoningMessage` | `reasoning_message.content`, `.id` |
| `tool_call_message` | `ToolCallMessage` | `tool_call`: `tool_call_id`, `name`, `arguments` |
| `approval_request_message` | `ToolCallMessage` | `tool_call`: **`tool_call_id`** (nested!), `name`, `arguments`; `id` → uuid |
| `tool_return_message` | `ToolResultMessage` | `tool_call_id`, `content`, `is_error` |
| `error_message` | `ErrorMessage` | `error` (message), `error_code`, `stop_reason`, `error_detail`, `recoverable` |
| `loop_error` | `ErrorMessage` | `error` |
| `retry` | `RetryMessage` | `retry_reason`, `retry_attempt`, `retry_max_attempts`, `retry_delay_ms` |
| `ping` | `PingMessage` | `id` |
| anything else | `StreamEventMessage` | raw delta kept in `.event` |

Top-level delta fields with meaning:

| field | meaning |
| --- | --- |
| `stop_reason` | run stopped: `end_turn`, `max_steps`, `requires_approval`, `interrupted`, ... |
| `usage_statistics` | `{"prompt_tokens", "completion_tokens", "total_tokens", "step_count", "run_ids"}` → `UsageMessage` |
| `approval_classification_end` | `{"auto_allowed_tool_call_ids": [...], "auto_denied_tool_call_ids": [...]}` — the server decided approvals itself; the follow-up run continues the same turn |
| `total_tokens` | cumulative token hint (used as fallback usage) |
| `run_id` | a turn may span several server runs (e.g. approval continuation); the SDK collects them into `ResultMessage.run_ids` |

### Other pushes

| push type | → SDK message |
| --- | --- |
| `update_loop_status` | `LoopStatusMessage` (`status`, `active_run_ids`) — e.g. `WAITING_ON_INPUT`, `WAITING_ON_APPROVAL` |
| `update_queue` | `QueueUpdateMessage` |
| `turn_finished` | no message — finalizes the current turn with the latest observed stop reason |
| `external_tool_call_request` | not a message — triggers the local `ToolSpec.execute` (see §6) |
| `control_request` | not a message — approval policy request (see §6.2) |
| `memory_updated` | no message (metadata only) |

## 5. `input` payload kinds

```json
{"type": "input",
 "runtime": {"agent_id": ..., "conversation_id": ...},
 "payload": {
   "kind": "create_message",
   "messages": [{"role": "user", "content": "text or parts",
                 "client_message_id": "sdk-message-<uuid>"}],
   "exclude_interactive_tools": true,
   "client_toolset": {"base": "auto", "include": ["..."]}}}
```

`client_toolset` is present in **every** `create_message` payload of a
session configured with `toolset` (request-scoped; the server applies it
per turn).

Other kinds:

- `approval_response`: `{"kind": "approval_response", "request_id": "...",
  "decision": {"behavior": "allow"|"deny", "message"?: str,
  "updated_input"?: object, "selected_permission_suggestion_ids"?: [str]}}`
- external tool result: `{"kind": "external_tool_response", ...}` with the
  `tool_call_id`, `result` or `error` from `ToolSpec.execute`.

## 6. Tooling

### 6.1 External (client-side) tools

1. `runtime_start` body includes `external_tools: [{name, label,
   description, parameters}]` for each `ToolSpec`.
2. When the model calls one, the server pushes `external_tool_call_request`
   (same run, turn stays open).
3. The SDK runs the registered `execute(tool_call_id, args)` in this
   process and answers with the tool-result `input` payload (matching the
   TypeScript SDK's `execute(toolCallId, args)` contract). A missing or
   raising `execute` is reported back as a typed error — never a hang.

### 6.2 Server-side tool approvals

Two mechanisms, both supported:

**A. Auto-approval (default on the local harness).** A gated tool call
pauses the run with `stop_reason: requires_approval`; the server then
emits `approval_classification_end` with `auto_allowed_tool_call_ids` and
starts a **follow-up run** that finishes the turn. The SDK therefore
*keeps the turn open* on `requires_approval` when no `can_use_tool`
callback is registered (the turn completes on the follow-up run's terminal
stop reason).

**B. `control_request`.** Some backends instead push:

```json
{"type": "control_request", "subtype": "can_use_tool",
 "request_id": "req-...", "tool_name": "memory",
 "input": {"command": "str_replace"},
 "permission_suggestions": [...], "runtime": {...}}
```

The SDK resolves the decision with the TypeScript SDK's
`resolveAppServerToolApproval` ordering and answers via an `input` payload
of kind `approval_response`:

1. tool in `{AskUserQuestion, ExitPlanMode}` (runtime user input) without
   a `can_use_tool` callback → `deny`;
2. session `permission_mode` normalizes to unrestricted
   (`unrestricted` / legacy `bypassPermissions` / `fullAccess`) and the
   tool is not runtime-user-input → `allow` (callback not consulted);
3. `can_use_tool` callback → its `CanUseToolDecision` (raising callback →
   `deny`);
4. `EnterPlanMode` → `allow` (headless default);
5. otherwise → `deny`.

### 6.3 MCP servers (stdio bridge)

`mcp_servers` servers are started **in the SDK process** before
`runtime_start` (JSON-RPC 2.0 over the subprocess's stdin/stdout,
newline-delimited: `initialize` → `notifications/initialized` →
`tools/list`). Each MCP tool is bridged as an external tool named
`mcp__<server>__<tool>` and rides the exact §6.1 protocol: registered in
`runtime_start`'s `external_tools`, executed by the SDK on
`external_tool_call_request`, answered with the tool-result `input` payload
(MCP content blocks are mapped: `text` → text part, `image` → image part,
anything else → JSON text). Server requests such as `ping` are answered;
notifications are ignored. The bridge is torn down (process terminated)
when the session closes or initialization fails.

## 7. Conversations, the `default` virtual conversation

- `create_session` always makes a **real** conversation
  (`runtime.conversation_id` = `local-conv-<n>`), listable via
  `conversation_messages_list`.
- `resume_session("<agent_id>")` uses `conversation_id: "default"`, which is
  **virtual**: turns work, but `conversation_retrieve("default")` and
  message listing fail server-side (`Agent agent-local-default not found`).
  Use `create_session()` when you need a listable conversation.

## 8. Turn lifecycle (as the SDK observes it)

```text
send(input) ──▶ run start ──▶ stream_delta* ──▶ stop_reason
                                                      │
                        ┌────────────────────────────┴──────────────────────┐
                  requires_approval                                  end_turn / max_steps / ...
                        │ (auto-approval or control_request)               │
                        ▼                                                    │
                  approval response ──▶ follow-up run ─▶ stream_delta* ─▶ stop_reason
                                                                  │
                                                            usage_statistics ──▶ UsageMessage
                                                            turn_finished ──▶ ResultMessage (turn complete)
```

Failure stops (`error`, `llm_api_error`, `loop_error`, `stream_closed`,
`cancelled`, `max_steps` without completion) yield an `ErrorMessage`/failed
`ResultMessage`. A trailing-usage grace window (0.15 s) ensures a
`usage_statistics` delta arriving after the stop reason still folds into the
`ResultMessage`.
