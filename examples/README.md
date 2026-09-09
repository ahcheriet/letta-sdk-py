# Examples & tutorial

Each example is a small, self-contained program. None of them hardcode a
server address, token, or model — everything comes from the environment
(see below), so the same files run against any Letta Code app server.

## Setup

1. Install the SDK:

   ```bash
   pip install -e .
   ```

2. Export your connection settings:

   ```bash
   export LETTA_MODEL="<provider>/<model>"     # required — a handle your server has
   # only when your server is not at the SDK's default endpoint:
   export LETTA_APP_SERVER_URL="ws://<host>:<port>/ws"
   # only when the server requires authentication:
   export LETTA_TOKEN_FILE="<path to the capability token file>"
   # (or: export LETTA_TOKEN="<token>")
   ```

   Find out which model handles your server has:

   ```bash
   python - <<'PY'
   import asyncio
   from letta_sdk import LettaAgentClient
   from _env import client_kwargs  # run this from examples/

   async def main():
       client = LettaAgentClient(**client_kwargs())
       print(await client.models.list())
       await client.close()

   asyncio.run(main())
   PY
   ```

## The examples

| file | what it shows |
| --- | --- |
| [`quickstart.py`](quickstart.py) | the 3-step tour: create an agent → one-shot `prompt()` → read `result.result` |
| [`streaming.py`](streaming.py) | `create_session()` + `send()` + `stream()` — tokens, tool calls, and usage in real time |
| [`resume.py`](resume.py) | persisting agent + conversation ids so state survives across process runs (run it twice) |
| [`images.py`](images.py) | multimodal input: `send([text, image_from_file(...)])` |
| [`external_tools.py`](external_tools.py) | local Python functions as agent tools (`ToolSpec` + `execute`) |
| [`approvals.py`](approvals.py) | the `can_use_tool` callback — allow/deny server-side tool calls from your code |
| [`chat.py`](chat.py) | a full chat app with persistent memory (`/memory`, `/reset`, `/exit`) |

Suggested order: quickstart → streaming → resume → external_tools →
approvals → images → chat.

## Tutorial: your first turn

```python
import asyncio
from letta_sdk import AssistantMessage, CreateAgentOptions, LettaAgentClient

async def main():
    client = LettaAgentClient()          # endpoint/token from env (see above)
    agent_id = await client.create_agent(
        CreateAgentOptions(
            name="demo",
            model="<provider>/<model>",  # from LETTA_MODEL
            system_prompt="You are a concise assistant.",
        )
    )
    session = client.create_session(agent_id)   # new real conversation
    await session.ready()
    await session.send("Say hello in three words.")
    async for message in session.stream():
        if isinstance(message, AssistantMessage):
            print(message.content, end="", flush=True)
    print()
    await session.close()
    await client.close()

asyncio.run(main())
```

Every turn ends with a terminal message — either `ResultMessage` (inspect
`.success`, `.result`, `.stop_reason`, `.duration_ms`) or an error. If you
only care about the answer, `await client.prompt(agent_id, "...")` does
send + drain for you.

## Tutorial: a chat that remembers

`chat.py` is the complete pattern:

```text
run 1:  python examples/chat.py
you>    remember that I like dark roast coffee
nova>   Got it — dark roast it is! ...

run 2:  python examples/chat.py        (new process)
you>    what coffee do I like?
nova>   Dark roast!
```

Why it works:

1. **The agent is the memory.** `chat.py` saves only the *agent id*
   (`agent.json`). The persona, core memory blocks, and the conversation
   all live on the app server; `resume_session(agent_id)` reattaches.
2. **The agent decides what to store.** It has a `memory` tool (a server
   tool, gated by approval) that writes facts to its memory files — the
   SDK's approval handling (see `approvals.py`) keeps those turns
   completing automatically.
3. **Local state is disposable.** Delete `agent.json` (or type `/reset`)
   and the next run starts a brand-new agent.

## Troubleshooting

| symptom | likely cause |
| --- | --- |
| `LETTA_MODEL is not set` | export the model handle (see Setup) |
| `AppServerConnectionError` | wrong `LETTA_APP_SERVER_URL`, or the server isn't running |
| `AppServerRequestError: ...401...` | token missing/wrong — set `LETTA_TOKEN_FILE`/`LETTA_TOKEN` |
| `AppServerTimeoutError` | server busy or model overloaded — retry, or raise `request_timeout` |
| `query()` fails with a 401 cloud message | your server doesn't support agent-free conversations — use an agent-based session instead |
| model says "image omitted: model does not support images" | the SDK sent the image fine — the **model handle is text-only**. Point `LETTA_MODEL` at a vision-capable model (e.g. a Qwen-VL / GPT-4o-class endpoint) and re-run |
