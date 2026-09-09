"""Tool helpers: typed parameter reading + pretty JSON tool results.

The SDK's ``tool_helpers`` module gives tool handlers the same
argument-discipline helpers the TypeScript SDK ships:

- ``read_string_param`` / ``read_number_param`` / ``read_boolean_param`` /
  ``read_string_array_param`` — validated, labeled parameter extraction
  (required fields raise ``ValueError("... required")`` for the model to
  see and retry),
- ``json_result`` — pretty-prints a payload as the tool's text output and
  attaches it as structured ``details``.

Run:
    export LETTA_MODEL=<model handle your app server has>
    python examples/tool_helpers.py
"""

from __future__ import annotations

import asyncio

from letta_sdk import (
    CreateAgentOptions,
    CreateSessionOptions,
    LettaAgentClient,
    ToolSpec,
)
from letta_sdk.tool_helpers import (
    json_result,
    read_number_param,
    read_string_param,
)

from _env import client_kwargs, model


async def scale_text(tool_call_id: str, args: dict):
    """Repeat a string N times — with strictly validated parameters."""
    text = read_string_param(args, "text", required=True, label="text")
    factor = read_number_param(args, "factor", required=True, integer=True)
    if text is None or factor is None:
        raise ValueError("text and factor are both required")
    if float(factor) != int(factor):
        raise ValueError("factor must be a whole number")
    n = int(factor)
    return json_result(
        {
            "original": text,
            "factor": n,
            "repeated": text * n,
            "length": len(text * n),
        }
    )


async def main() -> None:
    client = LettaAgentClient(**client_kwargs())
    agent_id: str | None = None
    try:
        agent_id = await client.create_agent(
            CreateAgentOptions(name="tool-helpers", model=model())
        )
        session = client.create_session(
            agent_id,
            CreateSessionOptions(
                tools=[
                    ToolSpec(
                        name="scale_text",
                        description=(
                            "Repeat a string an integer number of times. "
                            "Parameters: text (string, required), "
                            "factor (integer, required)."
                        ),
                        parameters={
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "factor": {"type": "integer"},
                            },
                            "required": ["text", "factor"],
                        },
                        execute=scale_text,
                    )
                ]
            ),
        )
        result = await session.prompt(
            "Use the scale_text tool with text='ok' and factor=3, "
            "then tell me the resulting length."
        )
        print(result.result)
        await session.close()
    finally:
        if agent_id:
            await client.agents.delete(agent_id)
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
