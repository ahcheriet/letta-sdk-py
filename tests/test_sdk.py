from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from letta_sdk import (
    CreateAgentOptions,
    CreateSessionOptions,
    LettaAgentClient,
    QueryOptions,
    ResultMessage,
    TranscriptAccumulator,
    resume_session,
    text_content,
)


@dataclass
class FakeStream:
    items: list[object]

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for item in self.items:
            yield item


class FakeMessagesAPI:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.create_calls: list[tuple[str, dict[str, object]]] = []
        self.list_calls: list[tuple[str, dict[str, object]]] = []

    async def create(self, conversation_id: str, **kwargs: object) -> FakeStream:
        self.create_calls.append((conversation_id, kwargs))
        return FakeStream(self.responses)

    async def list(self, conversation_id: str, **kwargs: object) -> dict[str, object]:
        self.list_calls.append((conversation_id, kwargs))
        return {"conversation_id": conversation_id, "items": []}


class FakeConversationsAPI:
    def __init__(self, responses: list[object]) -> None:
        self.messages = FakeMessagesAPI(responses)
        self.created: list[dict[str, object]] = []
        self.deleted: list[str] = []

    async def create(self, **kwargs: object) -> object:
        self.created.append(kwargs)
        return SimpleNamespace(id="conv-created")

    async def list(self, **kwargs: object) -> list[str]:
        return ["conv-created"]

    async def retrieve(self, conversation_id: str) -> object:
        return SimpleNamespace(id=conversation_id)

    async def update(self, conversation_id: str, **kwargs: object) -> object:
        return SimpleNamespace(id=conversation_id, **kwargs)

    async def delete(self, conversation_id: str) -> None:
        self.deleted.append(conversation_id)


class FakeAgentsAPI:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, object]] = []
        self.deleted: list[str] = []

    async def create(self, **kwargs: object) -> object:
        self.create_calls.append(kwargs)
        return SimpleNamespace(id="agent-created")

    async def list(self, **kwargs: object) -> list[str]:
        return ["agent-created"]

    async def retrieve(self, agent_id: str) -> object:
        return SimpleNamespace(id=agent_id)

    async def update(self, agent_id: str, **kwargs: object) -> object:
        return SimpleNamespace(id=agent_id, **kwargs)

    async def delete(self, agent_id: str) -> None:
        self.deleted.append(agent_id)


class FakeModelsAPI:
    async def list(self, **kwargs: object) -> list[str]:
        return ["model-a"]


class FakeAsyncLetta:
    def __init__(self, responses: list[object] | None = None) -> None:
        self.agents = FakeAgentsAPI()
        self.conversations = FakeConversationsAPI(responses or [])
        self.models = FakeModelsAPI()
        self.post_calls: list[tuple[str, dict[str, object]]] = []
        self.closed = False

    async def post(self, path: str, *, body: dict[str, object]) -> dict[str, str]:
        self.post_calls.append((path, body))
        return {"id": "conv-ephemeral"}

    async def close(self) -> None:
        self.closed = True


class LettaSdkTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_agent_maps_pythonic_options(self) -> None:
        fake = FakeAsyncLetta()
        client = LettaAgentClient(client=fake)

        agent_id = await client.create_agent(
            CreateAgentOptions(
                name="Nora",
                model="openai/gpt-5.6-luna",
                system_prompt="You are helpful.",
                tags=["research"],
                hidden=True,
            )
        )

        self.assertEqual(agent_id, "agent-created")
        self.assertEqual(
            fake.agents.create_calls[0],
            {
                "name": "Nora",
                "model": "openai/gpt-5.6-luna",
                "system": "You are helpful.",
                "tags": ["research"],
                "hidden": True,
            },
        )

    async def test_resume_session_streams_messages_and_result(self) -> None:
        fake = FakeAsyncLetta(
            responses=[
                SimpleNamespace(message_type="assistant_message", id="m1", content="Hello"),
                SimpleNamespace(message_type="usage_statistics", prompt_tokens=3, completion_tokens=5, total_tokens=8),
                SimpleNamespace(message_type="stop_reason", stop_reason="end_turn"),
            ]
        )
        client = LettaAgentClient(client=fake)
        session = client.resume_session("agent-123", CreateSessionOptions(model="model-a", max_steps=7))

        await session.send([text_content("Hi")])
        messages = [message async for message in session.stream()]

        self.assertEqual(messages[0].type, "assistant")
        self.assertEqual(messages[0].content, "Hello")
        self.assertEqual(messages[-1].type, "result")
        self.assertTrue(messages[-1].duration_ms is not None and messages[-1].duration_ms >= 0)
        conversation_id, payload = fake.conversations.messages.create_calls[0]
        self.assertEqual(conversation_id, "default")
        self.assertEqual(payload["agent_id"], "agent-123")
        self.assertEqual(payload["override_model"], "model-a")
        self.assertEqual(payload["max_steps"], 7)
        self.assertEqual(payload["input"], [{"type": "text", "text": "Hi"}])
        self.assertEqual(messages[-1], ResultMessage(type="result", raw={"stop_reason": "end_turn"}, success=True, stop_reason="end_turn", conversation_id=None, duration_ms=messages[-1].duration_ms, usage=messages[1], error=None))

    async def test_create_session_creates_conversation_and_lists_history(self) -> None:
        fake = FakeAsyncLetta()
        client = LettaAgentClient(client=fake)

        session = await client.create_session(
            "agent-xyz",
            CreateSessionOptions(
                model="model-b",
                summary="Sprint",
                description="Planning",
                hidden=True,
                extra_body={"metadata": {"team": "sdk"}},
            ),
        )
        history = await session.list_messages(limit=20)

        self.assertEqual(fake.conversations.created[0], {
            "agent_id": "agent-xyz",
            "model": "model-b",
            "summary": "Sprint",
            "description": "Planning",
            "hidden": True,
            "metadata": {"team": "sdk"},
        })
        self.assertEqual(history["conversation_id"], "conv-created")
        self.assertEqual(fake.conversations.messages.list_calls[0], ("conv-created", {"limit": 20}))

    async def test_agents_manager_can_create_agents(self) -> None:
        fake = FakeAsyncLetta()
        client = LettaAgentClient(client=fake)

        agent = await client.agents.create(name="Nora")

        self.assertEqual(agent.id, "agent-created")
        self.assertEqual(fake.agents.create_calls, [{"name": "Nora"}])

    async def test_query_uses_ephemeral_conversation_endpoint(self) -> None:
        fake = FakeAsyncLetta(
            responses=[SimpleNamespace(message_type="assistant_message", id="m1", content="Done")]
        )
        client = LettaAgentClient(client=fake)

        messages = [
            message
            async for message in client.query(
                "Ping",
                QueryOptions(
                    model="openai/gpt-5.6-luna",
                    system_prompt="Be terse.",
                    max_steps=3,
                ),
            )
        ]

        self.assertEqual(messages[0].type, "assistant")
        self.assertEqual(fake.post_calls[0], ("/v1/conversations/ephemeral", {"model": "openai/gpt-5.6-luna", "system": "Be terse."}))
        self.assertEqual(fake.conversations.messages.create_calls[0][0], "conv-ephemeral")
        self.assertEqual(fake.conversations.messages.create_calls[0][1]["max_steps"], 3)
        self.assertEqual(fake.agents.create_calls, [])
        self.assertEqual(fake.agents.deleted, [])

    async def test_query_requires_model_and_system_prompt(self) -> None:
        fake = FakeAsyncLetta()
        client = LettaAgentClient(client=fake)

        with self.assertRaisesRegex(ValueError, "QueryOptions.model"):
            [message async for message in client.query("Ping", QueryOptions(system_prompt="Be terse."))]
        with self.assertRaisesRegex(ValueError, "QueryOptions.system_prompt"):
            [message async for message in client.query("Ping", QueryOptions(model="openai/gpt-5.6-luna"))]

    async def test_prompt_returns_result_message(self) -> None:
        fake = FakeAsyncLetta(
            responses=[
                SimpleNamespace(message_type="assistant_message", id="m1", content="Done"),
                SimpleNamespace(message_type="stop_reason", stop_reason="end_turn"),
            ]
        )
        client = LettaAgentClient(client=fake)

        result = await client.prompt("agent-abc", "Hello")

        self.assertEqual(result.type, "result")
        self.assertTrue(result.success)
        self.assertEqual(result.stop_reason, "end_turn")

    async def test_transcript_accumulator_and_session_cleanup(self) -> None:
        fake = FakeAsyncLetta(
            responses=[SimpleNamespace(message_type="assistant_message", id="m1", content="A")] 
        )
        session = resume_session("agent-999", client=LettaAgentClient(client=fake))
        accumulator = TranscriptAccumulator()

        await session.send("hello")
        async for message in session.stream():
            accumulator.add(message)
        await session.close()

        self.assertEqual(accumulator.assistant_text, "A")
        with self.assertRaises(RuntimeError):
            await session.send("again")


if __name__ == "__main__":
    unittest.main()
