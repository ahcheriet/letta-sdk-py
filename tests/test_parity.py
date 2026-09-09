"""Parity tests for the ports from the TypeScript SDK.

Covers the v0.3 gap-fill modules:

- ``letta_sdk.transcript`` — the four reconciliation rules
  (family+otid keying, per-run seqId replay suppression, tool_call_id
  merging, rebase backfill) plus the preserved legacy API.
- ``letta_sdk.images.image_from_url`` — media-type detection with the
  network mocked out (no real I/O in tests).
- ``letta_sdk.stream_events.extract_stream_text_delta`` — both wire shapes.
- ``letta_sdk.tool_helpers`` — json_result + the four typed param readers.
"""

from __future__ import annotations

import base64
import json
from typing import Any, cast

import pytest

from letta_sdk import (
    AssistantMessage,
    ReasoningMessage,
    ResultMessage,
    StreamEventMessage,
    ToolCallMessage,
    ToolResultMessage,
)
from letta_sdk.images import image_from_url
from letta_sdk.stream_events import StreamTextDelta, extract_stream_text_delta
from letta_sdk.tool_helpers import (
    json_result,
    read_boolean_param,
    read_number_param,
    read_string_array_param,
    read_string_param,
)
from letta_sdk.transcript import TranscriptAccumulator
from letta_sdk.types import ImageContentPart, ImageSource, ListMessagesResult


# ═══════════════════════════════════════════════════════════════
# transcript: the four reconciliation rules
# ═══════════════════════════════════════════════════════════════


def test_transcript_rule1_same_family_otid_merges():
    acc = TranscriptAccumulator()
    acc.add(AssistantMessage(type="assistant", content="Hel", otid="t1"))
    acc.add(AssistantMessage(type="assistant", content="lo", otid="t1"))
    rows = acc.rows
    assert len(rows) == 1
    assert rows[0].kind == "assistant"
    assert rows[0].text == "Hello"
    assert rows[0].otid == "t1"
    assert rows[0].key == "assistant:otid:t1"


def test_transcript_rule1_same_otid_across_families_stays_split():
    acc = TranscriptAccumulator()
    acc.add(ReasoningMessage(type="reasoning", content="hmm", otid="t1"))
    acc.add(AssistantMessage(type="assistant", content="hi", otid="t1"))
    rows = acc.rows
    assert len(rows) == 2
    kinds = {row.kind for row in rows}
    assert kinds == {"assistant", "reasoning"}
    assert {row.key for row in rows} == {
        "assistant:otid:t1",
        "reasoning:otid:t1",
    }


def test_transcript_rule2_replay_suppressed_new_run_clean_threshold():
    acc = TranscriptAccumulator()
    acc.add(
        AssistantMessage(type="assistant", content="a1", run_id="r1", seq_id=1)
    )
    acc.add(
        AssistantMessage(
            type="assistant", content="a2", uuid="m1", run_id="r1", seq_id=2
        )
    )
    assert len(acc.rows) == 2

    # Replayed position (seq <= high-water mark of run r1): dropped.
    acc.add(
        AssistantMessage(
            type="assistant", content="REPLAY", uuid="m1", run_id="r1", seq_id=2
        )
    )
    rows = acc.rows
    assert len(rows) == 2
    assert "REPLAY" not in "".join(r.text for r in rows)

    # A new run starts from a clean threshold: seq 1 is accepted again.
    acc.add(
        AssistantMessage(
            type="assistant", content="b1", run_id="r2", seq_id=1
        )
    )
    rows = acc.rows
    assert len(rows) == 3
    assert rows[2].text == "b1"
    assert rows[2].run_id == "r2"


def test_transcript_rule2_messages_without_seq_never_suppressed():
    acc = TranscriptAccumulator()
    acc.add(AssistantMessage(type="assistant", content="x", uuid="u1"))
    # Same uuid, no seq id: identity-based dedup keeps one row, no crash.
    acc.add(AssistantMessage(type="assistant", content="y", uuid="u1"))
    rows = acc.rows
    assert len(rows) == 1
    assert rows[0].text == "xy"


def test_transcript_rule3_tool_fragments_and_result_one_row():
    acc = TranscriptAccumulator()
    acc.add(
        ToolCallMessage(
            type="tool_call",
            tool_call_id="tc1",
            tool_name="get_weather",
            raw_arguments='{"city":',
            uuid="env-call-1",
        )
    )
    rows = acc.rows
    assert len(rows) == 1
    assert rows[0].status == "streaming"
    assert rows[0].tool_name == "get_weather"
    assert rows[0].arguments_complete is False

    acc.add(
        ToolCallMessage(
            type="tool_call",
            tool_call_id="tc1",
            raw_arguments=' "nyc"}',
            uuid="env-call-2",
        )
    )
    row = acc.rows[0]
    assert row.status == "ready"
    assert row.tool_input == {"city": "nyc"}
    assert row.raw_arguments == '{"city": "nyc"}'
    assert row.arguments_complete is True

    acc.add(
        ToolResultMessage(
            type="tool_result",
            tool_call_id="tc1",
            content="sunny",
            uuid="env-result-1",
        )
    )
    rows = acc.rows
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "complete"
    assert row.result is not None
    assert row.result.content == "sunny"
    assert row.result.is_error is False
    # Envelope identities stay separately visible: the row's uuid is the
    # tool_call envelope; the result envelope is on the result.
    assert row.uuid == "env-call-1"
    assert row.result.uuid == "env-result-1"
    assert row.key == "tool_call:id:tc1"


def test_transcript_rule3_distinct_tool_calls_stay_separate():
    acc = TranscriptAccumulator()
    acc.add(
        ToolCallMessage(
            type="tool_call",
            tool_call_id="tc1",
            tool_name="a",
            raw_arguments="{}",
        )
    )
    acc.add(
        ToolCallMessage(
            type="tool_call",
            tool_call_id="tc2",
            tool_name="b",
            raw_arguments="{}",
        )
    )
    rows = acc.rows
    assert len(rows) == 2
    assert [r.tool_call_id for r in rows] == ["tc1", "tc2"]


def test_transcript_rule3_raw_wrapper_never_becomes_tool_input():
    # The protocol layer wraps unparseable fragments as {"raw": ...}; that
    # wrapper must not be reported as parsed arguments.
    acc = TranscriptAccumulator()
    acc.add(
        ToolCallMessage(
            type="tool_call",
            tool_call_id="tc1",
            tool_name="t",
            tool_input={"raw": '{"a":'},
            raw_arguments='{"a":',
        )
    )
    row = acc.rows[0]
    assert row.tool_input == {}
    assert row.arguments_complete is False
    assert row.raw_arguments == '{"a":'


def test_transcript_rule4_rebase_backfill_replaces_and_reorders():
    acc = TranscriptAccumulator()
    # Live-only row first.
    acc.add(AssistantMessage(type="assistant", content="live tail", otid="live1"))

    # Newest-first history page (as list_messages() returns by default).
    page = [
        {
            "id": "h2",
            "message_type": "assistant_message",
            "content": "hi there",
            "seq_id": 2,
        },
        {
            "id": "h1",
            "message_type": "user_message",
            "content": "hello",
            "seq_id": 1,
        },
    ]
    rows = acc.rebase(page)
    assert [r.kind for r in rows] == ["user", "assistant", "assistant"]
    assert rows[0].text == "hello"
    assert rows[1].text == "hi there"
    assert rows[2].text == "live tail"  # live-only row stays last

    # The page proves seq 2 for the anonymous run: a replayed live delta
    # at or below it is now suppressed.
    acc.add(
        AssistantMessage(type="assistant", content="REPLAY", seq_id=2)
    )
    assert len(acc.rows) == 3
    assert "REPLAY" not in "".join(r.text for r in acc.rows)


def test_transcript_rule4_rebase_explicit_order_and_replace_semantics():
    acc = TranscriptAccumulator()
    # Live partial row that the history page completes (same uuid).
    acc.add(AssistantMessage(type="assistant", content="par", uuid="h1"))

    rows = acc.rebase(
        [{"id": "h1", "message_type": "assistant_message", "content": "partial"}],
        order="asc",
    )
    assert len(rows) == 1
    assert rows[0].text == "partial"  # replace, not append


def test_transcript_rule4_rebase_accepts_list_messages_result():
    acc = TranscriptAccumulator()
    result = ListMessagesResult(
        messages=[
            {"id": "h1", "message_type": "user_message", "content": "one", "seq_id": 1},
            {"id": "h2", "message_type": "user_message", "content": "two", "seq_id": 2},
        ]
    )
    rows = acc.rebase(result)
    assert [r.text for r in rows] == ["one", "two"]


def test_transcript_stream_events_fold_into_live_rows():
    acc = TranscriptAccumulator()
    acc.add(
        StreamEventMessage(
            type="stream_event",
            event={"type": "content_block_delta", "delta": {"text": "ab"}},
        )
    )
    acc.add(
        StreamEventMessage(
            type="stream_event",
            event={"type": "content_block_delta", "delta": {"text": "cd"}},
        )
    )
    acc.add(
        StreamEventMessage(
            type="stream_event",
            event={"type": "content_block_delta", "delta": {"reasoning": "think"}},
        )
    )
    rows = acc.rows
    assert [r.kind for r in rows] == ["assistant", "reasoning"]
    assert rows[0].text == "abcd"
    assert rows[1].text == "think"

    # A stream_event with no text payload is ignored.
    acc.add(StreamEventMessage(type="stream_event", event={"type": "message_stop"}))
    assert len(acc.rows) == 2


def test_transcript_legacy_api_preserved():
    acc = TranscriptAccumulator()
    first = acc.add(AssistantMessage(type="assistant", content="Hel"))
    assert first is not None  # add() returns the message
    acc.add(ResultMessage(type="result", success=True, stop_reason="end_turn"))
    acc.add(AssistantMessage(type="assistant", content="lo"))
    assert acc.assistant_text == "Hello"
    assert len(acc.messages) == 3
    # Unknown families do not create rows.
    assert len(acc.rows) == 2  # two anonymous assistant slices

    acc.reset()
    assert acc.rows == []
    assert acc.assistant_text == ""
    assert acc.messages == []


# ═══════════════════════════════════════════════════════════════
# images: image_from_url (network mocked)
# ═══════════════════════════════════════════════════════════════


class _FakeHTTPResponse:
    def __init__(self, body: bytes, content_type: str, status: int = 200) -> None:
        self._body = body
        self.status = status
        self.headers: dict[str, str] = {"Content-Type": content_type}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _mock_urlopen(monkeypatch: pytest.MonkeyPatch, response: _FakeHTTPResponse) -> None:
    def fake_urlopen(url: str, timeout: float = 30.0) -> _FakeHTTPResponse:
        assert url
        assert timeout > 0
        return response

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def _assert_image_part(part: Any, expected_media: str) -> ImageSource:
    image = cast(ImageContentPart, part)
    assert image["type"] == "image"
    source = image["source"]
    assert source.get("media_type") == expected_media
    return source


def test_image_from_url_media_type_from_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = {
        "image/jpeg": "image/jpeg",
        "image/gif": "image/gif",
        "image/webp": "image/webp",
        "image/png": "image/png",
        "application/octet-stream": "image/png",  # default
    }
    for content_type, expected in cases.items():
        _mock_urlopen(monkeypatch, _FakeHTTPResponse(b"DATA", content_type))
        part = image_from_url("https://example.com/img")
        source = _assert_image_part(part, expected)
        assert source.get("data") == base64.b64encode(b"DATA").decode("ascii")


def test_image_from_url_extension_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generic = _FakeHTTPResponse(b"X", "application/octet-stream")
    for url, expected in [
        ("https://example.com/pic.jpg", "image/jpeg"),
        ("https://example.com/pic.jpeg", "image/jpeg"),
        ("https://example.com/pic.GIF", "image/gif"),
        ("https://example.com/pic.webp", "image/webp"),
        ("https://example.com/pic.png", "image/png"),
    ]:
        _mock_urlopen(monkeypatch, generic)
        assert _assert_image_part(image_from_url(url), expected), url


def test_image_from_url_base64_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    body = bytes(range(256))
    _mock_urlopen(monkeypatch, _FakeHTTPResponse(body, "image/png"))
    source = _assert_image_part(image_from_url("https://example.com/x.png"), "image/png")
    assert base64.b64decode(source.get("data", "")) == body


def test_image_from_url_non_2xx_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_urlopen(monkeypatch, _FakeHTTPResponse(b"", "image/png", status=404))
    with pytest.raises(ValueError, match="404"):
        image_from_url("https://example.com/missing.png")


def test_image_from_url_network_error_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.error

    def fake_urlopen(url: str, timeout: float = 30.0) -> Any:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ValueError, match="Failed to fetch"):
        image_from_url("https://example.com/x.png")


# ═══════════════════════════════════════════════════════════════
# stream_events: extract_stream_text_delta
# ═══════════════════════════════════════════════════════════════


def test_stream_delta_reasoning_field() -> None:
    delta = extract_stream_text_delta({"delta": {"reasoning": "think"}})
    assert delta == StreamTextDelta(kind="reasoning", text="think")


def test_stream_delta_text_field() -> None:
    delta = extract_stream_text_delta(
        {"type": "content_block_delta", "delta": {"text": "hi"}}
    )
    assert delta == StreamTextDelta(kind="assistant", text="hi")


def test_stream_delta_delta_shape_wins_over_message_type() -> None:
    delta = extract_stream_text_delta(
        {"message_type": "assistant_message", "delta": {"text": "x"}}
    )
    assert delta == StreamTextDelta(kind="assistant", text="x")


def test_stream_delta_reasoning_message_string_field() -> None:
    delta = extract_stream_text_delta(
        {"message_type": "reasoning_message", "reasoning": "R"}
    )
    assert delta == StreamTextDelta(kind="reasoning", text="R")


def test_stream_delta_reasoning_message_content_parts() -> None:
    delta = extract_stream_text_delta(
        {
            "message_type": "reasoning_message",
            "content": [{"type": "text", "text": "R2"}],
        }
    )
    assert delta == StreamTextDelta(kind="reasoning", text="R2")


def test_stream_delta_assistant_message_string_content() -> None:
    delta = extract_stream_text_delta(
        {"message_type": "assistant_message", "content": "plain"}
    )
    assert delta == StreamTextDelta(kind="assistant", text="plain")


def test_stream_delta_assistant_message_list_content() -> None:
    delta = extract_stream_text_delta(
        {
            "message_type": "assistant_message",
            "content": [{"text": "a"}, "b", 123, {}, None],
        }
    )
    assert delta == StreamTextDelta(kind="assistant", text="ab")


def test_stream_delta_assistant_message_dict_content() -> None:
    delta = extract_stream_text_delta(
        {"message_type": "assistant_message", "content": {"text": "d"}}
    )
    assert delta == StreamTextDelta(kind="assistant", text="d")


def test_stream_delta_no_match_returns_none() -> None:
    assert extract_stream_text_delta({}) is None
    assert extract_stream_text_delta({"delta": {"text": ""}}) is None
    assert extract_stream_text_delta({"message_type": "tool_call_message"}) is None
    assert extract_stream_text_delta(
        {"message_type": "assistant_message", "content": [42, None]}
    ) is None


# ═══════════════════════════════════════════════════════════════
# tool_helpers
# ═══════════════════════════════════════════════════════════════


def test_json_result_shape() -> None:
    result = json_result({"ok": True, "n": 2})
    assert result.details == {"ok": True, "n": 2}
    (part,) = result.content
    assert part["type"] == "text"
    assert json.loads(part["text"]) == {"ok": True, "n": 2}
    assert "\n" in part["text"]  # pretty-printed (indent=2)


def test_read_string_param_happy_paths() -> None:
    assert read_string_param({"q": " hi "}, "q") == "hi"
    assert read_string_param({"q": "x"}, "q", trim=False) == "x"
    # trim=False: the raw value is returned as-is (whitespace is content),
    # matching the TS `trim ? raw.trim() : raw` semantics.
    assert read_string_param({"q": "  "}, "q", trim=False) == "  "
    assert read_string_param({"q": ""}, "q", allow_empty=True) == ""
    assert read_string_param({}, "q") is None
    assert read_string_param({"q": 5}, "q") is None


def test_read_string_param_required_raises() -> None:
    for params in ({}, {"q": ""}, {"q": 5}):
        with pytest.raises(ValueError, match="^q required$"):
            read_string_param(params, "q", required=True)
    with pytest.raises(ValueError, match="^Query required$"):
        read_string_param({}, "q", required=True, label="Query")
    assert read_string_param({"q": "ok"}, "q", required=True) == "ok"


def test_read_number_param_happy_paths() -> None:
    assert read_number_param({"n": 3}, "n") == 3
    assert read_number_param({"n": 2.5}, "n") == 2.5
    assert read_number_param({"n": " 4 "}, "n") == 4.0
    assert read_number_param({"n": "2.5", "x": 0}, "n", integer=True) == 2
    assert read_number_param({"n": -2.9}, "n", integer=True) == -2  # toward zero
    assert read_number_param({}, "n") is None
    assert read_number_param({"n": "abc"}, "n") is None
    assert read_number_param({"n": float("nan")}, "n") is None
    assert read_number_param({"n": True}, "n") is None  # bool is not a number


def test_read_number_param_required_raises() -> None:
    for params in ({}, {"n": "abc"}, {"n": float("inf")}, {"n": True}):
        with pytest.raises(ValueError, match="^n required$"):
            read_number_param(params, "n", required=True)


def test_read_boolean_param_coercions() -> None:
    assert read_boolean_param({"b": True}, "b") is True
    assert read_boolean_param({"b": False}, "b") is False
    for value in ("true", "TRUE", " True ", "1", "yes", "YES"):
        assert read_boolean_param({"b": value}, "b") is True, value
    for value in ("false", "False", "0", "no", "NO"):
        assert read_boolean_param({"b": value}, "b") is False, value
    assert read_boolean_param({"b": "maybe"}, "b") is None
    assert read_boolean_param({}, "b") is None


def test_read_boolean_param_required_raises() -> None:
    for params in ({}, {"b": "maybe"}, {"b": 1}):
        with pytest.raises(ValueError, match="^b required$"):
            read_boolean_param(params, "b", required=True)


def test_read_string_array_param_coercions() -> None:
    assert read_string_array_param({"a": [" x ", 1, "y", None, ""]}, "a") == [
        "x",
        "y",
    ]
    assert read_string_array_param({"a": " solo "}, "a") == ["solo"]
    assert read_string_array_param({"a": []}, "a") is None
    assert read_string_array_param({"a": "  "}, "a") is None
    assert read_string_array_param({"a": 7}, "a") is None
    assert read_string_array_param({}, "a") is None


def test_read_string_array_param_required_raises() -> None:
    for params in ({}, {"a": []}, {"a": "  "}, {"a": 7}):
        with pytest.raises(ValueError, match="^a required$"):
            read_string_array_param(params, "a", required=True)
    assert read_string_array_param({"a": ["z"]}, "a", required=True) == ["z"]


# ═══════════════════════════════════════════════════════════════
# options: toolset / dreaming / skills / personality (TS parity)
# ═══════════════════════════════════════════════════════════════

from letta_sdk import (  # noqa: E402
    AgentSkill,
    CreateAgentOptions,
    CreateSessionOptions,
    DreamingOptions,
    LettaAgentClient,
    LettaSession,
    ToolsetConfig,
    load_skill_directory,
    parse_skill_markdown,
    resolve_skill_items,
    skill_memory_blocks,
    skills_have_support_files,
)
from letta_sdk.client import LettaAgentClient as _Client  # noqa: E402
from letta_sdk.session import LettaSession as _Session  # noqa: E402


class _FakeConn:
    """Minimal AppServerConnectionLike fake for option wiring tests."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.request_log: list[tuple[str, dict[str, Any]]] = []
        self.responses: dict[str, dict[str, Any]] = {}
        self.handlers: list[Any] = []
        self.closed = False

    def set_response(self, command: str, response: dict[str, Any]) -> None:
        self.responses[command] = response

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
        if type_ not in self.responses:
            raise AssertionError(f"fake: no handler for '{type_}'")
        return self.responses[type_]

    async def send(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    def on_message(self, handler: Any) -> Any:
        self.handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self.handlers:
                self.handlers.remove(handler)

        return unsubscribe

    async def close(self) -> None:
        self.closed = True

    def last_payload(self) -> dict[str, Any]:
        for payload in reversed(self.sent):
            if payload.get("type") == "input":
                return payload["payload"]
        raise AssertionError("no input payload sent")

    def set_runtime_start(self, agent_id: str = "agent-x") -> None:
        self.set_response(
            "runtime_start",
            {
                "type": "runtime_start_response",
                "success": True,
                "runtime": {
                    "agent_id": agent_id,
                    "conversation_id": f"conv-{agent_id}",
                },
                "agent": {"id": agent_id, "model": "m"},
            },
        )


# ── toolset ─────────────────────────────────────────────────────


def test_toolset_config_wire_shape_and_dedup() -> None:
    assert ToolsetConfig().to_wire() == {}
    assert ToolsetConfig(base="codex").to_wire() == {"base": "codex"}
    wire = ToolsetConfig(base=None, include=["a", "b", "a"]).to_wire()
    assert wire == {"include": ["a", "b"]}


def test_normalize_toolset_dict_and_validation() -> None:
    config = normalize_toolset_from({"base": "none", "include": ["x"]})
    assert config.to_wire() == {"base": "none", "include": ["x"]}
    assert normalize_toolset_from(None) is None
    with pytest.raises(ValueError, match="Invalid toolset.base"):
        normalize_toolset_from({"base": "gpt"})
    with pytest.raises(ValueError, match="non-empty strings"):
        normalize_toolset_from({"include": ["ok", ""]})
    with pytest.raises(ValueError, match="Unknown toolset"):
        normalize_toolset_from({"base": "codex", "extra": 1})


def normalize_toolset_from(value: Any) -> Any:
    from letta_sdk.types import normalize_toolset

    return normalize_toolset(value)


async def test_session_toolset_sent_on_every_turn() -> None:
    conn = _FakeConn()
    conn.set_runtime_start()
    session = LettaSession(
        connection=conn,
        owns_connection=False,
        agent_id="agent-x",
        options=CreateSessionOptions(
            toolset={"base": "gemini", "include": ["t1", "t2", "t1"]}
        ),
    )
    await session.send("hi")
    await session.send("again")
    first, second = conn.sent
    assert first["payload"]["client_toolset"] == {
        "base": "gemini",
        "include": ["t1", "t2"],
    }
    assert second["payload"]["client_toolset"] == first["payload"]["client_toolset"]


async def test_session_without_toolset_sends_no_field() -> None:
    conn = _FakeConn()
    conn.set_runtime_start()
    session = LettaSession(
        connection=conn, owns_connection=False, agent_id="agent-x"
    )
    await session.send("hi")
    assert "client_toolset" not in conn.sent[0]["payload"]


# ── dreaming ────────────────────────────────────────────────────


def test_dreaming_settings_defaults() -> None:
    assert DreamingOptions().to_settings() == {
        "trigger": "step-count",
        "step_count": 5,
    }
    assert DreamingOptions(trigger="off", step_count=9).to_settings() == {
        "trigger": "off",
        "step_count": 9,
    }


def test_dreaming_validation() -> None:
    from letta_sdk.types import normalize_dreaming

    assert normalize_dreaming(None, allow_behavior=True) is None
    normalized = normalize_dreaming(
        {"trigger": "compaction-event", "step_count": 3},
        allow_behavior=False,
    )
    assert normalized is not None
    assert normalized.to_settings() == {
        "trigger": "compaction-event",
        "step_count": 3,
    }
    with pytest.raises(ValueError, match="Invalid dreaming.trigger"):
        normalize_dreaming({"trigger": "sometimes"}, allow_behavior=True)
    with pytest.raises(ValueError, match="Invalid dreaming.behavior"):
        normalize_dreaming({"behavior": "nag"}, allow_behavior=True)
    with pytest.raises(
        ValueError, match="not supported when opening an existing"
    ):
        normalize_dreaming({"behavior": "reminder"}, allow_behavior=False)
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            normalize_dreaming({"step_count": bad}, allow_behavior=True)
    with pytest.raises(ValueError, match="Unknown dreaming"):
        normalize_dreaming({"bogus": 1}, allow_behavior=True)


async def test_session_dreaming_applied_after_start() -> None:
    conn = _FakeConn()
    conn.set_runtime_start("agent-d")
    conn.set_response(
        "set_reflection_settings",
        {"type": "set_reflection_settings_response", "success": True},
    )
    session = LettaSession(
        connection=conn,
        owns_connection=False,
        agent_id="agent-d",
        options=CreateSessionOptions(
            dreaming={"trigger": "compaction-event", "step_count": 9}
        ),
    )
    await session.ready()
    entries = [b for t, b in conn.request_log if t == "set_reflection_settings"]
    assert len(entries) == 1
    assert entries[0]["scope"] == "both"
    assert entries[0]["settings"] == {
        "trigger": "compaction-event",
        "step_count": 9,
    }
    assert entries[0]["runtime"] == {
        "agent_id": "agent-d",
        "conversation_id": "conv-agent-d",
    }


async def test_session_dreaming_skipped_when_stateless() -> None:
    conn = _FakeConn()
    conn.set_runtime_start("agent-d")
    session = LettaSession(
        connection=conn,
        owns_connection=False,
        agent_id="agent-d",
        options=CreateSessionOptions(
            dreaming=DreamingOptions(trigger="off"), stateless=True
        ),
    )
    await session.ready()
    assert not [
        t for t, _ in conn.request_log if t == "set_reflection_settings"
    ]


# ── skills ──────────────────────────────────────────────────────


def test_parse_skill_markdown_no_frontmatter() -> None:
    assert parse_skill_markdown("just body") == (None, None, "just body")
    assert parse_skill_markdown("---\nno fence") == (None, None, "---\nno fence")


def test_parse_skill_markdown_frontmatter() -> None:
    name, description, body = parse_skill_markdown(
        "---\nname: my-skill\ndescription: \"does things\"\n---\nBody here."
    )
    assert name == "my-skill"
    assert description == "does things"
    assert body == "Body here."


def test_parse_skill_markdown_folded_description() -> None:
    _, description, body = parse_skill_markdown(
        "---\ndescription: >-\n  line one\n  line two\n---\nB"
    )
    assert description == "line one line two"
    assert body == "B"


def test_load_skill_directory(tmp_path: Any) -> None:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: Test skill trigger\n---\nDo the thing.",
        encoding="utf-8",
    )
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "run.sh").write_text("echo hi", encoding="utf-8")
    skill = load_skill_directory(skill_dir)
    assert skill.name == "my-skill"
    assert skill.description == "Test skill trigger"
    assert skill.instructions == "Do the thing."
    assert skill.files == {"scripts/run.sh": b"echo hi"}
    assert skills_have_support_files([skill])


def test_load_skill_directory_errors(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        load_skill_directory(tmp_path / "nope")
    empty = tmp_path / "empty-skill"
    empty.mkdir()
    with pytest.raises(ValueError, match="has no SKILL.md"):
        load_skill_directory(empty)
    nodef = tmp_path / "no-desc"
    nodef.mkdir()
    (nodef / "SKILL.md").write_text("body only", encoding="utf-8")
    with pytest.raises(ValueError, match="no frontmatter description"):
        load_skill_directory(nodef)


def test_resolve_skill_items_inline_dict_and_errors() -> None:
    skill = resolve_skill_items(
        [
            AgentSkill(name="a", description="d", instructions="i"),
            {"name": "b", "description": "d2", "instructions": "i2"},
        ]
    )
    assert [s.name for s in skill] == ["a", "b"]
    with pytest.raises(ValueError, match="Duplicate skill name"):
        resolve_skill_items(
            [
                {"name": "dup", "description": "d", "instructions": "i"},
                AgentSkill(name="dup", description="d", instructions="i"),
            ]
        )
    with pytest.raises(ValueError, match="Invalid skill name"):
        resolve_skill_items(
            [{"name": "Bad Name", "description": "d", "instructions": "i"}]
        )
    with pytest.raises(ValueError, match="empty instructions"):
        resolve_skill_items(
            [{"name": "a", "description": "d", "instructions": "  "}]
        )
    with pytest.raises(ValueError, match="no description"):
        resolve_skill_items(
            [{"name": "a", "description": "", "instructions": "i"}]
        )


def test_skill_memory_blocks_shape() -> None:
    blocks = skill_memory_blocks(
        [AgentSkill(name="greet", description="Greet users", instructions="Say hi")]
    )
    assert blocks == [
        {
            "label": "skills/greet",
            "value": "Say hi",
            "description": "Greet users",
        }
    ]


def test_apply_skill_seed_validation() -> None:
    opts = CreateAgentOptions(
        skills=[AgentSkill(name="g", description="d", instructions="i")]
    )
    body: dict[str, Any] = {}
    LettaAgentClient._apply_skill_seed(opts, body)
    assert body["memory_blocks"] == [
        {"label": "skills/g", "value": "i", "description": "d"}
    ]
    no_memfs = CreateAgentOptions(
        skills=[AgentSkill(name="g", description="d", instructions="i")],
        memfs=False,
    )
    with pytest.raises(ValueError, match="memory filesystem"):
        LettaAgentClient._apply_skill_seed(no_memfs, {})
    with_files = CreateAgentOptions(
        skills=[
            AgentSkill(
                name="g",
                description="d",
                instructions="i",
                files={"scripts/x.sh": b"echo"},
            )
        ]
    )
    with pytest.raises(ValueError, match="skill support files"):
        LettaAgentClient._apply_skill_seed(with_files, {})


async def test_create_agent_skills_seed_memory_blocks() -> None:
    conn = _FakeConn()
    conn.set_runtime_start("agent-new")
    client = LettaAgentClient(connection=conn)
    await client.create_agent(
        CreateAgentOptions(
            name="skiller",
            skills=[AgentSkill(name="greet", description="d", instructions="i")],
        )
    )
    runtime_start_body = [
        b for t, b in conn.request_log if t == "runtime_start"
    ][0]
    blocks = runtime_start_body["create_agent"]["body"]["memory_blocks"]
    assert blocks == [
        {"label": "skills/greet", "value": "i", "description": "d"}
    ]


# ── personality ─────────────────────────────────────────────────


async def test_create_agent_personality_uses_native_command() -> None:
    conn = _FakeConn()
    conn.set_response(
        "create_agent",
        {
            "type": "create_agent_response",
            "success": True,
            "agent_id": "agent-p",
            "name": "Memo",
        },
    )
    client = LettaAgentClient(connection=conn)
    agent_id = await client.create_agent(
        CreateAgentOptions(personality="memo", model="m/1", tags=["t"])
    )
    assert agent_id == "agent-p"
    type_, body = conn.request_log[0]
    assert type_ == "create_agent"
    assert body == {
        "personality": "memo",
        "model": "m/1",
        "tags": ["t"],
        "pin_global": True,
    }


async def test_create_agent_personality_validation() -> None:
    conn = _FakeConn()
    client = LettaAgentClient(connection=conn)
    with pytest.raises(ValueError, match="Unknown personality"):
        await client.create_agent(CreateAgentOptions(personality="yoda"))
    with pytest.raises(ValueError, match="cannot be combined with persona"):
        await client.create_agent(
            CreateAgentOptions(personality="blank", persona="custom")
        )
    with pytest.raises(ValueError, match="cannot be combined with memory_blocks"):
        await client.create_agent(
            CreateAgentOptions(
                personality="blank",
                memory_blocks=[{"label": "persona", "value": "x"}],
            )
        )
    assert not conn.request_log


async def test_create_agent_personality_unpinned_when_hidden() -> None:
    conn = _FakeConn()
    conn.set_response(
        "create_agent",
        {
            "type": "create_agent_response",
            "success": True,
            "agent_id": "agent-p",
        },
    )
    client = LettaAgentClient(connection=conn)
    await client.create_agent(
        CreateAgentOptions(personality="blank", hidden=True)
    )
    _, body = conn.request_log[0]
    assert body["pin_global"] is False
    with_explicit = CreateAgentOptions(
        personality="blank", hidden=True, pin_global=True
    )
    await client.create_agent(with_explicit)
    assert conn.request_log[1][1]["pin_global"] is True
