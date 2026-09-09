"""Transcript accumulator.

Turns an :class:`~letta_sdk.types.SDKMessage` stream into stable, render-ready
rows so consumers stop hand-rolling stream reconciliation. This is a port of
the TypeScript SDK's ``transcript-accumulator.ts`` and owns the four
reconciliation rules the wire protocol requires:

1. **Typed-by-family accumulation.** Text slices are keyed on
   ``message family + otid``, falling back to ``uuid`` *within the same
   family*. A bare ``otid``/``uuid`` key would collapse an assistant slice
   into a reasoning slice whenever a provider reuses an identifier across
   kinds.
2. **Per-``run_id`` ``seq_id`` replay suppression.** Each run keeps its own
   high-water mark, so a resumed stream that replays positions is dropped
   while a new run starts from a clean threshold.
3. **``tool_call_id``-keyed merging.** Tool argument fragments and the
   eventual tool result merge into one row keyed on the payload identity
   (``tool_call_id``), while the envelope identities (the ``uuid`` of the
   ``tool_call`` message and of the ``tool_result`` message) stay
   separately visible.
4. **``rebase()`` for mid-run backfill.** A history page is merged in place
   with replace semantics, reordered ahead of live-only rows, and raises
   the replay thresholds it proves.

The accumulator is pure: no I/O, no timers.

Backward compatibility
----------------------
The minimal pre-v0.3 API is unchanged: ``TranscriptAccumulator()`` with a
``messages`` list, ``add(message) -> SDKMessage``, and the ``assistant_text``
property. The row-based surface (``rows``, ``rebase``, ``reset``) is additive.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TypeAlias

from .types import (
    AssistantMessage,
    ListMessagesResult,
    ReasoningMessage,
    SDKMessage,
    StreamEventMessage,
    ToolCallMessage,
    ToolResultMessage,
)

__all__ = [
    "TranscriptAccumulator",
    "TranscriptHistoryPage",
    "TranscriptRow",
    "TranscriptRowKind",
    "TranscriptTextKind",
    "TranscriptToolCallStatus",
    "TranscriptToolResult",
]

#: Message families the accumulator projects into rows.
TranscriptRowKind: TypeAlias = Literal["user", "assistant", "reasoning", "tool_call"]

#: Text families. Rows in different families never share a key.
TranscriptTextKind: TypeAlias = Literal["user", "assistant", "reasoning"]

#: Lifecycle of a tool row: ``streaming`` (argument fragments still
#: arriving, not yet parsed), ``ready`` (arguments parsed, no result),
#: ``complete`` (a tool result merged into the row).
TranscriptToolCallStatus: TypeAlias = Literal["streaming", "ready", "complete"]

#: A history page accepted by :meth:`TranscriptAccumulator.rebase`: a bare
#: list of Letta API message dicts, a dict with a ``messages`` key, or a
#: :class:`~letta_sdk.types.ListMessagesResult`.
TranscriptHistoryPage: TypeAlias = (
    list[dict[str, Any]] | dict[str, Any] | ListMessagesResult
)

#: Key segment separator. Every key carries a family segment and an
#: identifier kind segment ("otid"/"uuid"), so no wire identifier can
#: produce a key that collides with a key from another family or another
#: identifier kind.
_SEP = ":"

#: Bound on tracked replay thresholds so a long session cannot grow forever.
_MAX_TRACKED_RUNS = 64

#: Bucket used for streams that do not carry a ``run_id``.
_ANONYMOUS_RUN = ""


# ═══════════════════════════════════════════════════════════════
# PUBLIC ROW TYPES
# ═══════════════════════════════════════════════════════════════


@dataclass(slots=True, kw_only=True)
class TranscriptToolResult:
    """Tool result merged into a tool row.

    ``uuid`` is the envelope id of the ``tool_result`` message —
    deliberately distinct from the row's ``uuid``, which identifies the
    ``tool_call`` envelope.
    """

    content: str
    is_error: bool = False
    uuid: str | None = None


@dataclass(slots=True, kw_only=True)
class TranscriptRow:
    """One stable, render-ready transcript row.

    Text rows (``user`` / ``assistant`` / ``reasoning``) carry ``text``;
    tool rows (``tool_call``) carry the ``tool_*`` fields. ``key`` is the
    stable render key, namespaced by message family.
    """

    kind: TranscriptRowKind
    #: Stable render key. Namespaced by message family, so a provider that
    #: reuses an ``otid`` or a message id across kinds still produces
    #: separate rows.
    key: str
    #: Envelope id of the message that opened this row, when known.
    uuid: str | None = None
    #: Lineage key for this typed slice, when the stream supplied one.
    otid: str | None = None
    #: Run that most recently contributed to this row.
    run_id: str | None = None
    #: Highest replay cursor observed for this row.
    seq_id: int | None = None
    #: Accumulated text for text rows.
    text: str = ""
    #: Payload identity of a tool row; this is what the row is keyed on.
    tool_call_id: str | None = None
    tool_name: str | None = None
    #: Best known parsed arguments. Never the transitional ``{"raw": ...}``
    #: wrapper the protocol layer emits for an argument fragment that does
    #: not parse.
    tool_input: dict[str, Any] = field(default_factory=dict)
    #: Argument fragments concatenated in arrival order, when the wire sent any.
    raw_arguments: str | None = None
    #: Whether ``tool_input`` reflects fully parsed arguments.
    arguments_complete: bool = False
    result: TranscriptToolResult | None = None
    #: Tool rows only.
    status: TranscriptToolCallStatus | None = None


# ═══════════════════════════════════════════════════════════════
# INTERNALS
# ═══════════════════════════════════════════════════════════════


@dataclass(slots=True)
class _TextSlice:
    kind: TranscriptTextKind
    text: str
    uuid: str | None = None
    otid: str | None = None
    run_id: str | None = None
    seq_id: int | None = None


@dataclass(slots=True)
class _ToolCallMerge:
    tool_call_id: str
    #: ``"fragment"`` appends streamed argument text; ``"whole"`` treats the
    #: arguments as an authoritative complete value (history backfill).
    mode: Literal["fragment", "whole"]
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    raw_arguments: str | None = None
    uuid: str | None = None
    run_id: str | None = None
    seq_id: int | None = None


@dataclass(slots=True)
class _ToolResultMerge:
    tool_call_id: str
    content: str
    is_error: bool
    uuid: str | None = None
    run_id: str | None = None
    seq_id: int | None = None


@dataclass(slots=True)
class _ToolArguments:
    raw_arguments: str | None = None
    tool_input: dict[str, Any] = field(default_factory=dict)
    arguments_complete: bool = False


@dataclass(slots=True)
class _StreamTextDelta:
    kind: Literal["assistant", "reasoning"]
    text: str


def _read_string(record: dict[str, Any], field_name: str) -> str | None:
    value = record.get(field_name)
    return value if isinstance(value, str) and value else None


def _read_number(record: dict[str, Any], field_name: str) -> int | float | None:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _max_defined(
    next_value: int | None, previous: int | None
) -> int | None:
    if next_value is None:
        return previous
    if previous is None:
        return next_value
    return max(next_value, previous)


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    """Parse a JSON object, returning ``None`` for partial or non-object JSON."""
    trimmed = raw.strip()
    if not trimmed:
        return None
    try:
        parsed = json.loads(trimmed)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_raw_arguments_wrapper(
    tool_input: dict[str, Any] | None, raw_arguments: str | None
) -> bool:
    """Detect the protocol layer's transitional ``{"raw": ...}`` wrapper.

    ``toolInputFromArguments()`` wraps an unparseable argument fragment as
    ``{"raw": "<fragment>"}``. That wrapper is a parse failure, not
    arguments, and must never overwrite previously parsed input.
    """
    if not tool_input:
        return False
    if len(tool_input) != 1 or "raw" not in tool_input:
        return False
    wrapped = tool_input["raw"]
    if not isinstance(wrapped, str):
        return False
    return raw_arguments is None or wrapped == raw_arguments


def _tool_status(
    row: TranscriptRow,
) -> TranscriptToolCallStatus:
    if row.result is not None:
        return "complete"
    return "ready" if row.arguments_complete else "streaming"


def _seq_id_from_raw(raw: Any) -> int | None:
    """Recover ``seq_id`` from the raw wire delta.

    The Python ``ToolCallMessage`` / ``ToolResultMessage`` projections do not
    carry a ``seq_id`` field (the wire does); it is read back from the raw
    ``stream_delta`` payload they always retain.
    """
    if not isinstance(raw, dict):
        return None
    delta = raw.get("stream_delta")
    if not isinstance(delta, dict):
        return None
    value = _read_number(delta, "seq_id")
    return value if isinstance(value, int) else None


def _extract_text_from_content(content: Any) -> str | None:
    """Flatten wire ``content`` (str | list[part] | part) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
                continue
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                pieces.append(part["text"])
        joined = "".join(pieces)
        return joined if joined else None
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return None


def _first_tool_call(record: dict[str, Any]) -> dict[str, Any] | None:
    tool_calls = record.get("tool_calls")
    if isinstance(tool_calls, list):
        first = tool_calls[0] if tool_calls else None
        return first if isinstance(first, dict) else None
    if isinstance(tool_calls, dict):
        return tool_calls
    tool_call = record.get("tool_call")
    return tool_call if isinstance(tool_call, dict) else None


def _first_tool_return(record: dict[str, Any]) -> dict[str, Any] | None:
    tool_returns = record.get("tool_returns")
    if isinstance(tool_returns, list):
        first = tool_returns[0] if tool_returns else None
        return first if isinstance(first, dict) else None
    return None


def _extract_stream_text_delta(event: Any) -> _StreamTextDelta | None:
    """Extract appendable assistant/reasoning text from a stream_event payload.

    Supports both shapes currently emitted by headless mode:

    1. content_block style: ``{"type": ..., "delta": {"text"|"reasoning": ...}}``
    2. message chunk style: ``{"message_type": "assistant_message"|"reasoning_message", ...}``

    (Port of ``stream-events.ts:extractStreamTextDelta``; kept private here
    so this module stays self-contained.)
    """
    if not isinstance(event, dict):
        return None

    maybe_delta = event.get("delta")
    if isinstance(maybe_delta, dict):
        reasoning = maybe_delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            return _StreamTextDelta(kind="reasoning", text=reasoning)
        text = maybe_delta.get("text")
        if isinstance(text, str) and text:
            return _StreamTextDelta(kind="assistant", text=text)

    message_type = event.get("message_type")
    if message_type == "reasoning_message":
        reasoning = event.get("reasoning")
        reasoning_text = (
            reasoning
            if isinstance(reasoning, str)
            else _extract_text_from_content(event.get("content"))
        )
        if isinstance(reasoning_text, str) and reasoning_text:
            return _StreamTextDelta(kind="reasoning", text=reasoning_text)

    if message_type == "assistant_message":
        assistant_text = _extract_text_from_content(event.get("content"))
        if isinstance(assistant_text, str) and assistant_text:
            return _StreamTextDelta(kind="assistant", text=assistant_text)

    return None


def _parse_date(date: str) -> float | None:
    try:
        # ``Z``-suffixed ISO timestamps are accepted on 3.11+; the replace
        # keeps older interpreters in the >=3.11 line safe as well.
        return datetime.fromisoformat(date.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _detect_descending(messages: list[Any]) -> bool:
    """Detect newest-first ordering from the page itself.

    ``list_messages()`` defaults to newest-first; detecting it from the
    page means callers do not have to restate the order they requested.
    """
    seq_ids: list[int | float] = []
    dates: list[float] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        seq_id = _read_number(message, "seq_id")
        if seq_id is not None:
            seq_ids.append(seq_id)
        date = _read_string(message, "date")
        if date is None:
            continue
        parsed = _parse_date(date)
        if parsed is not None:
            dates.append(parsed)
    ordered = seq_ids if len(seq_ids) >= 2 else dates
    if len(ordered) < 2:
        return False
    return ordered[-1] < ordered[0]


def _normalize_history_page(
    page: TranscriptHistoryPage, order: str | None
) -> list[Any]:
    if isinstance(page, (list, tuple)):
        messages: list[Any] = list(page)
    elif isinstance(page, dict):
        candidate = page.get("messages")
        messages = list(candidate) if isinstance(candidate, (list, tuple)) else []
    else:
        candidate = getattr(page, "messages", None)
        messages = list(candidate) if isinstance(candidate, (list, tuple)) else []
    if len(messages) < 2:
        return messages
    descending = order == "desc" if order is not None else _detect_descending(messages)
    return list(reversed(messages)) if descending else messages


def _merge_tool_arguments(
    previous: _ToolArguments, merge: _ToolCallMerge
) -> _ToolArguments:
    """Fold one argument delivery into the arguments known so far.

    The wire can deliver arguments three ways for the same call: streamed
    JSON fragments, one complete JSON string, or an already-decoded object.
    Only a successful parse is allowed to change ``tool_input``.
    """
    raw_arguments = previous.raw_arguments
    tool_input = previous.tool_input
    arguments_complete = previous.arguments_complete
    fragment = merge.raw_arguments
    wrapped = _is_raw_arguments_wrapper(merge.tool_input, fragment)

    if fragment is not None and fragment:
        whole = _parse_json_object(fragment)
        if whole is not None:
            # A delivery that parses on its own is the complete argument
            # value: a final non-chunked ``tool_call_message``, or a
            # backfilled history row. Replace rather than append so a
            # repeated terminal message cannot corrupt the accumulation.
            return _ToolArguments(
                raw_arguments=fragment, tool_input=whole, arguments_complete=True
            )
        if arguments_complete:
            # Arguments already parsed; a trailing partial (a replayed
            # fragment after backfill) must not corrupt them.
            return previous
        if merge.mode == "whole":
            return _ToolArguments(
                raw_arguments=(
                    raw_arguments if raw_arguments is not None else fragment
                ),
                tool_input=tool_input,
                arguments_complete=arguments_complete,
            )
        raw_arguments = (raw_arguments or "") + fragment
        parsed = _parse_json_object(raw_arguments)
        if parsed is not None:
            return _ToolArguments(
                raw_arguments=raw_arguments,
                tool_input=parsed,
                arguments_complete=True,
            )
        # Keep the previous parse. Never promote the ``{"raw": ...}`` wrapper.
        return _ToolArguments(
            raw_arguments=raw_arguments,
            tool_input=tool_input,
            arguments_complete=False,
        )

    if not wrapped and merge.tool_input:
        if len(merge.tool_input) > 0:
            return _ToolArguments(
                raw_arguments=raw_arguments,
                tool_input=merge.tool_input,
                arguments_complete=True,
            )
        if raw_arguments is None:
            # Genuinely argument-free call: ``{}`` with no streamed fragments.
            return _ToolArguments(
                raw_arguments=raw_arguments,
                tool_input=tool_input,
                arguments_complete=True,
            )

    return previous


class TranscriptAccumulator:
    """Fold streamed SDK messages (and history pages) into stable rows.

    Use :meth:`add` for each streamed message and read :attr:`rows` for the
    current render-ready rows; :meth:`rebase` merges a history page mid-run
    without duplicating live rows. The legacy ``messages`` /
    ``assistant_text`` surface from the minimal pre-v0.3 accumulator is
    preserved.
    """

    def __init__(
        self,
        messages: list[SDKMessage] | None = None,
        assistant_chunks: list[str] | None = None,
    ) -> None:
        self.messages: list[SDKMessage] = []
        # Backwards-compatible legacy constructor kwarg (the v0.2
        # accumulator was a plain dataclass with this field).
        self.assistant_chunks: list[str] = list(assistant_chunks) if assistant_chunks else []
        # Row key -> row. Dict insertion order is the transcript order.
        self._by_key: dict[str, TranscriptRow] = {}
        # ``family + otid`` -> row key.
        self._alias_by_otid: dict[str, str] = {}
        # ``family + uuid`` -> row key.
        self._alias_by_uuid: dict[str, str] = {}
        # ``run_id`` -> highest accepted ``seq_id`` for that run.
        self._seq_thresholds: dict[str, int] = {}
        self._snapshot: list[TranscriptRow] | None = None
        self._anonymous_counter = 0
        if messages:
            for message in messages:
                self.add(message)

    # ── public API ──────────────────────────────────────────────

    def add(self, message: SDKMessage) -> SDKMessage:
        """Fold one streamed message into the transcript.

        Returns the message (legacy API); the reconciled view is
        :attr:`rows`.
        """
        self.messages.append(message)
        if isinstance(message, AssistantMessage) and message.content:
            self.assistant_chunks.append(message.content)
        self._apply(message)
        return message

    @property
    def assistant_text(self) -> str:
        """Concatenated assistant text (legacy API)."""
        return "".join(self.assistant_chunks)

    @property
    def rows(self) -> list[TranscriptRow]:
        """Current rows in transcript order.

        The returned list is referentially stable when a subsequent
        :meth:`add` changes nothing (a replayed position, or a message
        family the accumulator ignores), so it can be handed straight to a
        memoizing renderer.
        """
        if self._snapshot is None:
            self._snapshot = list(self._by_key.values())
        return self._snapshot

    def rebase(
        self,
        page: TranscriptHistoryPage,
        *,
        order: Literal["asc", "desc"] | None = None,
    ) -> list[TranscriptRow]:
        """Merge a history page into the transcript. Safe to call mid-run.

        ``order`` is the order of the supplied page; omitted means
        auto-detect from ``seq_id``/``date``. ``list_messages()`` defaults
        to ``"desc"`` (newest first).
        """
        messages = _normalize_history_page(page, order)
        if not messages:
            return self.rows

        history_keys: list[str] = []
        seen: set[str] = set()
        for message in messages:
            key = self._apply_history_message(message)
            if key is None or key in seen:
                continue
            seen.add(key)
            history_keys.append(key)

        self._reorder(history_keys)
        return self.rows

    def reset(self) -> None:
        """Drop all rows, replay state, and the legacy message list."""
        self.messages.clear()
        self.assistant_chunks.clear()
        self._by_key.clear()
        self._alias_by_otid.clear()
        self._alias_by_uuid.clear()
        self._seq_thresholds.clear()
        self._anonymous_counter = 0
        self._snapshot = None

    # ── live stream folding ─────────────────────────────────────

    def _apply(self, message: SDKMessage) -> None:
        if isinstance(message, AssistantMessage):
            self._apply_text(
                _TextSlice(
                    kind="assistant",
                    text=message.content,
                    uuid=message.uuid,
                    otid=message.otid,
                    run_id=message.run_id,
                    seq_id=message.seq_id,
                )
            )
            return
        if isinstance(message, ReasoningMessage):
            self._apply_text(
                _TextSlice(
                    kind="reasoning",
                    text=message.content,
                    uuid=message.uuid,
                    otid=message.otid,
                    run_id=message.run_id,
                    seq_id=message.seq_id,
                )
            )
            return
        if isinstance(message, ToolCallMessage):
            self._merge_tool_call(
                _ToolCallMerge(
                    tool_call_id=message.tool_call_id,
                    mode="fragment",
                    tool_name=message.tool_name,
                    tool_input=message.tool_input,
                    raw_arguments=message.raw_arguments,
                    uuid=message.uuid,
                    run_id=message.run_id,
                    seq_id=_seq_id_from_raw(message.raw),
                )
            )
            return
        if isinstance(message, ToolResultMessage):
            self._merge_tool_result(
                _ToolResultMerge(
                    tool_call_id=message.tool_call_id,
                    content=message.content,
                    is_error=message.is_error,
                    uuid=message.uuid,
                    run_id=message.run_id,
                    seq_id=_seq_id_from_raw(message.raw),
                )
            )
            return
        if isinstance(message, StreamEventMessage):
            self._apply_stream_event(message)
            return
        # init/result/error/retry/queue_update/loop_status/usage are
        # turn-level signals rather than transcript content; consumers
        # handle them directly.

    def _apply_stream_event(self, message: StreamEventMessage) -> None:
        """Fold raw stream events the session layer passes through uncooked.

        Identity comes from the payload when it has any (``id``, ``otid``,
        ``seq_id``, ``run_id``). Content-block deltas carry none, so they
        fold into a single live row per family — the most an anonymous
        delta stream can support.
        """
        delta = _extract_stream_text_delta(message.event)
        if delta is None:
            return
        payload = message.event if isinstance(message.event, dict) else {}
        otid = _read_string(payload, "otid")
        payload_id = _read_string(payload, "id")
        identified = otid is not None or payload_id is not None
        payload_seq = _read_number(payload, "seq_id")
        self._apply_text(
            _TextSlice(
                kind=delta.kind,
                text=delta.text,
                uuid=payload_id if identified else f"{delta.kind}{_SEP}live",
                otid=otid,
                run_id=_read_string(payload, "run_id"),
                seq_id=payload_seq if isinstance(payload_seq, int) else None,
            )
        )

    # ── replay suppression ──────────────────────────────────────

    def _is_replay(
        self, run_id: str | None, seq_id: int | None
    ) -> bool:
        """Per-run replay guard.

        Thresholds are bucketed by ``run_id``, so a brand new run starts
        with no threshold (a natural reset) while a resumed run keeps
        suppressing the positions it already delivered. Messages without a
        ``seq_id`` are never suppressed here: their families are
        deduplicated by identity instead.
        """
        if seq_id is None:
            return False
        bucket = run_id if run_id is not None else _ANONYMOUS_RUN
        threshold = self._seq_thresholds.get(bucket)
        if threshold is not None and seq_id <= threshold:
            return True
        self._remember_seq(bucket, seq_id)
        return False

    def _remember_seq(self, bucket: str, seq_id: int) -> None:
        threshold = self._seq_thresholds.get(bucket)
        if threshold is not None and threshold >= seq_id:
            return
        self._seq_thresholds[bucket] = seq_id
        while len(self._seq_thresholds) > _MAX_TRACKED_RUNS:
            # Oldest bucket first (dict insertion order).
            oldest = next(iter(self._seq_thresholds), None)
            if oldest is None:
                break
            del self._seq_thresholds[oldest]

    # ── text families ───────────────────────────────────────────

    def _apply_text(self, slice: _TextSlice) -> None:
        if self._is_replay(slice.run_id, slice.seq_id):
            return
        self._write_text(slice, "append")

    def _write_text(self, slice: _TextSlice, write: str) -> str:
        key = self._resolve_text_key(slice.kind, slice.uuid, slice.otid)
        existing = self._by_key.get(key)
        previous = (
            existing if existing is not None and existing.kind == slice.kind else None
        )
        previous_text = previous.text if previous is not None else ""
        text = (
            previous_text + slice.text if write == "append" else slice.text
        )
        self._set_row(
            key,
            TranscriptRow(
                kind=slice.kind,
                key=key,
                text=text,
                uuid=self._coalesce(previous, slice.uuid),
                otid=(
                    slice.otid
                    if slice.otid is not None
                    else (previous.otid if previous is not None else None)
                ),
                run_id=(
                    slice.run_id
                    if slice.run_id is not None
                    else (previous.run_id if previous is not None else None)
                ),
                seq_id=_max_defined(
                    slice.seq_id,
                    previous.seq_id if previous is not None else None,
                ),
            ),
        )
        return key

    @staticmethod
    def _coalesce(previous: TranscriptRow | None, value: str | None) -> str | None:
        if previous is not None and previous.uuid is not None:
            return previous.uuid
        return value

    def _resolve_text_key(
        self,
        kind: TranscriptTextKind,
        uuid: str | None,
        otid: str | None,
    ) -> str:
        """Resolve the row key for a text slice.

        ``otid`` is the lineage key when present, but a stream can
        transition: early fragments may carry only the message id and later
        fragments add an ``otid``. Both identifiers are aliased to one row so
        the transition does not split it. When a message id is reused by a
        *second* slice carrying a different ``otid`` (a provider emitting
        ``[text, thinking, text]`` under one message id), the new ``otid``
        opens its own row instead of appending to the previous one.
        """
        otid_alias = f"{kind}{_SEP}otid{_SEP}{otid}" if otid else None
        uuid_alias = f"{kind}{_SEP}uuid{_SEP}{uuid}" if uuid else None
        from_otid = self._alias_by_otid.get(otid_alias) if otid_alias else None
        from_uuid = self._alias_by_uuid.get(uuid_alias) if uuid_alias else None

        key = from_otid if from_otid is not None else from_uuid

        if key is not None and from_otid is None and otid is not None:
            existing = self._by_key.get(key)
            if (
                existing is not None
                and existing.otid is not None
                and existing.otid != otid
            ):
                # The envelope already committed to a different lineage: this
                # is a new slice sharing a message id, not a continuation of
                # the old one.
                key = None

        if key is None:
            if otid_alias is not None:
                key = otid_alias
            elif uuid_alias is not None:
                key = uuid_alias
            else:
                self._anonymous_counter += 1
                key = f"{kind}{_SEP}auto{_SEP}{self._anonymous_counter}"

        if otid_alias is not None:
            self._alias_by_otid[otid_alias] = key
        # The newest slice owns the envelope, so later id-only fragments
        # continue it rather than the slice that closed before it.
        if uuid_alias is not None:
            self._alias_by_uuid[uuid_alias] = key

        return key

    # ── tool families ───────────────────────────────────────────

    def _merge_tool_call(self, merge: _ToolCallMerge) -> str:
        key = f"tool_call{_SEP}id{_SEP}{merge.tool_call_id}"
        existing = self._by_key.get(key)
        previous = (
            existing if existing is not None and existing.kind == "tool_call" else None
        )

        args = _merge_tool_arguments(
            _ToolArguments(
                raw_arguments=(
                    previous.raw_arguments if previous is not None else None
                ),
                tool_input=(
                    dict(previous.tool_input) if previous is not None else {}
                ),
                arguments_complete=(
                    previous.arguments_complete if previous is not None else False
                ),
            ),
            merge,
        )

        if merge.tool_name and merge.tool_name != "?":
            tool_name = merge.tool_name
        elif previous is not None and previous.tool_name is not None:
            tool_name = previous.tool_name
        elif merge.tool_name is not None:
            tool_name = merge.tool_name
        else:
            tool_name = "?"

        row = TranscriptRow(
            kind="tool_call",
            key=key,
            tool_call_id=merge.tool_call_id,
            tool_name=tool_name,
            tool_input=args.tool_input,
            raw_arguments=args.raw_arguments,
            arguments_complete=args.arguments_complete,
            # Envelope identity stays pinned to the ``tool_call`` message that
            # opened the row; the payload identity is ``tool_call_id``.
            uuid=(
                previous.uuid
                if previous is not None and previous.uuid is not None
                else merge.uuid
            ),
            run_id=(
                merge.run_id
                if merge.run_id is not None
                else (previous.run_id if previous is not None else None)
            ),
            seq_id=_max_defined(
                merge.seq_id,
                previous.seq_id if previous is not None else None,
            ),
            result=previous.result if previous is not None else None,
        )
        row.status = _tool_status(row)
        self._set_row(key, row)
        return key

    def _merge_tool_result(self, merge: _ToolResultMerge) -> str:
        key = f"tool_call{_SEP}id{_SEP}{merge.tool_call_id}"
        existing = self._by_key.get(key)
        previous = (
            existing if existing is not None and existing.kind == "tool_call" else None
        )

        self._set_row(
            key,
            TranscriptRow(
                kind="tool_call",
                key=key,
                tool_call_id=merge.tool_call_id,
                tool_name=(
                    previous.tool_name
                    if previous is not None and previous.tool_name is not None
                    else "?"
                ),
                tool_input=dict(previous.tool_input) if previous is not None else {},
                raw_arguments=(
                    previous.raw_arguments if previous is not None else None
                ),
                arguments_complete=(
                    previous.arguments_complete if previous is not None else False
                ),
                result=TranscriptToolResult(
                    content=merge.content,
                    is_error=merge.is_error,
                    # The result envelope is a different message than the call
                    # envelope, so it is recorded beside the row's ``uuid``,
                    # not over it.
                    uuid=merge.uuid,
                ),
                status="complete",
                uuid=previous.uuid if previous is not None else None,
                run_id=(
                    merge.run_id
                    if merge.run_id is not None
                    else (previous.run_id if previous is not None else None)
                ),
                seq_id=_max_defined(
                    merge.seq_id,
                    previous.seq_id if previous is not None else None,
                ),
            ),
        )
        return key

    # ── history backfill ────────────────────────────────────────

    def _apply_history_message(self, message: Any) -> str | None:
        record = message if isinstance(message, dict) else None
        if record is None:
            return None
        message_type = _read_string(record, "message_type")
        if message_type is None:
            return None

        uuid = _read_string(record, "id")
        otid = _read_string(record, "otid")
        run_id = _read_string(record, "run_id")
        raw_seq_id = _read_number(record, "seq_id")
        # The wire carries integer cursors (session.py's ``_as_int`` agrees);
        # non-integral values are treated as absent.
        seq_id = raw_seq_id if isinstance(raw_seq_id, int) else None

        # A history page proves every position up to its own cursor for that
        # run, so replayed deltas at or below it are suppressed after the
        # merge.
        if seq_id is not None:
            self._remember_seq(run_id if run_id is not None else _ANONYMOUS_RUN, seq_id)

        if message_type in ("user_message", "assistant_message"):
            text = _extract_text_from_content(record.get("content"))
            if text is None:
                return None
            return self._write_text(
                _TextSlice(
                    kind=("user" if message_type == "user_message" else "assistant"),
                    text=text,
                    uuid=uuid,
                    otid=otid,
                    run_id=run_id,
                    seq_id=seq_id,
                ),
                "replace",
            )

        if message_type == "reasoning_message":
            reasoning = record.get("reasoning")
            text = (
                reasoning
                if isinstance(reasoning, str)
                else _extract_text_from_content(record.get("content"))
            )
            if not isinstance(text, str) or not text:
                return None
            return self._write_text(
                _TextSlice(
                    kind="reasoning",
                    text=text,
                    uuid=uuid,
                    otid=otid,
                    run_id=run_id,
                    seq_id=seq_id,
                ),
                "replace",
            )

        if message_type in ("tool_call_message", "approval_request_message"):
            tool_call = _first_tool_call(record)
            if tool_call is None:
                return None
            fn = tool_call.get("function")
            fn = fn if isinstance(fn, dict) else {}
            tool_call_id = _read_string(tool_call, "tool_call_id")
            if tool_call_id is None:
                tool_call_id = _read_string(tool_call, "id")
            if not tool_call_id:
                return None
            args = tool_call.get("arguments")
            if args is None:
                args = fn.get("arguments")
            tool_name = _read_string(tool_call, "name")
            if tool_name is None:
                tool_name = _read_string(fn, "name")
            return self._merge_tool_call(
                _ToolCallMerge(
                    tool_call_id=tool_call_id,
                    mode="whole",
                    tool_name=tool_name,
                    tool_input=args if isinstance(args, dict) else None,
                    raw_arguments=args if isinstance(args, str) else None,
                    uuid=uuid,
                    run_id=run_id,
                    seq_id=seq_id,
                )
            )

        if message_type == "tool_return_message":
            tool_return = _first_tool_return(record)
            if tool_return is None:
                tool_return = record
            tool_call_id = _read_string(record, "tool_call_id")
            if tool_call_id is None:
                tool_call_id = _read_string(tool_return, "tool_call_id")
            if not tool_call_id:
                return None
            content_value = record.get("tool_return")
            if content_value is None:
                content_value = tool_return.get("tool_return")
            if content_value is None:
                content_value = tool_return.get("content")
            content = _extract_text_from_content(content_value) or ""
            status = _read_string(record, "status")
            if status is None:
                status = _read_string(tool_return, "status")
            return self._merge_tool_result(
                _ToolResultMerge(
                    tool_call_id=tool_call_id,
                    content=content,
                    is_error=status == "error",
                    uuid=uuid,
                    run_id=run_id,
                    seq_id=seq_id,
                )
            )

        # system/summary/event/hidden-reasoning/approval-response messages
        # are not transcript content this accumulator claims to own.
        return None

    # ── row bookkeeping ─────────────────────────────────────────

    def _set_row(self, key: str, row: TranscriptRow) -> None:
        self._by_key[key] = row
        self._snapshot = None

    def _reorder(self, history_keys: list[str]) -> None:
        """Move backfilled rows ahead of rows that only exist in the live stream."""
        if not history_keys:
            return
        history_set = set(history_keys)
        reordered: dict[str, TranscriptRow] = {}
        for key in history_keys:
            row = self._by_key.get(key)
            if row is not None:
                reordered[key] = row
        for key, row in self._by_key.items():
            if key in history_set:
                continue
            reordered[key] = row
        self._by_key = reordered
        self._snapshot = None