"""Tool helpers for writing local tool handlers.

Mirrors the pi-coding-agent tool helper API (same as the TypeScript
original, ``tool-helpers.ts``): result construction plus typed readers
for tool arguments, so handlers parse ``args`` with consistent
validation instead of ad-hoc checks.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .types import ToolResult


def json_result(payload: Any) -> ToolResult:
    """Create a JSON tool result: pretty-printed text plus raw ``details``."""
    return ToolResult(
        content=[{"type": "text", "text": json.dumps(payload, indent=2, ensure_ascii=False)}],
        details=payload,
    )


def read_string_param(
    params: dict[str, Any],
    key: str,
    *,
    required: bool = False,
    trim: bool = True,
    label: str | None = None,
    allow_empty: bool = False,
) -> str | None:
    """Read a string parameter from tool args.

    A missing or non-string value (and an empty value when
    ``allow_empty`` is false) raises ``ValueError`` if ``required`` is
    true, otherwise returns ``None``.
    """
    lbl = label if label is not None else key
    raw = params.get(key)
    if not isinstance(raw, str):
        if required:
            raise ValueError(f"{lbl} required")
        return None
    value = raw.strip() if trim else raw
    if not value and not allow_empty:
        if required:
            raise ValueError(f"{lbl} required")
        return None
    return value


def read_number_param(
    params: dict[str, Any],
    key: str,
    *,
    required: bool = False,
    label: str | None = None,
    integer: bool = False,
) -> int | float | None:
    """Read a number parameter from tool args.

    Accepts finite ints/floats and numeric strings (trimmed, float
    parse). ``integer=True`` truncates toward zero.
    """
    lbl = label if label is not None else key
    raw = params.get(key)
    value: int | float | None = None
    if isinstance(raw, int) and not isinstance(raw, bool):
        value = raw
    elif isinstance(raw, float) and math.isfinite(raw):
        value = raw
    elif isinstance(raw, str):
        trimmed = raw.strip()
        if trimmed:
            try:
                parsed: float = float(trimmed)
            except ValueError:
                parsed = math.nan
            if math.isfinite(parsed):
                value = parsed
    if value is None:
        if required:
            raise ValueError(f"{lbl} required")
        return None
    return int(value) if integer else value


def read_boolean_param(
    params: dict[str, Any],
    key: str,
    *,
    required: bool = False,
    label: str | None = None,
) -> bool | None:
    """Read a boolean parameter from tool args.

    Accepts booleans as-is and the strings ``true``/``1``/``yes`` and
    ``false``/``0``/``no`` (case-insensitive, trimmed).
    """
    lbl = label if label is not None else key
    raw = params.get(key)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lower = raw.strip().lower()
        if lower in ("true", "1", "yes"):
            return True
        if lower in ("false", "0", "no"):
            return False
    if required:
        raise ValueError(f"{lbl} required")
    return None


def read_string_array_param(
    params: dict[str, Any],
    key: str,
    *,
    required: bool = False,
    label: str | None = None,
) -> list[str] | None:
    """Read a string-array parameter from tool args.

    A list keeps only its string items (trimmed, empties dropped); a
    bare string becomes a single-element list. An empty result — or a
    value of any other type — raises if ``required``, else returns
    ``None``.
    """
    lbl = label if label is not None else key
    raw = params.get(key)
    if isinstance(raw, list):
        values = [item.strip() for item in raw if isinstance(item, str)]
        values = [v for v in values if v]
        if not values:
            if required:
                raise ValueError(f"{lbl} required")
            return None
        return values
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            if required:
                raise ValueError(f"{lbl} required")
            return None
        return [value]
    if required:
        raise ValueError(f"{lbl} required")
    return None
