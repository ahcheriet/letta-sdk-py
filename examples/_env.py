"""Shared environment configuration for the examples.

Every example reads its connection settings from the environment, so the
same code runs against any Letta Code app server — nothing here is
hardcoded to a machine:

    LETTA_APP_SERVER_URL   WebSocket endpoint (optional — the SDK's built-in
                           default is used when unset)
    LETTA_TOKEN_FILE       path to a file containing the bearer token
    LETTA_TOKEN            bearer token directly (alternative to a file)
    LETTA_MODEL            model handle your server has configured (required
                           by examples that create agents or conversations)

Local dev servers without authentication need no token settings at all.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__all__ = ["client_kwargs", "model"]


def app_server_url() -> str | None:
    """Endpoint from the environment, or None to use the SDK default."""
    url = os.environ.get("LETTA_APP_SERVER_URL", "").strip()
    return url or None


def auth_token() -> str | None:
    """Bearer token from LETTA_TOKEN_FILE or LETTA_TOKEN, or None."""
    token_file = os.environ.get("LETTA_TOKEN_FILE", "").strip()
    if token_file:
        return Path(token_file).expanduser().read_text().strip()
    return os.environ.get("LETTA_TOKEN", "").strip() or None


def model() -> str:
    """Model handle from LETTA_MODEL (required, with a helpful error)."""
    value = os.environ.get("LETTA_MODEL", "").strip()
    if not value:
        raise SystemExit(
            "LETTA_MODEL is not set.\n"
            "Export the model handle your app server has configured, e.g.:\n"
            "    export LETTA_MODEL=<provider>/<model>\n"
            "(list available handles with the 'models' example or "
            "client.models.list())"
        )
    return value


def client_kwargs() -> dict[str, Any]:
    """Constructor kwargs for LettaAgentClient from the environment."""
    kwargs: dict[str, Any] = {"request_timeout": 120}
    url = app_server_url()
    if url:
        kwargs["url"] = url
    token = auth_token()
    if token:
        kwargs["auth_token"] = token
    return kwargs
