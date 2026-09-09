from __future__ import annotations

import base64
import re
import urllib.error
import urllib.request
from pathlib import Path

from .types import ImageContentPart, MessageContentPart, infer_media_type


def image_from_base64(data: str, media_type: str = "image/png") -> ImageContentPart:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def image_from_file(path: str | Path, media_type: str | None = None) -> ImageContentPart:
    file_path = Path(path)
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return image_from_base64(encoded, media_type or infer_media_type(file_path))


def image_from_url(url: str, timeout: float = 30.0) -> MessageContentPart:
    """Create image content from a URL.

    Fetches the image (stdlib ``urllib``) and converts it to a base64
    content part. The media type is detected from the ``Content-Type``
    header, with a URL-extension fallback (``.jpe?g`` / ``.gif`` /
    ``.webp``); anything else defaults to ``image/png``.

    Raises ``ValueError`` on a non-2xx HTTP status or an unreadable body.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if not 200 <= status < 300:
                raise ValueError(
                    f"Unexpected HTTP status {status} fetching image from {url}"
                )
            body = response.read()
            content_type = response.headers.get("Content-Type", "")
    except ValueError:
        raise
    except (urllib.error.URLError, OSError) as exc:
        raise ValueError(f"Failed to fetch image from {url}: {exc}") from exc

    data = base64.b64encode(body).decode("ascii")
    lowered = content_type.lower()
    if (
        "jpeg" in lowered
        or "jpg" in lowered
        or re.search(r"\.jpe?g$", url, re.IGNORECASE)
    ):
        media_type = "image/jpeg"
    elif "gif" in lowered or url.lower().endswith(".gif"):
        media_type = "image/gif"
    elif "webp" in lowered or url.lower().endswith(".webp"):
        media_type = "image/webp"
    else:
        media_type = "image/png"
    return image_from_base64(data, media_type)
