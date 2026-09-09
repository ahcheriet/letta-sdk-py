from __future__ import annotations

import base64
from pathlib import Path

from .types import ImageContentPart, infer_media_type


def image_from_base64(data: str, media_type: str = "image/png") -> ImageContentPart:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def image_from_file(path: str | Path, media_type: str | None = None) -> ImageContentPart:
    file_path = Path(path)
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return image_from_base64(encoded, media_type or infer_media_type(file_path))
