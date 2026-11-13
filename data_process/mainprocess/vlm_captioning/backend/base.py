"""Abstract caption backend."""

from __future__ import annotations

import base64
import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from PIL import Image

from ..config import VLMCaptioningConfig
from ..prompt import IMAGE_PLACEHOLDER, ChatMessages
from ..schema import CaptionRequest, CaptionResult


logger = logging.getLogger(__name__)

# Images are shrunk to fit this size (keeping aspect ratio) and JPEG-encoded
# before being sent to vLLM.
_IMG_MAX_SIZE = (1152, 864)


def _pil_to_data_uri(img: "Image.Image") -> str:
    """Encode a PIL image as a JPEG base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def inline_images(
    chat_messages: ChatMessages,
    image_paths: List[Path],
) -> ChatMessages:
    """Replace IMAGE_PLACEHOLDER image parts with inline data URIs, in order.

    Inlining the (resized) images means the backend does not re-read the originals
    from a network filesystem.
    """
    resolved_messages: ChatMessages = []

    img_idx = 0
    for msg in chat_messages:
        new_msg = dict(msg)
        content = msg.get("content")

        if isinstance(content, list):
            new_parts = []
            for part in content:
                if (
                    part.get("type") == "image_url"
                    and part.get("image_url", {}).get("url") == IMAGE_PLACEHOLDER
                ):
                    if img_idx < len(image_paths):
                        pil_img = Image.open(image_paths[img_idx]).convert("RGB")
                        pil_img.thumbnail(_IMG_MAX_SIZE, Image.LANCZOS)
                        new_parts.append({
                            "type": "image_url",
                            "image_url": {"url": _pil_to_data_uri(pil_img)},
                        })
                        img_idx += 1
                    else:
                        logger.warning(
                            "More image placeholders than image paths; skipping."
                        )
                else:
                    new_parts.append(part)
            new_msg["content"] = new_parts

        resolved_messages.append(new_msg)

    return resolved_messages


class CaptionBackend(ABC):
    """Uniform interface for caption generation backends."""

    @abstractmethod
    def setup(self, config: VLMCaptioningConfig) -> None:
        """Initialise the backend (load model). Called once before ``generate``."""
        ...

    @abstractmethod
    def generate(self, requests: List[CaptionRequest]) -> List[CaptionResult]:
        """Run inference on a batch; results are in the same order as *requests*."""
        ...

    def shutdown(self) -> None:
        """Optional cleanup (release GPU memory)."""
        pass
