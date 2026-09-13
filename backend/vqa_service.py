"""VQA model calls live here and only here (Agent.md §2).

Phase 1 uses Design.md §5 **Option A**: a hosted VLM served through Hugging
Face Inference Providers' OpenAI-compatible endpoint. Swapping to a custom
ZeroGPU Space later is a change to this file alone.
"""

import base64
import io
import os
from typing import Any

from openai import OpenAI
from PIL import Image

BASE_URL = "https://router.huggingface.co/v1"
# Design.md cites Qwen2.5-VL-3B-Instruct, but that model isn't served on the
# stable router for every account; Qwen3-VL-30B-A3B is the smallest VL model
# available and verified working (Agent.md §5: pick the simplest option that works).
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


class VQAServiceError(RuntimeError):
    """Raised when the model call itself fails; mapped to 502 upstream."""


def _image_to_data_uri(image: Image.Image) -> str:
    """Encode a PIL image as a base64 JPEG data URI (Design.md §5, Option A)."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _completion(messages: list[dict[str, Any]]) -> Any:
    """Run one chat completion against the router; raises ``VQAServiceError``."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise VQAServiceError(
            "HF_TOKEN is not set. Copy .env.example to .env and fill it in, "
            "then expose it as an environment variable."
        )
    client = OpenAI(base_url=BASE_URL, api_key=token)
    try:
        return client.chat.completions.create(
            model=os.environ.get("VQA_MODEL", DEFAULT_MODEL),
            messages=messages,
        )
    except Exception as exc:  # network/HTTP/auth errors from the OpenAI SDK
        raise VQAServiceError(f"Model call failed: {exc}") from exc


def answer_question(image: Image.Image, question: str) -> str:
    """Ask the hosted VLM what is in ``image`` and return the text answer.

    Args:
        image: The uploaded image (any mode/format; converted to RGB JPEG).
        question: A plain-English question about the image.

    Returns:
        The model's raw text answer.

    Raises:
        VQAServiceError: If the upstream model call fails or returns no text.
    """
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": _image_to_data_uri(image)}},
            ],
        }
    ]
    completion = _completion(messages)
    content = completion.choices[0].message.content
    if not content:
        raise VQAServiceError("Model returned an empty answer.")
    return content