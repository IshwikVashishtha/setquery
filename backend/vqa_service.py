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


def _extract_answer(completion: Any) -> str:
    """Pull the text answer out of a completion, or raise a clear error."""
    content = completion.choices[0].message.content
    if not content:
        raise VQAServiceError("Model returned an empty answer.")
    return content


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
    return _extract_answer(_completion(messages))


def answer_index_question(question: str, indices: dict[str, float]) -> str:
    
    """Answer a question about a GeoTIFF from computed multispectral indices.

    The GeoTIFF itself isn't sent to the model (it's not a viewable image) —
    instead the *measured* index values are placed in the prompt so the model
    reasons from real numbers instead of guessing (the Phase 2 cross-check).

    Args:
        question: A plain-English question about the raster, e.g. about
            vegetation or water health.
        indices: Measured values, e.g. ``{"NDVI": 0.42, "NDWI": -0.31}``.

    Returns:
        The model's raw text answer.

    Raises:
        VQAServiceError: If the upstream model call fails or returns no text.
    """
    measurements = ", ".join(f"{name}: {value:.3f}" for name, value in indices.items())
    prompt = (
        "The user uploaded a multispectral GeoTIFF. The following index values "
        f"were computed from its bands (means over valid pixels): {measurements}. "
        f"Answer the user's question using these measured values where relevant.\n\n"
        f"Question: {question}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    return _extract_answer(_completion(messages))


def answer_with_tools(
    image: Image.Image, question: str, tool_outputs: list[dict]
) -> str:
    """Answer ``question`` from the image *together with* tool measurements.

    The tool-augmented analysis path (backend/analyzer.py) runs the upload
    through the top-relevant MCP tools, then hands the resulting measurements
    to the model alongside the image so it reasons from real numbers, not
    guesses — the image-and-tools input the user asked for.

    Args:
        image: The uploaded image (GeoTIFFs arrive as a band-stretched preview).
        question: A plain-English question about the image/raster.
        tool_outputs: Tool results consumed by ``analyzer.call_tools``, each
            ``{"server", "name", "output"|"error", "skipped": bool}``.

    Returns:
        The model's raw text answer.

    Raises:
        VQAServiceError: If the upstream model call fails or returns no text.
    """
    used = [
        o
        for o in tool_outputs
        if not o.get("skipped") and (o.get("output") or o.get("error"))
    ]
    if used:
        lines = "\n".join(
            f"- {o['name']} ({o['server']}): {o.get('output') or o.get('error')}"
            for o in used
        )
        measurements = f"The following analysis tools ran on the uploaded image:\n{lines}"
    else:
        skipped = [o["name"] for o in tool_outputs if o.get("skipped")]
        measurements = (
            "No tool produced a measurement for this upload."
            + (f" Retrieved tools ({', '.join(skipped[:5])}) were skipped because "
               "their inputs could not be supplied from the uploaded file." if skipped else "")
        )
    prompt = (
        "A user asked a question about an uploaded remote-sensing image/raster.\n"
        f"{measurements}\n"
        "Incorporate these measured values where relevant, then answer the "
        f"user's question.\n\nQuestion: {question}"
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _image_to_data_uri(image)}},
            ],
        }
    ]
    return _extract_answer(_completion(messages))


_GROUNDING_PROMPT = """\
You are an object-detection assistant. Return ONLY a JSON array of the objects
matching the user's request that are visible in the image. Each element is an
object with exactly these keys:
  {{"label": "<object name>", "confidence": <0-1>, "xmin": <int>, "ymin": <int>, "xmax": <int>, "ymax": <int>}}
Rules:
- All coordinates are integers in [0, 1000] in normalized image space (0 = left/top edge, 1000 = right/bottom edge); xmin < xmax and ymin < ymax.
- One element per matching object; include every distinct instance.
- If nothing matches the request, return [].
- Do not output a single word outside the JSON array.

Requested object: {query}"""


def ground_objects(image: Image.Image, query: str) -> str:
    """Ask the hosted VLM to localize every instance of ``query`` in ``image``.

    Phase 6 grounding path (Backlog.md): the model returns a JSON array of
    bounding boxes for the requested object as **raw text** — parsing and
    rescaling are done in ``backend/grounding.py``, keeping the model-facing
    boundary isolated here (Agent.md §2). Coordinates come back in 0-1000
    normalized space.

    Args:
        image: The uploaded image (any mode/format; converted to RGB JPEG).
        query: The object class to localize, e.g. "buildings".

    Returns:
        The model's raw text answer (near-JSON array of boxes).

    Raises:
        VQAServiceError: If the upstream model call fails or returns no text.
    """
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _GROUNDING_PROMPT.format(query=query)},
                {"type": "image_url", "image_url": {"url": _image_to_data_uri(image)}},
            ],
        }
    ]
    return _extract_answer(_completion(messages))


def answer_change_question(
    image_before: Image.Image,
    image_after: Image.Image,
    question: str,
    date_before: str | None = None,
    date_after: str | None = None,
) -> str:
    """Describe changes between two co-registered images (Phase 5 ChangeVQA).

    Both images and, if given, their capture dates are sent to the hosted VLM in
    a single message so it can compare them directly. The model is set by
    ``VQA_MODEL``; today that's a general VLM (Qwen3-VL) — swap it for a
    purpose-built ChangeVQA model without touching this function.

    Args:
        image_before: The earlier capture.
        image_after: The later capture.
        question: A plain-English change-detection question.
        date_before, date_after: Optional capture dates for temporal context.

    Returns:
        The model's raw text answer.

    Raises:
        VQAServiceError: If the upstream model call fails or returns no text.
    """
    temporal = ""
    if date_before and date_after:
        temporal = f" Captured on {date_before} (before) and {date_after} (after)."
    prompt = (
        "Two images of the same region are provided at different times."
        f"{temporal}"
        " Compare them and describe what changed between the two; "
        f"then answer the user's question.\n\nQuestion: {question}"
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _image_to_data_uri(image_before)}},
                {"type": "image_url", "image_url": {"url": _image_to_data_uri(image_after)}},
            ],
        }
    ]
    return _extract_answer(_completion(messages))