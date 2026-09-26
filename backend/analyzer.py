"""Tool-augmented analysis: every upload goes through MCP tools, then the
tool outputs + the image go to the VLM for the final answer.

Flow (backend/app.py ``/api/analyze``):
  upload (any format) → tmp file on disk
  ChromaDB retrieves the top-8 tools most relevant to the question
  the arg-mappable subset is called over MCP (grouped per server)
  a band-stretched preview (GeoTIFF) or the PIL image (raster) is sent to the
  VLM together with the tool measurements → final text answer.
"""
from __future__ import annotations

import logging
import os
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from . import tool_router, vqa_service
from .grounding import render_rgb_preview
from .vqa_service import VQAServiceError

from logger import get_logger
logger = get_logger(__name__)

GEOTIFF_EXTENSIONS = {"tif", "tiff"}


async def _ensure_index() -> None:
    try:
        built = await tool_router.build_index()
        if built:
            logger.info("Tool index built for %s tools", tool_router.retrieve_tools)
        else:
            logger.info("Tool index already present — reused")
    except Exception as exc:
        logger.warning("Tool index unavailable: %s", exc)


async def analyze(
    question: str, data: bytes, filename: str
) -> dict:
    """Answer ``question`` about the uploaded file using MCP tools + the VLM.

    Returns ``{"answer": str, "tools_used": [{"server","name","output"|...}]}``.

    Raises:
        VQAServiceError: model call failure (maps to 502 upstream).
        ValueError / UnidentifiedImageError: unusable upload (maps to 400).
    """
    ext = (filename or "").rsplit(".", 1)[-1].lower()
    is_geotiff = ext in GEOTIFF_EXTENSIONS

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=f".{ext}" if ext else ".bin", delete=False
        ) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        await _ensure_index()
        selected = await tool_router.retrieve_tools(question, k=8)
        tools_used = await tool_router.call_tools(selected, str(tmp_path), question)

        if is_geotiff:
            image = render_rgb_preview(str(tmp_path))[0]
        else:
            try:
                image = Image.open(BytesIO(data))
                image.load()
            except UnidentifiedImageError:
                raise ValueError("image is not a valid image file.")

        answer = vqa_service.answer_with_tools(image, question, tools_used)
        return {"answer": answer, "tools_used": tools_used}
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass