"""LangGraph orchestration for the VQA agent (Phase 4, Design.md §7).

Replaces the endpoint's hardcoded single path with a small state machine:
a router node dispatches on request shape (image count, file kind, radar flag)
to single-image / bi-temporal / optical-SAR nodes.

Scope guard: only the **single-image** branch is implemented, and it is the
exact logic that shipped in Phases 1–3 (no regression — see the manual test in
Backlog.md). The bi-temporal and optical-SAR branches exist as reachable nodes
that return an explicit "not implemented" message; they get real behavior in
Phases 5 (bi-temporal) and beyond.
"""

from __future__ import annotations

import logging
import os
import tempfile
from io import BytesIO
from typing import Literal, TypedDict

from fastapi import HTTPException
from langgraph.graph import END, START, StateGraph
from PIL import Image, UnidentifiedImageError

from . import grounding
from .grounding import GroundingError
from .vqa_service import (
    VQAServiceError,
    answer_change_question,
    answer_question,
)

logger = logging.getLogger(__name__)

FileKind = Literal["raster", "geotiff"]
Route = Literal["single_image", "bi_temporal", "optical_sar"]

#: Branches that exist in the graph but have no implementation yet.
_NOT_IMPLEMENTED: dict[Route, str] = {
    "optical_sar": (
        "Optical-SAR fusion is a deferred path and is not implemented yet."
    ),
}


class OrchestrationState(TypedDict, total=False):
    """State threaded through the orchestration graph."""

    image_bytes: bytes
    image2_bytes: bytes
    question: str
    file_kind: FileKind
    n_images: int
    sar_request: bool
    date_before: str | None
    date_after: str | None
    answer: str
    error: str


def decide_route(state: OrchestrationState) -> Route:
    """Route on request shape: image count -> type -> radar marker.

    This is the router node. Placeholder axes (``n_images >= 2``,
    ``sar_request``) are not reachable from the current single-image API; they
    exist so Phase 5 slots in without restructuring the graph.
    """
    if state.get("n_images", 1) >= 2:
        return "bi_temporal"
    if state.get("sar_request"):
        return "optical_sar"
    return "single_image"


def _decode_image(data: bytes) -> Image.Image:
    """Decode an uploaded image, or fail with the standard 400 message."""
    try:
        pil_image = Image.open(BytesIO(data))
        pil_image.load()  # fail fast on corrupt files
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="image is not a valid image file.")
    return pil_image


def _answer_raster_image(data: bytes, question: str) -> str:
    """Phase 1 logic: send the image to the hosted VLM (unchanged)."""
    try:
        return answer_question(_decode_image(data), question)
    except VQAServiceError as exc:
        logger.error("VQA model call failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


async def _answer_geotiff(data: bytes, question: str) -> str:
    """Legacy ``/api/vqa`` GeoTIFF path: band-stretch to an RGB preview.

    The NDVI/NDWI-over-MCP pipeline that once lived here moved to the
    tool-augmented ``/api/analyze`` path (``backend/analyzer.py``), which sends
    every upload — any format — through the top-relevant MCP tools and hands
    measurements plus the image to the VLM. This route now renders the raster
    as a viewable RGB preview and answers directly, same as JPEG/PNG.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            preview, _w, _h = grounding.render_rgb_preview(tmp_path)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="This GeoTIFF's bands could not be rendered as an RGB "
                       "preview. Add band descriptions (e.g. 'Red', 'NIR') or use "
                       "the tool-augmented /api/analyze endpoint.",
            ) from exc

        try:
            return answer_question(preview, question)
        except VQAServiceError as exc:
            logger.error("VQA model call failed: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        if tmp_path:
            os.unlink(tmp_path)


async def single_image_node(state: OrchestrationState) -> dict:
    """Execute the single-image path."""
    kind = state.get("file_kind", "raster")
    answer = (
        await _answer_geotiff(state["image_bytes"], state["question"])
        if kind == "geotiff"
        else _answer_raster_image(state["image_bytes"], state["question"])
    )
    return {"answer": answer}


async def bi_temporal_node(state: OrchestrationState) -> dict:
    """Phase 5: compare two co-registered images and describe the change.

    Both images (and dates, if given) go to the VLM in one message via
    ``vqa_service.answer_change_question`` — the model-facing boundary stays
    isolated in that module.
    """
    try:
        return {
            "answer": answer_change_question(
                _decode_image(state["image_bytes"]),
                _decode_image(state["image2_bytes"]),
                state["question"],
                state.get("date_before"),
                state.get("date_after"),
            )
        }
    except VQAServiceError as exc:
        logger.error("VQA model call failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def not_implemented_node(state: OrchestrationState) -> dict:
    """Deferred branches report an explicit 501-style message."""
    route = decide_route(state)
    return {"error": _NOT_IMPLEMENTED[route]}


def build_graph():
    """Compile the orchestration state machine."""
    graph = StateGraph(OrchestrationState)
    graph.add_node("single_image_node", single_image_node)
    graph.add_node("bi_temporal_node", bi_temporal_node)
    graph.add_node("not_implemented", not_implemented_node)
    graph.add_conditional_edges(
        START,
        decide_route,
        {
            "single_image": "single_image_node",
            "bi_temporal": "bi_temporal_node",
            "optical_sar": "not_implemented",
        },
    )
    graph.add_edge("single_image_node", END)
    graph.add_edge("bi_temporal_node", END)
    graph.add_edge("not_implemented", END)
    return graph.compile()


GRAPH = build_graph()