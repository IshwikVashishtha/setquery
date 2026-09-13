"""Phase 4 tests: orchestration graph routing + no regression."""

import asyncio
from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from backend import mcp_client, vqa_service
from backend.orchestration import GRAPH, _NOT_IMPLEMENTED, decide_route


def _jpg_bytes() -> bytes:
    """A small valid JPEG."""
    image = Image.new("RGB", (16, 16), "blue")
    ImageDraw.Draw(image).rectangle([2, 2, 8, 8], fill="green")
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


class _FakeMessage:
    content = "A few shapes on a blue field."


class _FakeChoice:
    message = _FakeMessage()


class _FakeCompletion:
    choices = [_FakeChoice()]


# --- router decisions ------------------------------------------------------


def test_router_single_image_default():
    assert decide_route({"file_kind": "raster", "n_images": 1}) == "single_image"


def test_router_geotiff_is_still_single_image():
    assert decide_route({"file_kind": "geotiff", "n_images": 1}) == "single_image"


def test_router_two_images_routes_to_bi_temporal():
    assert decide_route({"n_images": 2}) == "bi_temporal"


def test_router_sar_marker_routes_to_optical_sar():
    assert decide_route({"sar_request": True}) == "optical_sar"


# --- graph behavior --------------------------------------------------------


def test_graph_single_image_returns_answer(monkeypatch):
    """The single-image branch runs the preserved answering logic."""
    monkeypatch.setattr(
        vqa_service, "_completion",
        lambda messages: _FakeCompletion(),
    )
    result = asyncio.run(
        GRAPH.ainvoke(
            {"image_bytes": _jpg_bytes(), "question": "What is here?",
             "file_kind": "raster", "n_images": 1}
        )
    )
    assert result.get("answer")
    assert not result.get("error")


def test_graph_bi_temporal_returns_answer(monkeypatch):
    """Phase 5: two-image requests reach the real bi-temporal branch."""
    captured = {}

    def capture_completion(messages):
        captured["messages"] = messages
        return _FakeCompletion()

    monkeypatch.setattr(vqa_service, "_completion", capture_completion)
    result = asyncio.run(
        GRAPH.ainvoke(
            {
                "image_bytes": _jpg_bytes(),
                "image2_bytes": _jpg_bytes(),
                "question": "What changed?",
                "n_images": 2,
                "date_before": "2025-01-01",
                "date_after": "2025-06-01",
            }
        )
    )
    assert result.get("answer")
    assert not result.get("error")
    content = captured["messages"][0]["content"]
    image_parts = [c for c in content if c["type"] == "image_url"]
    text_parts = [c for c in content if c["type"] == "text"]
    assert len(image_parts) == 2  # both images reach the model
    assert "2025-01-01" in text_parts[0]["text"]
    assert "2025-06-01" in text_parts[0]["text"]


def test_graph_sar_reports_not_implemented():
    result = asyncio.run(
        GRAPH.ainvoke({"sar_request": True, "question": "Anything?"})
    )
    assert result.get("error") == _NOT_IMPLEMENTED["optical_sar"]


def test_graph_geotiff_node_gets_measured_values(monkeypatch):
    """The geotiff node passes indices from the (mocked) MCP client to the model."""
    captured = {}

    def capture_completion(messages):
        captured["messages"] = messages
        return _FakeCompletion()

    async def fake_compute_indices(path, bbox=None):
        return {"NDVI": 0.7}

    monkeypatch.setattr(vqa_service, "_completion", capture_completion)
    monkeypatch.setattr(mcp_client, "compute_indices", fake_compute_indices)

    result = asyncio.run(
        GRAPH.ainvoke(
            {"image_bytes": b"not-used", "question": "Healthy?",
             "file_kind": "geotiff", "n_images": 1}
        )
    )
    assert result.get("answer") == _FakeMessage.content
    prompt_text = captured["messages"][0]["content"][0]["text"]
    assert "0.700" in prompt_text  # measured NDVI formatted to 3 decimals