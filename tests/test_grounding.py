"""Phase 6 tests: grounding — box parsing/rescaling, geo-mapping, Folium overlay,
and the POST /api/ground endpoint (Backlog.md Phase 6)."""

from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from backend import grounding, vqa_service
from backend.app import app
from backend.grounding import (
    GroundingError,
    parse_boxes,
    rescale_boxes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TIF = REPO_ROOT / "sample_multispectral.tif"


class _FakeMessage:
    content = '[{"label": "water", "xmin": 146, "ymin": 146, "xmax": 409, "ymax": 409}]'


class _FakeChoice:
    message = _FakeMessage()


class _FakeCompletion:
    choices = [_FakeChoice()]


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def mock_completion(monkeypatch):
    monkeypatch.setattr(vqa_service, "_completion", lambda messages: _FakeCompletion())


def _jpg_bytes(color: str = "blue") -> bytes:
    image = Image.new("RGB", (16, 16), "white")
    ImageDraw.Draw(image).rectangle([2, 2, 12, 12], fill=color)
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


# --- parsing ---------------------------------------------------------------


def test_parse_well_formed_json():
    raw = '[{"label":"pond","xmin":146,"ymin":146,"xmax":409,"ymax":409}]'
    boxes = parse_boxes(raw)
    assert len(boxes) == 1
    assert boxes[0].label == "pond"
    assert (boxes[0].xmin, boxes[0].ymin, boxes[0].xmax, boxes[0].ymax) == (
        146, 146, 409, 409)


def test_parse_strips_markdown_fences():
    raw = "```json\n[{\"label\":\"pond\",\"xmin\":146,\"ymin\":146,\"xmax\":409,\"ymax\":409}]\n```"
    assert len(parse_boxes(raw)) == 1


def test_parse_recovers_from_malformed_entries():
    """Near-JSON output (probe-observed) loses one box but keeps the other."""
    raw = ('[{"label":"square","xmin":146,"ymin":146,"xmax":409,"ymax":409},'
           ' {"label":"square","xmin": [582, 378, "ymin":378,"xmax":845,"ymax":641}]')
    boxes = parse_boxes(raw)
    assert len(boxes) == 1
    assert boxes[0].label == "square"


def test_parse_captures_confidence():
    raw = '[{"label":"pond","xmin":1,"ymin":1,"xmax":5,"ymax":5,"confidence":0.93}]'
    assert parse_boxes(raw)[0].confidence == 0.93


def test_parse_no_boxes_raises():
    """Prose with no array at all is unparseable -> error."""
    with pytest.raises(GroundingError):
        parse_boxes("There are no buildings in this image.")


def test_parse_clean_empty_array_is_valid_no_match():
    """A clean ``[]`` means 'no matching objects', not an error."""
    assert parse_boxes("[]") == []
    assert parse_boxes("No matches.\n```[]```") == []


# --- rescaling -------------------------------------------------------------


def test_rescale_normalized_to_pixels():
    boxes = parse_boxes('[{"label":"x","xmin":250,"ymin":0,"xmax":750,"ymax":1000}]')
    px = rescale_boxes(boxes, 400, 300)[0]
    # 0-1000 normalized -> square 400x400 space -> pixel coords.
    assert (px.xmin, px.xmax) == (100, 300)
    assert (px.ymin, px.ymax) == (0, 400)


def test_ground_box_as_dict_ints():
    boxes = parse_boxes('[{"label":"x","xmin":250,"ymin":250,"xmax":750,"ymax":750}]')
    as_dict = rescale_boxes(boxes, 100, 100)[0].as_dict()
    assert isinstance(as_dict["xmin"], int) and as_dict["xmin"] == 25


# --- GeoTIFF geo-mapping + Folium overlay ----------------------------------


def test_geotiff_geo_bounds_and_map(sample_tif=SAMPLE_TIF):
    if not sample_tif.exists():
        pytest.skip("sample_multispectral.tif not present")
    preview, w, h = grounding.render_rgb_preview(str(sample_tif))
    assert (w, h) == (100, 100)

    boxes = rescale_boxes(
        parse_boxes('[{"label":"x","xmin":0,"ymin":0,"xmax":500,"ymax":500}]'),
        w, h,
    )
    geo = grounding.georeference_boxes(str(sample_tif), boxes)
    assert "geo" in geo[0] and "lonlat" in geo[0]
    # EPSG:3857 geotiff -> WGS84 longitudes remain near 0.
    assert abs(geo[0]["lonlat"][0]) < 1e-3

    # Folium overlay renders a self-contained map with a rectangle.
    html = grounding.build_folium_overlay(
        preview, [[0.0, 0.0], [1e-3, 1e-3]], geo,
    )
    assert "leaflet" in html.lower()
    assert "data:image/png;base64," in html
    assert "rectangle" in html.lower() or "L.rectangle" in html


# --- endpoint --------------------------------------------------------------


def test_ground_endpoint_jpg_returns_boxes_and_no_map(client, mock_completion):
    response = client.post(
        "/api/ground",
        files=[("image", ("x.jpg", _jpg_bytes(), "image/jpeg"))],
        data={"query": "water"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["boxes"][0]["label"] == "water"
    assert payload["preview"].startswith("iVBOR")  # PNG base64
    assert payload["map_html"] is None


def test_ground_endpoint_geotiff_returns_map(client, mock_completion):
    if not SAMPLE_TIF.exists():
        pytest.skip("sample_multispectral.tif not present")
    response = client.post(
        "/api/ground",
        files=[("image", ("s.tif", SAMPLE_TIF.read_bytes(), "image/tiff"))],
        data={"query": "water"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["geo_bounds"] is not None
    assert payload["map_html"] is not None
    assert payload["boxes"][0]["lonlat"] is not None


def test_ground_endpoint_rejects_blank_query(client):
    response = client.post(
        "/api/ground",
        files=[("image", ("x.jpg", _jpg_bytes(), "image/jpeg"))],
        data={"query": "   "},
    )
    assert response.status_code == 400


def test_ground_endpoint_rejects_corrupt_file(client, mock_completion):
    response = client.post(
        "/api/ground",
        files=[("image", ("x.tif", b"not a tiff", "image/tiff"))],
        data={"query": "water"},
    )
    assert response.status_code == 400


def test_ground_endpoint_maps_model_failure_to_502(client, monkeypatch):
    def boom(messages):
        raise vqa_service.VQAServiceError("model call failed")

    monkeypatch.setattr(vqa_service, "_completion", boom)
    response = client.post(
        "/api/ground",
        files=[("image", ("x.jpg", _jpg_bytes(), "image/jpeg"))],
        data={"query": "water"},
    )
    assert response.status_code == 502
    assert "model call failed" in response.json()["detail"]


def test_ground_endpoint_parses_realistic_model_output(client, monkeypatch):
    """Model output with fences + prose ('Here are the boxes:') still parses."""
    class _ProseMessage:
        content = (
            "Here are the detected water bodies:\n```\n"
            '[{"label": "water", "xmin": 100, "ymin": 100, "xmax": 400, '
            '"ymax": 400, "confidence": 0.82}]\n```'
        )

    class _ProseChoice:
        message = _ProseMessage()

    class _ProseCompletion:
        choices = [_ProseChoice()]

    monkeypatch.setattr(vqa_service, "_completion", lambda messages: _ProseCompletion())
    response = client.post(
        "/api/ground",
        files=[("image", ("x.jpg", _jpg_bytes(), "image/jpeg"))],
        data={"query": "water"},
    )
    assert response.status_code == 200
    box = response.json()["boxes"][0]
    assert box["label"] == "water" and 0 < box["xmin"] < box["xmax"] < 16