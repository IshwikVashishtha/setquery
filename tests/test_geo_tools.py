"""Tests for Phase 2 GeoTIFF index computation and the GeoTIFF API path."""

import numpy as np
import pytest
import rasterio as rio
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from rasterio.transform import from_origin

from backend import grounding, vqa_service
from backend.app import app
from backend.geo_tools import BandResolutionError, compute_ndvi, compute_ndwi


def _write_geotiff(path: str, *, descriptions: list[str] | None = None) -> str:
    """10x10, 4-band RGBN GeoTIFF.

    Top half = vegetation (NDVI 0.8), bottom half = bare (NDVI 0.0);
    whole-image NDVI = 0.4, NDWI = -0.2222.
    """
    w = h = 10
    red = np.full((h, w), 150, dtype="uint8")
    red[0:5, :] = 20
    green = np.full((h, w), 120, dtype="uint8")
    green[0:5, :] = 90
    blue = np.full((h, w), 60, dtype="uint8")
    nir = np.full((h, w), 150, dtype="uint8")
    nir[0:5, :] = 180

    with rio.open(
        path, "w", driver="GTiff", width=w, height=h, count=4,
        dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 10, 1, 1),
    ) as ds:
        ds.write(red, 1)
        ds.write(green, 2)
        ds.write(blue, 3)
        ds.write(nir, 4)
        if descriptions:
            for band, name in enumerate(descriptions, start=1):
                ds.set_band_description(band, name)
    return path


def _write_bare_geotiff(path: str) -> str:
    """A 3-band GeoTIFF with no band descriptions (undecodable for NDVI)."""
    with rio.open(
        path, "w", driver="GTiff", width=4, height=4, count=3, dtype="uint8",
    ) as ds:
        ds.write(np.full((3, 4, 4), 100, dtype="uint8"))
    return path


# --- geo_tools unit tests -------------------------------------------------


def test_compute_ndvi_known_values(tmp_path):
    """NDVI matches hand-calculated values (0.8 vegetation / 0.4 whole)."""
    path = _write_geotiff(str(tmp_path / "sample_ms.tif"),
                          descriptions=["Red", "Green", "Blue", "NIR"])
    whole = compute_ndvi(path)
    assert whole == pytest.approx(0.4, abs=1e-6)
    bbox_top = compute_ndvi(path, [0, 5, 10, 10])  # only the vegetation half
    assert bbox_top == pytest.approx(0.8, abs=1e-6)


def test_compute_ndwi_known_values(tmp_path):
    """NDWI (McFeeters) matches hand-calculated values."""
    path = _write_geotiff(str(tmp_path / "sample_ms.tif"),
                          descriptions=["Red", "Green", "Blue", "NIR"])
    assert compute_ndwi(path) == pytest.approx(-0.222222, abs=1e-6)


def test_explicit_band_indices(tmp_path):
    """Explicit band indices bypass description resolution."""
    path = _write_geotiff(str(tmp_path / "sample_ms.tif"))
    assert compute_ndvi(path, nir=4, red=1) == pytest.approx(0.4, abs=1e-6)


def test_no_descriptions_raises(tmp_path):
    """Without descriptions/explicit bands, a clear BandResolutionError."""
    path = _write_bare_geotiff(str(tmp_path / "no_desc.tif"))
    with pytest.raises(BandResolutionError):
        compute_ndvi(path)


# --- /api/vqa GeoTIFF path -----------------------------------------------


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_vqa_geotiff_returns_answer(client, monkeypatch, tmp_path):
    """A GeoTIFF upload renders an RGB preview and answers directly."""
    captured = {}

    def capture_completion(messages):
        captured["messages"] = messages
        return _FakeCompletion()

    monkeypatch.setattr(vqa_service, "_completion", capture_completion)
    monkeypatch.setattr(
        grounding, "render_rgb_preview",
        lambda path: (PILImage.new("RGB", (8, 8), "green"), 8, 8),
    )

    path = _write_geotiff(str(tmp_path / "upload.tif"),
                          descriptions=["Red", "Green", "Blue", "NIR"])
    with open(path, "rb") as fh:
        response = client.post(
            "/api/vqa",
            files={"image": ("upload.tif", fh, "application/octet-stream")},
            data={"question": "Is the vegetation healthy?"},
        )
    assert response.status_code == 200
    assert len(response.json()["answer"]) > 0

    content = captured["messages"][0]["content"]
    assert any(c["type"] == "image_url" for c in content)  # preview reached the model
    assert any(
        c["type"] == "text" and "vegetation" in c["text"] for c in content
    )


def test_vqa_geotiff_unrenderable_returns_400(client, monkeypatch, tmp_path):
    """A GeoTIFF whose bands can't be stretched to an RGB preview -> 400."""
    def broken_preview(path):
        raise ValueError("no RGB bands")

    monkeypatch.setattr(vqa_service, "_completion", lambda messages: _FakeCompletion())
    monkeypatch.setattr(grounding, "render_rgb_preview", broken_preview)

    path = _write_bare_geotiff(str(tmp_path / "upload.tif"))
    with open(path, "rb") as fh:
        response = client.post(
            "/api/vqa",
            files={"image": ("upload.tif", fh, "application/octet-stream")},
            data={"question": "Is the vegetation healthy?"},
        )
    assert response.status_code == 400
    assert "preview" in response.json()["detail"]


class _FakeMessage:
    content = "placeholder"


class _FakeChoice:
    message = _FakeMessage()


class _FakeCompletion:
    choices = [_FakeChoice()]