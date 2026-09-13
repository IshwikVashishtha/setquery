"""Phase 3 integration tests: geo computation routed through the MCP subprocess."""

import asyncio
import os
import sys

import numpy as np
import pytest
import rasterio as rio
from mcp import ClientSession, StdioServerParameters, stdio_client
from rasterio.transform import from_origin

from backend.mcp_client import REPO_ROOT, compute_indices


@pytest.fixture()
def geotiff(tmp_path):
    """10x10 4-band RGBN GeoTIFF; whole-image NDVI = 0.4, NDWI = -0.2222."""
    path = tmp_path / "ms.tif"
    w = h = 10
    red = np.full((h, w), 150, dtype="uint8")
    red[0:5, :] = 20
    green = np.full((h, w), 120, dtype="uint8")
    green[0:5, :] = 90
    nir = np.full((h, w), 150, dtype="uint8")
    nir[0:5, :] = 180
    with rio.open(
        path, "w", driver="GTiff", width=w, height=h, count=4,
        dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 10, 1, 1),
    ) as ds:
        ds.write(red, 1)
        ds.write(green, 2)
        ds.write(nir, 4)
        for band, name in [(1, "Red"), (2, "Green"), (4, "NIR")]:
            ds.set_band_description(band, name)
    return str(path)


@pytest.fixture()
def bare_geotiff(tmp_path):
    """3-band GeoTIFF with no descriptions -> no indexable bands."""
    path = tmp_path / "bare.tif"
    with rio.open(path, "w", driver="GTiff", width=4, height=4, count=3,
                  dtype="uint8") as ds:
        ds.write(np.full((3, 4, 4), 100, dtype="uint8"))
    return str(path)


def test_compute_indices_via_mcp_subprocess(geotiff):
    """The same NDVI/NDWI values as the direct geo_tools path, via MCP."""
    indices = asyncio.run(compute_indices(geotiff))
    assert indices["NDVI"] == pytest.approx(0.4, abs=1e-6)
    assert indices["NDWI"] == pytest.approx(-0.222222, abs=1e-6)


def test_missing_bands_are_skipped_not_fatal(bare_geotiff):
    """A raster with no identifiable bands yields {} — not an exception."""
    assert asyncio.run(compute_indices(bare_geotiff)) == {}


def test_server_tool_count_below_truncation_threshold():
    """Only 2 tools: context-aware truncation is correctly not built yet."""

    async def list_tools():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "earth_agent.mcp_server"],
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return (await session.list_tools()).tools

    tools = asyncio.run(list_tools())
    names = {t.name for t in tools}
    assert {"compute_ndvi", "compute_ndwi"} <= names
    assert len(tools) < 10  # Backlog.md: truncation only once tools exceed ~10