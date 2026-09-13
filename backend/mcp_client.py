"""MCP client: compute GeoTIFF indices by talking to ``earth_agent.mcp_server``.

Phase 3 (Backlog): the index computation that /api/vqa previously ran by calling
``geo_tools`` directly now goes through an MCP subprocess instead. A fresh server
is spawned per call and torn down cleanly afterwards (the source doc's subprocess
lifecycle pattern).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client

from .geo_tools import GeoError

REPO_ROOT = Path(__file__).resolve().parents[1]

# The tool raises this exact wording when a required band is missing; the client
# uses it to skip that index rather than failing the whole request.
_BAND_RESOLUTION_MARKER = "Could not identify"


def _server_params() -> StdioServerParameters:
    """Point ``python -m earth_agent.mcp_server`` at this venv, cwd=repo root."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "earth_agent.mcp_server"],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


async def _call_tool(session: ClientSession, name: str, arguments: dict) -> str:
    """Call one tool and return its text payload, converting failures to GeoError."""
    result = await session.call_tool(name, arguments)
    if result.isError:
        text = result.content[0].text if result.content else f"MCP tool {name} failed"
        raise GeoError(f"MCP tool {name} failed: {text}")
    if not result.content:
        raise GeoError(f"MCP tool {name} returned no content")
    return result.content[0].text


async def compute_indices(path: str | Path, bbox: list[float] | None = None) -> dict[str, float]:
    """Compute NDVI and NDWI for a GeoTIFF via the MCP subprocess.

    Args:
        path: Path to a GeoTIFF (band descriptions or known order needed).
        bbox: Optional [minx, miny, maxx, maxy] in the raster's CRS.

    Returns:
        A mapping of computable indices, e.g. ``{"NDVI": 0.389}``. Indices whose
        bands aren't present in the raster are simply omitted.

    Raises:
        GeoError: If the subprocess can't be reached or a tool call fails for a
            reason other than a missing band.
    """
    arguments: dict = {"path": str(path)}
    if bbox is not None:
        arguments["bbox"] = bbox

    indices: dict[str, float] = {}
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for name, tool in (("NDVI", "compute_ndvi"), ("NDWI", "compute_ndwi")):
                try:
                    indices[name] = float(await _call_tool(session, tool, arguments))
                except GeoError as exc:
                    if _BAND_RESOLUTION_MARKER in str(exc):
                        continue  # band absent in this raster — skip, don't fail
                    raise
    return indices