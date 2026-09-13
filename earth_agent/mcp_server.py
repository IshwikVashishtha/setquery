"""Earth-Agent MCP server (Phase 3, Design.md §7 / Backlog Phase 3).

Exposes the Phase 2 GeoTIFF index functions as Model Context Protocol tools
over stdio. The arithmetic lives in ``backend.geo_tools.py`` — this module only
*exposes* it, so there is exactly one source of truth for the band math.

Run with:
    python -m earth_agent.mcp_server

Context-aware tool truncation is intentionally deferred: Backlog.md says it is
needed only once the tool count exceeds ~10. The current surface (NDVI, NDWI)
does not risk overflowing a 7B model's context window.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from backend.geo_tools import compute_ndvi as _compute_ndvi
from backend.geo_tools import compute_ndwi as _compute_ndwi

mcp = FastMCP("earth-agent")


@mcp.tool(description="Mean NDVI ((NIR-Red)/(NIR+Red)) over a GeoTIFF region.")
def compute_ndvi(path: str, bbox: list[float] | None = None) -> float:
    """Compute the mean NDVI of the GeoTIFF at ``path``.

    Args:
        path: Absolute path to a GeoTIFF with red and NIR bands.
        bbox: Optional [minx, miny, maxx, maxy] in the raster's CRS.
    """
    return _compute_ndvi(path, bbox)


@mcp.tool(description="Mean NDWI (McFeeters (Green-NIR)/(Green+NIR)) over a region.")
def compute_ndwi(path: str, bbox: list[float] | None = None) -> float:
    """Compute the mean NDWI of the GeoTIFF at ``path``.

    Args:
        path: Absolute path to a GeoTIFF with green and NIR bands.
        bbox: Optional [minx, miny, maxx, maxy] in the raster's CRS.
    """
    return _compute_ndwi(path, bbox)


if __name__ == "__main__":
    mcp.run(transport="stdio")