"""Pydantic v2 models for the VQA API."""

from pydantic import BaseModel


class VQAResponse(BaseModel):
    """Response body returned by ``POST /api/vqa``."""

    answer: str


class GroundBoxResponse(BaseModel):
    """One grounding detection: pixel box (+ optional real-world coordinates)."""

    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float | None = None
    #: EPSG:4326 ``[minx, miny, maxx, maxy]`` — present only for GeoTIFF uploads.
    geo: list[float] | None = None
    #: EPSG:4326 center point ``[lon, lat]`` — present only for GeoTIFF uploads.
    lonlat: list[float] | None = None


class GroundResponse(BaseModel):
    """Body returned by ``POST /api/ground``."""

    width: int
    height: int
    boxes: list[GroundBoxResponse]
    #: Base64-encoded PNG preview the boxes are drawn over (same pixel space).
    preview: str
    #: Leaflet bounds ``[[south, west], [north, east]]`` — only for GeoTIFF.
    geo_bounds: list[list[float]] | None = None
    #: Self-contained Folium/Leaflet HTML with the image + box rectangles
    #: overlaid at their real locations — only for GeoTIFF uploads.
    map_html: str | None = None