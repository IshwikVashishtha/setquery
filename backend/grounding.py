"""Phase 6: grounding — parse VLM boxes, map them onto imagery and a Leaflet map.

The hosted VLM (``vqa_service.ground_objects``) returns a *human-shaped* near-JSON
array of bounding boxes in **0-1000 normalized** coordinates. This module turns
that into typed, pixel-accurate boxes (``GroundBox``), rescales them to the
actual image dimensions, and — when the upload is a georeferenced GeoTIFF —
maps each pixel box to real-world coordinates and renders an interactive
**Folium/Leaflet** map with the image overlaid and a rectangle per box
(Backlog.md Phase 6).

Pipeline::

    raw text            parse_boxes()          rescale_boxes()       boxes (pixels)
    ─────────────────► normalized GroundBoxes ───────────────────►  xmin..ymax [px]
        then, for GeoTIFF:
        pixel box -> rasterio transform -> geo box -> warp to EPSG:4326
        + build_folium_overlay(preview, geo_bounds, geo_boxes) -> HTML
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass

import numpy as np
import rasterio as rio
import rasterio.warp
from PIL import Image, UnidentifiedImageError

from .vqa_service import ground_objects

#: Normalized coordinate space the model is asked to emit (0..NORMALIZED).
NORMALIZED = 1000

_BOX_KEYS = ("xmin", "ymin", "xmax", "ymax")
_NUM_RE = re.compile(r'"(xmin|ymin|xmax|ymax|confidence)"\s*:\s*([0-9.eE+-]+)')
_LABEL_RE = re.compile(r'"label"\s*:\s*"([^"]*)"')


class GroundingError(RuntimeError):
    """No usable boxes could be parsed out of the model's output (maps to 4xx)."""


@dataclass
class GroundBox:
    """One detected object instance.

    ``xmin/ymin/xmax/ymax`` are in the **normalized** 0-1000 space as emitted by
    the model; use ``rescale_boxes`` to convert to pixel coordinates.
    """

    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float | None = None

    def as_dict(self, scale: float | None = None) -> dict:
        """Serializable dict with integer coords.

        ``scale`` rescales a *normalized* (0-1000) box to pixel space; when None,
        the stored (already-pixel) coords are passed through and coerced to int.
        """
        def v(value: float) -> int:
            px = value / NORMALIZED * scale if scale else value
            return int(round(min(max(px, 0), scale))) if scale else int(round(px))

        box = {"label": self.label, "xmin": v(self.xmin), "ymin": v(self.ymin),
               "xmax": v(self.xmax), "ymax": v(self.ymax)}
        if self.confidence is not None:
            box["confidence"] = round(float(self.confidence), 3)
        return box


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    """Remove markdown code fences the model sometimes wraps JSON in."""
    return re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)


def _candidate_json(text: str) -> str | None:
    """Best-effort slice of ``text`` holding the outermost JSON array."""
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]


def _numeric(value: object) -> float:
    """Parse an int/float or numeric string; raise ValueError if not numeric."""
    if isinstance(value, bool):  # bool is an int subclass; treat as invalid
        raise ValueError(f"not numeric: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value.strip())  # type: ignore[union-attr]
    except AttributeError:
        raise ValueError(f"not numeric: {value!r}") from None


def _make_box(fields: dict, label: str) -> GroundBox | None:
    """Build a valid GroundBox from key->value fields, or None if malformed."""
    try:
        box = GroundBox(
            label=label.strip() or "object",
            xmin=_numeric(fields["xmin"]),
            ymin=_numeric(fields["ymin"]),
            xmax=_numeric(fields["xmax"]),
            ymax=_numeric(fields["ymax"]),
            confidence=(
                _numeric(fields["confidence"])
                if fields.get("confidence") is not None
                else None
            ),
        )
    except (KeyError, ValueError):
        return None
    if box.confidence is not None and not 0 <= box.confidence <= 1:
        box.confidence = None
    if not (0 <= box.xmin < box.xmax <= NORMALIZED and 0 <= box.ymin < box.ymax <= NORMALIZED):
        return None
    return box


def _parse_json_array(candidate: str) -> list[GroundBox]:
    """Strict-ish path: json.loads, validating each element via _make_box."""
    try:
        raw = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        raise ValueError("not valid JSON") from None
    if not isinstance(raw, list):
        raise ValueError("JSON is not an array")
    boxes: list[GroundBox] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "object"))
        box = _make_box(
            {key: item.get(key) for key in _BOX_KEYS + ("confidence",)},
            label,
        )
        if box is not None:
            boxes.append(box)
    return boxes


def _parse_by_regex(candidate: str) -> list[GroundBox]:
    """Fallback for near-JSON output: scrape per-object fields with regexes.

    The model occasionally mangles a value (e.g. ``"xmin": [582, 378``) which
    breaks json.loads but leaves the other fields on the same object recoverable.
    """
    boxes: list[GroundBox] = []
    for obj in re.findall(r"\{[^{}]*\}", candidate):
        fields: dict[str, str] = {}
        for key, value in _NUM_RE.findall(obj):
            fields.setdefault(key, value)
        label = ""
        match = _LABEL_RE.search(obj)
        if match:
            label = match.group(1)
        missing_nums = [k for k in _BOX_KEYS if k not in fields]
        if missing_nums:
            continue  # incomplete box — skip rather than emit a partial one
        box = _make_box(fields, label)
        if box is not None:
            boxes.append(box)
    return boxes


def parse_boxes(raw_text: str) -> list[GroundBox]:
    """Parse the model's grounding output into validated normalized boxes.

    Strategies, in order: strip fences -> json.loads; if that fails, regex-scrape
    per-object fields. Both paths drop malformed entries instead of failing the
    whole request. A clean empty array ``[]`` is a **valid** answer meaning "no
    matching objects"; raise ``GroundingError`` only when the output contains no
    parseable box data at all.

    Raises:
        GroundingError: if the output contains no parseable bounding boxes.
    """
    text = _strip_fences(raw_text)
    candidate = _candidate_json(text)
    if candidate is None:
        raise GroundingError(
            "The model's grounding output contained no JSON array "
            f"(raw: {raw_text[:200]!r})."
        )
    try:
        boxes = _parse_json_array(candidate)
    except ValueError:
        boxes = _parse_by_regex(candidate)
    if boxes:
        return boxes
    if _is_clean_empty_array(candidate):
        return []
    raise GroundingError(
        "The model returned grounding output that contained no usable "
        f"bounding boxes. Raw output was: {raw_text[:200]!r}"
    )


def _is_clean_empty_array(candidate: str) -> bool:
    """True when the JSON candidate is literally an empty array ``[]``."""
    try:
        return json.loads(candidate) == [] and isinstance(json.loads(candidate), list)
    except (json.JSONDecodeError, ValueError):
        return False


def rescale_boxes(boxes: list[GroundBox], width: int, height: int) -> list[GroundBox]:
    """Rescale normalized (0-1000) boxes to pixel coordinates.

    The normalized space is x/y-agnostic, so width and height are scaled by the
    same factor to keep boxes square in pixel space when the image is non-square.
    """
    scale = max(width, height)
    rescaled = []
    for box in boxes:
        rescaled.append(
            GroundBox(
                label=box.label,
                xmin=round(box.xmin / NORMALIZED * scale),
                ymin=round(box.ymin / NORMALIZED * scale),
                xmax=round(box.xmax / NORMALIZED * scale),
                ymax=round(box.ymax / NORMALIZED * scale),
                confidence=box.confidence,
            )
        )
    return rescaled


# ---------------------------------------------------------------------------
# Image / geo helpers
# ---------------------------------------------------------------------------


def render_rgb_preview(path: str, max_pixels: int = 1200) -> tuple[Image.Image, int, int]:
    """Render the first 3 bands of a GeoTIFF as a viewable RGB PIL image.

    Returns ``(preview, src_width, src_height)``. The preview keeps the source
    pixel dimensions (so grounding boxes on the preview map 1:1 onto the raster);
    ``max_pixels`` only caps display so huge rasters stay responsive.
    """
    with rio.open(path) as src:
        width, height = src.width, src.height
        window = None
        # Downsample via windowed read when the raster is huge.
        if width > max_pixels or height > max_pixels:
            scale = max_pixels / max(width, height)
            new_w = max(1, int(width * scale))
            new_h = max(1, int(height * scale))
            window = rio.windows.from_bounds(
                *src.bounds, width=new_w, height=new_h, transform=src.transform
            )
            width, height = new_w, new_h

        band_count = min(3, src.count)
        arrays = [
            src.read(i, window=window, masked=True).astype("float32")
            for i in range(1, band_count + 1)
        ]
        if band_count == 1:
            arrays = arrays + arrays + arrays  # grayscale -> RGB

        def stretch(arr: np.ndarray) -> np.ndarray:
            data = np.ma.filled(arr, np.nan)  # drop mask before percentile (numpy warning)
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                return np.zeros_like(data, dtype="uint8")
            low, high = np.percentile(finite, 2), np.percentile(finite, 98)
            if high <= low:
                return np.full_like(arr, 128, dtype="uint8")
            scaled = np.clip((arr - low) / (high - low) * 255, 0, 255)
            return scaled.astype("uint8")

        rgb = np.dstack([stretch(a) for a in arrays])
        preview = Image.fromarray(rgb)
        return preview, width, height


def _geo_bounds(src: rio.DatasetReader) -> list[list[float]]:
    """Raster bounds in WGS84 as Leaflet bounds ``[[south, west], [north, east]]``."""
    west, south, east, north = rasterio.warp.transform_bounds(
        src.crs, "EPSG:4326", *src.bounds
    )
    return [[south, west], [north, east]]


def _box_to_geo(src: rio.DatasetReader, box: GroundBox) -> dict:
    """Convert a *pixel-space* box (xmin..ymax in image px) to an EPSG:4326 box.

    Uses the raster transform: pixel (col, row) -> geo (x, y) via ``src.xy``,
    then warps the raster CRS to WGS84 for Leaflet (any source CRS works).
    Returns ``{geo: [minx, miny, maxx, maxy], lonlat: [lon, lat]}`` (center
    point) ready for JSON/Leaflet.
    """
    # Top-left and bottom-right pixel centers -> geo coordinates.
    xs, ys = [], []
    for (row, col) in ((box.ymin + 0.5, box.xmin + 0.5),
                       (box.ymax + 0.5, box.xmax + 0.5)):
        x, y = src.xy(row, col)
        xs.append(x)
        ys.append(y)
    lon, lat = rasterio.warp.transform(src.crs, "EPSG:4326", xs, ys)
    return {
        "geo": [min(lon), min(lat), max(lon), max(lat)],
        "lonlat": [(min(lon) + max(lon)) / 2, (min(lat) + max(lat)) / 2],
    }


def georeference_boxes(path: str, boxes_px: list[GroundBox]) -> list[dict]:
    """Add ``geo`` / ``lonlat`` (EPSG:4326) to a GeoTIFF's pixel-space boxes."""
    with rio.open(path) as src:
        return [_box_to_geo(src, box) for box in boxes_px]


# ---------------------------------------------------------------------------
# Folium / Leaflet overlay
# ---------------------------------------------------------------------------


def build_folium_overlay(
    preview: Image.Image,
    bounds_lonlat: list[list[float]],
    geo_boxes: list[dict],
    image_label: str = "Aerial image",
) -> str:
    """Build an interactive Leaflet map (Folium) with the image + box rectangles.

    Args:
        preview: The RGB preview rendered from the imagery; it is saved to a temp
            PNG and base64-embedded (data URI) so the returned HTML is
            self-contained and tiles are the only network dependency.
        bounds_lonlat: Leaflet bounds ``[[south, west], [north, east]]``.
        geo_boxes: Per-box ``{"label", "geo": [minx,miny,maxx,maxy], "lonlat"}``.

    Returns:
        Standalone HTML string (Leaflet + OpenStreetMap tiles) for the frontend.
    """
    import folium
    from folium import raster_layers

    south, west, north, east = (
        bounds_lonlat[0][0],
        bounds_lonlat[0][1],
        bounds_lonlat[1][0],
        bounds_lonlat[1][1],
    )
    center = [(south + north) / 2, (west + east) / 2]
    zoom = 16  # close enough to see the overlay; user can zoom

    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")
    # Image overlay: stretch the preview to the raster's real geo footprint.
    # Folium embeds a file path as a base64 data URI, keeping the HTML portable.
    png_path = None
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            png_path = tmp.name
        preview.convert("RGB").save(png_path, format="PNG")
        raster_layers.ImageOverlay(
            name=image_label,
            image=png_path,
            bounds=[[south, west], [north, east]],
            opacity=1.0,
        ).add_to(m)
    finally:
        if png_path:
            import os

            os.unlink(png_path)

    for box in geo_boxes:
        minx, miny, maxx, maxy = box["geo"]
        folium.Rectangle(
            bounds=[[miny, minx], [maxy, maxx]],
            color="#ff0000",
            weight=2,
            fill=True,
            fill_opacity=0.15,
            popup=box.get("label", "object"),
        ).add_to(m)
    folium.LayerControl().add_to(m)
    return m.get_root().render()


# ---------------------------------------------------------------------------
# Top-level entry: image bytes + query -> everything the API needs
# ---------------------------------------------------------------------------


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def ground_image(
    data: bytes,
    query: str,
    is_geotiff: bool,
    tmp_path: str | None = None,
) -> dict:
    """Run the full Phase 6 grounding pipeline for one uploaded file.

    For plain images the model sees the image directly and the preview is the
    same image. For GeoTIFFs the model sees an RGB band preview and each pixel
    box is additionally mapped to real-world coordinates so the result can be
    overlaid on a map.

    Returns a dict for the API response::

        {"boxes": [GroundBox pixel dicts (+ geo for geotiffs)],
         "preview": base64 PNG,
         "geo_bounds": [[south, west], [north, east]] | None,
         "map_html": folium HTML | None}

    Raises:
        GroundingError: no usable boxes returned by the model.
        VQAServiceError: upstream model call failed (mapped to 502).
    """
    if is_geotiff:
        assert tmp_path, "GeoTIFF grounding requires a path on disk."
        try:
            preview, width, height = render_rgb_preview(tmp_path)
        except rio.errors.RasterioIOError as exc:
            raise GroundingError(f"Could not read the GeoTIFF: {exc}") from exc
        raw = ground_objects(preview, query)
    else:
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise GroundingError("image is not a valid image file.") from exc
        width, height = image.size
        preview = image
        raw = ground_objects(image, query)

    boxes = rescale_boxes(parse_boxes(raw), width, height)
    # ``as_dict()`` (no scale) passes the pixel values through unchanged, since
    # ``rescale_boxes`` already converted from the 0-1000 normalized space.
    box_dicts = [box.as_dict() for box in boxes]
    result: dict = {
        "width": width,
        "height": height,
        "boxes": box_dicts,
        "preview": base64.b64encode(_png_bytes(preview)).decode("ascii"),
        "geo_bounds": None,
        "map_html": None,
    }

    if is_geotiff:
        geo = georeference_boxes(tmp_path, boxes)
        result["boxes"] = [dict(box, **geo[i]) for i, box in enumerate(box_dicts)]
        with rio.open(tmp_path) as src:
            bounds = _geo_bounds(src)
        result["geo_bounds"] = bounds
        result["map_html"] = build_folium_overlay(preview, bounds, geo)
    return result