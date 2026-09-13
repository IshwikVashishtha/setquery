"""GeoTIFF index computation for Phase 2 (Design.md §7, Phase 2).

Real band math with rasterio — no placeholders. NDVI/NDWI are computed from
actual reflected-band arrays, masked for nodata, windowed to a bbox.

Band resolution (order of preference):
  1. Explicit ``*_band`` argument, if given.
  2. Band *descriptions* (``dataset.descriptions`` or band tags) matched on
     keywords: ``red``/``redband``, ``green``, ``nir``, ``swir``.
  3. If none match, raise ``BandResolutionError`` with an actionable message —
     we do not guess per-satellite band conventions.

Formulas:
  NDVI = (NIR - Red) / (NIR + Red)
  NDWI = (Green - NIR) / (Green + NIR)   % McFeeters (1996)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import rasterio as rio

BandKind = Literal["red", "green", "nir", "swir"]

_KIND_TO_TOKENS: dict[BandKind, tuple[str, ...]] = {
    "red": ("redband", "red band", "red", "sr_b4"),
    "green": ("greenband", "green band", "green", "sr_b3"),
    "nir": ("nirband", "nir band", "nir", "near infrared", "sr_b5", "sr_b8"),
    "swir": ("swirband", "swir band", "swir"),
}


class GeoError(RuntimeError):
    """Base error for geospatial index computation (maps to HTTP 4xx upstream)."""


class BandResolutionError(GeoError):
    """Raised when a required band cannot be identified in the raster."""


def _window_for(src: rio.DatasetReader, bbox: list[float] | None):
    """Rasterio window covering ``bbox`` (raster CRS), or the full image if None."""
    if bbox is None:
        return None
    if len(bbox) != 4:
        raise GeoError("bbox must have 4 values: [minx, miny, maxx, maxy].")
    try:
        return src.window(bbox[0], bbox[1], bbox[2], bbox[3])
    except rio.errors.WindowError as exc:
        raise GeoError(f"Invalid bbox {bbox}: {exc}") from exc


def _description(src: rio.DatasetReader, band: int) -> str:
    """Best-effort human name for a band, from descriptions or tags."""
    desc = (src.descriptions[band - 1] or "").strip()
    if not desc:
        desc = src.tags(band).get("DESCRIPTION") or src.tags(band).get("band_name") or ""
    return desc.lower()


def _resolve_bands(
    src: rio.DatasetReader,
    needed: tuple[BandKind, ...],
    **explicit: int,
) -> dict[BandKind, int]:
    """Map each needed band to a 1-based band index in the dataset.

    ``explicit`` may carry pre-resolved indices (e.g. ``red=3, nir=4``).
    """
    resolved: dict[BandKind, int] = {}
    for kind in needed:
        if kind in explicit and explicit[kind] is not None:
            band = int(explicit[kind])
            if not 1 <= band <= src.count:
                raise GeoError(f"Band {band} is out of range (dataset has {src.count} bands).")
            resolved[kind] = band
            continue

        # No explicit index: try to match over 1..count by keyword first match.
        for band in range(1, src.count + 1):
            desc = _description(src, band)
            if any(tok in desc for tok in _KIND_TO_TOKENS[kind]):
                resolved[kind] = band
                break
        else:
            raise BandResolutionError(
                f"Could not identify the '{kind}' band in this GeoTIFF "
                f"({src.count} bands, descriptions: {list(src.descriptions)}). "
                f"Pass it explicitly, e.g. compute_ndvi(path, bbox, {kind}=<n>)."
            )
    return resolved


def _band_array(src: rio.DatasetReader, band: int, window) -> np.ndarray:
    """Read one band as a float64 array; nodata pixels become NaN."""
    arr = src.read(band, window=window, masked=True).astype("float64")
    return np.ma.filled(arr, np.nan)


def _index_from_bands(src: rio.DatasetReader, band_map: dict[BandKind, int], window, numerator_kind, denominator) -> float:
    """Mean of ``value = (A - B) / (A + B)`` over valid pixels in the window."""
    a = _band_array(src, band_map[numerator_kind], window)
    b = _band_array(src, band_map[denominator], window)
    denom = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.where(denom != 0, (a - b) / denom, np.nan)
    return float(np.nanmean(value))


def compute_ndvi(path: str, bbox: list[float] | None = None, *, red: int | None = None, nir: int | None = None) -> float:
    """Normalized Difference Vegetation Index, averaged over ``bbox``.

    Args:
        path: Path to a GeoTIFF containing near-infrared and red bands.
        bbox: ``[minx, miny, maxx, maxy]`` in the raster's CRS, or None for the
            whole image.
        red, nir: Optional explicit 1-based band indices, bypassing description
            resolution.

    Returns:
        Mean NDVI in [-1, 1] over valid (non-nodata) pixels.

    Raises:
        GeoError: if the file can't be opened or ``bbox`` is invalid.
        BandResolutionError: if the red/NIR bands can't be identified.
    """
    with rio.open(path) as src:
        bands = _resolve_bands(src, ("red", "nir"), red=red, nir=nir)
        window = _window_for(src, bbox)
        return _index_from_bands(src, bands, window, "nir", "red")


def compute_ndwi(path: str, bbox: list[float] | None = None, *, green: int | None = None, nir: int | None = None) -> float:
    """Normalized Difference Water Index (McFeeters), averaged over ``bbox``.

    Args:
        path: Path to a GeoTIFF containing green and near-infrared bands.
        bbox: ``[minx, miny, maxx, maxy]`` in the raster's CRS, or None for the
            whole image.
        green, nir: Optional explicit 1-based band indices, bypassing
            description resolution.

    Returns:
        Mean NDWI in [-1, 1] over valid pixels.

    Raises:
        GeoError: if the file can't be opened or ``bbox`` is invalid.
        BandResolutionError: if the green/NIR bands can't be identified.
    """
    with rio.open(path) as src:
        bands = _resolve_bands(src, ("green", "nir"), green=green, nir=nir)
        window = _window_for(src, bbox)
        return _index_from_bands(src, bands, window, "green", "nir")