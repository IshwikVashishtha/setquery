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
from logger import get_logger
logger = get_logger(__name__)

BandKind = Literal["red", "green", "nir", "swir"]

_KIND_TO_TOKENS: dict[BandKind, tuple[str, ...]] = {
    "red": ("redband", "red band", "red", "sr_b4"),
    "green": ("greenband", "green band", "green", "sr_b3"),
    "nir": ("nirband", "nir band", "nir", "near infrared", "sr_b5", "sr_b8"),
    "swir": ("swirband", "swir band", "swir"),
}

# ---------------------------------------------------------------------------
# Band-count-based positional mapping (fallback when descriptions are missing)
# ---------------------------------------------------------------------------

# Each entry maps band count → {band_kind: 1-based_index}.
# Only the bands needed for NDVI/NDWI (red, green, nir) are mapped; SWIR is
# included where a standard format makes it unambiguous.
_BAND_POSITION_MAP: dict[int, dict[BandKind, int]] = {
    # 4-band drone / multispectral: R, G, B, NIR
    4: {"red": 1, "green": 2, "nir": 4},
    # Landsat 7 ETM+ (8 bands): B1-Blue, B2-Green, B3-Red, B4-NIR, B5-SWIR1, B6-TIR, B7-SWIR2, B8-Pan
    8: {"green": 2, "red": 3, "nir": 4},
    # Landsat 8/9 OLI (11 bands): B1-Coastal, B2-Blue, B3-Green, B4-Red, B5-NIR, B6-SWIR1, B7-SWIR2, B8-Pan, B9-Cirrus, B10-TIR1, B11-TIR2
    11: {"green": 3, "red": 4, "nir": 5},
    # Sentinel-2 (13 bands): B01-Coastal, B02-Blue, B03-Green, B04-Red, B05-B07-RE, B08-NIR, B08A-Narrow, B09-WV, B10-Cirrus, B11-SWIR1, B12-SWIR2
    13: {"green": 3, "red": 4, "nir": 8},
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

    Resolution order per band:
      1. Explicit ``*_band`` argument, if given.
      2. Band *descriptions* matched on keywords (``red``, ``nir``, …).
      3. Positional map for known satellite formats keyed by band count
         (Sentinel-2 = 13, Landsat 8/9 = 11, Landsat 7 = 8, drone RGBN = 4).

    ``explicit`` may carry pre-resolved indices (e.g. ``red=3, nir=4``).
    """
    resolved: dict[BandKind, int] = {}
    unresolved: list[BandKind] = []

    # --- pass 1: explicit indices + keyword matching ---
    for kind in needed:
        if kind in explicit and explicit[kind] is not None:
            band = int(explicit[kind])
            if not 1 <= band <= src.count:
                raise GeoError(f"Band {band} is out of range (dataset has {src.count} bands).")
            resolved[kind] = band
            continue

        for band in range(1, src.count + 1):
            desc = _description(src, band)
            if any(tok in desc for tok in _KIND_TO_TOKENS[kind]):
                resolved[kind] = band
                break
        else:
            unresolved.append(kind)

    # --- pass 2: positional fallback for known satellite formats ---
    if unresolved and src.count in _BAND_POSITION_MAP:
        pos = _BAND_POSITION_MAP[src.count]
        still_missing: list[BandKind] = []
        for kind in unresolved:
            if kind in pos:
                band = pos[kind]
                if 1 <= band <= src.count:
                    resolved[kind] = band
                else:
                    still_missing.append(kind)
            else:
                still_missing.append(kind)
        unresolved = still_missing

    if unresolved:
        raise BandResolutionError(
            f"Could not identify the '{unresolved[0]}' band in this GeoTIFF "
            f"({src.count} bands, descriptions: {list(src.descriptions)}). "
            f"Pass it explicitly, e.g. compute_ndvi(path, bbox, {unresolved[0]}=<n>)."
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