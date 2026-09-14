"""GDAL-free stand-in for the upstream `utils.py`.

The upstream tools do ``from utils import read_image, read_image_uint8`` and
``from osgeo import gdal``. This repo's GDAL comes from ``rasterio`` (the
``osgeo`` Python bindings have no Windows wheel for Python 3.13), so this module
re-implements the same three functions on rasterio with identical semantics:

- ``read_image(path)``        -> 2D array for single band; (H, W, B) for multi
- ``read_image_uint8(path)``  -> same shapes, min-max normalized to uint8
- ``get_geotransform(path)``  -> (GDAL geotransform tuple, WKT) or (None, None)
"""

import numpy as np
import rasterio


def read_image(file_path: str) -> np.ndarray:
    with rasterio.open(file_path) as src:
        if src.count == 1:
            return src.read(1)
        img = src.read()                # (bands, H, W)
        return np.transpose(img, (1, 2, 0))


def read_image_uint8(file_path: str) -> np.ndarray:
    img = read_image(file_path).astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val) * 255
    else:
        img = np.zeros_like(img)

    return img.astype(np.uint8)


def get_geotransform(file_path) -> tuple:
    with rasterio.open(file_path) as src:
        transform = src.transform
        # Affine(a, b, c, d, e, f) -> GDAL (c, a, b, f, d, e)
        geo = (transform.c, transform.a, transform.b,
               transform.f, transform.d, transform.e)
        proj = src.crs.to_wkt() if src.crs else None
    if geo == (0, 1.0, 0, 0, 0, 1.0):
        return None, None
    else:
        return geo, proj