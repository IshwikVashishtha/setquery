import * as GeoTIFF from 'geotiff'

/**
 * Render any GeoTIFF or standard TIFF file into a viewable PNG data URL.
 * Handles 1-band (grayscale/SAR), 3-band (RGB), multi-band (Landsat/Sentinel),
 * uint8, uint16, int16, and float32 rasters with percentile contrast stretching.
 *
 * @param {File | Blob} file
 * @returns {Promise<string>} PNG Data URL
 */
export async function renderTiffPreview(file) {
  try {
    const arrayBuffer = await file.arrayBuffer()
    const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer)
    const image = await tiff.getImage()

    const width = image.getWidth()
    const height = image.getHeight()

    // Read all raster bands
    const rasters = await image.readRasters()
    if (!rasters || rasters.length === 0) {
      throw new Error('No raster data found in TIFF')
    }

    const numBands = rasters.length

    // Create an off-screen canvas (downscale large imagery for snappy rendering)
    const maxDim = 1200
    let renderWidth = width
    let renderHeight = height
    if (width > maxDim || height > maxDim) {
      const scale = maxDim / Math.max(width, height)
      renderWidth = Math.max(1, Math.round(width * scale))
      renderHeight = Math.max(1, Math.round(height * scale))
    }

    const canvas = document.createElement('canvas')
    canvas.width = renderWidth
    canvas.height = renderHeight
    const ctx = canvas.getContext('2d')
    const imgData = ctx.createImageData(renderWidth, renderHeight)

    // Compute robust 2% - 98% percentile stretch for a band
    function computePercentileStretch(arr) {
      const step = Math.max(1, Math.floor(arr.length / 40000))
      const sample = []
      for (let i = 0; i < arr.length; i += step) {
        const v = arr[i]
        if (!isNaN(v) && isFinite(v) && v !== 0) {
          sample.push(v)
        }
      }
      if (sample.length < 50) {
        for (let i = 0; i < arr.length; i += step) {
          const v = arr[i]
          if (!isNaN(v) && isFinite(v)) sample.push(v)
        }
      }
      if (sample.length === 0) return { min: 0, max: 255 }

      sample.sort((a, b) => a - b)
      const low = sample[Math.floor(sample.length * 0.02)]
      const high = sample[Math.floor(sample.length * 0.98)]
      return { min: low, max: high > low ? high : low + 1 }
    }

    const ranges = []
    for (let b = 0; b < Math.min(3, numBands); b++) {
      ranges.push(computePercentileStretch(rasters[b]))
    }

    const rBand = rasters[0]
    const gBand = numBands >= 2 ? rasters[1] : rasters[0]
    const bBand = numBands >= 3 ? rasters[2] : rasters[0]

    const rRange = ranges[0]
    const gRange = ranges[Math.min(1, numBands - 1)]
    const bRange = ranges[Math.min(2, numBands - 1)]

    const xRatio = width / renderWidth
    const yRatio = height / renderHeight
    const data = imgData.data
    let ptr = 0

    for (let y = 0; y < renderHeight; y++) {
      const srcY = Math.min(height - 1, Math.floor(y * yRatio))
      const rowOffset = srcY * width

      for (let x = 0; x < renderWidth; x++) {
        const srcX = Math.min(width - 1, Math.floor(x * xRatio))
        const srcIdx = rowOffset + srcX

        const rVal = rBand[srcIdx]
        const gVal = gBand[srcIdx]
        const bVal = bBand[srcIdx]

        const rNorm = Math.min(
          255,
          Math.max(
            0,
            Math.round(((rVal - rRange.min) / (rRange.max - rRange.min)) * 255)
          )
        )
        const gNorm = Math.min(
          255,
          Math.max(
            0,
            Math.round(((gVal - gRange.min) / (gRange.max - gRange.min)) * 255)
          )
        )
        const bNorm = Math.min(
          255,
          Math.max(
            0,
            Math.round(((bVal - bRange.min) / (bRange.max - bRange.min)) * 255)
          )
        )

        data[ptr++] = isNaN(rNorm) ? 0 : rNorm
        data[ptr++] = isNaN(gNorm) ? 0 : gNorm
        data[ptr++] = isNaN(bNorm) ? 0 : bNorm
        data[ptr++] = 255 // fully opaque
      }
    }

    ctx.putImageData(imgData, 0, 0)
    return canvas.toDataURL('image/png')
  } catch (err) {
    console.error('Client-side GeoTIFF decoding error:', err)
    throw err
  }
}
