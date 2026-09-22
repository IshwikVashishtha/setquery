import axios from 'axios'
import { renderTiffPreview } from './renderTiff.js'

const API_BASE = import.meta.env.VITE_BACKEND_URL || ''

const api = axios.create({
  baseURL: API_BASE,
  timeout: 300000, 
})

/**
 * Perform tool-augmented analysis on an uploaded image.
 * @param {File} file - Image or GeoTIFF file
 * @param {string} question - Question to ask about the image
 * @returns {Promise<{answer: string, tools_used?: Array<{name: string, server: string, input?: any, output?: any}>}>}
 */
export async function analyzeImage(file, question) {
  const formData = new FormData()
  formData.append('image', file)
  formData.append('question', question)

  const response = await api.post('/api/analyze', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

/**
 * Standard VQA request for single image
 * @param {File} file
 * @param {string} question
 */
export async function vqaImage(file, question) {
  const formData = new FormData()
  formData.append('image', file)
  formData.append('question', question)

  const response = await api.post('/api/vqa', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })

  return response.data
}

/**
 * Grounding request for object localization
 * @param {File} file
 * @param {string} query
 */
export async function groundImage(file, query) {
  const formData = new FormData()
  formData.append('image', file)
  formData.append('query', query)

  const response = await api.post('/api/ground', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })

  return response.data
}

/**
 * Bi-temporal change detection
 * @param {File} file1
 * @param {File} file2
 * @param {string} question
 * @param {string} [date1]
 * @param {string} [date2]
 */
export async function detectChange(file1, file2, question, date1, date2) {
  const formData = new FormData()
  formData.append('image1', file1)
  formData.append('image2', file2)
  formData.append('question', question)
  if (date1) formData.append('date1', date1)
  if (date2) formData.append('date2', date2)

  const response = await api.post('/api/change', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })

  return response.data
}

/**
 * Generates or retrieves a preview URL for a GeoTIFF or standard image.
 * Uses client-side GeoTIFF raster extraction and percentile contrast stretching,
 * with graceful fallback to backend /api/preview or standard object URL.
 * @param {File} file
 * @returns {Promise<string>} Preview URL (Data URL or Blob URL)
 */
export async function getPreviewUrl(file) {
  const isTiff = file.name.toLowerCase().match(/\.(tif|tiff)$/) !== null
  if (isTiff) {
    try {
      // 1. Try client-side GeoTIFF decoding first (instant, works offline/online)
      const dataUrl = await renderTiffPreview(file)
      return dataUrl
    } catch (clientErr) {
      console.warn('Client-side GeoTIFF render failed, trying backend /api/preview...', clientErr)
      try {
        // 2. Try backend preview service
        const formData = new FormData()
        formData.append('image', file)
        const response = await api.post('/api/preview', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          responseType: 'blob',
        })
        return URL.createObjectURL(response.data)
      } catch (backendErr) {
        console.warn('Backend preview failed:', backendErr)
        return null
      }
    }
  }
  return URL.createObjectURL(file)
}

/**
 * Check backend health status
 */
export async function checkHealth() {
  const response = await api.get('/health')
  return response.data
}

export default api
