import React, { useCallback, useState, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'
import { FiUpload, FiFile, FiX, FiInfo } from 'react-icons/fi'
import { getPreviewUrl } from '../utils/api.js'
import './ImagePanel.css'

const ImagePanel = ({ file, previewUrl, onFileSelect, onReset, isTiff }) => {
  const [geoTiffPreview, setGeoTiffPreview] = useState(null)
  const [loadingPreview, setLoadingPreview] = useState(false)

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0])
    }
  }, [onFileSelect])

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.webp'],
      'application/octet-stream': ['.tif', '.tiff']
    },
    multiple: false,
    maxSize: 100 * 1024 * 1024 // 100MB
  })

  useEffect(() => {
    let active = true
    let createdUrl = null

    if (file && file.name.toLowerCase().match(/\.(tif|tiff)$/)) {
      setLoadingPreview(true)
      getPreviewUrl(file)
        .then((url) => {
          if (active) {
            createdUrl = url
            setGeoTiffPreview(url)
          }
        })
        .catch((err) => {
          console.error('GeoTIFF preview failed:', err)
          if (active) setGeoTiffPreview(null)
        })
        .finally(() => {
          if (active) setLoadingPreview(false)
        })
    } else {
      setGeoTiffPreview(null)
    }

    return () => {
      active = false
      if (createdUrl) {
        URL.revokeObjectURL(createdUrl)
      }
    }
  }, [file])

  const displayImage = file
    ? (file.name.toLowerCase().match(/\.(tif|tiff)$/) ? (geoTiffPreview || previewUrl) : previewUrl)
    : null

  return (
    <div className="image-panel">
      <div className="panel-header">
        <h2>Image Source</h2>
        {file && (
          <button className="clear-btn" onClick={onReset} title="Clear image">
            <FiX size={16} />
          </button>
        )}
      </div>

      {!file ? (
        <div
          {...getRootProps()}
          className={`dropzone ${isDragActive ? 'active' : ''}`}
        >
          <input {...getInputProps()} />
          <div className="dropzone-content">
            <FiUpload size={40} className="dropzone-icon" />
            <p className="dropzone-text">
              {isDragActive
                ? 'Drop your image here...'
                : 'Drag & drop image, or click to browse'}
            </p>
            <span className="dropzone-hint">
              JPG, PNG, GeoTIFF (.tif / .tiff)
            </span>
          </div>
        </div>
      ) : (
        <div className="image-display">
          <div className="image-frame">
            {loadingPreview ? (
              <div className="preview-loading">
                <div className="spinner"></div>
                <p>Rendering GeoTIFF preview...</p>
              </div>
            ) : displayImage ? (
              <img src={displayImage} alt="Uploaded" className="uploaded-image" />
            ) : (
              <div className="no-preview">
                <FiInfo size={28} />
                <p>Generating preview for {file.name}</p>
              </div>
            )}
            {isTiff && (
              <div className="tiff-badge">
                <FiFile size={13} /> GeoTIFF
              </div>
            )}
          </div>
          <div className="file-info">
            <span className="file-name" title={file.name}>{file.name}</span>
            <span className="file-size">
              {(file.size / (1024 * 1024)).toFixed(2)} MB
            </span>
          </div>
        </div>
      )}

      {fileRejections.length > 0 && (
        <div className="file-error">
          <FiInfo size={14} />
          <span>{fileRejections[0].errors[0].message}</span>
        </div>
      )}
    </div>
  )
}

export default ImagePanel
