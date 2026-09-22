import React from 'react'
import { FiCloud, FiActivity, FiMessageSquare } from 'react-icons/fi'
import './Header.css'

const Header = () => {
  return (
    <div className="header">
      <div className="header-content">
        <div className="header-title">
          <h1>Remote Sensing VQA Agent</h1>
          <p>Upload satellite/aerial imagery and ask questions in plain English</p>
        </div>
        <div className="header-modules">
          <div className="module">
            <FiCloud size={24} className="module-icon" />
            <div>
              <h3>Image Analysis</h3>
              <p>Multi-format support</p>
            </div>
          </div>
          <div className="module">
            <FiActivity size={24} className="module-icon" />
            <div>
              <h3>Processing Pipeline</h3>
              <p>Real-time tool visualization</p>
            </div>
          </div>
          <div className="module">
            <FiMessageSquare size={24} className="module-icon" />
            <div>
              <h3>Interactive Chat</h3>
              <p>Tool-augmented responses</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Header
