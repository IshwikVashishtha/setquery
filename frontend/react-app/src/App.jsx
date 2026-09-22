import React, { useState, useCallback } from 'react'
import ChatPanel from './components/ChatPanel.jsx'
import ImagePanel from './components/ImagePanel.jsx'
import FlowGraph from './components/FlowGraph.jsx'
import Header from './components/Header.jsx'
import { useAnalysis } from './hooks/useAnalysis.js'
import { useProcessingSteps } from './hooks/useProcessingSteps.js'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [isTiff, setIsTiff] = useState(false)

  const { steps, setCurrentStep, resetSteps, completeStep } = useProcessingSteps()
  const { sendMessage, isLoading, error, toolsUsed, answer } = useAnalysis({
    onComplete: (answer) => {
      completeStep('vlm_inference')
      completeStep('vlm_answer')
    },
    onToolStart: (stepName) => {
      setCurrentStep(stepName)
    },
    onToolComplete: (stepName, result) => {
      completeStep(stepName)
    },
    onToolsUsed: (tools) => {
      if (tools?.length > 0) {
        setCurrentStep('vlm_inference')
      }
    }
  })

  const handleFileSelect = useCallback((file) => {
    setSelectedFile(file)
    setIsTiff(file.name.toLowerCase().match(/\.(tif|tiff)$/) !== null)

    const url = URL.createObjectURL(file)
    setPreviewUrl(url)

    resetSteps()
  }, [resetSteps])

  const handleSendMessage = useCallback(async (message) => {
    if (!selectedFile || !message.trim()) return

    await sendMessage(selectedFile, message)
  }, [selectedFile, sendMessage])

  const resetContext = useCallback(() => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setIsTiff(false)
    resetSteps()
  }, [resetSteps])

  React.useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  return (
    <div className="app-container">
      <Header />
      <div className="three-column-layout">
        {/* Left Column - Image */}
        <div className="left-column">
          <ImagePanel
            file={selectedFile}
            previewUrl={previewUrl}
            onFileSelect={handleFileSelect}
            isTiff={isTiff}
            onReset={resetContext}
          />
        </div>

        {/* Center Column - 3D Flow Graph */}
        <div className="center-column">
          <FlowGraph steps={steps} currentStep={steps.currentStep} />
        </div>

        {/* Right Column - Chat */}
        <div className="right-column">
          <ChatPanel
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
            error={error}
            toolsUsed={toolsUsed}
            selectedFile={selectedFile}
            answer={answer}
          />
        </div>
      </div>
    </div>
  )
}

export default App