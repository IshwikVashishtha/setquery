import { useState, useCallback } from 'react'

const STEP_ORDER = [
  'init',
  'tool_index',
  'retrieve_tools',
  'call_tools',
  'render_preview',
  'vlm_inference',
  'vlm_answer',
]

const initialNodes = () => ({
  init: { status: 'pending' },
  tool_index: { status: 'pending' },
  retrieve_tools: { status: 'pending' },
  call_tools: { status: 'pending' },
  render_preview: { status: 'pending' },
  vlm_inference: { status: 'pending' },
  vlm_answer: { status: 'pending' },
})

export function useProcessingSteps() {
  const [nodes, setNodes] = useState(initialNodes)
  const [currentStep, setCurrentStepState] = useState(null)

  const setCurrentStep = useCallback((stepId) => {
    setCurrentStepState(stepId)
    if (stepId) {
      setNodes((prev) => {
        const next = { ...prev }
        // Mark current step as running
        if (next[stepId]) {
          next[stepId] = { ...next[stepId], status: 'running' }
        }
        // Mark earlier steps in pipeline as completed if still pending
        const currentIndex = STEP_ORDER.indexOf(stepId)
        if (currentIndex > 0) {
          for (let i = 0; i < currentIndex; i++) {
            const prevId = STEP_ORDER[i]
            if (next[prevId] && next[prevId].status === 'pending') {
              next[prevId] = { ...next[prevId], status: 'completed' }
            }
          }
        }
        return next
      })
    }
  }, [])

  const completeStep = useCallback((stepId) => {
    setNodes((prev) => ({
      ...prev,
      [stepId]: { ...prev[stepId], status: 'completed' },
    }))
  }, [])

  const failStep = useCallback((stepId) => {
    setNodes((prev) => ({
      ...prev,
      [stepId]: { ...prev[stepId], status: 'error' },
    }))
    setCurrentStepState(null)
  }, [])

  const resetSteps = useCallback(() => {
    setNodes(initialNodes())
    setCurrentStepState(null)
  }, [])

  const completedCount = Object.values(nodes).filter(
    (n) => n.status === 'completed'
  ).length

  const steps = {
    nodes,
    currentStep,
    stats: {
      completed: completedCount,
      total: STEP_ORDER.length,
    },
  }

  return {
    steps,
    currentStep,
    setCurrentStep,
    completeStep,
    failStep,
    resetSteps,
  }
}

export default useProcessingSteps
