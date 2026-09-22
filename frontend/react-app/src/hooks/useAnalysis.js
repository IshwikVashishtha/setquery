import { useState, useCallback } from 'react'
import { analyzeImage } from '../utils/api.js'

export function useAnalysis({
  onComplete,
  onToolStart,
  onToolComplete,
  onToolsUsed,
} = {}) {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [toolsUsed, setToolsUsed] = useState([])
  const [answer, setAnswer] = useState(null)

  const sendMessage = useCallback(
    async (file, question) => {
      if (!file || !question.trim()) return

      setIsLoading(true)
      setError(null)
      setAnswer(null)
      setToolsUsed([])

      try {
        // Step 1: Upload & Initialise
        onToolStart?.('init')
        await new Promise((r) => setTimeout(r, 200))
        onToolComplete?.('init')

        // Step 2: Tool Index / Vector Search
        onToolStart?.('tool_index')
        await new Promise((r) => setTimeout(r, 200))
        onToolComplete?.('tool_index')

        // Step 3: Retrieve MCP Tools
        onToolStart?.('retrieve_tools')
        await new Promise((r) => setTimeout(r, 200))
        onToolComplete?.('retrieve_tools')

        // Step 4: Execute Tools
        onToolStart?.('call_tools')

        // Call backend /api/analyze
        const response = await analyzeImage(file, question)

        onToolComplete?.('call_tools')

        const tools = response.tools_used || []
        setToolsUsed(tools)
        onToolsUsed?.(tools)

        // Step 5: Render RGB Preview
        onToolStart?.('render_preview')
        await new Promise((r) => setTimeout(r, 250))
        onToolComplete?.('render_preview')

        // Step 6 & 7: VLM Inference & Answer
        onToolStart?.('vlm_inference')
        const finalAnswer = response.answer || '(No answer returned)'
        setAnswer(finalAnswer)

        onComplete?.(finalAnswer)
        return response
      } catch (err) {
        const errorMsg =
          err.response?.data?.detail ||
          err.message ||
          'Failed to analyze image. Please verify backend is running.'
        setError(errorMsg)
        throw err
      } finally {
        setIsLoading(false)
      }
    },
    [onComplete, onToolStart, onToolComplete, onToolsUsed]
  )

  return {
    sendMessage,
    isLoading,
    error,
    toolsUsed,
    answer,
  }
}

export default useAnalysis
