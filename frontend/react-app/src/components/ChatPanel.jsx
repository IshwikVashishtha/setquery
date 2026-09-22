import React, { useState, useRef, useEffect } from 'react'
import {
  FiSend,
  FiCpu,
  FiUser,
  FiCheckCircle,
  FiAlertCircle,
  FiLayers,
  FiTerminal,
} from 'react-icons/fi'
import './ChatPanel.css'

const SUGGESTIONS = [
  'What is the vegetation health (NDVI) in this area?',
  'Detect and count all aircraft or ships visible.',
  'Analyze land use and classify the terrain.',
  'Identify water bodies and calculate moisture index.',
]

function formatMessage(text) {
  if (!text) return text
  // Preserve line breaks by splitting and wrapping in <span> with white-space
  // Handle **bold**, `code`, and bullet lines
  const lines = text.split(/\r?\n/)
  const elements = []
  lines.forEach((line, i) => {
    const trimmed = line.trim()
    // Bullet / list
    if (trimmed.match(/^[\-\*]\s+/)) {
      elements.push(
        <div key={i} className="msg-list-item">{line.replace(/^[\-\*]\s+/, '')}</div>
      )
      return
    }
    // Code block marker
    if (trimmed === '```') {
      // skip markers; simple approach leaves inline only
      return
    }
    // Simple bold + backtick pass on remaining inline text
    const parts = line.split(/(\*\*.*?\*\*|`.*?`)/g)
    const inline = parts.map((part, idx) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={idx}>{part.slice(2, -2)}</strong>
      }
      if (part.startsWith('`') && part.endsWith('`')) {
        return <code key={idx} className="msg-inline-code">{part.slice(1, -1)}</code>
      }
      return <span key={idx}>{part}</span>
    })
    elements.push(<div key={i} className="msg-line">{inline}</div>)
  })
  return elements
}

const ChatPanel = ({
  onSendMessage,
  isLoading,
  error,
  toolsUsed = [],
  selectedFile,
  answer,
}) => {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      content:
        'Hello! Upload a satellite or aerial image (PNG, JPG, or GeoTIFF) and ask questions about land use, vegetation, objects, or change detection.',
      timestamp: new Date().toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
      }),
    },
  ])

  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isLoading])

  // When a new answer arrives from the pipeline hook, append it to messages
  useEffect(() => {
    if (answer) {
      setMessages((prev) => {
        // Prevent duplicate appending of the same answer
        const lastMsg = prev[prev.length - 1]
        if (lastMsg && lastMsg.role === 'assistant' && lastMsg.content === answer) {
          return prev
        }
        return [
          ...prev,
          {
            id: `assistant-${Date.now()}`,
            role: 'assistant',
            content: answer,
            tools: toolsUsed,
            timestamp: new Date().toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            }),
          },
        ]
      })
    }
  }, [answer, toolsUsed])

  const handleSend = async (e) => {
    e?.preventDefault()
    const text = input.trim()
    if (!text || isLoading || !selectedFile) return

    const userMsg = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date().toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
      }),
    }

    setMessages((prev) => [...prev, userMsg])
    setInput('')

    try {
      await onSendMessage(text)
    } catch {
      // Error handled by hook and passed in props
    }
  }

  const handleSuggestionClick = (prompt) => {
    if (isLoading || !selectedFile) return
    setInput(prompt)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="chat-panel">
      {/* Panel Header */}
      <div className="chat-header">
        <div className="chat-header-title">
          <FiCpu className="chat-header-icon" />
          <h2>VQA & Tool Assistant</h2>
        </div>
        <div className="chat-status-indicator">
          <span className={`status-dot ${isLoading ? 'active' : 'idle'}`}></span>
          <span className="status-text">{isLoading ? 'Processing' : 'Ready'}</span>
        </div>
      </div>

      {/* Messages Container */}
      <div className="chat-messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`message-row ${msg.role}`}>
            <div className="message-avatar">
              {msg.role === 'assistant' ? (
                <FiCpu size={14} />
              ) : (
                <FiUser size={14} />
              )}
            </div>
            <div className="message-bubble-container">
              <div className={`message-bubble ${msg.role}`}>
                <div className="message-text formatted">{formatMessage(msg.content)}</div>

                {/* Render tool executions badges if any */}
                {msg.tools && msg.tools.length > 0 && (
                  <div className="message-tools-badge">
                    <div className="tools-badge-header">
                      <FiLayers size={11} />
                      <span>Tools Executed ({msg.tools.length}):</span>
                    </div>
                    <div className="tools-list">
                      {msg.tools.map((tool, idx) => (
                        <div key={idx} className="tool-chip" title={tool.server ? `${tool.server}/${tool.name}` : tool.name}>
                          <FiTerminal size={10} />
                          <span>{tool.server ? `${tool.server}/${tool.name}` : tool.name || 'Tool'}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              <span className="message-time">{msg.timestamp}</span>
            </div>
          </div>
        ))}

        {/* Loading / Thinking State */}
        {isLoading && (
          <div className="message-row assistant">
            <div className="message-avatar">
              <FiCpu size={14} />
            </div>
            <div className="message-bubble-container">
              <div className="message-bubble assistant loading">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <span className="thinking-text">Executing remote sensing tools & VLM...</span>
              </div>
            </div>
          </div>
        )}

        {/* Error Notification */}
        {error && (
          <div className="chat-error-banner">
            <FiAlertCircle size={14} />
            <span>{error}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Prompt Suggestions */}
      {messages.length <= 2 && selectedFile && !isLoading && (
        <div className="chat-suggestions">
          <span className="suggestions-label">Suggested prompts:</span>
          <div className="suggestions-chips">
            {SUGGESTIONS.map((s, idx) => (
              <button
                key={idx}
                className="suggestion-chip"
                onClick={() => handleSuggestionClick(s)}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input Area */}
      <form className="chat-input-area" onSubmit={handleSend}>
        {!selectedFile && (
          <div className="upload-warning">
            <FiAlertCircle size={13} />
            <span>Upload an image in the left panel to begin querying.</span>
          </div>
        )}
        <div className="input-box-wrapper">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              selectedFile
                ? 'Ask a question about this satellite image... (Press Enter to send)'
                : 'Upload an image first...'
            }
            disabled={isLoading || !selectedFile}
            rows={1}
            className="chat-textarea"
          />
          <button
            type="submit"
            disabled={isLoading || !selectedFile || !input.trim()}
            className="send-button"
            title="Send query"
          >
            <FiSend size={15} />
          </button>
        </div>
      </form>
    </div>
  )
}

export default ChatPanel
