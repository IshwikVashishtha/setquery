import React, { Suspense, useRef, useMemo, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Sphere, Html, Stars, Sparkles } from '@react-three/drei'
import { FiClock, FiActivity } from 'react-icons/fi'
import * as THREE from 'three'
import './FlowGraph.css'

// Node definitions — one per backend pipeline stage
const NODE_DEFINITIONS = [
  { id: 'init', label: 'Image Upload', description: 'Upload satellite imagery' },
  { id: 'tool_index', label: 'Tool Index', description: 'Build ChromaDB tool index' },
  { id: 'retrieve_tools', label: 'Retrieve Tools', description: 'Find relevant MCP tools' },
  { id: 'call_tools', label: 'Call Tools', description: 'Execute selected tools' },
  { id: 'render_preview', label: 'Render Preview', description: 'Preview image for VLM' },
  { id: 'vlm_inference', label: 'VLM Inference', description: 'Send to Vision-Language Model' },
  { id: 'vlm_answer', label: 'Generate Answer', description: 'Final response generation' },
]


const NODE_POSITIONS = [
  [-7, 0, 3],
  [-4.5, 1.5, 1.5],
  [-2, -0.5, 0],
  [0.5, 1.5, -1.5],
  [3, -0.5, -3],
  [5.5, 1, -4.5],
  [7.5, -1, -6],
]

function statusColor(status, isActive) {
  if (status === 'completed') return '#008cffff'
  if (status === 'running' || isActive) return '#0004ffff'
  if (status === 'error') return '#ff3355'
  if (status === 'skipped') return '#777799'
  return '#3a3a5c'
}

// A single glowing node
function GlowingNode({ position, status, label, description, isActive }) {
  const glowRef = useRef()
  const color = statusColor(status, isActive)
  const emissive = status === 'completed' || status === 'running' || isActive

  useFrame(({ clock }) => {
    if (glowRef.current && (isActive || status === 'running')) {
      const s = 1.3 + Math.sin(clock.elapsedTime * 3) * 0.15
      glowRef.current.scale.set(s, s, s)
      glowRef.current.material.opacity = 0.25 + Math.sin(clock.elapsedTime * 3) * 0.1
    }
  })

  return (
    <group position={position}>
      {(isActive || status === 'running') && (
        <Sphere ref={glowRef} args={[1.0, 24, 24]}>
          <meshBasicMaterial color={color} transparent opacity={0.3} depthWrite={false} />
        </Sphere>
      )}
      <Sphere args={[0.7, 32, 32]}>
        <meshStandardMaterial
          color={color}
          emissive={emissive ? color : '#0dee27ff'}
          emissiveIntensity={isActive || status === 'running' ? 0.9 : status === 'completed' ? 0.5 : 0.1}
          roughness={0.35}
          metalness={0.7}
        />
      </Sphere>
      <Html distanceFactor={12} position={[0, 1.3, 0]} center>
        <div className={`node-label ${isActive ? 'active' : ''} ${status === 'completed' ? 'completed' : ''}`}>
          <div className="node-label-text">{label}</div>
          <div className="node-label-desc">{description}</div>
        </div>
      </Html>
    </group>
  )
}

// A pulse of light that travels the edge as the glow passes from node to node
function EdgePulse({ start, end, color }) {
  const ref = useRef()
  const s = useMemo(() => new THREE.Vector3(...start), [start])
  const e = useMemo(() => new THREE.Vector3(...end), [end])

  useFrame(({ clock }) => {
    if (ref.current) {
      const t = (clock.elapsedTime % 1.4) / 1.4
      ref.current.position.lerpVectors(s, e, t)
    }
  })

  return (
    <Sphere ref={ref} args={[0.12, 12, 12]}>
      <meshBasicMaterial color={color} transparent opacity={0.9} />
    </Sphere>
  )
}

// Static connection line between two nodes
function ConnectionLine({ start, end, status, isActive }) {
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([...start, ...end], 3)
    )
    return g
  }, [start, end])

  const active = isActive || status === 'running'
  const completed = status === 'completed'
  const color = completed ? '#008cffff' : active ? '#00aaff' : '#333355'

  return (
    <>
      <line geometry={geometry}>
        <lineBasicMaterial
          color={color}
          transparent
          opacity={completed ? 0.65 : active ? 0.85 : 0.25}
        />
      </line>
      {(active || completed) && <EdgePulse start={start} end={end} color={color} />}
    </>
  )
}

function NodeNetwork({ steps, currentStep }) {
  const nodes = steps?.nodes || {}

  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[0, 5, 10]} intensity={0.6} color="#00aaff" />
      <pointLight position={[0, -5, -10]} intensity={0.4} color="#008cffff" />

      {NODE_DEFINITIONS.slice(0, -1).map((node, idx) => {
        const nextId = NODE_DEFINITIONS[idx + 1].id
        const srcStatus = nodes[node.id]?.status || 'pending'
        const isActive = currentStep === node.id || currentStep === nextId
        return (
          <ConnectionLine
            key={`edge-${node.id}`}
            start={NODE_POSITIONS[idx]}
            end={NODE_POSITIONS[idx + 1]}
            status={srcStatus}
            isActive={isActive}
          />
        )
      })}

      {NODE_DEFINITIONS.map((node, idx) => {
        const status = nodes[node.id]?.status || 'pending'
        const isActive = currentStep === node.id
        return (
          <GlowingNode
            key={node.id}
            position={NODE_POSITIONS[idx]}
            status={status}
            label={node.label}
            description={node.description}
            isActive={isActive}
          />
        )
      })}
    </>
  )
}

function LoadingFallback() {
  return (
    <div className="graph-loading">
      <FiActivity size={32} className="spinning" />
      <p>Loading 3D visualization…</p>
    </div>
  )
}

const FlowGraph = ({ steps, currentStep }) => {
  const [hovered, setHovered] = useState(false)
  const stats = steps?.stats || { completed: 0, total: 7 }

  return (
    <div className="flow-graph-container">
      <div className="graph-header">
        <h2>Processing Pipeline</h2>
        <div className="graph-stats">
          <span className="stat-item">
            <FiClock size={14} />
            <span>{stats.completed} / {stats.total} steps</span>
          </span>
          <span className="stat-item">
            <FiActivity size={14} className={hovered ? 'spinning' : ''} />
            <span className="current-step-badge">{currentStep || 'idle'}</span>
          </span>
        </div>
      </div>

      <div
        className="graph-canvas"
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        <Canvas camera={{ position: [0, 0, 16], fov: 60 }}>
          <color attach="background" args={['#0a0a1a']} />
          <Stars
            radius={80}
            depth={50}
            count={4000}
            factor={4}
            saturation={0.5}
            fade
            speed={1.5}
          />
          <Sparkles
            count={60}
            scale={[30, 15, 25]}
            size={2.5}
            speed={0.4}
            opacity={0.6}
            color="#00c8ff"
          />
          <Suspense fallback={<Html center><LoadingFallback /></Html>}>
            <NodeNetwork steps={steps} currentStep={currentStep} />
            <OrbitControls
              enablePan
              enableZoom
              enableRotate
              zoomSpeed={0.6}
              rotateSpeed={0.5}
              minDistance={8}
              maxDistance={30}
            />
          </Suspense>
        </Canvas>
      </div>

      <div className="graph-legend">
        <div className="legend-item"><span className="legend-dot completed"></span><span>Completed</span></div>
        <div className="legend-item"><span className="legend-dot active"></span><span>Active</span></div>
        <div className="legend-item"><span className="legend-dot pending"></span><span>Pending</span></div>
      </div>
    </div>
  )
}

export default FlowGraph
