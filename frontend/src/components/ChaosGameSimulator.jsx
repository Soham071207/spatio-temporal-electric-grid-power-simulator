import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import GridSimulator3D from './GridSimulator3D'
import DisasterControl from './DisasterControl'
import { X, Play, Pause, SkipForward, SkipBack, Zap, AlertTriangle, Shield, ArrowRight, Battery, Maximize, Minimize } from 'lucide-react'


const API = 'http://127.0.0.1:8000'

// ─── Narrative Engine ──────────────────────────────────────────────────────────
// Translates raw simulation JSON into timestamped, human-readable log entries.
function buildNarrative(simulationState) {
  if (!simulationState) return []

  const entries = []
  const baseTime = new Date('2025-06-15T14:00:00')
  let sec = 0

  const ts = () => {
    const t = new Date(baseTime.getTime() + sec * 1000)
    return t.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }

  // 1. Initial fault
  entries.push({
    type: 'fault',
    time: ts(),
    icon: '⚡',
    title: 'FAULT INJECTED',
    body: simulationState.fault_description || 'Unknown fault',
    detail: `Initial deficit: ${simulationState.post_fault_state?.deficit_mw?.toFixed(0) || '?'} MW`,
  })
  sec += 2

  // 2. Cascade steps
  const cascadeSteps = simulationState.cascade_history || []
  cascadeSteps.forEach((step, i) => {
    sec += 1
    const parts = []
    if (step.tripped_lines?.length > 0) {
      parts.push(`${step.tripped_lines.length} transmission line${step.tripped_lines.length > 1 ? 's' : ''} overloaded & tripped`)
    }
    if (step.tripped_generators?.length > 0) {
      parts.push(`${step.tripped_generators.length} generator${step.tripped_generators.length > 1 ? 's' : ''} tripped (under-frequency)`)
    }
    if (parts.length === 0) parts.push('System propagation check — no new failures')

    entries.push({
      type: 'cascade',
      time: ts(),
      icon: '🔻',
      title: `CASCADE — Depth ${step.depth}`,
      body: parts.join(' · '),
      detail: `Deficit: ${step.deficit_mw?.toFixed(0) || 0} MW  |  Freq: ${step.frequency_hz?.toFixed(2) || '50.00'} Hz  |  Regions: ${step.affected_regions?.length || 0}`,
    })
  })

  // 3. Rule-based healing
  const ruleActions = simulationState.comparison?.rule_based?.actions || []
  if (ruleActions.length > 0) {
    sec += 2
    entries.push({
      type: 'healing-header',
      time: ts(),
      icon: '🔧',
      title: 'RULE-BASED HEALING',
      body: `${ruleActions.length} redispatch action${ruleActions.length > 1 ? 's' : ''} executed`,
      detail: `Cost: €${simulationState.comparison?.rule_based?.cost_eur?.toLocaleString() || 0} | Remaining deficit: ${simulationState.comparison?.rule_based?.deficit_after_mw?.toFixed(0) || 0} MW`,
    })

    ruleActions.forEach(action => {
      if (action.type === 'ramp_up') {
        const id = action.generator_id || 'Unknown Generator'
        const rampMw = action.ramped_mw ?? '?'
        const newMw = action.new_output_mw ?? '?'
        const region = action.region || 'Unknown Region'
        const cost = action.cost_eur?.toLocaleString() || 0

        entries.push({
          type: 'healing',
          time: '',
          icon: '⚙️',
          title: `${action.technology || 'Resource'} ramp-up`,
          body: `"${id}" ramped +${rampMw} MW → now ${newMw} MW`,
          detail: `Region: ${region} | Cost: €${cost}`,
        })
      } else if (action.type === 'emergency_import') {
        const importedMw = action.imported_mw ?? '?'
        const sources = Object.entries(action.sources || {}).map(([k, v]) => `${k}: ${v} MW`).join(', ')
        entries.push({
          type: 'import',
          time: '',
          icon: '🔌',
          title: 'Emergency Import',
          body: `Pulled ${importedMw} MW from interconnections`,
          detail: sources || 'Unknown sources',
        })
      }
    })
  }

  // 4. AI healing
  const aiActions = simulationState.comparison?.ai_healing?.actions || []
  if (aiActions.length > 0) {
    sec += 2
    entries.push({
      type: 'ai-header',
      time: ts(),
      icon: '🧠',
      title: 'AI HEALING AGENT',
      body: `${aiActions.length} AI-optimized action${aiActions.length > 1 ? 's' : ''} proposed`,
      detail: `Predicted cost: €${simulationState.comparison?.ai_healing?.predicted_cost_eur?.toLocaleString() || 0} | Remaining deficit: ${simulationState.comparison?.ai_healing?.deficit_after_mw?.toFixed(0) || 0} MW`,
    })

    aiActions.forEach(action => {
      if (action.type === 'ramp_up') {
        const id = action.generator_id || 'Unknown Generator'
        const rampMw = action.ramped_mw ?? action.recommended_mw ?? '?'
        const newMw = action.new_output_mw ?? '?'
        const region = action.region || 'Unknown Region'
        const cost = action.cost_eur?.toLocaleString() || 0
        // Fix #6: show if generator was ramp-rate capped
        const maxRamp = { Hydro: 750, Gas: 300, Coal: 120, Nuclear: 75, BESS: 999 }
        const tech = action.technology || ''
        const wasCapped = maxRamp[tech] && parseFloat(rampMw) >= maxRamp[tech] * 0.95

        entries.push({
          type: 'ai-action',
          time: '',
          icon: '⚙️',
          title: `${tech || 'Resource'} ramp-up${wasCapped ? ' ⚠ RAMP-CAPPED' : ''}`,
          body: `"${id}" ramped +${rampMw} MW → now ${newMw} MW`,
          detail: `Region: ${region} | Cost: €${cost}${wasCapped ? ' | At physical ramp limit' : ''}`,
        })
      } else if (action.type === 'emergency_import') {
        const importedMw = action.imported_mw ?? action.recommended_mw ?? '?'
        const sources = Object.entries(action.sources || {}).map(([k, v]) => `${k}: ${v} MW`).join(', ')
        entries.push({
          type: 'import',
          time: '',
          icon: '🔌',
          title: 'Emergency Import',
          body: `Pulled ${importedMw} MW from interconnections`,
          detail: sources || 'Unknown sources',
        })
      } else if (action.type === 'power_transfer') {
        // Fix: show distance and efficiency loss from physics upgrade
        const dist = action.distance_km ? `${action.distance_km} km` : null
        const eff = action.efficiency ? `${(action.efficiency * 100).toFixed(1)}% efficiency` : null
        const sent = action.sent_mw ?? action.transferred_mw
        const arrived = action.transferred_mw
        const lostMw = sent && arrived ? (sent - arrived).toFixed(1) : null

        entries.push({
          type: 'transfer',
          time: '',
          icon: '➡️',
          title: 'Regional Power Transfer',
          body: `${action.from_region} → ${action.to_region}: ${arrived} MW delivered`,
          detail: [
            dist && eff ? `${dist} · ${eff}` : 'Local transfer',
            lostMw > 0 ? `${lostMw} MW lost in transmission` : null,
          ].filter(Boolean).join(' · ') || 'Instantaneous surplus routed via transmission grid',
        })
      } else if (action.type === 'buffer_discharge') {
        const sent = action.sent_mw ?? action.discharged_mw
        const arrived = action.discharged_mw
        const lostMw = sent && arrived ? (sent - arrived).toFixed(1) : null
        const dist = action.distance_km ? `${action.distance_km} km` : null
        const eff = action.transmission_efficiency ? `${(action.transmission_efficiency * 100).toFixed(1)}%` : null
        entries.push({
          type: 'buffer',
          time: '',
          icon: '🔋',
          title: 'Buffer Discharge',
          body: `${action.from_region} → ${action.to_region}: ${arrived} MW delivered`,
          detail: [
            `Cost: €${action.cost_eur || 0}`,
            dist && eff ? `${dist} · ${eff} efficiency` : null,
            lostMw > 0 ? `${lostMw} MW lost in line` : null,
          ].filter(Boolean).join(' · '),
        })
      }
    })
  }


  // 5. Final status
  sec += 2
  const aiScore = simulationState.comparison?.ai_healing?.resilience_score
  const ruleScore = simulationState.comparison?.rule_based?.resilience_score
  entries.push({
    type: 'stable',
    time: ts(),
    icon: '✅',
    title: 'SIMULATION COMPLETE',
    body: `AI Resilience: ${aiScore?.toFixed(1) || '—'} / 100  |  Rule-based: ${ruleScore?.toFixed(1) || '—'} / 100`,
    detail: aiScore > ruleScore
      ? `AI healing outperformed rule-based by ${(aiScore - ruleScore).toFixed(1)} points`
      : ruleScore > aiScore
        ? `Rule-based outperformed AI by ${(ruleScore - aiScore).toFixed(1)} points`
        : 'Both approaches performed equally',
  })

  return entries
}

// ─── Narrative Log Entry UI ────────────────────────────────────────────────────
const TYPE_STYLES = {
  fault:            { border: 'border-red-500/40',     bg: 'bg-red-500/10',      text: 'text-red-400' },
  cascade:          { border: 'border-amber-500/30',   bg: 'bg-amber-500/8',     text: 'text-amber-400' },
  'healing-header': { border: 'border-cyan-500/30',    bg: 'bg-cyan-500/8',      text: 'text-cyan-400' },
  healing:          { border: 'border-emerald-500/20', bg: 'bg-emerald-500/5',   text: 'text-emerald-400' },
  import:           { border: 'border-blue-500/30',    bg: 'bg-blue-500/8',      text: 'text-blue-400' },
  'ai-header':      { border: 'border-purple-500/30',  bg: 'bg-purple-500/8',    text: 'text-purple-400' },
  'ai-action':      { border: 'border-purple-500/20',  bg: 'bg-purple-500/5',    text: 'text-purple-300' },
  transfer:         { border: 'border-sky-500/30',     bg: 'bg-sky-500/8',       text: 'text-sky-400' },
  buffer:           { border: 'border-amber-400/40',   bg: 'bg-amber-400/8',     text: 'text-amber-300' },
  stable:           { border: 'border-emerald-500/40', bg: 'bg-emerald-500/10',  text: 'text-emerald-400' },
}


function LogEntry({ entry, isVisible }) {
  const s = TYPE_STYLES[entry.type] || TYPE_STYLES.cascade
  return (
    <motion.div
      initial={{ opacity: 0, x: 12, height: 0 }}
      animate={isVisible ? { opacity: 1, x: 0, height: 'auto' } : { opacity: 0, x: 12, height: 0 }}
      transition={{ duration: 0.35, ease: [0.32, 0.72, 0, 1] }}
      className="overflow-hidden"
    >
      <div className={`p-2.5 rounded-lg border ${s.border} ${s.bg} mb-1.5`}>
        <div className="flex items-start gap-2">
          {entry.time && (
            <span className="text-[10px] text-white/30 font-mono mt-0.5 shrink-0 w-16">{entry.time}</span>
          )}
          {!entry.time && <span className="w-16 shrink-0" />}
          <div className="flex-1 min-w-0">
            <div className={`text-[11px] font-bold uppercase tracking-wider ${s.text} flex items-center gap-1.5`}>
              <span>{entry.icon}</span> {entry.title}
            </div>
            <div className="text-xs text-white/80 mt-0.5 leading-relaxed">{entry.body}</div>
            {entry.detail && (
              <div className="text-[10px] text-white/35 mt-1 font-mono">{entry.detail}</div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  )
}

// ─── Frequency Gauge ───────────────────────────────────────────────────────────
function FrequencyGauge({ frequency }) {
  const freq = frequency ?? 50.0
  const minHz = 47.0, maxHz = 50.5
  const clamped = Math.max(minHz, Math.min(maxHz, freq))
  const pct = (clamped - minHz) / (maxHz - minHz)
  const angle = -135 + pct * 270

  let color = '#22c55e'
  let label = 'NORMAL'
  if (freq < 49.0) { color = '#ef4444'; label = 'CRITICAL' }
  else if (freq < 49.5) { color = '#f97316'; label = 'WARNING' }
  else if (freq < 49.8) { color = '#eab308'; label = 'ALERT' }

  const r = 44
  const cx = 56, cy = 56

  const radAngle = (angle * Math.PI) / 180
  const nx = cx + r * 0.72 * Math.cos(radAngle)
  const ny = cy + r * 0.72 * Math.sin(radAngle)

  const arcPath = (startAngle, endAngle, radius) => {
    const s = (startAngle * Math.PI) / 180
    const e = (endAngle * Math.PI) / 180
    const x1 = cx + radius * Math.cos(s)
    const y1 = cy + radius * Math.sin(s)
    const x2 = cx + radius * Math.cos(e)
    const y2 = cy + radius * Math.sin(e)
    const large = endAngle - startAngle > 180 ? 1 : 0
    return `M ${x1} ${y1} A ${radius} ${radius} 0 ${large} 1 ${x2} ${y2}`
  }

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 112 90" className="w-[140px] h-[112px]">
        <path d={arcPath(-135, 135, r)} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="6" strokeLinecap="round" />
        <path d={arcPath(-135, -135 + (2/3.5)*270, r)} fill="none" stroke="rgba(239,68,68,0.15)" strokeWidth="6" strokeLinecap="round" />
        <path d={arcPath(-135, angle, r)} fill="none" stroke={color} strokeWidth="6" strokeLinecap="round" style={{ filter: `drop-shadow(0 0 4px ${color}40)` }} />
        <line x1={cx} y1={cy} x2={nx} y2={ny} stroke={color} strokeWidth="2.5" strokeLinecap="round" style={{ filter: `drop-shadow(0 0 3px ${color})`, transition: 'all 0.8s cubic-bezier(0.32, 0.72, 0, 1)' }} />
        <circle cx={cx} cy={cy} r="3" fill={color} />
        <text x={cx} y={cy + 18} textAnchor="middle" fill="white" fontSize="14" fontWeight="700" fontFamily="monospace">
          {freq.toFixed(2)}
        </text>
        <text x={cx} y={cy + 28} textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="7" fontWeight="500">
          Hz
        </text>
      </svg>
      <div className="text-[9px] font-bold uppercase tracking-[0.2em] mt-0.5" style={{ color }}>
        {label}
      </div>
    </div>
  )
}

// ─── Timeline Scrubber ─────────────────────────────────────────────────────────
function TimelineScrubber({ totalSteps, currentStep, onStepChange, isPlaying, onTogglePlay, playSpeed, onSpeedChange, isHealing }) {
  const totalNodes = totalSteps + 1
  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-2 px-1">
        {Array.from({ length: totalNodes }, (_, i) => {
          const isLast = i === totalNodes - 1
          const isCurrent = isLast ? isHealing : i === currentStep
          const isPast = isLast ? false : i < currentStep

          return (
            <React.Fragment key={i}>
              <button
                onClick={() => {
                  if (isLast) {
                    onStepChange(totalSteps - 1)
                  } else {
                    onStepChange(i)
                  }
                }}
                className={`relative w-5 h-5 rounded-full border-2 transition-all duration-300 flex items-center justify-center
                  ${isCurrent
                    ? isLast ? 'border-emerald-400 bg-emerald-400/30 shadow-lg shadow-emerald-500/30' : 'border-red-400 bg-red-400/30 shadow-lg shadow-red-500/30'
                    : isPast ? 'border-white/30 bg-white/15' : 'border-white/10 bg-transparent'
                  }`}
              >
                {isCurrent && (
                  <motion.div
                    animate={{ scale: [1, 1.6, 1] }}
                    transition={{ repeat: Infinity, duration: 1.5 }}
                    className={`absolute w-full h-full rounded-full ${isLast ? 'bg-emerald-400/20' : 'bg-red-400/20'}`}
                  />
                )}
                <span className={`text-[7px] font-bold ${isCurrent ? 'text-white' : isPast ? 'text-white/50' : 'text-white/20'}`}>
                  {isLast ? '✓' : i + 1}
                </span>
              </button>
              {i < totalNodes - 1 && (
                <div className={`flex-1 h-0.5 mx-0.5 transition-colors duration-500 ${isPast ? 'bg-white/20' : 'bg-white/5'}`} />
              )}
            </React.Fragment>
          )
        })}
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          <button onClick={() => onStepChange(Math.max(0, currentStep - 1))} className="w-7 h-7 flex items-center justify-center rounded-lg bg-white/5 hover:bg-white/10 text-white/40 hover:text-white transition-all">
            <SkipBack className="w-3 h-3" />
          </button>
          <button onClick={onTogglePlay} className="w-8 h-8 flex items-center justify-center rounded-lg bg-white/10 hover:bg-white/15 text-white/60 hover:text-white transition-all">
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          </button>
          <button onClick={() => onStepChange(Math.min(totalSteps - 1, currentStep + 1))} className="w-7 h-7 flex items-center justify-center rounded-lg bg-white/5 hover:bg-white/10 text-white/40 hover:text-white transition-all">
            <SkipForward className="w-3 h-3" />
          </button>
        </div>
        <div className="flex items-center gap-1.5">
          {[0.5, 1, 2].map(s => (
            <button
              key={s}
              onClick={() => onSpeedChange(s)}
              className={`px-2 py-0.5 text-[10px] font-medium rounded transition-all
                ${playSpeed === s ? 'bg-white/15 text-white' : 'text-white/30 hover:text-white/60'}`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

// ─── Main Component ────────────────────────────────────────────────────────────
export default function ChaosGameSimulator({ onClose }) {
  const [selectedNode, setSelectedNode] = useState(null)
  const [simulationState, setSimulationState] = useState(null)
  const [isSimulating, setIsSimulating] = useState(false)
  const [simError, setSimError] = useState(null)
  const [currentStepIndex, setCurrentStepIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(true)
  const [playSpeed, setPlaySpeed] = useState(1)
  const [visibleLogCount, setVisibleLogCount] = useState(0)
  const [simDate, setSimDate] = useState('2025-06-15')
  const [simHour, setSimHour] = useState('14')
  const [isUIVisible, setIsUIVisible] = useState(true)
  const logContainerRef = useRef(null)

  // Derived datetime string for the API
  const simDatetime = `${simDate} ${simHour.padStart(2, '0')}:00:00`

  // Build narrative from simulation state
  const narrative = useMemo(() => buildNarrative(simulationState), [simulationState])

  const cascadeSteps = simulationState?.cascade_history || []
  const totalSteps = cascadeSteps.length

  // Is cascade finished? (at the last step)
  const isFinished = simulationState && currentStepIndex >= totalSteps - 1
  const healingActions = isFinished ? (simulationState?.comparison?.ai_healing?.actions || []) : []

  // Current frequency from the active cascade step
  const currentFrequency = useMemo(() => {
    if (!simulationState) return 50.0
    if (isFinished) {
      const postState = simulationState.comparison?.ai_healing?.post_healing_state
      if (postState?.frequency_hz) return postState.frequency_hz
      return 49.8
    }
    const step = cascadeSteps[currentStepIndex]
    return step?.frequency_hz ?? 50.0
  }, [simulationState, currentStepIndex, isFinished, cascadeSteps])

  // Extract specific generator IDs for 3D
  const trippedGeneratorIds = useMemo(() => {
    if (!simulationState) return []
    const ids = new Set()
    for (let i = 0; i <= currentStepIndex && i < cascadeSteps.length; i++) {
      (cascadeSteps[i].tripped_generators || []).forEach(id => ids.add(id))
    }
    return [...ids]
  }, [simulationState, currentStepIndex, cascadeSteps])

  // Bug 5 Fix: pass ALL ramp_up actions (not just top 1)
  const allHealingActions = useMemo(() => {
    if (!isFinished || !healingActions || healingActions.length === 0) return []
    // Include ramp_up, power_transfer, and buffer_discharge in the 3D map
    return healingActions.filter(a =>
      a.type === 'ramp_up' ||
      a.type === 'power_transfer' ||
      a.type === 'buffer_discharge'
    )
  }, [healingActions, isFinished])

  const rampingGeneratorIds = useMemo(() => {
    return allHealingActions.filter(a => a.type === 'ramp_up' && a.generator_id).map(a => a.generator_id)
  }, [allHealingActions])


  // Auto-play cascade steps
  useEffect(() => {
    if (!isPlaying || !simulationState || totalSteps === 0) return
    if (currentStepIndex >= totalSteps - 1) {
      setIsPlaying(false)
      return
    }
    const delay = 1500 / playSpeed
    const timer = setTimeout(() => setCurrentStepIndex(i => i + 1), delay)
    return () => clearTimeout(timer)
  }, [currentStepIndex, simulationState, isPlaying, playSpeed, totalSteps])

  // Reveal narrative entries progressively
  useEffect(() => {
    if (!simulationState || narrative.length === 0) return
    if (visibleLogCount >= narrative.length) return
    const delay = 400 / playSpeed
    const timer = setTimeout(() => setVisibleLogCount(c => c + 1), delay)
    return () => clearTimeout(timer)
  }, [visibleLogCount, narrative.length, simulationState, playSpeed])

  // Auto-scroll log to bottom
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [visibleLogCount])

  const handleNodeClick = (node) => {
    if (!isSimulating) {
      setSelectedNode(node)
    }
  }

  const handleTriggerDisaster = async (scenarioType, node) => {
    setIsSimulating(true)
    setSimulationState(null)
    setSimError(null)
    setCurrentStepIndex(0)
    setVisibleLogCount(0)
    setIsPlaying(true)

    try {
      const res = await fetch(`${API}/api/chaos/v2/heal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          datetime: simDatetime,
          scenario_type: scenarioType,
          params: { region: node.region, generator_id: node.id },
        }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || `HTTP ${res.status}`)
      }
      const data = await res.json()
      setSimulationState(data)
    } catch (err) {
      console.error('Simulation failed:', err)
      setSimError(err.message || 'Simulation failed — is the backend running?')
    } finally {
      setIsSimulating(false)
      setSelectedNode(null)
    }
  }

  const currentCascadeStep = cascadeSteps[currentStepIndex]

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 1.05 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="fixed inset-0 z-50 bg-black overflow-hidden flex"
    >
      {/* 3D Map Background */}
      <div className="absolute inset-0 z-0">
        <GridSimulator3D
          cascadeStep={currentCascadeStep}
          healingActions={allHealingActions}
          trippedGeneratorIds={trippedGeneratorIds}
          rampingGeneratorIds={rampingGeneratorIds}
          onNodeClick={handleNodeClick}
        />
      </div>


      {/* Top Bar */}
      <div className="absolute top-0 left-0 right-0 p-6 z-10 flex justify-between items-start pointer-events-none">
        <div className="pointer-events-auto">
          <h1 className="text-3xl font-medium tracking-tighter text-white drop-shadow-xl">
            Chaos Engine <span className="text-red-500 font-bold">SIMULATOR</span>
          </h1>
          <p className="text-white/50 text-sm mt-1">Click a power plant node on the 3D grid to inject a fault.</p>

          {/* Simulation date/time picker */}
          <div className="flex items-center gap-2 mt-3">
            <span className="text-[10px] uppercase tracking-widest text-white/30 font-semibold">Sim Date</span>
            <input
              type="date"
              value={simDate}
              min="2019-01-01"
              max="2024-12-31"
              onChange={e => setSimDate(e.target.value)}
              className="bg-white/5 border border-white/10 rounded-lg px-2 py-1 text-xs text-white/70 focus:outline-none focus:border-white/30 transition-colors"
            />
            <select
              value={simHour}
              onChange={e => setSimHour(e.target.value)}
              className="bg-white/5 border border-white/10 rounded-lg px-2 py-1 text-xs text-white/70 focus:outline-none focus:border-white/30 transition-colors"
            >
              {Array.from({ length: 24 }, (_, i) => (
                <option key={i} value={i} className="bg-black">
                  {String(i).padStart(2, '0')}:00
                </option>
              ))}
            </select>
            <span className="text-[10px] text-white/20 italic">Grid state calibrated from real REE data</span>
          </div>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setIsUIVisible(!isUIVisible)}
            className="pointer-events-auto w-12 h-12 flex items-center justify-center rounded-2xl bg-white/5 border border-white/10 text-white/50 hover:text-white hover:bg-white/10 hover:border-white/20 transition-all backdrop-blur-md"
            title={isUIVisible ? "Hide UI Panels" : "Show UI Panels"}
          >
            {isUIVisible ? <Maximize className="w-5 h-5" /> : <Minimize className="w-5 h-5" />}
          </button>
          <button
            onClick={onClose}
            className="pointer-events-auto w-12 h-12 flex items-center justify-center rounded-2xl bg-white/5 border border-white/10 text-white/50 hover:text-white hover:bg-red-500/20 hover:border-red-500/50 transition-all backdrop-blur-md"
          >
            <X className="w-6 h-6" />
          </button>
        </div>
      </div>

      {/* Disaster Control Overlay */}
      <DisasterControl
        selectedNode={selectedNode}
        onClose={() => setSelectedNode(null)}
        onTriggerDisaster={handleTriggerDisaster}
        isSimulating={isSimulating}
      />

      {/* ── Left panel: Frequency Gauge + Buffer ─────────────────────── */}
      <AnimatePresence>
      {simulationState && isUIVisible && (
        <motion.div
          initial={{ opacity: 0, y: -10, x: -20 }}
          animate={{ opacity: 1, y: 0, x: 0 }}
          exit={{ opacity: 0, x: -20, transition: { duration: 0.2 } }}
          className="absolute top-36 left-6 z-10 pointer-events-none flex flex-col gap-3"
        >
          <div className="bg-black/60 backdrop-blur-xl border border-white/10 rounded-2xl p-3">
            <div className="text-[9px] uppercase tracking-[0.2em] text-white/30 text-center mb-1 font-bold">
              Grid Frequency
            </div>
            <FrequencyGauge frequency={currentFrequency} />
          </div>

          {/* Virtual Buffer panel with explanation */}
          {simulationState.buffer_summary && simulationState.buffer_summary.total_stored_mwh > 0 && (
            <div className="bg-black/60 backdrop-blur-xl border border-amber-500/20 rounded-2xl p-3">
              <div className="flex items-center gap-1.5 mb-1">
                <Battery className="w-3 h-3 text-amber-400" />
                <div className="text-[9px] uppercase tracking-[0.15em] text-amber-400/70 font-bold">
                  Virtual Buffer
                </div>
              </div>
              <div className="text-[9px] text-white/30 mb-2 leading-relaxed">
                Excess energy stored throughout the day from surplus regions,
                dispatched during faults — cheaper than any fossil fuel.
              </div>
              <div className="text-lg font-bold text-amber-300 tabular-nums">
                {simulationState.buffer_summary.total_stored_mwh?.toLocaleString() || '0'}
                <span className="text-xs font-normal text-amber-400/60 ml-1">MWh stored</span>
              </div>
              {simulationState.buffer_dispatched_mw > 0 && (
                <div className="mt-2">
                  <div className="text-[9px] text-amber-400/50 mb-1">Dispatched this event</div>
                  <div className="flex items-center gap-1.5">
                    <div className="flex-1 h-1 bg-white/5 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.min(100, (simulationState.buffer_dispatched_mw / (simulationState.buffer_summary.total_stored_mwh || 1)) * 100)}%` }}
                        transition={{ duration: 1, ease: 'easeOut' }}
                        className="h-full bg-amber-400/60 rounded-full"
                      />
                    </div>
                    <span className="text-[10px] text-amber-300 font-bold tabular-nums">
                      {simulationState.buffer_dispatched_mw} MW
                    </span>
                  </div>
                  <div className="text-[9px] text-amber-400/40 mt-1">
                    @€2/MWh — cheapest available response
                  </div>
                </div>
              )}
              {/* Top donor regions */}
              <div className="mt-2 space-y-0.5">
                {Object.entries(simulationState.buffer_summary.per_region || {}).slice(0, 3).map(([region, mwh]) => (
                  <div key={region} className="flex items-center justify-between text-[9px]">
                    <span className="text-white/40 truncate max-w-[80px]">{region}</span>
                    <span className="text-amber-400/70 tabular-nums">{mwh} MWh</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </motion.div>
      )}
      </AnimatePresence>


      {/* ── Right Panel: Narrative Log + Timeline ──────────────────────── */}
      <AnimatePresence>
      {isUIVisible && (
        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 20, transition: { duration: 0.2 } }}
          className="absolute right-6 top-24 bottom-6 w-[420px] z-10 pointer-events-none flex flex-col gap-3"
        >
        {/* Simulation Log Panel */}
        <div className="pointer-events-auto flex-1 min-h-0 flex flex-col bg-black/60 backdrop-blur-2xl border border-white/10 rounded-2xl shadow-2xl shadow-black/80 overflow-hidden">
          {/* Header */}
          <div className="p-4 pb-3 border-b border-white/8 shrink-0">
            <div className="flex items-center justify-between">
              <div className="text-xs uppercase tracking-widest text-white/40 font-bold flex items-center gap-2">
                <AlertTriangle className="w-3.5 h-3.5" />
                Simulation Log
              </div>
              {simulationState && (
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-1.5 bg-purple-500/10 border border-purple-500/20 rounded-lg px-2 py-1">
                    <span className="text-[9px] text-purple-300/70 uppercase tracking-wider">AI</span>
                    <span className="text-sm font-bold text-purple-400">
                      {simulationState.comparison?.ai_healing?.resilience_score?.toFixed(0) || '—'}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 bg-cyan-500/10 border border-cyan-500/20 rounded-lg px-2 py-1">
                    <span className="text-[9px] text-cyan-300/70 uppercase tracking-wider">Rule</span>
                    <span className="text-sm font-bold text-cyan-400">
                      {simulationState.comparison?.rule_based?.resilience_score?.toFixed(0) || '—'}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Log entries */}
          <div ref={logContainerRef} className="flex-1 overflow-y-auto p-3 custom-scrollbar">
            {isSimulating && (
              <div className="flex items-center gap-3 p-4 text-amber-400">
                <div className="w-5 h-5 rounded-full border-2 border-amber-400/30 border-t-amber-400 animate-spin shrink-0" />
                <div>
                  <div className="text-sm font-medium animate-pulse">Computing grid stability & AI response...</div>
                  <div className="text-[10px] text-amber-400/50 mt-1">
                    Snapshot: {simDatetime} · Running cascade + rule-based + AI healing
                  </div>
                </div>
              </div>
            )}

            {simError && !isSimulating && (
              <div className="m-3 p-3 rounded-xl border border-red-500/30 bg-red-500/10">
                <div className="text-xs font-bold text-red-400 mb-1">⚠ Simulation Error</div>
                <div className="text-[11px] text-red-300/70 leading-relaxed">{simError}</div>
                <div className="text-[10px] text-white/30 mt-2">Make sure the backend is running: <code className="text-white/50">uvicorn backend.main:app</code></div>
              </div>
            )}

            {narrative.slice(0, visibleLogCount).map((entry, i) => (
              <LogEntry key={i} entry={entry} isVisible={true} />
            ))}

            {!simulationState && !isSimulating && (
              <div className="flex flex-col items-center justify-center text-white/30 py-16 gap-3">
                <Zap className="w-8 h-8 opacity-40" />
                <div className="text-sm font-medium">Awaiting fault injection</div>
                <div className="text-[11px] text-white/20 max-w-[200px] text-center">Click a power plant on the 3D grid, then choose a disaster type</div>
              </div>
            )}
          </div>
        </div>

        {/* Timeline Scrubber */}
        {simulationState && totalSteps > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="pointer-events-auto bg-black/60 backdrop-blur-xl border border-white/10 rounded-2xl p-4 shrink-0"
          >
            <div className="text-[9px] uppercase tracking-[0.2em] text-white/30 mb-3 font-bold">
              Cascade Timeline — Step {currentStepIndex + 1} of {totalSteps}
            </div>
            <TimelineScrubber
              totalSteps={totalSteps}
              currentStep={currentStepIndex}
              onStepChange={setCurrentStepIndex}
              isPlaying={isPlaying}
              onTogglePlay={() => setIsPlaying(p => !p)}
              playSpeed={playSpeed}
              onSpeedChange={setPlaySpeed}
              isHealing={isFinished}
            />
          </motion.div>
        )}
      </motion.div>
      )}
      </AnimatePresence>
    </motion.div>
  )
}
