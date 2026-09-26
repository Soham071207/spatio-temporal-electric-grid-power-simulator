import React, { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import {
  AlertTriangle, Zap, Activity, Shield, Brain,
  Play, RotateCcw, ChevronDown, Loader2
} from 'lucide-react'

const API = 'http://127.0.0.1:8000'

const SCENARIO_TYPES = [
  { id: 'single_gen_failure', label: 'Generator Trip', icon: Zap, desc: 'One plant goes offline' },
  { id: 'n_of_k_failure', label: 'Multi-Failure', icon: AlertTriangle, desc: 'Multiple plants fail simultaneously' },
  { id: 'renewable_collapse', label: 'Renewable Collapse', icon: Activity, desc: 'Wind or Solar output drops 30-80%' },
  { id: 'demand_spike', label: 'Demand Surge', icon: Zap, desc: 'Regional demand spikes 10-40%' },
  { id: 'storm_regional', label: 'Regional Storm', icon: AlertTriangle, desc: 'All plants in a region fail' },
  { id: 'compound', label: 'Compound Event', icon: AlertTriangle, desc: 'Multiple failures at once' },
  { id: 'heatwave', label: 'Heatwave', icon: Activity, desc: 'Demand +15%, thermal derate 10%' },
]

function ResilienceGauge({ score, label, color }) {
  const circumference = 2 * Math.PI * 45
  const strokeDashoffset = circumference - (score / 100) * circumference

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative w-24 h-24">
        <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
          <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="6" />
          <motion.circle
            cx="50" cy="50" r="45" fill="none"
            stroke={color}
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xl font-bold text-white">{Math.round(score)}</span>
        </div>
      </div>
      <span className="text-[10px] uppercase tracking-wider text-white/40">{label}</span>
    </div>
  )
}

function ActionCard({ action, index }) {
  const isImport = action.type === 'emergency_import'
  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1 }}
      className="flex items-center gap-3 px-3 py-2 rounded-lg bg-white/[0.03] border border-white/[0.06]"
    >
      <div className={`w-2 h-2 rounded-full ${isImport ? 'bg-amber-400' : 'bg-emerald-400'}`} />
      <div className="flex-1">
        <div className="text-xs font-medium text-white/80">
          {isImport ? 'Emergency Import' : `Ramp ${action.technology}`}
        </div>
        <div className="text-[10px] text-white/40">
          {action.recommended_mw?.toFixed(0) || action.ramped_mw?.toFixed(0) || '?'} MW
        </div>
      </div>
    </motion.div>
  )
}

export default function ChaosDashboard() {
  const [selectedScenario, setSelectedScenario] = useState('single_gen_failure')
  const [datetime, setDatetime] = useState('2025-06-15 14:00:00')
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [showDropdown, setShowDropdown] = useState(false)
  const [mode, setMode] = useState('compare') // 'scenario' | 'compare' | 'batch'

  const runScenario = useCallback(async () => {
    setIsRunning(true)
    setResult(null)
    try {
      const endpoint = mode === 'compare' ? '/api/chaos/v2/heal' : '/api/chaos/v2/run-scenario'
      const res = await fetch(`${API}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          datetime,
          scenario_type: selectedScenario,
          params: {},
        }),
      })
      const data = await res.json()
      setResult(data)
    } catch (err) {
      console.error('Chaos scenario failed:', err)
      setResult({ status: 'error', message: err.message })
    } finally {
      setIsRunning(false)
    }
  }, [selectedScenario, datetime, mode])

  const selectedInfo = SCENARIO_TYPES.find(s => s.id === selectedScenario) || SCENARIO_TYPES[0]

  return (
    <div className="w-full flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center justify-center">
          <AlertTriangle className="w-4 h-4 text-red-400" />
        </div>
        <div>
          <h2 className="text-sm font-semibold text-white tracking-tight">Chaos Engine v2</h2>
          <p className="text-[10px] text-white/40">ML-powered self-healing grid resilience</p>
        </div>
      </div>

      {/* Controls */}
      <div className="space-y-3">
        {/* Mode toggle */}
        <div className="flex gap-1 p-0.5 rounded-lg bg-white/[0.03] border border-white/[0.06]">
          {[
            { id: 'compare', label: 'AI vs Rules', icon: Brain },
            { id: 'scenario', label: 'Scenario', icon: Zap },
          ].map(m => (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-[11px] font-medium transition-all ${
                mode === m.id
                  ? 'bg-white/10 text-white shadow-sm'
                  : 'text-white/40 hover:text-white/60'
              }`}
            >
              <m.icon className="w-3 h-3" />
              {m.label}
            </button>
          ))}
        </div>

        {/* Scenario selector */}
        <div className="relative">
          <button
            onClick={() => setShowDropdown(!showDropdown)}
            className="w-full flex items-center justify-between px-3 py-2.5 rounded-lg bg-white/[0.03] border border-white/[0.06] hover:border-white/[0.12] transition-all"
          >
            <div className="flex items-center gap-2">
              <selectedInfo.icon className="w-3.5 h-3.5 text-red-400" />
              <span className="text-xs font-medium text-white/80">{selectedInfo.label}</span>
            </div>
            <ChevronDown className={`w-3.5 h-3.5 text-white/30 transition-transform ${showDropdown ? 'rotate-180' : ''}`} />
          </button>

          <AnimatePresence>
            {showDropdown && (
              <motion.div
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                className="absolute top-full left-0 right-0 mt-1 z-50 rounded-lg bg-[#0d0d0d] border border-white/[0.08] shadow-xl overflow-hidden"
              >
                {SCENARIO_TYPES.map(s => (
                  <button
                    key={s.id}
                    onClick={() => { setSelectedScenario(s.id); setShowDropdown(false) }}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 text-left hover:bg-white/[0.04] transition-colors ${
                      selectedScenario === s.id ? 'bg-white/[0.06]' : ''
                    }`}
                  >
                    <s.icon className="w-3.5 h-3.5 text-red-400/60" />
                    <div>
                      <div className="text-xs font-medium text-white/80">{s.label}</div>
                      <div className="text-[10px] text-white/30">{s.desc}</div>
                    </div>
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Datetime input */}
        <input
          type="text"
          value={datetime}
          onChange={e => setDatetime(e.target.value)}
          placeholder="2025-06-15 14:00:00"
          className="w-full px-3 py-2 rounded-lg bg-white/[0.03] border border-white/[0.06] text-xs text-white/70 placeholder-white/20 focus:outline-none focus:border-white/[0.15]"
        />

        {/* Run button */}
        <button
          onClick={runScenario}
          disabled={isRunning}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-red-500/15 hover:bg-red-500/25 border border-red-500/20 text-red-300 text-xs font-semibold transition-all disabled:opacity-40 active:scale-[0.98]"
        >
          {isRunning ? (
            <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Running Chaos...</>
          ) : (
            <><Play className="w-3.5 h-3.5" /> Inject Fault</>
          )}
        </button>
      </div>

      {/* Results */}
      <AnimatePresence mode="wait">
        {result && result.status === 'success' && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-4"
          >
            {/* Fault description */}
            <div className="px-3 py-2 rounded-lg bg-red-500/[0.06] border border-red-500/[0.12]">
              <div className="text-[10px] uppercase tracking-wider text-red-400/60 mb-1">Fault Injected</div>
              <div className="text-xs text-white/70 leading-relaxed">
                {result.fault_description || result.description || 'Scenario executed'}
              </div>
            </div>

            {/* Comparison mode: AI vs Rules */}
            {mode === 'compare' && result.comparison && (
              <>
                {/* Gauges */}
                <div className="flex justify-around py-2">
                  <ResilienceGauge
                    score={result.comparison.rule_based.resilience_score}
                    label="Rule-Based"
                    color="#f97316"
                  />
                  <ResilienceGauge
                    score={result.comparison.ai_healing.resilience_score}
                    label="AI Healing"
                    color="#22d3ee"
                  />
                </div>

                {/* Stats comparison */}
                <div className="grid grid-cols-2 gap-2">
                  {/* Rule-based */}
                  <div className="rounded-lg bg-orange-500/[0.04] border border-orange-500/[0.1] p-3">
                    <div className="text-[10px] uppercase tracking-wider text-orange-400/60 mb-2 flex items-center gap-1">
                      <Shield className="w-3 h-3" /> Rule-Based
                    </div>
                    <div className="space-y-1">
                      <div className="text-[10px] text-white/40">
                        Deficit: <span className="text-white/70">{result.comparison.rule_based.deficit_after_mw?.toFixed(0)} MW</span>
                      </div>
                      <div className="text-[10px] text-white/40">
                        Cost: <span className="text-white/70">EUR {result.comparison.rule_based.cost_eur?.toFixed(0)}</span>
                      </div>
                    </div>
                  </div>

                  {/* AI */}
                  <div className="rounded-lg bg-cyan-500/[0.04] border border-cyan-500/[0.1] p-3">
                    <div className="text-[10px] uppercase tracking-wider text-cyan-400/60 mb-2 flex items-center gap-1">
                      <Brain className="w-3 h-3" /> AI Healing
                    </div>
                    <div className="space-y-1">
                      <div className="text-[10px] text-white/40">
                        Deficit: <span className="text-white/70">{result.comparison.ai_healing.deficit_after_mw?.toFixed(0)} MW</span>
                      </div>
                      <div className="text-[10px] text-white/40">
                        Cost: <span className="text-white/70">EUR {result.comparison.ai_healing.predicted_cost_eur?.toFixed(0)}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* AI Actions */}
                {result.comparison.ai_healing.actions?.length > 0 && (
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-cyan-400/40 mb-2">AI Healing Actions</div>
                    <div className="space-y-1">
                      {result.comparison.ai_healing.actions.map((a, i) => (
                        <ActionCard key={i} action={a} index={i} />
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}

            {/* Simple scenario mode */}
            {mode === 'scenario' && (
              <div className="space-y-3">
                <div className="flex justify-around py-2">
                  <ResilienceGauge
                    score={result.resilience_after_fault || 0}
                    label="After Fault"
                    color="#ef4444"
                  />
                  <ResilienceGauge
                    score={result.resilience_after_healing || 0}
                    label="After Healing"
                    color="#22c55e"
                  />
                </div>

                <div className="grid grid-cols-2 gap-2 text-[10px]">
                  <div className="px-3 py-2 rounded-lg bg-white/[0.02] border border-white/[0.06]">
                    <span className="text-white/40">Gen Lost</span>
                    <div className="text-white/80 font-medium">{result.generation_lost_mw} MW</div>
                  </div>
                  <div className="px-3 py-2 rounded-lg bg-white/[0.02] border border-white/[0.06]">
                    <span className="text-white/40">Cascade</span>
                    <div className="text-white/80 font-medium">Depth {result.cascade_depth}, Size {result.cascade_size}</div>
                  </div>
                  <div className="px-3 py-2 rounded-lg bg-white/[0.02] border border-white/[0.06]">
                    <span className="text-white/40">Freq Dev</span>
                    <div className="text-white/80 font-medium">{result.frequency_deviation_hz} Hz</div>
                  </div>
                  <div className="px-3 py-2 rounded-lg bg-white/[0.02] border border-white/[0.06]">
                    <span className="text-white/40">Cost</span>
                    <div className="text-white/80 font-medium">EUR {result.healing_cost_eur?.toFixed(0)}</div>
                  </div>
                </div>
              </div>
            )}

            {/* Reset */}
            <button
              onClick={() => setResult(null)}
              className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-white/[0.03] border border-white/[0.06] text-white/40 text-[11px] hover:text-white/60 hover:bg-white/[0.06] transition-all"
            >
              <RotateCcw className="w-3 h-3" /> Reset
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error state */}
      {result && result.status === 'error' && (
        <div className="px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-300">
          Error: {result.message || result.detail || 'Unknown error'}
        </div>
      )}
    </div>
  )
}
