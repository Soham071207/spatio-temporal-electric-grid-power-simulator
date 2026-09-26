import React from 'react'
import { motion, AnimatePresence } from 'motion/react'
import { AlertTriangle, Wind, Zap, ThermometerSun, X, CloudLightning, TrendingUp, Layers } from 'lucide-react'

const DISASTERS = [
  {
    id: 'single_gen_failure',
    name: 'Generator Trip',
    icon: Zap,
    desc: 'Instantly trips this specific power plant offline.',
    detail: 'Most common real-world fault — relay failure, mechanical fault, or operator error.',
    color: 'text-red-400',
    bg: 'bg-red-500/10',
    border: 'border-red-500/20',
  },
  {
    id: 'n_of_k_failure',
    name: 'Multi-Plant Failure',
    icon: Layers,
    desc: 'Trips 2–5 generators simultaneously across the grid.',
    detail: 'N-1-1 contingency — simulates a second fault while the grid is already stressed.',
    color: 'text-red-400',
    bg: 'bg-red-900/10',
    border: 'border-red-900/30',
  },
  {
    id: 'storm_regional',
    name: 'Tornado / Storm',
    icon: Wind,
    desc: 'Devastates the local region — multiple lines and plants fail.',
    detail: 'High-impact low-probability event: towers down, substations damaged.',
    color: 'text-blue-400',
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/20',
  },
  {
    id: 'heatwave',
    name: 'Heatwave',
    icon: ThermometerSun,
    desc: 'Spikes regional demand by 20% and derates thermal plant capacity.',
    detail: 'AC load surge + thermal efficiency loss at high ambient temperatures.',
    color: 'text-amber-400',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/20',
  },
  {
    id: 'renewable_collapse',
    name: 'Renewable Shock',
    icon: CloudLightning,
    desc: 'Wind or solar output collapses 50–80% — sudden renewable loss.',
    detail: 'Cloud cover, wind lull, or curtailment order hits clean generation hard.',
    color: 'text-yellow-400',
    bg: 'bg-yellow-500/10',
    border: 'border-yellow-500/20',
  },
  {
    id: 'demand_spike',
    name: 'Demand Spike',
    icon: TrendingUp,
    desc: 'Sudden 30% demand surge in this region — grid scrambles to respond.',
    detail: 'Industrial plant starts up unexpectedly, or mass EV charging event.',
    color: 'text-orange-400',
    bg: 'bg-orange-500/10',
    border: 'border-orange-500/20',
  },
  {
    id: 'compound',
    name: 'Compound Crisis',
    icon: AlertTriangle,
    desc: 'Simultaneous combination of generator trip + renewable shock + demand surge.',
    detail: 'The worst-case stress test — multiple simultaneous failures.',
    color: 'text-purple-400',
    bg: 'bg-purple-500/10',
    border: 'border-purple-500/20',
  },
]

export default function DisasterControl({ selectedNode, onClose, onTriggerDisaster, isSimulating }) {
  if (!selectedNode) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -20 }}
        className="absolute top-24 left-8 w-80 z-50 pointer-events-auto"
      >
        <div className="bg-black/80 backdrop-blur-xl border border-red-500/30 rounded-2xl p-5 shadow-2xl shadow-red-900/20">
          
          <div className="flex items-start justify-between mb-4">
            <div>
              <div className="flex items-center gap-2 text-red-400 text-xs font-bold uppercase tracking-wider mb-1">
                <AlertTriangle className="w-4 h-4" />
                Disaster Control
              </div>
              <h3 className="text-xl font-medium text-white">{selectedNode.region}</h3>
              <p className="text-xs text-white/50 mt-0.5">Target: {selectedNode.name}</p>
            </div>
            <button onClick={onClose} className="text-white/40 hover:text-white transition-colors">
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="space-y-2 mt-4 max-h-[60vh] overflow-y-auto pr-1 custom-scrollbar">
            {DISASTERS.map(d => (
              <button
                key={d.id}
                disabled={isSimulating}
                onClick={() => onTriggerDisaster(d.id, selectedNode)}
                className={`w-full flex items-start gap-3 p-3 rounded-xl border ${d.border} ${d.bg} hover:brightness-125 transition-all text-left ${isSimulating ? 'opacity-50 cursor-not-allowed' : 'hover:scale-[1.02] active:scale-[0.99]'}`}
              >
                <div className={`p-2 rounded-lg bg-black/30 ${d.color} shrink-0 mt-0.5`}>
                  <d.icon className="w-4 h-4" />
                </div>
                <div>
                  <div className={`text-sm font-semibold ${d.color}`}>{d.name}</div>
                  <div className="text-xs text-white/55 mt-0.5 leading-relaxed">
                    {d.desc}
                  </div>
                  {d.detail && (
                    <div className="text-[10px] text-white/30 mt-1 italic">{d.detail}</div>
                  )}
                </div>
              </button>
            ))}
          </div>

          {isSimulating && (
            <div className="mt-4 flex items-center justify-center gap-2 text-xs text-amber-400 animate-pulse font-medium">
              <div className="w-3 h-3 rounded-full border border-amber-400/60 border-t-amber-400 animate-spin" />
              Simulation in progress...
            </div>
          )}

        </div>
      </motion.div>
    </AnimatePresence>
  )
}
