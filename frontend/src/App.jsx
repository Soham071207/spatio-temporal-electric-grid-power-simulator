import React, { useState, useCallback, useEffect, useRef } from 'react'
import SpainMap from './components/SpainMap'
import ForecastPanel from './components/ForecastPanel'
import DateNavigator from './components/DateNavigator'
import ChaosDashboard from './components/ChaosDashboard'
import ChaosGameSimulator from './components/ChaosGameSimulator'
import { RefreshCw, Skull } from 'lucide-react'
import { motion, AnimatePresence } from 'motion/react'

const API = 'http://127.0.0.1:8000'

function App() {
  const [activeRegion, setActiveRegion]     = useState(null)
  const [testDate, setTestDate]             = useState("2025-07-20")
  const [forecastData, setForecastData]     = useState(null)
  const [weekendData, setWeekendData]       = useState(null)
  const [isFetching, setIsFetching]         = useState(false)
  const [availableDates, setAvailableDates] = useState([])
  const [regionStats, setRegionStats]       = useState(null)
  const [isLoadingMap, setIsLoadingMap]     = useState(false)
  const [showChaos, setShowChaos]           = useState(false)
  const [rightPanel, setRightPanel]         = useState('forecast')

  const fetchSeq = useRef(0)

  useEffect(() => {
    fetch(`${API}/api/forecast/dates`)
      .then(r => r.json())
      .then(data => {
        if (data.dates?.length) {
          setAvailableDates(data.dates)
          setTestDate(prev => {
            if (data.dates.includes(prev)) return prev
            const target = new Date(prev).getTime()
            const closest = data.dates.reduce((a, b) =>
              Math.abs(new Date(a) - target) < Math.abs(new Date(b) - target) ? a : b
            )
            return closest
          })
        }
      })
      .catch(err => console.warn('Could not fetch available dates:', err))
  }, [])

  useEffect(() => {
    if (!testDate) return
    setIsLoadingMap(true)
    fetch(`${API}/api/forecast/all-regions?date=${testDate}`)
      .then(r => r.json())
      .then(data => setRegionStats(data.regions || null))
      .catch(err => console.warn('All-regions fetch failed:', err))
      .finally(() => setIsLoadingMap(false))
  }, [testDate])

  const fetchForecast = useCallback((region, date) => {
    if (!region || !date) return
    const seq = ++fetchSeq.current
    setIsFetching(true)
    setForecastData(null)
    setWeekendData(null)
    Promise.all([
      fetch(`${API}/api/forecast?date=${date}&region=${encodeURIComponent(region)}`).then(r => r.json()),
      fetch(`${API}/api/forecast/weekend?date=${date}&region=${encodeURIComponent(region)}`).then(r => r.json()),
    ])
      .then(([dayData, wkData]) => {
        if (seq !== fetchSeq.current) return
        setForecastData(dayData)
        setWeekendData(wkData)
        setIsFetching(false)
      })
      .catch(err => {
        if (seq !== fetchSeq.current) return
        console.error('Forecast fetch failed', err)
        setIsFetching(false)
      })
  }, [])

  const handleRegionClick = useCallback((name) => {
    setActiveRegion(name)
    setForecastData(null)
    setWeekendData(null)
    setRightPanel('forecast')
    fetchForecast(name, testDate)
  }, [testDate, fetchForecast])

  const debounceTimer = useRef(null)
  const handleDateChange = useCallback((newDate) => {
    setTestDate(newDate)
    clearTimeout(debounceTimer.current)
    debounceTimer.current = setTimeout(() => {
      if (activeRegion) fetchForecast(activeRegion, newDate)
    }, 300)
  }, [activeRegion, fetchForecast])

  const handleRefresh = useCallback(() => {
    if (activeRegion && testDate) fetchForecast(activeRegion, testDate)
  }, [activeRegion, testDate, fetchForecast])

  return (
    <div className="min-h-[100dvh] w-full p-4 md:p-8 flex flex-col items-center">

      <header className="w-full max-w-7xl flex flex-col md:flex-row items-center justify-between gap-6 mb-12">
        <div className="flex flex-col">
          <div className="text-[10px] uppercase tracking-[0.2em] font-medium text-white/50 mb-2">
            Grid Topology Simulation
          </div>
          <h1 className="text-3xl font-medium tracking-tighter text-white flex items-center gap-2">
            Spain Digital Twin <span className="text-white/20">|</span>{' '}
            <span className="text-blue-400">GAT Forecaster</span>
          </h1>
        </div>

        <div className="card-shell shrink-0">
          <div className="card-inner flex items-center gap-3 px-4 py-3 bg-black/50">
            <DateNavigator
              availableDates={availableDates}
              currentDate={testDate}
              onChange={handleDateChange}
            />
            <button
              onClick={handleRefresh}
              disabled={!activeRegion || isFetching}
              title="Refresh forecast"
              className="ml-1 w-8 h-8 flex items-center justify-center rounded-xl border border-white/10 bg-white/5 text-white/40 hover:text-white hover:bg-white/10 disabled:opacity-20 disabled:cursor-not-allowed transition-all active:scale-90"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isFetching ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        <button
          onClick={() => setShowChaos(c => !c)}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl border text-xs font-semibold transition-all active:scale-95 ${
            showChaos
              ? 'bg-red-500/15 border-red-500/30 text-red-300'
              : 'bg-white/5 border-white/10 text-white/50 hover:text-white/80 hover:bg-white/10'
          }`}
        >
          <Skull className="w-3.5 h-3.5" />
          3D Simulator
        </button>
      </header>

      <main className="w-full max-w-7xl grid grid-cols-1 md:grid-cols-12 gap-8 md:gap-12 relative">

        <div className="col-span-1 md:col-span-7 flex flex-col">
          <SpainMap
            onRegionClick={handleRegionClick}
            activeRegion={activeRegion}
            testDate={testDate}
            regionStats={regionStats}
            isLoadingMap={isLoadingMap}
          />
          <div className="mt-4 text-center text-sm text-white/40">
            {activeRegion
              ? <>Showing forecast for <span className="text-white/70 font-medium">{activeRegion}</span> — use arrows to toggle dates</>
              : <>Hover a region for quick predictions. <span className="text-blue-400 font-medium">Click</span> to load full forecast.</>
            }
          </div>
        </div>

        <div className="col-span-1 md:col-span-5 flex flex-col min-h-[650px]">

          <div className="flex gap-1 p-0.5 mb-4 rounded-xl bg-white/[0.03] border border-white/[0.06]">
            {[
              { id: 'forecast', label: 'Forecast',     hasIcon: false },
              { id: 'chaos',    label: 'Chaos Engine', hasIcon: true  },
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setRightPanel(tab.id)}
                className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-[11px] font-medium transition-all ${
                  rightPanel === tab.id
                    ? tab.id === 'chaos'
                      ? 'bg-red-500/15 text-red-300 border border-red-500/20'
                      : 'bg-white/10 text-white shadow-sm'
                    : 'text-white/40 hover:text-white/60'
                }`}
              >
                {tab.hasIcon && <Skull className="w-3 h-3" />}
                {tab.label}
              </button>
            ))}
          </div>

          <AnimatePresence mode="wait">
            {rightPanel === 'forecast' ? (
              <motion.div
                key={`forecast-${activeRegion || 'empty'}`}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -8 }}
                transition={{ duration: 0.2 }}
                className="w-full h-full"
              >
                <ForecastPanel
                  region={activeRegion}
                  date={testDate}
                  forecastData={forecastData}
                  weekendData={weekendData}
                  onRefresh={handleRefresh}
                  isFetching={isFetching}
                  availableDates={availableDates}
                  onDateChange={handleDateChange}
                />
              </motion.div>
            ) : (
              <motion.div
                key="chaos-panel"
                initial={{ opacity: 0, x: 8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 8 }}
                transition={{ duration: 0.2 }}
                className="w-full h-full rounded-2xl border border-white/[0.06] bg-white/[0.015] backdrop-blur-sm p-4 overflow-y-auto"
              >
                <ChaosDashboard />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

      <AnimatePresence>
        {showChaos && (
          <ChaosGameSimulator onClose={() => setShowChaos(false)} />
        )}
      </AnimatePresence>
    </div>
  )
}

export default App
