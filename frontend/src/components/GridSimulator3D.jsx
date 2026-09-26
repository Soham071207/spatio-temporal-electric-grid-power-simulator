import React, { useMemo, useRef, useState, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Stars, Line, QuadraticBezierLine, Html } from '@react-three/drei'
import * as THREE from 'three'

import PLANT_COORDS from '../assets/plant_coords.json'

// Center of Spain roughly
const CENTER_LAT = 40.4168
const CENTER_LON = -3.7038
const SCALE = 1.2

function projectCoord(lat, lon) {
  const x = (lon - CENTER_LON) * SCALE
  const z = -(lat - CENTER_LAT) * SCALE
  return [x, 0, z]
}

const TECH_COLORS = {
  Nuclear: '#a855f7',
  Gas: '#3b82f6',
  Coal: '#71717a',
  Hydro: '#0ea5e9',
  Solar: '#f59e0b',
  Wind: '#10b981',
}

// ─── Power Plant Node ──────────────────────────────────────────────────────────
function PowerPlant({ plant, status, rampedMw, score, onClick }) {
  const [x, y, z] = useMemo(() => projectCoord(plant.lat, plant.lon), [plant])
  const groupRef = useRef()
  const ref = useRef()
  const haloRef = useRef()
  const [hovered, setHovered] = useState(false)

  let color = TECH_COLORS[plant.category] || '#ffffff'
  let scale = 1.0
  let emissiveIntensity = 1.5

  if (status === 'tripped') {
    color = '#ff0000' // Stronger Red
    emissiveIntensity = 4.0
    scale = 1.5
  } else if (status === 'ramping') {
    color = '#22c55e' // Green
    // Brighter glow for bigger contribution
    const rampFactor = Math.min(1, (rampedMw || 50) / 500)
    emissiveIntensity = 3.0 + rampFactor * 3.0
    scale = 1.3 + rampFactor * 0.3
  }



  const baseSize = 0.03 + (plant.capacity_mw / 3000) * 0.1
  const finalSize = baseSize * scale

  return (
    <group position={[x, y, z]} ref={groupRef}>
      <mesh
        ref={ref}
        onClick={(e) => { e.stopPropagation(); onClick(plant); }}
        onPointerOver={(e) => { e.stopPropagation(); setHovered(true); document.body.style.cursor = 'pointer'; }}
        onPointerOut={(e) => { e.stopPropagation(); setHovered(false); document.body.style.cursor = 'auto'; }}
      >
        <sphereGeometry args={[finalSize, 16, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={emissiveIntensity}
          toneMapped={false}
        />
      </mesh>

      {(status === 'tripped' || status === 'ramping') && <RippleRing color={color} />}

      {hovered && (
        <Html position={[0, 0.2, 0]} center className="pointer-events-none z-50">
          <div className="bg-black/90 border border-white/20 px-2 py-1 rounded text-[10px] text-white whitespace-nowrap shadow-xl">
            <div className="font-bold">{plant.name}</div>
            <div className="text-white/60">{plant.tech} - {plant.capacity_mw} MW</div>
            {status === 'tripped' && <div className="text-red-400 font-bold uppercase mt-1">OFFLINE</div>}
            {status === 'ramping' && (
              <div className="text-green-400 font-bold uppercase mt-1">
                RAMPING +{rampedMw || '?'} MW
              </div>
            )}
            {score !== undefined && (
              <div className="text-[9px] text-gray-300 opacity-80 normal-case mt-0.5">Score: {score}</div>
            )}
          </div>
        </Html>
      )}
    </group>
  )
}

function RippleRing({ color }) {
  const ref = useRef()
  const timeRef = useRef(0)

  useFrame((state, delta) => {
    timeRef.current += delta
    if (ref.current) {
      const t = (timeRef.current * 1.5) % 1
      ref.current.scale.set(1 + t * 3, 1, 1 + t * 3)
      ref.current.material.opacity = 0.6 * (1 - t)
    }
  })

  return (
    <mesh ref={ref} rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[0.04, 0.08, 32]} />
      <meshBasicMaterial color={color} transparent opacity={0.6} depthWrite={false} side={THREE.DoubleSide} />
    </mesh>
  )
}



// ─── Healing Arcs (animated energy flow from ramping → tripped generators) ────
function HealingArcs({ healingActions, trippedGeneratorIds }) {
  // Build a map of plant coords by ID
  const plantMap = useMemo(() => {
    const m = {}
    PLANT_COORDS.forEach(p => { m[p.id] = p })
    return m
  }, [])

  // For each ramp_up action, draw an arc from the ramping generator to the nearest tripped generator
  const arcs = useMemo(() => {
    if (!healingActions?.length || !trippedGeneratorIds?.length) return []

    const result = []
    const trippedPlants = trippedGeneratorIds.map(id => plantMap[id]).filter(Boolean)
    if (trippedPlants.length === 0) return []

    healingActions.forEach(action => {
      if (action.type !== 'ramp_up') return
      const source = plantMap[action.generator_id]
      if (!source) return

      // Find the nearest tripped plant
      let nearest = trippedPlants[0]
      let minDist = Infinity
      trippedPlants.forEach(tp => {
        const d = Math.sqrt(Math.pow(source.lat - tp.lat, 2) + Math.pow(source.lon - tp.lon, 2))
        if (d < minDist) { minDist = d; nearest = tp }
      })

      const start = projectCoord(source.lat, source.lon)
      const end = projectCoord(nearest.lat, nearest.lon)
      // Midpoint raised above the grid for a nice arc
      const mid = [
        (start[0] + end[0]) / 2,
        0.5 + (action.ramped_mw || 50) / 500 * 0.5, // Height scales with MW
        (start[2] + end[2]) / 2,
      ]

      result.push({
        start,
        mid,
        end,
        ramped_mw: action.ramped_mw || 50,
        technology: action.technology,
        id: `${action.generator_id}->${nearest.id}`,
      })
    })

    return result
  }, [healingActions, trippedGeneratorIds, plantMap])

  if (arcs.length === 0) return null

  return (
    <group>
      {arcs.map((arc) => (
        <AnimatedArc key={arc.id} arc={arc} color="#22c55e" />
      ))}
    </group>
  )
}

// ─── Buffer Discharge Arcs (gold — virtual storage draining to deficit) ─────
function BufferArcs({ healingActions }) {
  // Compute regional centroids
  const regionCoords = useMemo(() => {
    const coords = {}
    const counts = {}
    PLANT_COORDS.forEach(p => {
      if (!coords[p.region]) {
        coords[p.region] = { lat: 0, lon: 0 }
        counts[p.region] = 0
      }
      coords[p.region].lat += p.lat
      coords[p.region].lon += p.lon
      counts[p.region] += 1
    })
    for (const r in coords) {
      coords[r].lat /= counts[r]
      coords[r].lon /= counts[r]
    }
    return coords
  }, [])

  const arcs = useMemo(() => {
    if (!healingActions?.length) return []
    const result = []

    healingActions.forEach((action, i) => {
      if (action.type !== 'buffer_discharge') return

      const fromCoord = regionCoords[action.from_region]
      const toCoord = regionCoords[action.to_region]
      if (!fromCoord || !toCoord) return

      const start = projectCoord(fromCoord.lat, fromCoord.lon)
      const end = projectCoord(toCoord.lat, toCoord.lon)
      const mid = [
        (start[0] + end[0]) / 2,
        0.7 + (action.discharged_mw || 50) / 500 * 0.6,  // Taller arc than transfers
        (start[2] + end[2]) / 2,
      ]

      result.push({
        start,
        mid,
        end,
        // Use sent_mw for arc visual (shows gross energy dispatched, not just arrived)
        ramped_mw: action.sent_mw ?? action.discharged_mw ?? 50,
        arrived_mw: action.discharged_mw,
        efficiency: action.transmission_efficiency ?? 1.0,
        id: `buffer-${i}-${action.from_region}->${action.to_region}`,
      })
    })
    return result
  }, [healingActions, regionCoords])

  if (arcs.length === 0) return null

  return (
    <group>
      {arcs.map((arc) => (
        // Buffer arcs dim when efficiency is low (long-distance transfer)
        <AnimatedArc key={arc.id} arc={arc} color={`rgba(245,158,11,${0.4 + (arc.efficiency ?? 1) * 0.6})`} />
      ))}
    </group>
  )
}


function TransferArcs({ healingActions }) {
  // Compute regional centroids
  const regionCoords = useMemo(() => {
    const coords = {}
    const counts = {}
    PLANT_COORDS.forEach(p => {
      if (!coords[p.region]) {
        coords[p.region] = { lat: 0, lon: 0 }
        counts[p.region] = 0
      }
      coords[p.region].lat += p.lat
      coords[p.region].lon += p.lon
      counts[p.region] += 1
    })
    for (const r in coords) {
      coords[r].lat /= counts[r]
      coords[r].lon /= counts[r]
    }
    return coords
  }, [])

  const arcs = useMemo(() => {
    if (!healingActions?.length) return []
    const result = []

    healingActions.forEach((action, i) => {
      if (action.type !== 'power_transfer') return
      
      const fromCoord = regionCoords[action.from_region]
      const toCoord = regionCoords[action.to_region]
      
      if (!fromCoord || !toCoord) return

      const start = projectCoord(fromCoord.lat, fromCoord.lon)
      const end = projectCoord(toCoord.lat, toCoord.lon)
      
      // Scale arc height by distance_km — longer transfers arc higher
      const distFactor = action.distance_km ? Math.min(1.5, action.distance_km / 600) : 0.5
      const mid = [
        (start[0] + end[0]) / 2,
        0.4 + distFactor + (action.transferred_mw || 50) / 1000 * 0.3,
        (start[2] + end[2]) / 2,
      ]

      // Efficiency affects arc opacity — lower efficiency = dimmer line (more loss)
      const eff = action.efficiency ?? 1.0

      result.push({
        start,
        mid,
        end,
        ramped_mw: action.sent_mw ?? action.transferred_mw ?? 50,
        efficiency: eff,
        id: `transfer-${i}-${action.from_region}->${action.to_region}`,
      })
    })
    return result
  }, [healingActions, regionCoords])

  if (arcs.length === 0) return null

  return (
    <group>
      {arcs.map((arc) => (
        // Blue arcs fade with distance loss — vivid blue = local, pale blue = far away
        <AnimatedArc key={arc.id} arc={arc} color={`rgba(59,130,246,${0.35 + (arc.efficiency ?? 1) * 0.65})`} />
      ))}
    </group>
  )
}

function AnimatedArc({ arc, color }) {
  const ref = useRef()
  const lineWidth = Math.min(4, 1 + (arc.ramped_mw / 200))

  useFrame((state) => {
    if (ref.current) {
      ref.current.material.dashOffset -= 0.02
    }
  })

  return (
    <QuadraticBezierLine
      ref={ref}
      start={arc.start}
      mid={arc.mid}
      end={arc.end}
      color={color || "#22c55e"}
      lineWidth={lineWidth}
      dashed
      dashScale={20}
      dashSize={1}
      dashOffset={0}
      opacity={0.8}
      transparent
    />
  )
}

// ─── Grid Connection Lines ─────────────────────────────────────────────────────
function GridLines({ plants, trippedRegions }) {
  const lines = useMemo(() => {
    const l = []
    const regions = {}
    plants.forEach(p => {
      if (!regions[p.region]) regions[p.region] = []
      regions[p.region].push(p)
    })

    Object.keys(regions).forEach(reg => {
      const ps = regions[reg]
      if (ps.length > 0) {
        const center = ps[0]
        ps.forEach(p => {
          if (p !== center) {
            const p1 = projectCoord(p.lat, p.lon)
            const p2 = projectCoord(center.lat, center.lon)
            l.push({ p1, p2, region: reg, isMain: false })
          }
        })
      }
    })

    const regionalCenters = Object.values(regions).map(r => r[0])
    for (let i = 0; i < regionalCenters.length - 1; i++) {
      for (let j = i + 1; j < regionalCenters.length; j++) {
        const c1 = regionalCenters[i]
        const c2 = regionalCenters[j]
        const dist = Math.sqrt(Math.pow(c1.lat - c2.lat, 2) + Math.pow(c1.lon - c2.lon, 2))
        if (dist < 3.0) {
          l.push({
            p1: projectCoord(c1.lat, c1.lon),
            p2: projectCoord(c2.lat, c2.lon),
            region1: c1.region,
            region2: c2.region,
            isMain: true
          })
        }
      }
    }
    return l
  }, [plants])

  return (
    <group>
      {lines.map((line, i) => {
        const isTripped = trippedRegions.includes(line.region) ||
                         trippedRegions.includes(line.region1) ||
                         trippedRegions.includes(line.region2)

        return (
          <Line
            key={i}
            points={[line.p1, line.p2]}
            color={isTripped ? '#ff0000' : (line.isMain ? '#3b82f6' : '#ffffff')}
            lineWidth={line.isMain ? 1.5 : 0.5}
            transparent
            opacity={isTripped ? 0.8 : (line.isMain ? 0.3 : 0.1)}
          />
        )
      })}
    </group>
  )
}

// ─── Spain Map Outline ─────────────────────────────────────────────────────────
function SpainMapOutline() {
  const [geometry, setGeometry] = useState(null)

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/regions')
      .then(r => r.json())
      .then(data => {
        const pointsArray = []
        data.features.forEach(feature => {
          const polys = feature.geometry.type === 'MultiPolygon' 
            ? feature.geometry.coordinates.map(p => p[0])
            : [feature.geometry.coordinates[0]]
            
          polys.forEach(poly => {
            for (let i = 0; i < poly.length - 1; i++) {
              const p1 = projectCoord(poly[i][1], poly[i][0])
              const p2 = projectCoord(poly[i+1][1], poly[i+1][0])
              pointsArray.push(...p1, ...p2)
            }
          })
        })
        const geo = new THREE.BufferGeometry()
        geo.setAttribute('position', new THREE.Float32BufferAttribute(pointsArray, 3))
        setGeometry(geo)
      })
      .catch(err => console.warn('Could not load map outline', err))
  }, [])

  if (!geometry) return null

  return (
    <lineSegments geometry={geometry} position={[0, -0.09, 0]}>
      <lineBasicMaterial color="#555555" opacity={0.4} transparent />
    </lineSegments>
  )
}

// ─── Main Export ────────────────────────────────────────────────────────────────
export default function GridSimulator3D({
  cascadeStep,
  healingActions,
  trippedGeneratorIds,
  rampingGeneratorIds,
  onNodeClick
}) {
  // Use the explicitly passed IDs if available, fall back to cascadeStep data
  const trippedPlantIds = trippedGeneratorIds || cascadeStep?.tripped_generators || []
  const affectedRegions = cascadeStep?.affected_regions || []

  // Build a map from generator_id → ramped_mw for precise highlighting
  const rampedMwMap = useMemo(() => {
    const m = {}
    if (healingActions) {
      healingActions.forEach(a => {
        if (a.type === 'ramp_up' && a.generator_id) {
          m[a.generator_id] = a.ramped_mw || 0
        }
      })
    }
    return m
  }, [healingActions])

  const scoreMap = useMemo(() => {
    const m = {}
    if (healingActions) {
      healingActions.forEach(a => {
        if (a.generator_id && a.score !== undefined) {
          m[a.generator_id] = a.score
        }
      })
    }
    return m
  }, [healingActions])

  // Set of ramping IDs for fast lookup
  const rampingIdSet = useMemo(() => {
    return new Set(rampingGeneratorIds || [])
  }, [rampingGeneratorIds])

  return (
    <div className="w-full h-full absolute inset-0 bg-[#050505]">
      <Canvas camera={{ position: [0, 6, 8], fov: 45 }}>
        <color attach="background" args={['#050505']} />
        <ambientLight intensity={0.2} />

        {/* Base Grid Plane */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.1, 0]}>
          <planeGeometry args={[50, 50]} />
          <meshBasicMaterial color="#0a0a0a" transparent opacity={0.8} />
          <gridHelper args={[50, 50, '#111', '#111']} rotation={[Math.PI / 2, 0, 0]} />
        </mesh>

        <SpainMapOutline />

        <GridLines plants={PLANT_COORDS} trippedRegions={affectedRegions} />

        {/* Healing arcs — animated energy flow lines */}
        <HealingArcs
          healingActions={healingActions}
          trippedGeneratorIds={trippedPlantIds}
        />
        <TransferArcs 
          healingActions={healingActions} 
        />
        {/* Buffer discharge arcs — gold, taller than transfer arcs */}
        <BufferArcs
          healingActions={healingActions}
        />

        {PLANT_COORDS.map((plant) => {
          let status = 'normal'
          let rampedMw = 0
          let score = scoreMap[plant.id]
          if (trippedPlantIds.includes(plant.id)) {
            status = 'tripped'
          } else if (rampingIdSet.has(plant.id)) {
            status = 'ramping'
            rampedMw = rampedMwMap[plant.id] || 0
          }

          return (
            <PowerPlant
              key={plant.id}
              plant={plant}
              status={status}
              rampedMw={rampedMw}
              score={score}
              onClick={onNodeClick}
            />
          )
        })}

        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          maxPolarAngle={Math.PI / 2 - 0.05}
          minDistance={2}
          maxDistance={15}
        />
      </Canvas>
    </div>
  )
}
