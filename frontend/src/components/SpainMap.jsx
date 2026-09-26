import React, { useState, useEffect } from "react";
import { ComposableMap, Geographies, Geography } from "react-simple-maps";
import { cn } from "../utils";
import { motion, AnimatePresence } from "motion/react";
import { Zap, TrendingUp, Clock } from "lucide-react";

// Translates the GeoJSON short names → the UI/API names used by the backend
const GEO_TO_UI = {
  'Andalucia':        'Andalucía',
  'Aragon':           'Aragón',
  'Asturias':         'Principado de Asturias',
  'Baleares':         'Illes Balears',
  'Canarias':         'Canarias',
  'Cantabria':        'Cantabria',
  'Castilla-Leon':    'Castilla y León',
  'Castilla-La Mancha': 'Castilla - La Mancha',
  'Cataluña':         'Cataluña',
  // GeoJSON may encode the ñ differently — cover both
  'Catalu\u00f1a':    'Cataluña',
  'Valencia':         'Comunitat Valenciana',
  'Extremadura':      'Extremadura',
  'Galicia':          'Galicia',
  'Madrid':           'Comunidad de Madrid',
  'Murcia':           'Región de Murcia',
  'Navarra':          'Comunidad Foral de Navarra',
  'Pais Vasco':       'País Vasco',
  'La Rioja':         'La Rioja',
  'Ceuta':            'Ceuta',
  'Melilla':          'Melilla',
};

/** Translate a raw GeoJSON name to the canonical UI/API name. */
function toUIName(geoName) {
  return GEO_TO_UI[geoName] ?? geoName;
}

const geoUrl = "http://127.0.0.1:8000/api/regions";

/**
 * Linearly interpolates peak MW onto a colour from cool-blue → amber → orange-red.
 * Returns a CSS rgba string.
 */
function peakToColor(peakMw, minPeak, maxPeak, isActive, isHovered) {
  if (isActive) return "rgba(255,255,255,0.95)";
  if (!peakMw || maxPeak === minPeak) {
    return isHovered ? "rgba(96,165,250,0.45)" : "rgba(255,255,255,0.08)";
  }

  const t = Math.max(0, Math.min(1, (peakMw - minPeak) / (maxPeak - minPeak)));

  // Cool blue (low) → teal → amber → orange-red (high)
  // Palette waypoints at t=0, 0.4, 0.7, 1.0
  let r, g, b;
  if (t < 0.4) {
    const s = t / 0.4;
    // #1e40af → #0891b2  (blue → cyan-700)
    r = Math.round(30  + s * (8   - 30));
    g = Math.round(64  + s * (145 - 64));
    b = Math.round(175 + s * (178 - 175));
  } else if (t < 0.7) {
    const s = (t - 0.4) / 0.3;
    // #0891b2 → #d97706  (cyan → amber)
    r = Math.round(8   + s * (217 - 8));
    g = Math.round(145 + s * (119 - 145));
    b = Math.round(178 + s * (6   - 178));
  } else {
    const s = (t - 0.7) / 0.3;
    // #d97706 → #dc2626  (amber → red)
    r = Math.round(217 + s * (220 - 217));
    g = Math.round(119 + s * (38  - 119));
    b = Math.round(6   + s * (38  - 6));
  }

  const alpha = isHovered ? 0.85 : 0.55 + t * 0.35;
  return `rgba(${r},${g},${b},${alpha})`;
}

export default function SpainMap({
  onRegionClick,
  onRegionHover,
  activeRegion,
  testDate,
  regionStats,   // { [uiName]: { peak_predicted_mw, total_predicted_mwh } }
  isLoadingMap,
}) {
  const [hoveredRegion, setHoveredRegion]   = useState(null);
  const [hoverData,     setHoverData]       = useState(null);
  const [loadingHover,  setLoadingHover]    = useState(false);

  // Pre-compute min/max for the colour scale
  const peaks = regionStats
    ? Object.values(regionStats).map(v => v.peak_predicted_mw).filter(Boolean)
    : [];
  const minPeak = peaks.length ? Math.min(...peaks) : 0;
  const maxPeak = peaks.length ? Math.max(...peaks) : 1;

  // Hover tooltip — calls /api/forecast/hover
  useEffect(() => {
    if (!hoveredRegion || !testDate) { setHoverData(null); return; }
    setLoadingHover(true);
    const ctrl = new AbortController();
    const apiRegion = toUIName(hoveredRegion);
    fetch(
      `http://127.0.0.1:8000/api/forecast/hover?date=${testDate}&region=${encodeURIComponent(apiRegion)}`,
      { signal: ctrl.signal }
    )
      .then(r => r.json())
      .then(d => { setHoverData(d); setLoadingHover(false); })
      .catch(err => { if (err.name !== "AbortError") setLoadingHover(false); });
    return () => ctrl.abort();
  }, [hoveredRegion, testDate]);

  return (
    <div
      className="card-shell w-full h-[500px] md:h-[650px] flex items-center justify-center relative overflow-hidden group"
    >
      <div className="card-inner bg-black/60 pt-12">

        {/* Loading map overlay */}
        <AnimatePresence>
          {isLoadingMap && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute top-3 right-4 z-20 flex items-center gap-1.5 text-[10px] text-white/40 uppercase tracking-wider"
            >
              <div className="w-3 h-3 rounded-full border border-white/20 border-t-white/60 animate-spin" />
              Updating map…
            </motion.div>
          )}
        </AnimatePresence>

        {/* Colour scale legend */}
        {regionStats && peaks.length > 0 && (
          <div className="absolute top-3 left-4 z-20 flex items-center gap-2">
            <div className="text-[9px] uppercase tracking-wider text-white/30">
              Peak demand
            </div>
            <div
              className="w-24 h-2 rounded-full"
              style={{
                background: "linear-gradient(to right, rgba(30,64,175,0.7), rgba(8,145,178,0.7), rgba(217,119,6,0.7), rgba(220,38,38,0.8))",
              }}
            />
            <div className="flex justify-between text-[9px] text-white/30 w-24">
              <span>{Math.round(minPeak)} MW</span>
              <span>{Math.round(maxPeak)} MW</span>
            </div>
          </div>
        )}

        <ComposableMap
          projection="geoAzimuthalEqualArea"
          projectionConfig={{ rotate: [4, -40, 0], scale: 2800 }}
          className="w-full h-full"
        >
          <Geographies geography={geoUrl}>
            {({ geographies }) =>
              geographies.map((geo) => {
                const geoName   = geo.properties.name;
                const uiName    = toUIName(geoName);
                const isActive  = activeRegion === uiName;
                const isHovered = hoveredRegion === geoName;
                const peak      = regionStats?.[uiName]?.peak_predicted_mw;
                const fillColor = peakToColor(peak, minPeak, maxPeak, isActive, isHovered);
                const strokeColor = isActive
                  ? "rgba(0,0,0,0.8)"
                  : isHovered
                    ? "rgba(255,255,255,0.5)"
                    : "rgba(255,255,255,0.12)";

                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    onClick={() => onRegionClick(uiName)}
                    onMouseEnter={() => {
                      setHoveredRegion(geoName);
                      if (onRegionHover) onRegionHover(uiName);
                    }}
                    onMouseLeave={() => {
                      setHoveredRegion(null);
                      setHoverData(null);
                    }}
                    className="cursor-pointer outline-none"
                    style={{
                      default: {
                        fill: fillColor,
                        stroke: strokeColor,
                        strokeWidth: isActive ? 0.8 : 0.5,
                        transition: "fill 0.6s ease, stroke 0.3s ease",
                        outline: "none",
                      },
                      hover: { fill: fillColor, outline: "none" },
                      pressed: { fill: fillColor, outline: "none" },
                    }}
                  />
                );
              })
            }
          </Geographies>
        </ComposableMap>

        {/* Ambient glow */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[80%] h-[80%] bg-white/5 rounded-full blur-[120px] pointer-events-none transition-opacity duration-1000 group-hover:bg-white/8" />
      </div>

      {/* ── Hover Tooltip (Anchored Top Right) ───────────────────────── */}
      <AnimatePresence>
        {hoveredRegion && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -10, x: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0, x: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -5, x: 5 }}
            transition={{ duration: 0.2 }}
            className="absolute top-6 right-6 z-50 pointer-events-none"
          >
            <div className="bg-black/90 backdrop-blur-xl border border-white/15 rounded-2xl px-4 py-3 shadow-2xl min-w-[210px]">
              <div className="text-[10px] uppercase tracking-[0.2em] text-white/50 mb-1.5">
                Forecast Preview
              </div>
              <div className="text-sm font-medium text-white mb-3 tracking-tight">
                {toUIName(hoveredRegion)}
              </div>

              {/* Map heatmap stat */}
              {regionStats?.[toUIName(hoveredRegion)] && (
                <div className="flex items-center justify-between gap-6 mb-2 pb-2 border-b border-white/8">
                  <span className="text-[10px] text-white/40 uppercase tracking-wider">
                    Peak predicted
                  </span>
                  <span className="text-xs font-semibold text-amber-400">
                    {regionStats[toUIName(hoveredRegion)].peak_predicted_mw.toLocaleString()} MW
                  </span>
                </div>
              )}

              {loadingHover || !hoverData ? (
                <div className="flex items-center gap-2 text-white/40 text-xs py-2">
                  <div className="w-3 h-3 rounded-full border border-white/20 border-t-white/60 animate-spin" />
                  Computing…
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-6">
                    <span className="flex items-center gap-1.5 text-xs text-white/50">
                      <Clock className="w-3 h-3" /> Next Hour
                    </span>
                    <span className="text-sm font-medium text-blue-400">
                      {hoverData.next_hour_predicted_mw} MW
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-6">
                    <span className="flex items-center gap-1.5 text-xs text-white/50">
                      <Zap className="w-3 h-3" /> Next 24h
                    </span>
                    <span className="text-sm font-medium text-emerald-400">
                      {hoverData.next_day_predicted_mw?.toLocaleString()} MWh
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-6">
                    <span className="flex items-center gap-1.5 text-xs text-white/50">
                      <TrendingUp className="w-3 h-3" /> Peak
                    </span>
                    <span className="text-sm font-medium text-amber-400">
                      {hoverData.peak_predicted_mw} MW
                    </span>
                  </div>
                </div>
              )}

              <div className="mt-3 pt-2 border-t border-white/5 text-[9px] text-white/30 text-center uppercase tracking-wider">
                Click for full forecast →
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
