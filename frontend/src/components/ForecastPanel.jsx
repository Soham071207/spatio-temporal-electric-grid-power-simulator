import React, { useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Area, AreaChart, ReferenceLine, Dot,
} from "recharts";
import { Activity, Zap, TrendingUp, RefreshCw, BarChart2, Sun } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import DateNavigator from "./DateNavigator";

// ── Custom dot renderers (matching the reference matplotlib style) ──────────
const CircleDot = (props) => {
  const { cx, cy, stroke } = props;
  return <circle cx={cx} cy={cy} r={4} fill={stroke} stroke="#000" strokeWidth={0.5} />;
};

const TriangleDot = (props) => {
  const { cx, cy, stroke } = props;
  const size = 5;
  return (
    <polygon
      points={`${cx},${cy - size} ${cx - size},${cy + size * 0.6} ${cx + size},${cy + size * 0.6}`}
      fill={stroke}
      stroke="#000"
      strokeWidth={0.5}
    />
  );
};

// ── Custom tooltip ────────────────────────────────────────────────────────
const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-black/95 backdrop-blur-xl border border-white/15 rounded-xl px-3 py-2.5 shadow-2xl text-xs">
      <div className="text-white/50 mb-1.5">Hour {label}:00</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-2 py-0.5">
          <div className="w-2 h-2 rounded-full" style={{ background: p.stroke || p.fill }} />
          <span className="text-white/70">{p.name}:</span>
          <span className="font-medium text-white">{p.value?.toLocaleString()} MW</span>
        </div>
      ))}
    </div>
  );
};

const WeekendTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-black/95 backdrop-blur-xl border border-white/15 rounded-xl px-3 py-2.5 shadow-2xl text-xs">
      <div className="text-white/50 mb-1.5">{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-2 py-0.5">
          <div className="w-2 h-2 rounded-full" style={{ background: p.stroke || p.fill }} />
          <span className="text-white/70">{p.name}:</span>
          <span className="font-medium text-white">{p.value?.toLocaleString()} MW</span>
        </div>
      ))}
    </div>
  );
};

// ── Stat badge ────────────────────────────────────────────────────────────
const StatBadge = ({ icon: Icon, label, value, unit, color }) => (
  <div className="flex-1 bg-white/5 rounded-xl p-3 border border-white/10 min-w-[90px]">
    <div className="flex items-center gap-1.5 text-white/40 mb-1">
      <Icon className="w-3 h-3" />
      <span className="text-[9px] uppercase tracking-wider">{label}</span>
    </div>
    <div className={`text-base font-semibold ${color || "text-white"}`}>
      {value} <span className="text-xs font-normal text-white/30">{unit}</span>
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────────────────────
export default function ForecastPanel({
  region,
  date,
  forecastData,
  weekendData,
  onRefresh,
  isFetching,
  availableDates,
  onDateChange,
}) {
  // ── Derived stats (Hooks must be called unconditionally) ──────────────────
  const hourly = forecastData?.hourly_data || [];

  const stats = useMemo(() => {
    if (!hourly.length) return null;
    const actualArr   = hourly.map(d => d.actual_mw);
    const predictArr  = hourly.map(d => d.predicted_mw);
    const peakActual  = Math.max(...actualArr);
    const peakPred    = Math.max(...predictArr);
    const totalActual = actualArr.reduce((a, b) => a + b, 0);
    const totalPred   = predictArr.reduce((a, b) => a + b, 0);
    const mae         = actualArr.reduce((s, v, i) => s + Math.abs(v - predictArr[i]), 0) / actualArr.length;
    const mape        = actualArr.reduce((s, v, i) => s + Math.abs((v - predictArr[i]) / Math.max(v, 1)), 0) / actualArr.length * 100;
    return { peakActual, peakPred, totalActual, totalPred, mae, mape };
  }, [hourly]);

  // Max Y for the primary chart (add 20% headroom like the reference image)
  const chartMax = useMemo(() => {
    if (!hourly.length) return undefined;
    const m = Math.max(...hourly.map(d => Math.max(d.actual_mw, d.predicted_mw)));
    return Math.ceil(m * 1.2 / 500) * 500;
  }, [hourly]);

  // ── Empty state ──────────────────────────────────────────────────────────
  if (!region) {
    return (
      <div className="card-shell w-full h-full min-h-[400px] flex items-center justify-center">
        <div className="card-inner flex flex-col items-center justify-center text-white/40 p-8 text-center">
          <Activity className="w-12 h-12 mb-4 opacity-50" />
          <h3 className="text-xl font-medium tracking-tight">No Region Selected</h3>
          <p className="text-sm mt-2 max-w-[220px]">
            Click any region on the map to instantly load its 24-hour GAT forecast.
          </p>
        </div>
      </div>
    );
  }

  // ── Loading state ────────────────────────────────────────────────────────
  if (isFetching) {
    return (
      <div className="card-shell w-full h-full min-h-[400px]">
        <div className="card-inner flex items-center justify-center">
          <div className="flex flex-col items-center gap-4">
            <div className="relative w-12 h-12">
              <div className="absolute inset-0 rounded-full border-2 border-white/10" />
              <div className="absolute inset-0 rounded-full border-2 border-t-blue-400 animate-spin" />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-white/80">{region}</p>
              <p className="text-xs text-white/40 tracking-widest uppercase mt-1">
                Running inference…
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Label for the chart legend (mirrors the reference image)
  const maeLabelActual   = region;
  const maeLabelPred     = stats ? `GAT Regional (MAE: ${Math.round(stats.mae)} MW)` : "GAT Regional";

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, ease: [0.32, 0.72, 0, 1] }}
      className="flex flex-col gap-5 w-full h-full"
    >
      {/* ── Header card ───────────────────────────────────────────────── */}
      <div className="card-shell w-full">
        <div className="card-inner p-5 flex flex-col gap-4">
          {/* Region name + refresh */}
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="text-[10px] uppercase tracking-[0.2em] font-medium text-white/40 mb-1">
                Target Node
              </div>
              <h2 className="text-2xl font-semibold tracking-tighter text-white leading-none">
                {region}
              </h2>
            </div>
            <button
              onClick={onRefresh}
              disabled={isFetching}
              title="Refresh forecast"
              className="mt-0.5 flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium text-white/50 border border-white/10 bg-white/5 hover:bg-white/10 hover:text-white disabled:opacity-20 disabled:cursor-not-allowed transition-all active:scale-95"
            >
              <RefreshCw className="w-3 h-3" /> Refresh
            </button>
          </div>

          {/* Date navigator embedded in the panel */}
          <div className="flex items-center justify-center py-1 border-t border-b border-white/8">
            <DateNavigator
              availableDates={availableDates}
              currentDate={date}
              onChange={onDateChange}
            />
          </div>

          {/* Stats row */}
          {stats && (
            <div className="flex gap-2">
              <StatBadge
                icon={Zap}
                label="Peak actual"
                value={Math.round(stats.peakActual).toLocaleString()}
                unit="MW"
                color="text-white"
              />
              <StatBadge
                icon={BarChart2}
                label="MAE"
                value={Math.round(stats.mae).toLocaleString()}
                unit="MW"
                color="text-blue-400"
              />
              <StatBadge
                icon={TrendingUp}
                label="MAPE"
                value={stats.mape.toFixed(1)}
                unit="%"
                color={stats.mape < 5 ? "text-emerald-400" : stats.mape < 10 ? "text-amber-400" : "text-red-400"}
              />
            </div>
          )}
        </div>
      </div>

      {!forecastData ? (
        /* Waiting state (region selected but no data yet — shouldn't usually show) */
        <div className="card-shell w-full flex-grow min-h-[260px]">
          <div className="card-inner flex flex-col items-center justify-center text-white/30 p-8 gap-3">
            <div className="w-6 h-6 rounded-full border border-white/20 border-t-white/50 animate-spin" />
            <p className="text-sm">Loading forecast…</p>
          </div>
        </div>
      ) : (
        <>
          {/* ── Chart 1: 24-Hour Forecast vs Actual ──────────────────── */}
          <div className="card-shell w-full">
            <div className="card-inner p-5 flex flex-col">
              {/* Chart header */}
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold tracking-tight text-white flex items-center gap-2">
                  <BarChart2 className="w-4 h-4 text-white/40" />
                  24-Hour Forecast vs Actual
                </h3>
                <div className="flex items-center gap-4 text-[10px] text-white/50">
                  <span className="flex items-center gap-1.5">
                    <span className="inline-block w-5 h-0.5 bg-blue-500" style={{ borderTop: "2px solid #3b82f6" }} />
                    <span className="inline-block w-2.5 h-2.5 rounded-full bg-blue-500 -mt-px" />
                    Actual (MW)
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span
                      className="inline-block w-5 h-0"
                      style={{ borderTop: "2px dashed #22c55e" }}
                    />
                    <span
                      className="inline-block w-0 h-0 -mt-px"
                      style={{
                        borderLeft: "4px solid transparent",
                        borderRight: "4px solid transparent",
                        borderBottom: "7px solid #22c55e",
                      }}
                    />
                    GAT Regional
                  </span>
                </div>
              </div>

              {/* Title line matching the reference chart title */}
              <div className="text-[11px] text-white/30 text-center mb-2 tracking-tight">
                {forecastData.region} — {date}
              </div>

              <div className="w-full h-[230px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={forecastData.hourly_data}
                    margin={{ top: 8, right: 10, bottom: 5, left: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                    <XAxis
                      dataKey="hour"
                      stroke="rgba(255,255,255,0.25)"
                      fontSize={10}
                      tickLine={false}
                      axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                      tickFormatter={(v) => `${v}h`}
                      label={{ value: "Hours of the Day", position: "insideBottom", offset: -2, fill: "rgba(255,255,255,0.3)", fontSize: 9 }}
                    />
                    <YAxis
                      stroke="rgba(255,255,255,0.25)"
                      fontSize={10}
                      tickLine={false}
                      axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                      width={50}
                      tickFormatter={(v) => v >= 1000 ? `${(v/1000).toFixed(1)}k` : v}
                      domain={[0, chartMax]}
                      label={{ value: "Demand (MW)", angle: -90, position: "insideLeft", offset: 10, fill: "rgba(255,255,255,0.3)", fontSize: 9 }}
                    />
                    <Tooltip content={<ChartTooltip />} />
                    {/* Actual — blue solid with circle dots */}
                    <Line
                      type="monotone"
                      dataKey="actual_mw"
                      name={maeLabelActual}
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={<CircleDot stroke="#3b82f6" />}
                      activeDot={{ r: 5, fill: "#3b82f6", stroke: "#fff", strokeWidth: 1 }}
                    />
                    {/* Predicted — green dashed with triangle dots */}
                    <Line
                      type="monotone"
                      dataKey="predicted_mw"
                      name={maeLabelPred}
                      stroke="#22c55e"
                      strokeWidth={2}
                      strokeDasharray="6 3"
                      dot={<TriangleDot stroke="#22c55e" />}
                      activeDot={{ r: 5, fill: "#22c55e", stroke: "#fff", strokeWidth: 1 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Legend below chart — matching the reference image */}
              <div className="flex flex-col gap-1 mt-3 px-1">
                <div className="flex items-center gap-2 text-[10px] text-white/60">
                  <div className="flex items-center gap-1">
                    <span className="inline-block w-6 h-0" style={{ borderTop: "2px solid #3b82f6" }} />
                    <span className="inline-block w-2 h-2 rounded-full bg-blue-500" />
                  </div>
                  Actual {forecastData.region} (MW)
                </div>
                <div className="flex items-center gap-2 text-[10px] text-white/60">
                  <div className="flex items-center gap-1">
                    <span className="inline-block w-6 h-0" style={{ borderTop: "2px dashed #22c55e" }} />
                    <span
                      className="inline-block w-0 h-0"
                      style={{
                        borderLeft: "4px solid transparent",
                        borderRight: "4px solid transparent",
                        borderBottom: "7px solid #22c55e",
                      }}
                    />
                  </div>
                  {maeLabelPred}
                </div>
              </div>
            </div>
          </div>

          {/* ── Chart 2: 48-Hour Weekend Forecast ───────────────────────── */}
          <div className="card-shell w-full">
            <div className="card-inner p-5 flex flex-col">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold tracking-tight text-white flex items-center gap-2">
                  <Sun className="w-4 h-4 text-white/40" />
                  Weekend Forecast (48h)
                </h3>
                {weekendData && (
                  <div className="text-[10px] text-white/40">
                    {weekendData.saturday} — {weekendData.sunday}
                  </div>
                )}
              </div>

              <div className="w-full h-[200px]">
                {weekendData ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={weekendData.hourly_data}
                      margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
                    >
                      <defs>
                        <linearGradient id="wkActGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.18} />
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="wkPredGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%"  stopColor="#22c55e" stopOpacity={0.2} />
                          <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                      <XAxis
                        dataKey="hour"
                        stroke="rgba(255,255,255,0.25)"
                        fontSize={9}
                        tickLine={false}
                        axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                        interval={7}
                      />
                      <YAxis
                        stroke="rgba(255,255,255,0.25)"
                        fontSize={9}
                        tickLine={false}
                        axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                        width={45}
                        tickFormatter={(v) => v >= 1000 ? `${(v/1000).toFixed(1)}k` : v}
                      />
                      <ReferenceLine
                        x="Sun 0h"
                        stroke="rgba(255,255,255,0.2)"
                        strokeDasharray="4 4"
                        label={{ value: "Sunday", fill: "rgba(255,255,255,0.35)", fontSize: 9, position: "top" }}
                      />
                      <Tooltip content={<WeekendTooltip />} />
                      <Area type="monotone" dataKey="actual_mw"    name="Actual MW"    stroke="#3b82f6" strokeWidth={1.5} fill="url(#wkActGrad)"  dot={false} activeDot={{ r: 3, fill: "#3b82f6" }} />
                      <Area type="monotone" dataKey="predicted_mw" name="Predicted MW" stroke="#22c55e" strokeWidth={1.5} fill="url(#wkPredGrad)" dot={false} strokeDasharray="6 3" activeDot={{ r: 3, fill: "#22c55e" }} />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-white/30 text-sm gap-2">
                    <div className="w-4 h-4 rounded-full border border-white/20 border-t-white/50 animate-spin" />
                    Loading weekend data…
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </motion.div>
  );
}
