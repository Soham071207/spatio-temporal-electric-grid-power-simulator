import React from "react";
import { ChevronLeft, ChevronRight, Calendar } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

/**
 * DateNavigator — steps through the list of available model dates.
 * Props:
 *   availableDates: string[]   sorted list of "YYYY-MM-DD" strings from /api/forecast/dates
 *   currentDate:   string      currently selected date
 *   onChange:      (date: string) => void
 */
export default function DateNavigator({ availableDates, currentDate, onChange }) {
  const dates = availableDates || [];
  const currentIdx = dates.indexOf(currentDate);
  const hasPrev = currentIdx > 0;
  const hasNext = currentIdx < dates.length - 1;
  const displayNum = currentIdx >= 0 ? currentIdx + 1 : "—";

  const go = (delta) => {
    const next = dates[currentIdx + delta];
    if (next) onChange(next);
  };

  // Format date nicely: "Thu, 27 Nov 2024"
  const formatDate = (d) => {
    if (!d) return "—";
    try {
      return new Date(d + "T12:00:00Z").toLocaleDateString("en-GB", {
        weekday: "short",
        day: "numeric",
        month: "short",
        year: "numeric",
        timeZone: "UTC",
      });
    } catch {
      return d;
    }
  };

  return (
    <div className="flex items-center gap-2 select-none">
      {/* Prev arrow */}
      <button
        onClick={() => go(-1)}
        disabled={!hasPrev}
        className="w-8 h-8 flex items-center justify-center rounded-xl border border-white/10 bg-white/5 text-white/60 hover:bg-white/10 hover:text-white disabled:opacity-20 disabled:cursor-not-allowed transition-all active:scale-90"
        title="Previous date"
      >
        <ChevronLeft className="w-4 h-4" />
      </button>

      {/* Date display */}
      <div className="flex flex-col items-center min-w-[160px]">
        <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.2em] text-white/40 mb-0.5">
          <Calendar className="w-3 h-3" />
          {dates.length > 0
            ? `${displayNum} / ${dates.length}`
            : "Loading dates…"}
        </div>
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={currentDate}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
            className="text-sm font-medium text-white tracking-tight text-center"
          >
            {formatDate(currentDate)}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Next arrow */}
      <button
        onClick={() => go(1)}
        disabled={!hasNext}
        className="w-8 h-8 flex items-center justify-center rounded-xl border border-white/10 bg-white/5 text-white/60 hover:bg-white/10 hover:text-white disabled:opacity-20 disabled:cursor-not-allowed transition-all active:scale-90"
        title="Next date"
      >
        <ChevronRight className="w-4 h-4" />
      </button>
    </div>
  );
}
