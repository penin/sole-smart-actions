import { pct, thresholdPct } from "../lib/format";
import type { HistoryEntry } from "../types";

interface Props {
  history: HistoryEntry[];
  activeKey: string | null;
  onRestore: (entry: HistoryEntry) => void;
  onClear: () => void;
}

export function RecentPredictions({ history, activeKey, onRestore, onClear }: Props) {
  return (
    <section aria-labelledby="history-title" className="card p-5">
      <div className="flex items-center justify-between gap-2">
        <h2 id="history-title" className="card-title">
          Recent Predictions
        </h2>
        <button
          type="button"
          onClick={onClear}
          disabled={history.length === 0}
          className="rounded-lg px-2 py-1 text-xs font-medium text-slate-500 transition hover:bg-slate-100 hover:text-slate-700 disabled:opacity-40"
        >
          Clear History
        </button>
      </div>
      <p className="mt-0.5 text-xs text-slate-500">This session only (last 10). Not saved anywhere.</p>
      {history.length === 0 ? (
        <p className="mt-4 rounded-xl border border-dashed border-slate-200 p-4 text-center text-xs text-slate-400">
          Commands you test will appear here.
        </p>
      ) : (
        <ul className="mt-3 space-y-1.5" aria-label="Recent predictions">
          {history.map((h) => {
            const p = h.prediction;
            const fallback = h.finalIntentAtTime === "UNKNOWN_INTENT";
            return (
              <li key={h.key}>
                <button
                  type="button"
                  onClick={() => onRestore(h)}
                  className={`w-full rounded-xl border px-3 py-2 text-left transition ${
                    activeKey === h.key ? "border-teal-600 bg-teal-50/60" : "border-slate-200 hover:border-slate-300 hover:bg-slate-50"
                  }`}
                >
                  <p className="truncate text-sm font-medium text-navy-900">“{p.input}”</p>
                  <p className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-slate-500">
                    <span className="font-mono">{p.raw_prediction}</span>
                    <span className="tabular-nums">{pct(p.confidence)}</span>
                    <span aria-hidden>→</span>
                    <span className={`font-mono font-semibold ${fallback ? "text-amber-700" : "text-teal-700"}`}>{h.finalIntentAtTime}</span>
                    <span className="tabular-nums">@ {thresholdPct(h.thresholdUsed)}</span>
                  </p>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
