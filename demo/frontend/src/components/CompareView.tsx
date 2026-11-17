import { useCallback, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../lib/api";
import { intentLabel, pct, thresholdPct } from "../lib/format";
import { applyThreshold } from "../lib/threshold";
import { buildWorkflow } from "../lib/workflow";
import type { CompareEntry, ModelSummary } from "../types";
import { CommandInput } from "./CommandInput";
import { ComparisonDashboard, MODEL_COLORS } from "./ComparisonDashboard";
import { SampleCommands } from "./SampleCommands";
import { WorkflowPanel } from "./WorkflowPanel";

interface Props {
  models: ModelSummary[];
  input: string;
  onInput: (v: string) => void;
}

function ModelCard({ entry, threshold }: { entry: CompareEntry; threshold: number | null }) {
  const color = MODEL_COLORS[entry.model_id] ?? "#64748b";
  if (!entry.ok || !entry.prediction) {
    return (
      <article className="card border-t-4 p-4" style={{ borderTopColor: color }}>
        <h3 className="text-sm font-semibold text-navy-900">{entry.model_name}</h3>
        <p role="alert" className="mt-2 rounded-lg bg-amber-50 p-2 text-xs text-amber-800">
          Not available: {entry.error}
        </p>
      </article>
    );
  }
  const p = entry.prediction;
  const t = threshold ?? p.default_threshold;
  const d = applyThreshold(p.probabilities, t, p.class_order);
  return (
    <article className="card flex flex-col border-t-4 p-4" style={{ borderTopColor: color }} data-testid={`compare-${entry.model_id}`}>
      <h3 className="text-sm font-semibold text-navy-900">{entry.model_name}</h3>
      <p className="mt-0.5 text-[11px] text-slate-500">
        {p.inference_ms.toFixed(1)} ms · threshold {thresholdPct(t)} {threshold === null ? "(model default)" : "(shared)"}
      </p>
      <p className="eyebrow mt-3">Predicted intent</p>
      <p className="font-mono text-sm font-semibold break-all text-navy-900">{d.raw_prediction}</p>
      <p className="text-2xl font-semibold tabular-nums text-navy-900">{pct(d.confidence)}</p>
      <p className="mt-1">
        {d.fallback_applied ? (
          <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs font-semibold text-amber-800 ring-1 ring-amber-600/30">
            ▼ Below threshold · clarification needed
          </span>
        ) : (
          <span className="rounded-full bg-teal-50 px-2 py-0.5 text-xs font-semibold text-teal-800 ring-1 ring-teal-600/25">
            ✓ Passes threshold
          </span>
        )}
      </p>
      <p className="eyebrow mt-3">Top 3 intents</p>
      <ul className="mt-1 space-y-1">
        {p.ranked.slice(0, 3).map((r) => (
          <li key={r.intent} className="text-xs">
            <div className="flex justify-between gap-2">
              <span className="truncate text-slate-700">{intentLabel(r.intent)}</span>
              <span className="tabular-nums text-slate-500">{pct(r.probability)}</span>
            </div>
            <div className="mt-0.5 h-1.5 rounded-full bg-slate-100">
              <div className="h-1.5 rounded-full" style={{ width: `${Math.max(1, r.probability * 100)}%`, background: color }} />
            </div>
          </li>
        ))}
      </ul>
      <div className="mt-3 flex-1">
        <WorkflowPanel workflow={buildWorkflow(p.input, d, p.ranked)} compact />
      </div>
    </article>
  );
}

export function CompareView({ models, input, onInput }: Props) {
  const [entries, setEntries] = useState<CompareEntry[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shared, setShared] = useState<number | null>(null);
  const reqId = useRef(0);
  const primary = useMemo(() => models.filter((m) => (m.group ?? "primary") === "primary"), [models]);

  const run = useCallback(async (text: string) => {
    if (!text.trim()) return;
    const id = ++reqId.current;
    setLoading(true);
    setError(null);
    try {
      // Probabilities do not depend on the threshold, so request them once and re-apply thresholds locally.
      const res = await api.compare(text, null);
      if (id === reqId.current) setEntries(res);
    } catch (e) {
      if (id === reqId.current) setError(e instanceof ApiError ? e.message : "Comparison failed");
    } finally {
      if (id === reqId.current) setLoading(false);
    }
  }, []);

  const agree = entries?.filter((e) => e.ok && e.prediction).map((e) => e.prediction!.raw_prediction) ?? [];
  const unanimous = agree.length > 1 && agree.every((x) => x === agree[0]);

  return (
    <div className="flex flex-col gap-6">
      <CommandInput value={input} onChange={onInput} onSubmit={run} loading={loading} disabled={primary.length === 0} />
      <SampleCommands modelClasses={[]} onPick={(c) => { onInput(c); void run(c); }} activeCommand={entries?.[0]?.prediction?.input ?? null} disabled={loading} />

      <section aria-labelledby="cmp-title" className="card p-5 sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 id="cmp-title" className="card-title text-lg">
              All Models Side by Side
            </h2>
            <p className="mt-0.5 text-xs text-slate-500">
              Same command, four live models.{" "}
              {entries && agree.length > 1 && (unanimous ? "All available models agree." : "The models disagree on this command.")}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-xs text-slate-600">
            <label className="inline-flex items-center gap-1.5">
              <input type="checkbox" checked={shared === null} onChange={(e) => setShared(e.target.checked ? null : 0.5)} />
              Use each model's validation default
            </label>
            {shared !== null && (
              <label className="inline-flex items-center gap-2">
                Shared threshold {thresholdPct(shared)}
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={shared}
                  aria-label="Shared confidence threshold"
                  onChange={(e) => setShared(Number(e.target.value))}
                />
              </label>
            )}
          </div>
        </div>
        {error && (
          <p role="alert" className="mt-3 rounded-lg bg-red-50 p-3 text-sm text-red-800">
            {error}
          </p>
        )}
        {!entries && !error && (
          <div className="mt-4 rounded-xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500">
            {loading ? "Running every model… (the neural models load on first use)" : "Type a command or pick a sample to compare all models."}
          </div>
        )}
        {entries && (
          <div className={`mt-4 grid gap-4 sm:grid-cols-2 xl:grid-cols-4 ${loading ? "opacity-60" : ""}`} aria-live="polite">
            {entries.map((e) => (
              <ModelCard key={e.model_id} entry={e} threshold={shared} />
            ))}
          </div>
        )}
      </section>

      <ComparisonDashboard />
    </div>
  );
}
