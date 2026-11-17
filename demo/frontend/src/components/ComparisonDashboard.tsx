import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, ApiError } from "../lib/api";
import { pct, thresholdPct } from "../lib/format";
import type { Comparison, ComparisonRow } from "../types";

// One fixed colour per model (validated categorical slots 1-4); colour follows the model, never its rank.
export const MODEL_COLORS: Record<string, string> = {
  bow: "#2a78d6",
  tfidf: "#eb6834",
  transformer: "#1baf7a",
  distilbert: "#eda100",
};
const SHORT: Record<string, string> = { bow: "BoW", tfidf: "TF-IDF", transformer: "Transformer", distilbert: "DistilBERT" };

interface ChartProps {
  title: string;
  rows: ComparisonRow[];
  value: (r: ComparisonRow) => number | null;
  format: (v: number) => string;
  domain?: [number | string, number | string];
  log?: boolean;
}

function MetricChart({ title, rows, value, format, domain, log }: ChartProps) {
  const data = rows.map((r) => ({ model: r.model, name: SHORT[r.model] ?? r.display_name, v: value(r) ?? 0 }));
  // Log scale: start one decade below the smallest value so every bar (e.g. 0.19 ms) has visible height.
  const positive = data.map((d) => d.v).filter((v) => v > 0);
  const logDomain: [number, number] = [
    10 ** Math.floor(Math.log10(Math.min(...positive, 1)) - 0.5),
    10 ** Math.ceil(Math.log10(Math.max(...positive, 1)) + 0.2),
  ];
  return (
    <figure className="rounded-xl border border-slate-200 p-3">
      <figcaption className="text-xs font-semibold text-navy-900">{title}</figcaption>
      <div className="mt-2 h-44" role="img" aria-label={`${title}: ${data.map((d) => `${d.name} ${format(d.v)}`).join(", ")}`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 18, right: 6, bottom: 0, left: -18 }} barCategoryGap="28%">
            <CartesianGrid vertical={false} stroke="#e2e8f0" />
            <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={{ stroke: "#cbd5e1" }} interval={0} />
            <YAxis
              scale={log ? "log" : "auto"}
              domain={domain ?? (log ? logDomain : [0, "auto"])}
              tick={{ fontSize: 10, fill: "#64748b" }}
              tickFormatter={(v: number) => format(v)}
              axisLine={false}
              tickLine={false}
              allowDataOverflow={!log}
            />
            <Tooltip
              cursor={{ fill: "rgba(148,163,184,0.12)" }}
              formatter={(v) => [format(Number(v)), title]}
              labelStyle={{ fontWeight: 600 }}
              contentStyle={{ fontSize: 12, borderRadius: 8 }}
            />
            <Bar dataKey="v" radius={[4, 4, 0, 0]} isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.model} fill={MODEL_COLORS[d.model] ?? "#64748b"} />
              ))}
              <LabelList dataKey="v" position="top" formatter={(v) => format(Number(v))} style={{ fontSize: 10, fill: "#0b1f3a" }} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

export function ComparisonDashboard() {
  const [data, setData] = useState<Comparison | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api
      .comparison()
      .then(setData)
      .catch((e: unknown) => setError(e instanceof ApiError ? e.message : "Comparison results unavailable"));
  }, []);

  return (
    <section aria-labelledby="dash-title" className="card p-5 sm:p-6">
      <h2 id="dash-title" className="card-title text-lg">
        Model Comparison Dashboard
      </h2>
      <p className="mt-0.5 text-xs text-slate-500">
        Held-out test set (270 commands) and 300 unsupported commands, read from <code>{data?.source ?? "results/comparison/"}</code>
        {data ? ` (generated ${data.generated_utc.slice(0, 10)})` : ""}. Nothing here is computed or typed in by the demo.
      </p>
      {error && <p className="mt-3 rounded-lg bg-amber-50 p-3 text-sm text-amber-800">{error}</p>}
      {!data && !error && <p className="mt-3 text-sm text-slate-500">Loading…</p>}
      {data && (
        <>
          {data.missing.length > 0 && (
            <p className="mt-3 rounded-lg bg-amber-50 p-3 text-xs text-amber-800">Not yet evaluated: {data.missing.join(", ")}</p>
          )}
          <div className="mt-4 flex flex-wrap gap-3 text-xs text-slate-600" aria-label="Legend">
            {data.models.map((r) => (
              <span key={r.model} className="inline-flex items-center gap-1.5">
                <span aria-hidden className="h-2.5 w-2.5 rounded-sm" style={{ background: MODEL_COLORS[r.model] }} />
                {r.display_name}
              </span>
            ))}
          </div>
          <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            <MetricChart title="Raw test accuracy (no threshold)" rows={data.models} value={(r) => r.raw_test_accuracy} format={(v) => pct(v)} domain={[0, 1]} />
            <MetricChart title="Macro F1 (test)" rows={data.models} value={(r) => r.test_macro_f1} format={(v) => v.toFixed(3)} domain={[0, 1]} />
            <MetricChart title="Unknown commands safely handled" rows={data.models} value={(r) => r.unknown_safe_fallback_at_default} format={(v) => pct(v, 0)} domain={[0, 1]} />
            <MetricChart title="Latency P50, ms (log scale)" rows={data.models} value={(r) => r.latency_p50_ms} format={(v) => (v < 1 ? v.toFixed(2) : v.toFixed(1))} log />
            <MetricChart title="Model size, MB (log scale)" rows={data.models} value={(r) => r.model_size_mb} format={(v) => (v < 1 ? v.toFixed(2) : v.toFixed(0))} log />
            <MetricChart title="Accuracy with one typo per command" rows={data.models} value={(r) => r.typo_test_accuracy} format={(v) => pct(v)} domain={[0, 1]} />
          </div>

          <div className="mt-5 overflow-x-auto">
            <table className="w-full min-w-[46rem] text-left text-xs">
              <caption className="sr-only">Model comparison table</caption>
              <thead className="border-b border-slate-200 text-slate-500">
                <tr>
                  <th className="py-2 pr-3 font-semibold">Model</th>
                  <th className="py-2 pr-3 font-semibold">Raw accuracy</th>
                  <th className="py-2 pr-3 font-semibold">Macro F1</th>
                  <th className="py-2 pr-3 font-semibold">Default threshold</th>
                  <th className="py-2 pr-3 font-semibold">Correct routing at default</th>
                  <th className="py-2 pr-3 font-semibold">Unknown rejected / → help / → money flow</th>
                  <th className="py-2 pr-3 font-semibold">Latency P50 / P95</th>
                  <th className="py-2 pr-3 font-semibold">Size</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 tabular-nums text-navy-900">
                {data.models.map((r) => (
                  <tr key={r.model}>
                    <td className="py-2 pr-3 font-medium">
                      <span aria-hidden className="mr-1.5 inline-block h-2 w-2 rounded-sm" style={{ background: MODEL_COLORS[r.model] }} />
                      {r.display_name}
                    </td>
                    <td className="py-2 pr-3">
                      {pct(r.raw_test_accuracy)} <span className="text-slate-400">({r.raw_test_correct}/{r.raw_test_n})</span>
                    </td>
                    <td className="py-2 pr-3">{r.test_macro_f1.toFixed(3)}</td>
                    <td className="py-2 pr-3">{thresholdPct(r.default_threshold)}</td>
                    <td className="py-2 pr-3">{pct(r.overall_correct_routing_at_default)}</td>
                    <td className="py-2 pr-3">
                      {pct(r.unknown_rejection_at_default, 0)} / {pct(r.unknown_help_support_at_default, 0)} / {r.unknown_to_financial_at_default}
                    </td>
                    <td className="py-2 pr-3">
                      {r.latency_p50_ms?.toFixed(2)} / {r.latency_p95_ms?.toFixed(2)} ms
                      <span className="text-slate-400"> ({r.latency_device})</span>
                    </td>
                    <td className="py-2 pr-3">{r.model_size_mb < 1 ? r.model_size_mb.toFixed(2) : r.model_size_mb.toFixed(0)} MB</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
            Raw accuracy counts all 270 legitimate test commands before any threshold. Thresholds differ by model because each is
            calibrated the same way (95% coverage on validation). Synthetic data with shared sentence templates: real-world
            accuracy will be lower. Latency depends on hardware (neural models measured on a GPU).
          </p>
        </>
      )}
    </section>
  );
}
