import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { intentLabel, thresholdPct } from "../lib/format";
import type { Decision, Prediction } from "../types";

interface Props {
  prediction: Prediction;
  decision: Decision;
  /** Held-out test-set figures, used only to describe calibration honestly. */
  calibration?: { meanConfidence: number | null; accuracy: number | null } | null;
}

const TEAL = "#0d9488";
const AMBER = "#d97706";
const MUTED = "#94a3b8";

export function ConfidenceChart({ prediction, decision, calibration }: Props) {
  const [showTable, setShowTable] = useState(false);
  // Real predict_proba output, sorted high -> low. Values are displayed, never adjusted.
  const rows = [...prediction.class_order]
    .map((intent) => ({ intent, label: intentLabel(intent), pct: prediction.probabilities[intent] * 100 }))
    .sort((a, b) => b.pct - a.pct);
  const thr = decision.threshold * 100;
  const topColor = decision.fallback_applied ? AMBER : TEAL;

  return (
    <section aria-labelledby="chart-title" className="card p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 id="chart-title" className="card-title text-lg">
            Confidence Breakdown
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Probability for each of the {rows.length} intents the model knows, from <code>predict_proba()</code>
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs text-slate-600">
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-2.5 w-2.5 rounded-sm" style={{ background: topColor }} />
            Top intent {decision.fallback_applied ? "(below threshold)" : "(passes threshold)"}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-3 border-l-2 border-dashed border-navy-900" />
            Threshold {thresholdPct(decision.threshold)}
          </span>
        </div>
      </div>

      <div className="mt-4 h-[340px] w-full" role="img" aria-label={`Bar chart of intent probabilities. Highest: ${rows[0].label} at ${rows[0].pct.toFixed(1)}%. Threshold ${thresholdPct(decision.threshold)}.`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ top: 18, right: 56, bottom: 4, left: 4 }} barCategoryGap={6}>
            <CartesianGrid horizontal={false} stroke="#e2e8f0" />
            <XAxis
              type="number"
              domain={[0, 100]}
              ticks={[0, 25, 50, 75, 100]}
              tickFormatter={(v) => `${v}%`}
              tick={{ fontSize: 11, fill: "#64748b" }}
              axisLine={{ stroke: "#cbd5e1" }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={128}
              tick={{ fontSize: 12, fill: "#334155" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              cursor={{ fill: "rgba(11,31,58,0.04)" }}
              formatter={(v) => [`${Number(v).toFixed(2)}%`, "Probability"]}
              labelStyle={{ color: "#0b1f3a", fontWeight: 600 }}
              contentStyle={{ borderRadius: 10, borderColor: "#e2e8f0", fontSize: 12 }}
            />
            <Bar dataKey="pct" radius={[0, 4, 4, 0]} isAnimationActive={false} minPointSize={2}>
              {rows.map((r, i) => (
                <Cell key={r.intent} fill={i === 0 ? topColor : MUTED} />
              ))}
              <LabelList
                dataKey="pct"
                position="right"
                formatter={(v) => `${Number(v).toFixed(1)}%`}
                style={{ fontSize: 11, fill: "#334155", fontVariantNumeric: "tabular-nums" }}
              />
            </Bar>
            <ReferenceLine
              x={thr}
              stroke="#0b1f3a"
              strokeDasharray="5 4"
              strokeWidth={2}
              ifOverflow="visible"
              label={{ value: `Threshold ${thresholdPct(decision.threshold)}`, position: "top", fill: "#0b1f3a", fontSize: 11, fontWeight: 600 }}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="mt-3 text-xs leading-relaxed text-slate-500">
        These are <strong>model-estimated probabilities</strong>, not guarantees of correctness.
        {calibration?.meanConfidence != null && calibration.accuracy != null && (
          <>
            {" "}
            On the held-out test set the average top probability was {(calibration.meanConfidence * 100).toFixed(0)}%
            while accuracy was {(calibration.accuracy * 100).toFixed(0)}%
            {calibration.meanConfidence < calibration.accuracy - 0.05
              ? ", so this model tends to be under-confident."
              : calibration.meanConfidence > calibration.accuracy + 0.05
                ? ", so this model tends to be over-confident."
                : "."}
          </>
        )}
      </p>

      <button
        type="button"
        onClick={() => setShowTable((s) => !s)}
        className="mt-2 text-xs font-medium text-teal-700 underline-offset-2 hover:underline"
        aria-expanded={showTable}
      >
        {showTable ? "Hide" : "Show"} values as a table
      </button>
      {showTable && (
        <table className="mt-2 w-full text-left text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-xs text-slate-500">
              <th className="py-1.5 font-medium">Intent</th>
              <th className="py-1.5 text-right font-medium">Probability</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.intent} className="border-b border-slate-100">
                <td className="py-1.5 font-mono text-xs">{r.intent}</td>
                <td className="py-1.5 text-right tabular-nums">{r.pct.toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
