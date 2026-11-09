import { intentLabel, pct, thresholdPct } from "../lib/format";
import { UNKNOWN_INTENT } from "../lib/threshold";
import type { Decision, Prediction } from "../types";

interface Props {
  prediction: Prediction | null;
  decision: Decision | null;
  loading: boolean;
  error: string | null;
}

function Tile({ label, children, tone = "plain" }: { label: string; children: React.ReactNode; tone?: "plain" | "accent" }) {
  return (
    <div
      className={`rounded-xl border p-4 transition-colors ${
        tone === "accent" ? "border-navy-900/10 bg-navy-900 text-white" : "border-slate-200 bg-slate-50/60"
      }`}
    >
      <p className={`text-[11px] font-semibold uppercase tracking-[0.08em] ${tone === "accent" ? "text-slate-300" : "text-slate-500"}`}>
        {label}
      </p>
      <div className="mt-1.5">{children}</div>
    </div>
  );
}

export function StatusBadge({ decision }: { decision: Decision }) {
  if (!decision.fallback_applied) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-teal-50 px-3 py-1 text-sm font-semibold text-teal-800 ring-1 ring-teal-600/25">
        <span aria-hidden className="h-2 w-2 rounded-full bg-teal-600" />
        Accepted
      </span>
    );
  }
  return (
    <span className="flex flex-wrap gap-1.5">
      <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-50 px-3 py-1 text-sm font-semibold whitespace-nowrap text-amber-800 ring-1 ring-amber-600/30">
        <span aria-hidden>▼</span>
        Below Threshold
      </span>
      <span className="inline-flex items-center gap-1.5 rounded-full bg-slate-100 px-3 py-1 text-sm font-semibold whitespace-nowrap text-slate-700 ring-1 ring-slate-300">
        <span aria-hidden>↺</span>
        Fallback Applied
      </span>
    </span>
  );
}

export function ResultsPanel({ prediction, decision, loading, error }: Props) {
  return (
    <section aria-labelledby="results-title" aria-busy={loading} className="card p-5 sm:p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="results-title" className="card-title text-lg">
          Prediction Results
        </h2>
        {prediction && (
          <p className="text-xs text-slate-500">
            Live inference · {prediction.model_name} · {prediction.inference_ms.toFixed(2)} ms
          </p>
        )}
      </div>

      {error && (
        <div role="alert" className="mt-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          <p className="font-semibold">Could not classify this command</p>
          <p className="mt-1">{error}</p>
        </div>
      )}

      {!prediction && !error && (
        <div className="mt-4 rounded-xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500">
          {loading ? "Running the model…" : "Type a command or pick a sample below to see how the model classifies it."}
        </div>
      )}

      {prediction && decision && (
        <div aria-live="polite" className={`mt-4 transition-opacity ${loading ? "opacity-60" : "opacity-100"}`}>
          <p className="mb-4 text-sm text-slate-600">
            Command: <span className="font-medium text-navy-900">“{prediction.input}”</span>
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            <Tile label="Raw Predicted Intent">
              <p className="font-mono text-[15px] font-semibold break-all text-navy-900" data-testid="raw-intent">
                {decision.raw_prediction}
              </p>
              <p className="text-xs text-slate-500">Highest probability, before the threshold</p>
            </Tile>
            <Tile label="Confidence Score">
              <p className="text-2xl font-semibold tabular-nums text-navy-900" data-testid="confidence">
                {pct(decision.confidence)}
              </p>
              <p className="text-xs text-slate-500">
                {decision.below_threshold ? "Below" : "At or above"} the {thresholdPct(decision.threshold)} threshold
              </p>
            </Tile>
            <Tile label="Final Intent" tone="accent">
              <p className="font-mono text-[15px] font-semibold break-all" data-testid="final-intent">
                {decision.final_intent}
              </p>
              <p className="text-xs text-slate-300">
                {decision.final_intent === UNKNOWN_INTENT ? "Fallback: offer Help & Support or clarify" : `Route to ${intentLabel(decision.final_intent)}`}
              </p>
            </Tile>
            <Tile label="Classification Status">
              <div data-testid="status">
                <StatusBadge decision={decision} />
              </div>
            </Tile>
          </div>

          <div className="mt-4 rounded-xl bg-slate-50 p-3 text-sm text-slate-700">
            <span className="font-semibold text-navy-900">What the app would do: </span>
            {decision.fallback_applied ? (
              <>
                Not confident enough to act. Show the fallback: offer <span className="intent-chip">HELP_SUPPORT</span> or
                ask the customer to rephrase. The model's best guess was{" "}
                <span className="intent-chip">{decision.application.suggested_intent}</span>, which is shown but not
                executed.
              </>
            ) : decision.final_intent === "HELP_SUPPORT" ? (
              <>
                The model classified this as a Help &amp; Support request, so the customer is routed to support. This is a
                real prediction, not a fallback.
              </>
            ) : (
              <>
                Open the <span className="font-medium">{intentLabel(decision.final_intent)}</span> flow. Any money movement
                would still need the customer's confirmation.
              </>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
