import { thresholdPct } from "../lib/format";

export type Verification = "idle" | "pending" | "verified" | "mismatch" | "error";

interface Props {
  threshold: number;
  defaultThreshold: number | null;
  defaultSource: string | null;
  onChange: (t: number) => void;
  onReset: () => void;
  verification: Verification;
}

const VERIFY_TEXT: Record<Verification, string> = {
  idle: "",
  pending: "Confirming with backend…",
  verified: "Decision confirmed by backend",
  mismatch: "Backend decision differs — showing backend result",
  error: "Could not confirm with backend",
};

export function ThresholdControl({ threshold, defaultThreshold, defaultSource, onChange, onReset, verification }: Props) {
  const isDefault = defaultThreshold !== null && threshold === defaultThreshold;
  const fill = `${threshold * 100}%`;
  return (
    <section aria-labelledby="threshold-title" className="card p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 id="threshold-title" className="card-title">
            Confidence Threshold
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">Minimum confidence needed to accept a prediction</p>
        </div>
        <button
          type="button"
          onClick={onReset}
          disabled={defaultThreshold === null || isDefault}
          className="shrink-0 rounded-lg border border-slate-200 px-2.5 py-1 text-xs whitespace-nowrap font-medium text-slate-600 transition hover:border-slate-300 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Reset to Default
        </button>
      </div>

      <p className="mt-4 text-sm text-slate-600">
        Confidence Threshold:{" "}
        <output htmlFor="threshold" aria-live="polite" className="text-3xl font-semibold tabular-nums text-navy-900">
          {thresholdPct(threshold)}
        </output>
      </p>

      <label htmlFor="threshold" className="sr-only">
        Confidence threshold (percent)
      </label>
      <input
        id="threshold"
        type="range"
        min={0}
        max={100}
        step={1}
        value={threshold * 100}
        onChange={(e) => onChange(Number(e.target.value) / 100)}
        aria-valuetext={thresholdPct(threshold)}
        className="threshold-slider mt-3 w-full"
        style={{ "--fill": fill } as React.CSSProperties}
      />
      <div className="mt-1 flex justify-between text-[11px] text-slate-400 tabular-nums">
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>

      <p className="mt-3 text-xs leading-relaxed text-slate-600">
        Project default:{" "}
        <span className="font-semibold text-navy-900">
          {defaultThreshold === null ? "Not available" : thresholdPct(defaultThreshold)}
        </span>
        {defaultSource && <span className="text-slate-500"> (from {defaultSource})</span>}. If the top probability is{" "}
        <em>below</em> this value the model returns <code className="intent-chip">UNKNOWN_INTENT</code> and the app
        offers Help &amp; Support or asks the customer to rephrase. Moving the slider never retrains the model or
        changes saved settings.
      </p>
      {verification !== "idle" && (
        <p
          className={`mt-2 flex items-center gap-1.5 text-[11px] font-medium ${
            verification === "verified" ? "text-teal-700" : verification === "pending" ? "text-slate-500" : "text-amber-700"
          }`}
        >
          <span aria-hidden>{verification === "verified" ? "✓" : verification === "pending" ? "…" : "!"}</span>
          {VERIFY_TEXT[verification]}
        </p>
      )}
    </section>
  );
}
