import { num, pct, thresholdPct } from "../lib/format";
import type { ModelInfo } from "../types";

interface Props {
  info: ModelInfo | null;
  error: string | null;
  session: { tested: number; lastInferenceMs: number | null };
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5">
      <dt className="text-xs text-slate-500">{label}</dt>
      <dd className="text-right text-sm font-medium tabular-nums break-words text-navy-900">{value}</dd>
    </div>
  );
}

const ms = (x: number | null | undefined) => (x === null || x === undefined ? "Not available" : `${x.toFixed(2)} ms`);

export function ModelInfoPanel({ info, error, session }: Props) {
  const ev = info?.evaluation;
  const ci = ev?.test_macro_f1_ci95;
  return (
    <section aria-labelledby="info-title" className="card p-5 sm:p-6">
      <h2 id="info-title" className="card-title text-lg">
        Model Information
      </h2>
      {error && <p className="mt-2 text-sm text-red-700">{error}</p>}
      {!info && !error && <p className="mt-2 text-sm text-slate-500">Loading…</p>}
      {info && ev && (
        <div className="mt-4 grid gap-6 md:grid-cols-3">
          <div>
            <p className="eyebrow">Model</p>
            <dl className="mt-1 divide-y divide-slate-100">
              <Row label="Name" value={info.name} />
              <Row label="Approach" value={info.family} />
              {info.original_author && <Row label="Original implementation" value={info.original_author} />}
              <Row label="Training data" value={`${info.dataset_version ?? "Not available"}${info.trained_on ? ` (${info.trained_on.join(" + ")})` : ""}`} />
              {info.details.map((d) => (
                <Row key={d.label} label={d.label} value={d.value} />
              ))}
              <Row label="Intent classes" value={info.n_classes} />
              <Row label="Default threshold" value={thresholdPct(info.default_threshold)} />
            </dl>
            <p className="mt-2 font-mono text-[10px] break-all text-slate-400" title={info.model_sha256}>
              {info.model_file} · sha256 {info.model_sha256.slice(0, 12)}…
            </p>
          </div>
          <div>
            <p className="eyebrow">Evaluation (held-out test set, n = {ev.test_n ?? "?"})</p>
            <dl className="mt-1 divide-y divide-slate-100">
              <Row label="Raw accuracy (no threshold)" value={pct(ev.test_accuracy)} />
              <Row label="Macro F1" value={num(ev.test_macro_f1)} />
              {ci && <Row label="Macro F1 95% CI" value={`${num(ci[0])}–${num(ci[1])}`} />}
              <Row label="Weighted F1" value={num(ev.test_weighted_f1)} />
              <Row label="Macro ROC-AUC" value={num(ev.test_macro_roc_auc)} />
              <Row label="Top-2 accuracy" value={pct(ev.test_top2_accuracy)} />
              <Row label="Validation macro F1" value={num(ev.validation_macro_f1)} />
              <Row label="Calibration error (ECE)" value={num(ev.expected_calibration_error)} />
            </dl>
          </div>
          <div>
            <p className="eyebrow">At default threshold (evaluation)</p>
            <dl className="mt-1 divide-y divide-slate-100">
              <Row label="Known commands accepted" value={pct(ev.known_intent_coverage_at_default)} />
              <Row label="Unknown commands rejected" value={pct(ev.unknown_rejection_rate_at_default)} />
              {ev.unknown_safe_fallback_at_default != null && (
                <Row label="Unknown: safe fallback" value={pct(ev.unknown_safe_fallback_at_default)} />
              )}
              <Row label="Latency P50 (benchmark)" value={ms(ev.median_latency_ms)} />
              {ev.p95_latency_ms != null && <Row label="Latency P95 (benchmark)" value={ms(ev.p95_latency_ms)} />}
              <Row label="Model size" value={ev.model_size_mb == null ? "Not available" : `${ev.model_size_mb.toFixed(2)} MB`} />
            </dl>
            <p className="eyebrow mt-4">This demo session</p>
            <dl className="mt-1 divide-y divide-slate-100">
              <Row label="Commands tested" value={session.tested} />
              <Row label="Last inference (incl. explanation)" value={ms(session.lastInferenceMs)} />
            </dl>
          </div>
        </div>
      )}
      {info && (
        <p className="mt-4 text-xs leading-relaxed text-slate-500">
          Evaluation metrics are read from <code>{ev?.source}</code>. They are not computed by this demo. Session figures
          describe only the commands typed here and are not a measure of accuracy.{" "}
          {!info.unknown_label_is_model_class && (
            <>
              <code>UNKNOWN_INTENT</code> is not one of the model's classes. It is produced only by the confidence threshold.
            </>
          )}
        </p>
      )}
    </section>
  );
}
