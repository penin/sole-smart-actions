import { useEffect, useState } from "react";
import { intentLabel } from "../lib/format";
import type { Workflow } from "../types";

interface Props {
  workflow: Workflow;
  compact?: boolean;
}

/** Illustrative Sole workflow. Simulation only: confirming never moves money and no PIN is ever requested. */
export function WorkflowPanel({ workflow, compact = false }: Props) {
  const [confirmed, setConfirmed] = useState<"none" | "confirmed" | "cancelled">("none");
  useEffect(() => setConfirmed("none"), [workflow.message, workflow.kind]);

  const tone =
    workflow.kind === "clarify"
      ? "border-amber-200 bg-amber-50/60"
      : workflow.kind === "confirm"
        ? "border-sky-200 bg-sky-50/60"
        : "border-teal-200 bg-teal-50/50";

  return (
    <div className={`rounded-xl border p-3 text-sm ${tone}`} data-testid="workflow">
      <p className="eyebrow">
        Illustrative Sole workflow {workflow.screen ? `· ${workflow.screen}` : "· clarification"}
      </p>
      <p className="mt-1 font-medium text-navy-900" data-testid="workflow-message">
        {workflow.message}
      </p>

      {workflow.kind === "clarify" && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {workflow.options.map((o) => (
            <span key={o} className="rounded-full bg-white px-2.5 py-1 text-xs font-medium text-slate-700 ring-1 ring-slate-300">
              {o.includes(" ") ? o : intentLabel(o)}
            </span>
          ))}
        </div>
      )}

      {workflow.kind === "confirm" && (
        <div className="mt-2">
          {confirmed === "none" ? (
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => setConfirmed("confirmed")}
                className="rounded-lg bg-navy-900 px-3 py-1.5 text-xs font-semibold text-white hover:bg-navy-700"
              >
                Confirm (simulated)
              </button>
              <button
                type="button"
                onClick={() => setConfirmed("cancelled")}
                className="rounded-lg bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 ring-1 ring-slate-300 hover:bg-slate-50"
              >
                Cancel
              </button>
            </div>
          ) : (
            <p role="status" className="text-xs font-medium text-slate-700">
              {confirmed === "confirmed"
                ? "Simulated confirmation recorded. No money was moved: this demo never executes transactions."
                : "Cancelled. Nothing happened."}
            </p>
          )}
        </div>
      )}

      {!compact && (
        <ol className="mt-3 list-decimal space-y-0.5 pl-5 text-xs text-slate-600">
          {workflow.steps.map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ol>
      )}
      <p className="mt-2 text-[11px] text-slate-500">{workflow.note}</p>
    </div>
  );
}
