import { useState } from "react";
import samples from "../data/sampleCommands.json";
import type { SampleCategory } from "../types";

interface Props {
  modelClasses: string[];
  onPick: (command: string) => void;
  activeCommand: string | null;
  disabled: boolean;
}

const CATEGORIES = samples.categories as SampleCategory[];

export function SampleCommands({ modelClasses, onPick, activeCommand, disabled }: Props) {
  const [tab, setTab] = useState(CATEGORIES[0].id);
  const cat = CATEGORIES.find((c) => c.id === tab) ?? CATEGORIES[0];
  const hintInModel = cat.intent_hint === null || modelClasses.length === 0 || modelClasses.includes(cat.intent_hint);

  return (
    <section aria-labelledby="samples-title" className="card p-5 sm:p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="samples-title" className="card-title text-lg">
          Try Sample Commands
        </h2>
        <p className="text-xs text-slate-500">Click any command to run it through the live model</p>
      </div>

      <div role="tablist" aria-label="Sample command categories" className="-mx-1 mt-4 flex gap-1.5 overflow-x-auto px-1 pb-1 sm:flex-wrap sm:overflow-visible">
        {CATEGORIES.map((c) => (
          <button
            key={c.id}
            role="tab"
            type="button"
            id={`tab-${c.id}`}
            aria-selected={c.id === tab}
            aria-controls={`panel-${c.id}`}
            onClick={() => setTab(c.id)}
            className={`shrink-0 rounded-full px-3.5 py-1.5 text-sm font-medium whitespace-nowrap transition ${
              c.id === tab ? "bg-navy-900 text-white shadow-sm" : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            }`}
          >
            {c.label}
          </button>
        ))}
      </div>

      <div role="tabpanel" id={`panel-${cat.id}`} aria-labelledby={`tab-${cat.id}`} className="mt-4">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {cat.commands.map((cmd) => (
            <button
              key={cmd}
              type="button"
              disabled={disabled}
              onClick={() => onPick(cmd)}
              className={`group flex items-center justify-between gap-2 rounded-xl border px-3.5 py-2.5 text-left text-sm transition disabled:cursor-not-allowed disabled:opacity-60 ${
                activeCommand === cmd
                  ? "border-teal-600 bg-teal-50 text-navy-900 ring-2 ring-teal-600/15"
                  : "border-slate-200 bg-white text-slate-700 hover:border-teal-500 hover:bg-teal-50/40"
              }`}
            >
              <span>“{cmd}”</span>
              <span aria-hidden className="text-slate-300 transition group-hover:translate-x-0.5 group-hover:text-teal-600">
                →
              </span>
            </button>
          ))}
        </div>
        <p className="mt-3 text-xs text-slate-500">
          {cat.intent_hint === null ? (
            <>These commands are deliberately vague. They have no single correct intent.</>
          ) : hintInModel ? (
            <>
              Written as <span className="intent-chip">{cat.intent_hint}</span> requests. The result shown is whatever
              the model actually predicts, including mistakes.
            </>
          ) : (
            <>
              <span className="intent-chip">{cat.intent_hint}</span> is <strong>not a class in the loaded model</strong>.
              The model can only choose among its {modelClasses.length} trained intents; unsupported requests can only be
              caught by the confidence threshold.
            </>
          )}
        </p>
      </div>
    </section>
  );
}
