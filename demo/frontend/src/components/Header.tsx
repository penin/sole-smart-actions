import type { ModelSummary } from "../types";

interface Props {
  models: ModelSummary[];
  selectedModel: string;
  onSelectModel: (id: string) => void;
}

export function Header({ models, selectedModel, onSelectModel }: Props) {
  const current = models.find((m) => m.id === selectedModel);
  return (
    <header className="bg-navy-900 text-white">
      <div className="mx-auto flex max-w-7xl flex-col gap-5 px-4 py-6 sm:px-6 md:flex-row md:items-center md:justify-between lg:px-8">
        <div className="flex items-center gap-4">
          <div aria-hidden className="grid h-12 w-12 shrink-0 place-items-center rounded-xl bg-white/10 ring-1 ring-white/15">
            <svg viewBox="0 0 32 32" className="h-8 w-8">
              <path
                d="M21 11.5c-1-1.3-2.7-2-4.6-2-2.8 0-4.6 1.5-4.6 3.5 0 4.6 9.6 2.6 9.6 7.2 0 2-2 3.6-5 3.6-2.1 0-4-.8-5.1-2.3"
                fill="none"
                stroke="#2dd4bf"
                strokeWidth="2.6"
                strokeLinecap="round"
              />
            </svg>
          </div>
          <div>
            <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Sole Smart Actions</h1>
            <p className="text-sm text-slate-300">Natural Language Intent Classification</p>
          </div>
        </div>

        <div className="flex flex-col gap-1.5 md:items-end">
          <label htmlFor="model-select" className="text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-400">
            Classifier
          </label>
          <div className="relative">
            <select
              id="model-select"
              value={selectedModel}
              onChange={(e) => onSelectModel(e.target.value)}
              className="w-full appearance-none rounded-lg border border-white/15 bg-navy-800 py-2 pr-9 pl-3 text-sm font-medium text-white md:w-[31rem]"
            >
              {models.length === 0 && <option value={selectedModel}>Bag of Words + Logistic Regression</option>}
              {models.map((m) => (
                <option key={m.id} value={m.id} disabled={!m.available}>
                  {m.name} — {m.available ? "Available" : m.status === "error" ? "Failed to load" : "Coming Soon"}
                </option>
              ))}
            </select>
            <svg aria-hidden viewBox="0 0 20 20" className="pointer-events-none absolute top-1/2 right-3 h-4 w-4 -translate-y-1/2 text-slate-400">
              <path d="M5 7l5 5 5-5" fill="none" stroke="currentColor" strokeWidth="2" />
            </svg>
          </div>
          <p className="text-xs text-slate-400">
            {current?.available ? `Live model: ${current.name}` : "Model status loading…"}
          </p>
        </div>
      </div>
    </header>
  );
}
