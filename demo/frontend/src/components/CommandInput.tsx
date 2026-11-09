import { useState } from "react";

export const MAX_INPUT = 500;

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSubmit: (text: string) => void;
  loading: boolean;
  disabled: boolean;
}

export function CommandInput({ value, onChange, onSubmit, loading, disabled }: Props) {
  const [hint, setHint] = useState<string | null>(null);
  const blank = value.trim().length === 0;

  function submit() {
    if (blank) {
      setHint("Type a command first — for example “Send $50 to Jone”.");
      return;
    }
    setHint(null);
    onSubmit(value);
  }

  return (
    <section aria-labelledby="try-title" className="card p-5 sm:p-6">
      <div className="mb-4 flex items-baseline justify-between gap-3">
        <h2 id="try-title" className="text-xl font-semibold tracking-tight text-navy-900">
          Try Sole Smart Actions
        </h2>
        <span className="hidden text-xs text-slate-500 sm:inline">Press Enter to analyze</span>
      </div>
      <form
        className="flex flex-col gap-3 sm:flex-row"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <label htmlFor="command" className="sr-only">
          Customer command
        </label>
        <input
          id="command"
          type="text"
          value={value}
          maxLength={MAX_INPUT}
          autoComplete="off"
          placeholder="Try typing: Send $50 to Jone"
          onChange={(e) => {
            onChange(e.target.value);
            if (hint) setHint(null);
          }}
          aria-describedby="command-help"
          className="min-w-0 flex-1 rounded-xl border border-slate-300 bg-white px-4 py-3 text-base text-navy-900 shadow-inner shadow-slate-100 transition placeholder:text-slate-400 focus:border-teal-600 focus:ring-4 focus:ring-teal-600/15 focus:outline-none"
        />
        <button
          type="submit"
          disabled={loading || disabled}
          className="inline-flex items-center justify-center gap-2 rounded-xl bg-navy-900 px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-navy-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading && (
            <span aria-hidden className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
          )}
          {loading ? "Analyzing…" : "Analyze Command"}
        </button>
      </form>
      <div id="command-help" className="mt-2 flex min-h-5 justify-between gap-3 text-xs">
        <span role={hint ? "alert" : undefined} className={hint ? "font-medium text-amber-700" : "text-slate-500"}>
          {hint ?? "Commands are classified only — no payment is ever made."}
        </span>
        <span className={value.length >= MAX_INPUT ? "font-medium text-amber-700" : "text-slate-400"}>
          {value.length}/{MAX_INPUT}
        </span>
      </div>
    </section>
  );
}
