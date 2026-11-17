import { intentLabel, pct, thresholdPct } from "../lib/format";
import type { Decision, Prediction } from "../types";

interface Props {
  prediction: Prediction;
  decision: Decision;
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <li className="group relative flex gap-3 pb-5 last:pb-0">
      <span aria-hidden className="absolute top-8 bottom-0 left-[13px] w-px bg-slate-200 group-last:hidden" />
      <span className="z-10 grid h-7 w-7 shrink-0 place-items-center rounded-full bg-navy-900 text-xs font-semibold text-white">
        {n}
      </span>
      <div className="min-w-0 flex-1 pt-0.5">
        <p className="text-sm font-semibold text-navy-900">{title}</p>
        <div className="mt-1.5 text-sm text-slate-600">{children}</div>
      </div>
    </li>
  );
}

export function Interpretation({ prediction, decision }: Props) {
  const e = prediction.explanation;
  const linear = !e.kind || e.kind === "bow_linear";
  const maxAbs = Math.max(0.01, ...e.contributions.map((c) => Math.abs(c.contribution)));
  const MODEL_STEP: Record<string, { tokens: string; score: string; scoreText: string }> = {
    tfidf: { tokens: "TF-IDF builds word and character n-gram features", score: "Logistic Regression scores every intent",
      scoreText: "Each TF-IDF feature (words and 2-5 letter fragments) adds its learned weight to each intent's score; softmax turns scores into probabilities." },
    transformer: { tokens: "Tokenizer maps words to learned embeddings", score: "Self-attention encoder scores every intent",
      scoreText: "One Transformer block lets every word attend to every other word, the vectors are averaged and a small dense head outputs nine probabilities." },
    distilbert: { tokens: "WordPiece splits the text into sub-words", score: "Fine-tuned DistilBERT scores every intent",
      scoreText: "Six pretrained self-attention layers read the whole command; a classification head on the [CLS] vector outputs nine probabilities." },
  };
  const ms = e.kind ? MODEL_STEP[e.kind] : undefined;
  return (
    <section aria-labelledby="interp-title" className="card p-5 sm:p-6">
      <h2 id="interp-title" className="card-title text-lg">
        How the Model Interpreted This Command
      </h2>
      <p className="mt-0.5 text-xs text-slate-500">
        {linear
          ? "Computed directly from the trained vectorizer and Logistic Regression weights. No language model is involved."
          : `Shown from ${prediction.model_name}'s own tokenizer. Neural models have no exact per-word attribution, so none is invented here.`}
      </p>

      <ol className="mt-5">
        <Step n={1} title="Customer enters a command">
          <span className="font-medium text-navy-900">“{prediction.input}”</span>
        </Step>
        <Step n={2} title="Same preprocessing as training">
          {linear ? "Unicode normalization, lowercase, whitespace cleanup:" : "Preprocessed text:"}{" "}
          <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs text-navy-900">{e.normalized_text || "(empty)"}</code>
        </Step>
        <Step n={3} title={ms ? ms.tokens : "CountVectorizer builds a Bag of Words"}>
          <p>
            {ms
              ? e.description
              : (e.ngram_range?.[1] ?? 1) > 1
              ? "Word order is mostly ignored: the model counts single words and two-word phrases seen during training."
              : "Word order is ignored. Only words seen during training count."}
          </p>
          <div className="mt-2 space-y-2">
            <div>
              <p className="eyebrow">Recognized words ({e.recognized_tokens.length})</p>
              <div className="mt-1 flex flex-wrap gap-1.5" data-testid="recognized">
                {e.recognized_tokens.length === 0 && <span className="text-xs text-slate-400">None — the model has no words to go on</span>}
                {e.recognized_tokens.map((t, i) => (
                  <span key={`${t}-${i}`} className="rounded-md bg-teal-50 px-2 py-0.5 font-mono text-xs font-medium text-teal-800 ring-1 ring-teal-600/20">
                    {t}
                  </span>
                ))}
              </div>
            </div>
            {(e.recognized_phrases?.length ?? 0) > 0 && (
              <div>
                <p className="eyebrow">Recognized two-word phrases ({e.recognized_phrases!.length})</p>
                <div className="mt-1 flex flex-wrap gap-1.5" data-testid="phrases">
                  {e.recognized_phrases!.map((t, i) => (
                    <span key={`${t}-${i}`} className="rounded-md bg-sky-50 px-2 py-0.5 font-mono text-xs font-medium text-sky-800 ring-1 ring-sky-600/20">
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            )}
            <div>
              <p className="eyebrow">Unknown words — not in the {""}model vocabulary ({e.unknown_tokens.length})</p>
              <div className="mt-1 flex flex-wrap gap-1.5" data-testid="unknown">
                {e.unknown_tokens.length === 0 && <span className="text-xs text-slate-400">None</span>}
                {e.unknown_tokens.map((t, i) => (
                  <span key={`${t}-${i}`} className="rounded-md bg-amber-50 px-2 py-0.5 font-mono text-xs font-medium text-amber-800 line-through decoration-amber-500/60 ring-1 ring-amber-600/25">
                    {t}
                  </span>
                ))}
              </div>
              {e.unknown_tokens.length > 0 && (
                <p className="mt-1 text-xs text-slate-500">{linear ? "Unknown words are dropped. They cannot influence the prediction." : "How this model handles unknown words is described above."}</p>
              )}
            </div>
          </div>
        </Step>
        {ms ? (
          <Step n={4} title={ms.score}>
            <p>
              {ms.scoreText} Top intent: <span className="intent-chip">{decision.raw_prediction}</span> at {pct(decision.confidence)}.
            </p>
          </Step>
        ) : (
          <Step n={4} title="Logistic Regression scores every intent">
            <p>
              Each recognized word adds its learned weight to each intent's score. The scores become probabilities (softmax).
              Top intent: <span className="intent-chip">{decision.raw_prediction}</span> at {pct(decision.confidence)}.
            </p>
            <div className="mt-3 overflow-hidden rounded-xl border border-slate-200">
              <p className="border-b border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-600">
                Word contributions to the {intentLabel(e.explained_class ?? decision.raw_prediction)} score
              </p>
              <ul className="divide-y divide-slate-100" data-testid="contributions">
                {e.contributions.length === 0 && (
                  <li className="px-3 py-2 text-xs text-slate-500">No recognized words. The score comes from the bias term alone.</li>
                )}
                {e.contributions.map((c) => {
                  const w = (Math.abs(c.contribution) / maxAbs) * 50;
                  const positive = c.contribution >= 0;
                  return (
                    <li key={c.feature} className="grid grid-cols-[minmax(4.5rem,7rem)_1fr_4.5rem] items-center gap-2 px-3 py-1.5">
                      <span className="truncate font-mono text-xs text-navy-900">
                        {c.feature}
                        {c.count > 1 && <span className="text-slate-400"> ×{c.count}</span>}
                      </span>
                      <span className="relative h-3" aria-hidden>
                        <span className="absolute top-0 bottom-0 left-1/2 w-px bg-slate-300" />
                        <span
                          className={`absolute top-0.5 bottom-0.5 rounded-sm ${positive ? "bg-teal-600" : "bg-amber-500"}`}
                          style={positive ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }}
                        />
                      </span>
                      <span className={`text-right font-mono text-xs tabular-nums ${positive ? "text-teal-800" : "text-amber-800"}`}>
                        {c.contribution >= 0 ? "+" : ""}
                        {c.contribution.toFixed(3)}
                      </span>
                    </li>
                  );
                })}
              </ul>
              <p className="border-t border-slate-200 bg-slate-50 px-3 py-2 text-[11px] leading-relaxed text-slate-500">
                Score for {e.explained_class} = bias {(e.intercept ?? 0).toFixed(3)} + word contributions ={" "}
                <span className="font-semibold text-slate-700">{(e.class_score ?? 0).toFixed(3)}</span>. Runner-up intent:{" "}
                {e.runner_up_class}. Contributions show how each word moves this intent's score. They show
                association, not causation.
              </p>
            </div>
            {e.top_class_features.length > 0 && (
              <p className="mt-3 text-xs text-slate-500">
                Words the model most strongly associates with {intentLabel(e.explained_class ?? decision.raw_prediction)}:{" "}
                {e.top_class_features.map((f) => (
                  <span key={f.feature} className="mr-1 inline-block rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[11px] text-slate-700">
                    {f.feature}
                  </span>
                ))}
              </p>
            )}
          </Step>
        )}
        <Step n={5} title="Threshold and fallback rules decide the final intent">
          {decision.fallback_applied ? (
            <>
              {pct(decision.confidence)} is below the {thresholdPct(decision.threshold)} threshold, so the final intent is{" "}
              <span className="intent-chip">UNKNOWN_INTENT</span> and the app shows the help / clarify fallback.
            </>
          ) : (
            <>
              {pct(decision.confidence)} meets the {thresholdPct(decision.threshold)} threshold, so the final intent is{" "}
              <span className="intent-chip">{decision.final_intent}</span>.
            </>
          )}
        </Step>
      </ol>
    </section>
  );
}
