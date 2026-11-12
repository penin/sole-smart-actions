import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CommandInput } from "./components/CommandInput";
import { ConfidenceChart } from "./components/ConfidenceChart";
import { Header } from "./components/Header";
import { Interpretation } from "./components/Interpretation";
import { ModelInfoPanel } from "./components/ModelInfoPanel";
import { RecentPredictions } from "./components/RecentPredictions";
import { ResultsPanel } from "./components/ResultsPanel";
import { SampleCommands } from "./components/SampleCommands";
import { ThresholdControl, type Verification } from "./components/ThresholdControl";
import { api, ApiError } from "./lib/api";
import { applyThreshold } from "./lib/threshold";
import type { Decision, HistoryEntry, ModelInfo, ModelSummary, Prediction } from "./types";

const DEFAULT_MODEL = "bow_logreg";
const HISTORY_LIMIT = 10;

export default function App() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [modelId, setModelId] = useState(DEFAULT_MODEL);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);

  const [threshold, setThreshold] = useState(0);
  const [input, setInput] = useState("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [tested, setTested] = useState(0);
  const [verification, setVerification] = useState<Verification>("idle");
  const [backendDecision, setBackendDecision] = useState<Decision | null>(null);
  const requestId = useRef(0);

  useEffect(() => {
    api.models().then(setModels).catch(() => setModels([]));
  }, []);

  useEffect(() => {
    let cancelled = false;
    setInfo(null);
    setInfoError(null);
    api
      .modelInfo(modelId)
      .then((i) => {
        if (cancelled) return;
        setInfo(i);
        setThreshold(i.default_threshold);
      })
      .catch((e: unknown) => !cancelled && setInfoError(e instanceof ApiError ? e.message : "Model information unavailable"));
    return () => {
      cancelled = true;
    };
  }, [modelId]);

  // Instant re-decision from stored probabilities: no inference, no reload.
  const localDecision = useMemo(
    () => (prediction ? applyThreshold(prediction.probabilities, threshold, prediction.class_order) : null),
    [prediction, threshold],
  );

  // Backend remains authoritative: confirm the decision (debounced). On disagreement, show the backend result.
  useEffect(() => {
    if (!prediction || !localDecision) {
      setVerification("idle");
      setBackendDecision(null);
      return;
    }
    setVerification("pending");
    const handle = setTimeout(() => {
      api
        .decide(prediction.probabilities, threshold, prediction.model_id)
        .then((d) => {
          const same = d.final_intent === localDecision.final_intent && d.status === localDecision.status;
          setBackendDecision(same ? null : d);
          setVerification(same ? "verified" : "mismatch");
        })
        .catch(() => setVerification("error"));
    }, 250);
    return () => clearTimeout(handle);
  }, [prediction, threshold, localDecision]);

  const decision = backendDecision ?? localDecision;

  const analyze = useCallback(
    async (text: string, thresholdOverride?: number) => {
      if (!text.trim()) return; // never send blank commands
      const id = ++requestId.current;
      setLoading(true);
      setError(null);
      try {
        const p = await api.predict(text, thresholdOverride ?? threshold, modelId);
        if (id !== requestId.current) return; // a newer request superseded this one
        setPrediction(p);
        setBackendDecision(null);
        setTested((n) => n + 1);
        const key = `${p.model_id}::${p.input}`;
        setActiveKey(key);
        setHistory((h) =>
          [{ key, prediction: p, thresholdUsed: p.threshold, finalIntentAtTime: p.final_intent }, ...h.filter((e) => e.key !== key)].slice(
            0,
            HISTORY_LIMIT,
          ),
        );
      } catch (e) {
        if (id !== requestId.current) return;
        setError(e instanceof ApiError ? e.message : "Unexpected error while classifying the command.");
      } finally {
        if (id === requestId.current) setLoading(false);
      }
    },
    [threshold, modelId],
  );

  // Optional deep link for presentations: ?q=<command>&t=<threshold percent>
  const deepLinked = useRef(false);
  useEffect(() => {
    if (!info || deepLinked.current) return;
    deepLinked.current = true;
    const params = new URLSearchParams(window.location.search);
    const q = params.get("q");
    const t = params.get("t");
    const tNum = t === null ? NaN : Number(t);
    const thr = Number.isFinite(tNum) && tNum >= 0 && tNum <= 100 ? tNum / 100 : info.default_threshold;
    setThreshold(thr);
    if (q && q.trim()) {
      setInput(q.slice(0, 500));
      void analyze(q.slice(0, 500), thr);
    }
  }, [info, analyze]);

  // Switching models clears the current result: its probabilities belong to the previous model.
  const selectModel = (id: string) => {
    if (id === modelId) return;
    requestId.current++;
    setLoading(false);
    setError(null);
    setPrediction(null);
    setBackendDecision(null);
    setActiveKey(null);
    setModelId(id);
  };

  const pickSample = (cmd: string) => {
    setInput(cmd);
    void analyze(cmd);
  };

  const restore = (entry: HistoryEntry) => {
    requestId.current++; // ignore any in-flight request
    setLoading(false);
    setError(null);
    setInput(entry.prediction.input);
    setPrediction(entry.prediction); // original probabilities; current threshold is re-applied
    setBackendDecision(null);
    setActiveKey(entry.key);
  };

  const modelReady = info !== null;

  return (
    <div className="flex min-h-screen flex-col">
      <Header models={models} selectedModel={modelId} onSelectModel={selectModel} />

      <div className="border-b border-slate-200 bg-white">
        <p className="mx-auto max-w-7xl px-4 py-2.5 text-xs text-slate-600 sm:px-6 lg:px-8">
          <span className="font-semibold text-navy-900">{info?.name ?? "Bag of Words + Logistic Regression"}</span> · Baseline
          model trained on synthetic Sole wallet commands
          {info?.dataset_version ? ` (dataset ${info.dataset_version})` : ""} · Every result below is computed live by
          the trained model
        </p>
      </div>

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 lg:px-8">
        {infoError && (
          <div role="alert" className="mb-6 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
            <p className="font-semibold">The model is not available</p>
            <p className="mt-1">{infoError}</p>
          </div>
        )}
        {/* Mobile: single column using `order`; desktop: main column + sticky side column. */}
        <div className="flex flex-col gap-6 lg:grid lg:grid-cols-12 lg:items-start">
          <div className="contents lg:col-span-8 lg:flex lg:flex-col lg:gap-6">
            <div className="order-1 min-w-0 lg:order-none">
              <CommandInput value={input} onChange={setInput} onSubmit={analyze} loading={loading} disabled={!modelReady} />
            </div>
            <div className="order-2 min-w-0 lg:order-none">
              <SampleCommands
                modelClasses={info?.classes ?? []}
                onPick={pickSample}
                activeCommand={prediction?.input ?? null}
                disabled={!modelReady}
              />
            </div>
            <div className="order-4 min-w-0 lg:order-none">
              <ResultsPanel prediction={prediction} decision={decision} loading={loading} error={error} />
            </div>
            {prediction && decision && (
              <>
                <div className="order-5 min-w-0 lg:order-none">
                  <ConfidenceChart
                    prediction={prediction}
                    decision={decision}
                    calibration={info ? { meanConfidence: info.evaluation.test_mean_confidence, accuracy: info.evaluation.test_accuracy } : null}
                  />
                </div>
                <div className="order-6 min-w-0 lg:order-none">
                  <Interpretation prediction={prediction} decision={decision} />
                </div>
              </>
            )}
            <div className="order-8 min-w-0 lg:order-none">
              <ModelInfoPanel info={info} error={infoError} session={{ tested, lastInferenceMs: prediction?.inference_ms ?? null }} />
            </div>
          </div>

          <aside className="contents lg:sticky lg:top-6 lg:col-span-4 lg:flex lg:max-h-[calc(100vh-3rem)] lg:flex-col lg:gap-6 lg:overflow-y-auto lg:pb-1">
            <div className="order-3 min-w-0 lg:order-none">
              <ThresholdControl
                threshold={threshold}
                defaultThreshold={info?.default_threshold ?? null}
                defaultSource={info?.default_threshold_source ?? null}
                onChange={setThreshold}
                onReset={() => info && setThreshold(info.default_threshold)}
                verification={verification}
              />
            </div>
            <div className="order-7 min-w-0 lg:order-none">
              <RecentPredictions history={history} activeKey={activeKey} onRestore={restore} onClear={() => setHistory([])} />
            </div>
          </aside>
        </div>
      </main>

      <footer className="mt-8 bg-navy-900 py-5 text-center text-xs text-slate-300">
        MIT AI Edge | Sole Smart Actions | Bag of Words Baseline
      </footer>
    </div>
  );
}
