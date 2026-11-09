import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import samples from "../data/sampleCommands.json";
import { applyThreshold } from "../lib/threshold";
import type { Prediction } from "../types";
import fixture from "./fixtures/threshold_cases.json";

// Mock backend that serves the REAL model probabilities captured in the generated fixture.
const ORDER = fixture.class_order;
const REAL = fixture.sample_probabilities as Record<string, Record<string, number>>;
const DEFAULT = fixture.default_threshold;

type Call = { url: string; body: Record<string, unknown> | null };
let calls: Call[] = [];
let failPredict: { status: number; detail: string } | "network" | null = null;
let decideOverride: ((b: Record<string, unknown>) => unknown) | null = null;

function json(status: number, body: unknown) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
}

function predictionFor(text: string, threshold: number): Prediction {
  const probabilities = REAL[text] ?? Object.fromEntries(ORDER.map((c) => [c, 1 / ORDER.length]));
  const d = applyThreshold(probabilities, threshold, ORDER);
  return {
    ...d,
    model_id: "bow_logreg",
    model_name: "Bag of Words + Logistic Regression",
    input: text,
    class_order: ORDER,
    probabilities,
    ranked: [...ORDER].map((intent) => ({ intent, probability: probabilities[intent] })).sort((a, b) => b.probability - a.probability),
    default_threshold: DEFAULT,
    recognized_tokens: ["send"],
    unknown_tokens: ["zzqx"],
    explanation: {
      normalized_text: text.toLowerCase(),
      tokens: ["send", "zzqx"],
      recognized_tokens: ["send"],
      unknown_tokens: ["zzqx"],
      explained_class: d.raw_prediction,
      runner_up_class: ORDER[1],
      intercept: 0.1,
      class_score: 0.6,
      contributions: [{ feature: "send", count: 1, weight: 0.5, contribution: 0.5, contribution_vs_runner_up: 0.4 }],
      top_class_features: [{ feature: "send", weight: 0.5 }],
    },
    inference_ms: 0.4,
  };
}

const INFO = {
  id: "bow_logreg",
  name: "Bag of Words + Logistic Regression",
  family: "BoW",
  version: "1.0.0",
  model_file: "models/bow_logreg_v1_tuned_v1.0.0.joblib",
  model_sha256: "abc123abc123abc123",
  pipeline_steps: [],
  vectorizer: { type: "CountVectorizer", ngram_range: [1, 1], binary: false, token_pattern: "x", vocabulary_size: 250 },
  classifier: { type: "LogisticRegression", C: 0.1, n_parameters: 2259 },
  classes: ORDER,
  n_classes: ORDER.length,
  unknown_label: "UNKNOWN_INTENT",
  unknown_label_is_model_class: false,
  default_threshold: DEFAULT,
  default_threshold_source: "model bundle",
  fallback_rule: "",
  evaluation: {
    source: "results/metrics",
    test_n: 270,
    test_accuracy: 0.7926,
    test_macro_f1: 0.7963,
    test_weighted_f1: 0.7963,
    test_macro_precision: null,
    test_macro_recall: null,
    test_macro_roc_auc: 0.954,
    test_log_loss: null,
    test_top2_accuracy: null,
    test_mean_confidence: 0.49,
    test_macro_f1_ci95: null,
    validation_macro_f1: 0.735,
    expected_calibration_error: null,
    known_intent_coverage_at_default: 0.907,
    unknown_rejection_rate_at_default: 0.147,
    median_latency_ms: null,
    training_seconds: null,
    model_size_mb: null,
  },
};

const MODELS = [
  { id: "bow_logreg", name: "Bag of Words + Logistic Regression", available: true, status: "available" },
  { id: "tfidf_logreg", name: "TF-IDF + Logistic Regression", available: false, status: "coming_soon" },
  { id: "custom_transformer", name: "Custom Transformer", available: false, status: "coming_soon" },
  { id: "distilbert", name: "DistilBERT", available: false, status: "coming_soon" },
];

beforeEach(() => {
  calls = [];
  failPredict = null;
  decideOverride = null;
  vi.stubGlobal("fetch", (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const body = init?.body ? JSON.parse(String(init.body)) : null;
    calls.push({ url, body });
    if (url.startsWith("/api/models")) return json(200, MODELS);
    if (url.startsWith("/api/model-info")) return json(200, INFO);
    if (url.startsWith("/api/predict")) {
      if (failPredict === "network") return Promise.reject(new TypeError("Failed to fetch"));
      if (failPredict) return json(failPredict.status, { detail: failPredict.detail });
      return json(200, predictionFor(body.text, body.threshold));
    }
    if (url.startsWith("/api/decide")) {
      if (decideOverride) return json(200, decideOverride(body));
      return json(200, applyThreshold(body.probabilities, body.threshold, ORDER));
    }
    return json(404, { detail: "not found" });
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const predictCalls = () => calls.filter((c) => c.url.startsWith("/api/predict"));
const slider = () => screen.getByRole("slider", { name: /confidence threshold/i }) as HTMLInputElement;

async function renderReady() {
  render(<App />);
  await waitFor(() => expect(screen.getByText("19.57%", { selector: "output" })).toBeInTheDocument());
}

async function runSample(user: ReturnType<typeof userEvent.setup>, category: string, command: string) {
  await user.click(screen.getByRole("tab", { name: category }));
  await user.click(screen.getByRole("button", { name: `“${command}”` }));
  const results = screen.getByRole("region", { name: "Prediction Results" });
  await waitFor(() => expect(within(results).getByText(`“${command}”`)).toBeInTheDocument());
}

describe("layout and model selection", () => {
  it("shows title, subtitle, footer and the model dropdown with future models disabled", async () => {
    await renderReady();
    expect(screen.getByRole("heading", { name: "Sole Smart Actions" })).toBeInTheDocument();
    expect(screen.getByText("Natural Language Intent Classification")).toBeInTheDocument();
    expect(screen.getByText("MIT AI Edge | Sole Smart Actions | Bag of Words Baseline")).toBeInTheDocument();
    const select = screen.getByLabelText("Classifier");
    await waitFor(() => expect(within(select).getAllByRole("option")).toHaveLength(4));
    const options = within(select).getAllByRole("option") as HTMLOptionElement[];
    expect(options[0]).toHaveTextContent("Bag of Words + Logistic Regression — Available");
    expect(options[0].disabled).toBe(false);
    for (const o of options.slice(1)) {
      expect(o.disabled).toBe(true);
      expect(o).toHaveTextContent("Coming Soon");
    }
  });

  it("shows model information read from the API, with 'Not available' for missing metrics", async () => {
    await renderReady();
    expect(screen.getByText("79.3%")).toBeInTheDocument();
    expect(screen.getAllByText("Not available").length).toBeGreaterThan(0);
  });
});

describe("threshold slider", () => {
  it("spans 0–100% in 1% steps and loads the configured default", async () => {
    await renderReady();
    expect(slider().min).toBe("0");
    expect(slider().max).toBe("100");
    expect(slider().step).toBe("1");
  });

  it("re-decides without new inference, keeps raw prediction/probabilities, and resets", async () => {
    const user = userEvent.setup();
    await renderReady();
    await runSample(user, "Transfer Money", "Send $50 to Jone");
    const conf = REAL["Send $50 to Jone"].TRANSFER_MONEY;
    expect(screen.getByTestId("raw-intent")).toHaveTextContent("TRANSFER_MONEY");
    expect(screen.getByTestId("final-intent")).toHaveTextContent("TRANSFER_MONEY");
    expect(screen.getByTestId("status")).toHaveTextContent("Accepted");
    const before = predictCalls().length;

    fireEvent.change(slider(), { target: { value: "95" } });
    expect(screen.getByText("95%", { selector: "output" })).toBeInTheDocument();
    expect(screen.getByTestId("final-intent")).toHaveTextContent("UNKNOWN_INTENT");
    expect(screen.getByTestId("raw-intent")).toHaveTextContent("TRANSFER_MONEY");
    expect(screen.getByTestId("confidence")).toHaveTextContent(`${(conf * 100).toFixed(1)}%`);
    expect(screen.getByTestId("status")).toHaveTextContent("Below Threshold");
    expect(screen.getByTestId("status")).toHaveTextContent("Fallback Applied");

    fireEvent.change(slider(), { target: { value: "0" } });
    expect(screen.getByTestId("final-intent")).toHaveTextContent("TRANSFER_MONEY");
    expect(predictCalls().length).toBe(before); // no re-inference

    await user.click(screen.getByRole("button", { name: "Reset to Default" }));
    expect(screen.getByText("19.57%", { selector: "output" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Decision confirmed by backend")).toBeInTheDocument());
  });

  it("shows the backend's decision if it ever disagrees with the local one", async () => {
    const user = userEvent.setup();
    await renderReady();
    await runSample(user, "Transfer Money", "Send $50 to Jone");
    decideOverride = (b) => ({ ...applyThreshold(b.probabilities as Record<string, number>, 1, ORDER), threshold: b.threshold });
    fireEvent.change(slider(), { target: { value: "10" } });
    await waitFor(() => expect(screen.getByText(/Backend decision differs/)).toBeInTheDocument());
    expect(screen.getByTestId("final-intent")).toHaveTextContent("UNKNOWN_INTENT");
  });
});

describe("command input", () => {
  it("does not send blank commands", async () => {
    const user = userEvent.setup();
    await renderReady();
    await user.click(screen.getByRole("button", { name: "Analyze Command" }));
    await user.type(screen.getByLabelText("Customer command"), "   {Enter}");
    expect(predictCalls()).toHaveLength(0);
    expect(screen.getByRole("alert")).toHaveTextContent(/Type a command first/);
  });

  it("submits on Enter, keeps the input, and limits length", async () => {
    const user = userEvent.setup();
    await renderReady();
    const box = screen.getByLabelText("Customer command") as HTMLInputElement;
    expect(box.maxLength).toBe(500);
    await user.type(box, "Transfer 50 dollars to Jone{Enter}");
    await waitFor(() => expect(screen.getByTestId("raw-intent")).toBeInTheDocument());
    expect(predictCalls()[0].body).toMatchObject({ text: "Transfer 50 dollars to Jone", model_id: "bow_logreg" });
    expect(predictCalls()[0].body!.threshold).toBe(DEFAULT);
    expect(box.value).toBe("Transfer 50 dollars to Jone");
  });

  it("shows server errors and network failures gracefully", async () => {
    const user = userEvent.setup();
    await renderReady();
    failPredict = { status: 503, detail: "Model failed to load" };
    await user.type(screen.getByLabelText("Customer command"), "hello{Enter}");
    await waitFor(() => expect(screen.getByText("Model failed to load")).toBeInTheDocument());
    failPredict = "network";
    await user.click(screen.getByRole("button", { name: "Analyze Command" }));
    await waitFor(() => expect(screen.getByText(/Cannot reach the model server/)).toBeInTheDocument());
  });
});

describe("sample command library", () => {
  it("every sample command is clickable, fills the input and runs real-probability inference", async () => {
    const user = userEvent.setup();
    await renderReady();
    const box = screen.getByLabelText("Customer command") as HTMLInputElement;
    let n = 0;
    for (const cat of samples.categories) {
      for (const cmd of cat.commands) {
        await runSample(user, cat.label, cmd);
        n++;
        expect(box.value).toBe(cmd);
        expect(predictCalls().at(-1)!.body).toMatchObject({ text: cmd });
        const expected = applyThreshold(REAL[cmd], DEFAULT, ORDER);
        expect(screen.getByTestId("raw-intent")).toHaveTextContent(expected.raw_prediction);
        expect(screen.getByTestId("final-intent")).toHaveTextContent(expected.final_intent);
      }
    }
    expect(n).toBe(36);
    expect(predictCalls()).toHaveLength(36);
  }, 30000);

  it("flags UNKNOWN_INTENT as not a class of the loaded model", async () => {
    const user = userEvent.setup();
    await renderReady();
    await user.click(screen.getByRole("tab", { name: "Unknown / Out-of-Scope" }));
    expect(screen.getByText(/not a class in the loaded model/)).toBeInTheDocument();
  });

  it("preserves the current threshold when a sample is clicked", async () => {
    const user = userEvent.setup();
    await renderReady();
    fireEvent.change(slider(), { target: { value: "40" } });
    await runSample(user, "Ambiguous", "Payment");
    expect(predictCalls().at(-1)!.body!.threshold).toBe(0.4);
    expect(screen.getByText("40%", { selector: "output" })).toBeInTheDocument();
  });
});

describe("recent predictions", () => {
  it("records, restores with the current threshold, caps at 10, and clears", async () => {
    const user = userEvent.setup();
    await renderReady();
    const cmds = samples.categories.flatMap((c) => c.commands.map((cmd) => [c.label, cmd] as const)).slice(0, 12);
    for (const [cat, cmd] of cmds) await runSample(user, cat, cmd);
    const list = screen.getByRole("list", { name: "Recent predictions" });
    expect(within(list).getAllByRole("button")).toHaveLength(10);

    const target = cmds[11][1];
    await user.click(screen.getByRole("tab", { name: "Ambiguous" }));
    const before = predictCalls().length;
    fireEvent.change(slider(), { target: { value: "100" } });
    await user.click(within(list).getByText(`“${cmds[10][1]}”`));
    expect((screen.getByLabelText("Customer command") as HTMLInputElement).value).toBe(cmds[10][1]);
    expect(screen.getByTestId("final-intent")).toHaveTextContent("UNKNOWN_INTENT"); // current threshold applied
    expect(screen.getByTestId("raw-intent")).toHaveTextContent(applyThreshold(REAL[cmds[10][1]], 0, ORDER).raw_prediction);
    expect(predictCalls().length).toBe(before); // restored without inference
    expect(within(list).getByText(`“${target}”`)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Clear History" }));
    expect(screen.queryByRole("list", { name: "Recent predictions" })).not.toBeInTheDocument();
    await act(async () => {});
  }, 30000);
});
