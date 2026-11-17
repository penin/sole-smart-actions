import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import samples from "../data/sampleCommands.json";
import { thresholdPct } from "../lib/format";
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
    model_id: "bow",
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
  id: "bow",
  name: "Bag of Words + Logistic Regression",
  family: "BoW",
  version: "primary",
  original_author: "penin",
  dataset_version: "v5_deduplicated",
  trained_on: ["train"],
  model_file: "artifacts/bow/model.joblib",
  model_sha256: "abc123abc123abc123",
  details: [
    { label: "Vectorizer", value: "CountVectorizer (1-2-grams, binary)" },
    { label: "Parameters", value: "23,265" },
  ],
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
  { id: "bow", name: "Bag of Words + Logistic Regression", available: true, status: "available", group: "primary" },
  { id: "tfidf", name: "TF-IDF + Logistic Regression", available: true, status: "available", group: "primary" },
  { id: "transformer", name: "Custom Transformer", available: true, status: "available", group: "primary" },
  { id: "distilbert", name: "Fine-tuned DistilBERT", available: false, status: "error", group: "primary", error: "Not trained yet" },
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
  await waitFor(() => expect(screen.getByText(thresholdPct(DEFAULT), { selector: "output" })).toBeInTheDocument());
}

async function runSample(user: ReturnType<typeof userEvent.setup>, category: string, command: string) {
  await user.click(screen.getByRole("tab", { name: category }));
  await user.click(screen.getByRole("button", { name: `“${command}”` }));
  const results = screen.getByRole("region", { name: "Prediction Results" });
  await waitFor(() => expect(within(results).getByText(`“${command}”`)).toBeInTheDocument());
}

describe("layout and model selection", () => {
  it("shows title, subtitle, footer and all four models plus Compare All in the dropdown", async () => {
    await renderReady();
    expect(screen.getByRole("heading", { name: "Sole Smart Actions" })).toBeInTheDocument();
    expect(screen.getByText(/four models/)).toBeInTheDocument();
    expect(screen.getByText(/MIT AI Edge \| Sole Smart Actions/)).toBeInTheDocument();
    const select = screen.getByLabelText("Classifier");
    await waitFor(() => expect(within(select).getAllByRole("option")).toHaveLength(5));
    const options = within(select).getAllByRole("option") as HTMLOptionElement[];
    expect(options.map((o) => o.value)).toEqual(["bow", "tfidf", "transformer", "distilbert", "__compare__"]);
    expect(options[0]).toHaveTextContent("Bag of Words + Logistic Regression — Available");
    expect(options[3].disabled).toBe(true); // an unavailable model is shown, not hidden
    expect(options[3]).toHaveTextContent("Not available");
    expect(options[4]).toHaveTextContent("Compare All Models");
  });

  it("shows model information read from the API, with 'Not available' for missing metrics", async () => {
    await renderReady();
    expect(screen.getByText("79.3%")).toBeInTheDocument();
    expect(screen.getAllByText("Not available").length).toBeGreaterThan(0);
  });
});

describe("model switching", () => {
  it("clears the previous result and loads the newly selected model's information", async () => {
    const user = userEvent.setup();
    await renderReady();
    await runSample(user, "Transfer Money", "Send $50 to Jone");
    expect(screen.getByTestId("raw-intent")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Classifier"), "tfidf");
    expect(screen.queryByTestId("raw-intent")).not.toBeInTheDocument();
    await waitFor(() => expect(calls.some((c) => c.url.includes("model_id=tfidf"))).toBe(true));
    await user.type(screen.getByLabelText("Customer command"), "Pay my EFL bill{Enter}");
    await waitFor(() => expect(predictCalls().at(-1)!.body).toMatchObject({ model_id: "tfidf" }));
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

    fireEvent.change(slider(), { target: { value: "100" } });
    expect(screen.getByText("100%", { selector: "output" })).toBeInTheDocument();
    expect(screen.getByTestId("final-intent")).toHaveTextContent("UNKNOWN_INTENT");
    expect(screen.getByTestId("raw-intent")).toHaveTextContent("TRANSFER_MONEY");
    expect(screen.getByTestId("confidence")).toHaveTextContent(`${(conf * 100).toFixed(1)}%`);
    expect(screen.getByTestId("status")).toHaveTextContent("Below Threshold");
    expect(screen.getByTestId("status")).toHaveTextContent("Fallback Applied");

    fireEvent.change(slider(), { target: { value: "0" } });
    expect(screen.getByTestId("final-intent")).toHaveTextContent("TRANSFER_MONEY");
    expect(predictCalls().length).toBe(before); // no re-inference

    await user.click(screen.getByRole("button", { name: "Reset to Default" }));
    expect(screen.getByText(thresholdPct(DEFAULT), { selector: "output" })).toBeInTheDocument();
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
    expect(predictCalls()[0].body).toMatchObject({ text: "Transfer 50 dollars to Jone", model_id: "bow" });
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
    const total = samples.categories.reduce((k, c) => k + c.commands.length, 0);
    expect(n).toBe(total);
    expect(predictCalls()).toHaveLength(total);
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

describe("simulated workflow and safety", () => {
  it("asks for explicit confirmation of an ambiguous money request and never executes it", async () => {
    const user = userEvent.setup();
    REAL["I owe Mary $50"] = Object.fromEntries(ORDER.map((c) => [c, c === "TRANSFER_MONEY" ? 0.92 : 0.01]));
    await renderReady();
    await user.type(screen.getByLabelText("Customer command"), "I owe Mary $50{Enter}");
    await waitFor(() => expect(screen.getByTestId("workflow-message")).toHaveTextContent("Would you like to send $50 to Mary?"));
    await user.click(screen.getByRole("button", { name: "Confirm (simulated)" }));
    expect(within(screen.getByTestId("workflow")).getByRole("status")).toHaveTextContent("No money was moved");
    fireEvent.change(slider(), { target: { value: "100" } });
    expect(screen.getByTestId("workflow-message")).toHaveTextContent("Did you mean one of these?");
  });
});

describe("compare all models", () => {
  it("runs one command through every model and renders the dashboard from generated results", async () => {
    const user = userEvent.setup();
    const p = predictionFor("Send $50 to Jone", DEFAULT);
    const row = {
      model: "bow", display_name: "Bag of Words + Logistic Regression", original_author: "penin", raw_test_accuracy: 0.937,
      raw_test_correct: 253, raw_test_n: 270, test_macro_f1: 0.937, test_weighted_f1: 0.937, test_macro_roc_auc_ovr: 0.995,
      default_threshold: 0.3415, known_coverage_at_default: 0.96, overall_correct_routing_at_default: 0.92,
      unknown_rejection_at_default: 0.4, unknown_help_support_at_default: 0.16, unknown_safe_fallback_at_default: 0.56,
      unknown_to_financial_at_default: 30, latency_p50_ms: 0.33, latency_p95_ms: 0.7, latency_device: "cpu",
      model_size_mb: 0.22, n_parameters: 23265, typo_test_accuracy: 0.85,
    };
    vi.stubGlobal("fetch", (input: RequestInfo | URL) => {
      const url = String(input);
      calls.push({ url, body: null });
      if (url.startsWith("/api/models")) return json(200, MODELS);
      if (url.startsWith("/api/model-info")) return json(200, INFO);
      if (url.startsWith("/api/compare"))
        return json(200, [
          { model_id: "bow", model_name: "Bag of Words + Logistic Regression", ok: true, error: null, prediction: p },
          { model_id: "distilbert", model_name: "Fine-tuned DistilBERT", ok: false, error: "Not trained yet", prediction: null },
        ]);
      if (url.startsWith("/api/comparison"))
        return json(200, { generated_utc: "2026-11-16T00:00:00+00:00", source: "results/comparison/model_comparison.json", models: [row], missing: [] });
      return json(404, { detail: "not found" });
    });
    await renderReady();
    await user.selectOptions(screen.getByLabelText("Classifier"), "__compare__");
    await waitFor(() => expect(screen.getByRole("heading", { name: "Model Comparison Dashboard" })).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText("(253/270)")).toBeInTheDocument());
    await user.type(screen.getByLabelText("Customer command"), "Send $50 to Jone{Enter}");
    const card = await screen.findByTestId("compare-bow");
    expect(within(card).getByText("TRANSFER_MONEY")).toBeInTheDocument();
    expect(screen.getByText(/Not available: Not trained yet/)).toBeInTheDocument();
  });
});
