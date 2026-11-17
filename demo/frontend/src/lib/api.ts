import type { CompareEntry, Comparison, Decision, ModelInfo, ModelSummary, Prediction } from "../types";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  } catch {
    throw new ApiError("Cannot reach the model server. Is the backend running? (see demo/README.md)", 0);
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON error page */
  }
  if (!res.ok) {
    const b = body as { detail?: string; errors?: { field: string; message: string }[] } | null;
    const detail = b?.errors?.length ? b.errors.map((e) => `${e.field}: ${e.message}`).join("; ") : b?.detail;
    throw new ApiError(detail || `Request failed (HTTP ${res.status})`, res.status);
  }
  return body as T;
}

export const api = {
  models: () => request<ModelSummary[]>("/api/models"),
  modelInfo: (modelId: string) => request<ModelInfo>(`/api/model-info?model_id=${encodeURIComponent(modelId)}`),
  predict: (text: string, threshold: number, modelId: string) =>
    request<Prediction>("/api/predict", {
      method: "POST",
      body: JSON.stringify({ text, threshold, model_id: modelId }),
    }),
  decide: (probabilities: Record<string, number>, threshold: number, modelId: string) =>
    request<Decision>("/api/decide", {
      method: "POST",
      body: JSON.stringify({ probabilities, threshold, model_id: modelId }),
    }),
  /** Same command through every primary model; threshold null = each model's own default. */
  compare: (text: string, threshold: number | null) =>
    request<CompareEntry[]>("/api/compare", { method: "POST", body: JSON.stringify({ text, threshold }) }),
  comparison: () => request<Comparison>("/api/comparison"),
};
