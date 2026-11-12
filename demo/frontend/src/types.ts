// Shapes returned by the FastAPI backend (see demo/backend/schemas.py).

export interface ApplicationResponse {
  action: string;
  model_decision: string;
  route_to: string | null;
  offers: string[];
  suggested_intent: string | null;
}

export type DecisionStatus = "accepted" | "fallback_applied";

export interface Decision {
  raw_prediction: string;
  confidence: number;
  threshold: number;
  final_intent: string;
  status: DecisionStatus;
  below_threshold: boolean;
  fallback_applied: boolean;
  application: ApplicationResponse;
}

export interface Contribution {
  feature: string;
  count: number;
  weight: number;
  contribution: number;
  contribution_vs_runner_up: number;
}

export interface Explanation {
  normalized_text: string;
  tokens: string[];
  recognized_tokens: string[];
  unknown_tokens: string[];
  recognized_phrases?: string[];
  ngram_range?: number[];
  explained_class: string;
  runner_up_class: string;
  intercept: number;
  class_score: number;
  contributions: Contribution[];
  top_class_features: { feature: string; weight: number }[];
}

export interface Prediction extends Decision {
  model_id: string;
  model_name: string;
  input: string;
  class_order: string[];
  probabilities: Record<string, number>;
  ranked: { intent: string; probability: number }[];
  default_threshold: number;
  recognized_tokens: string[];
  unknown_tokens: string[];
  explanation: Explanation;
  inference_ms: number;
}

export interface ModelSummary {
  id: string;
  name: string;
  available: boolean;
  status: "available" | "coming_soon" | "error" | string;
  error?: string | null;
}

export interface ModelEvaluation {
  source: string;
  test_n: number | null;
  test_accuracy: number | null;
  test_macro_f1: number | null;
  test_weighted_f1: number | null;
  test_macro_precision: number | null;
  test_macro_recall: number | null;
  test_macro_roc_auc: number | null;
  test_log_loss: number | null;
  test_top2_accuracy: number | null;
  test_mean_confidence: number | null;
  test_macro_f1_ci95: [number, number] | null;
  validation_macro_f1: number | null;
  expected_calibration_error: number | null;
  known_intent_coverage_at_default: number | null;
  unknown_rejection_rate_at_default: number | null;
  median_latency_ms: number | null;
  training_seconds: number | null;
  model_size_mb: number | null;
}

export interface ModelInfo {
  id: string;
  name: string;
  family: string;
  version: string | null;
  dataset_version?: string | null;
  model_file: string;
  model_sha256: string;
  pipeline_steps: string[];
  vectorizer: { type: string; ngram_range: number[]; binary: boolean; token_pattern: string; vocabulary_size: number };
  classifier: { type: string; C: number; n_parameters: number };
  classes: string[];
  n_classes: number;
  unknown_label: string;
  unknown_label_is_model_class: boolean;
  default_threshold: number;
  default_threshold_source: string;
  fallback_rule: string;
  evaluation: ModelEvaluation;
}

export interface HistoryEntry {
  key: string;
  prediction: Prediction;
  thresholdUsed: number;
  finalIntentAtTime: string;
}

export interface SampleCategory {
  id: string;
  label: string;
  intent_hint: string | null;
  commands: string[];
}
