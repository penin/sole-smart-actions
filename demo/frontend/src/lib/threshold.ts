import type { ApplicationResponse, Decision } from "../types";

// Mirror of demo/backend/decision.py (which wraps sole_intent.rejection). Used only for instant slider
// feedback; the backend stays authoritative and both are tested against the same generated fixture.
export const UNKNOWN_INTENT = "UNKNOWN_INTENT";
export const FALLBACK_ACTION = "FALLBACK_HELP_OR_CLARIFY";
export const FALLBACK_OFFERS = ["HELP_SUPPORT", "CLARIFY"];

export function applyThreshold(
  probabilities: Record<string, number>,
  threshold: number,
  classOrder: string[],
): Decision {
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
    throw new RangeError("threshold must be between 0 and 1");
  }
  const missing = classOrder.filter((c) => !(c in probabilities));
  const extra = Object.keys(probabilities).filter((c) => !classOrder.includes(c));
  if (missing.length || extra.length) {
    throw new Error(`probabilities do not match model classes (missing=${missing}, unexpected=${extra})`);
  }
  // First maximum in class order wins, matching numpy.argmax.
  let best = 0;
  for (let i = 1; i < classOrder.length; i++) {
    if (probabilities[classOrder[i]] > probabilities[classOrder[best]]) best = i;
  }
  const raw = classOrder[best];
  const confidence = probabilities[raw];
  // Project rule: reject only when strictly below the threshold (equal is accepted).
  const accepted = !(confidence < threshold);
  const final = accepted ? raw : UNKNOWN_INTENT;
  const application: ApplicationResponse = accepted
    ? { action: `ROUTE_${raw}`, model_decision: raw, route_to: raw, offers: [], suggested_intent: null }
    : { action: FALLBACK_ACTION, model_decision: UNKNOWN_INTENT, route_to: null, offers: [...FALLBACK_OFFERS], suggested_intent: raw };
  return {
    raw_prediction: raw,
    confidence,
    threshold,
    final_intent: final,
    status: accepted ? "accepted" : "fallback_applied",
    below_threshold: !accepted,
    fallback_applied: !accepted,
    application,
  };
}
