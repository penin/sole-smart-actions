import { describe, expect, it } from "vitest";
import { buildWorkflow, extractEntities } from "../lib/workflow";
import type { Decision } from "../types";
import fixture from "./fixtures/workflow_cases.json";

// The TypeScript port must reproduce demo/backend/workflow.py exactly (fixture generated from Python).
describe("simulated workflow matches the backend", () => {
  for (const c of fixture.cases) {
    it(`${c.text} @ ${c.decision.threshold} -> ${c.decision.final_intent}`, () => {
      expect(buildWorkflow(c.text, c.decision as Decision, c.ranked)).toEqual(c.expected);
    });
  }
});

describe("financial safety", () => {
  it("turns an implicit money request into a question", () => {
    const d = { raw_prediction: "TRANSFER_MONEY", confidence: 0.9, threshold: 0.3, final_intent: "TRANSFER_MONEY" } as Decision;
    const w = buildWorkflow("I owe Mary $50", d, [{ intent: "TRANSFER_MONEY" }]);
    expect(w.message).toBe("Would you like to send $50 to Mary?");
    expect(w.requires_confirmation).toBe(true);
  });

  it("every money-moving intent requires explicit confirmation; nothing mentions a PIN prompt", () => {
    for (const intent of ["TRANSFER_MONEY", "BILL_PAY", "WITHDRAW_MONEY"]) {
      const d = { raw_prediction: intent, confidence: 0.99, threshold: 0.3, final_intent: intent } as Decision;
      const w = buildWorkflow("Send $5 to Jone", d, [{ intent }]);
      expect(w.kind).toBe("confirm");
      expect(w.requires_confirmation).toBe(true);
      expect(JSON.stringify(w).toLowerCase()).not.toMatch(/enter (your )?pin/);
    }
  });

  it("extracts simple slots", () => {
    expect(extractEntities("Pay $75 to EFL")).toEqual({ amount: "75", recipient: null, biller: "EFL" });
    expect(extractEntities("Send $50 to Jone")).toEqual({ amount: "50", recipient: "Jone", biller: null });
  });
});
