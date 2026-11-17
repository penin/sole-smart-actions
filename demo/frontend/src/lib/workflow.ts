// Illustrative Sole workflow - a line-by-line port of demo/backend/workflow.py so the simulated workflow follows the
// threshold slider instantly. SIMULATION ONLY: nothing is executed, and no PIN is ever requested.
// src/__tests__/workflow.test.ts checks this port against cases generated from the Python implementation.
import type { Decision, Workflow } from "../types";

export const FINANCIAL = new Set(["TRANSFER_MONEY", "BILL_PAY", "WITHDRAW_MONEY"]);
const SCREENS: Record<string, string> = {
  TRANSFER_MONEY: "Send Money",
  CHECK_BALANCE: "Wallet Balance",
  VIEW_TRANSACTIONS: "Transaction History",
  WITHDRAW_MONEY: "Cash Out",
  BILL_PAY: "Pay Bills",
  HELP_SUPPORT: "Help & Support",
  MANAGE_CONTACTS: "Contacts",
  VIEW_GROUP: "Groups",
  CHECK_SPENDING: "Spending Insights",
};
const AMOUNT = /(?:\$|fjd\s*)?\b(\d{1,6}(?:[.,]\d{1,2})?)\b\s*(?:dollars?|fjd|bucks)?/i;
const TO = /\b(?:to|for)\s+([A-Z][a-zA-Z'-]+)/;
const OWE = /\bowe\s+([A-Z][a-zA-Z'-]+)/;
const BILLER = /\b(EFL|Vodafone|Digicel|TFL|WAF|Water Authority|Fiji Water|electricity|water|internet|phone)\b/i;
const EXPLICIT = /\b(send|transfer|pay|withdraw|cash out|take out|give)\b/i;
const NOT_NAMES = new Set(["My", "The", "Me", "Him", "Her", "Them", "Sole", "EFL", "Vodafone", "Digicel"]);

export function extractEntities(text: string): Workflow["entities"] {
  let amount: string | null = null;
  const m = AMOUNT.exec(text);
  if (m && (m[0].includes("$") || /dollar|fjd|bucks/i.test(m[0]) || /\$|send|transfer|pay|withdraw|owe|give/i.test(text))) {
    amount = m[1].replace(",", ".");
  }
  let recipient: string | null = null;
  for (const pat of [OWE, TO]) {
    const r = pat.exec(text);
    if (r && !NOT_NAMES.has(r[1])) {
      recipient = r[1];
      break;
    }
  }
  const b = BILLER.exec(text);
  return { amount, recipient, biller: b ? b[1] : null };
}

function moneyPhrase(intent: string, e: Workflow["entities"]): string {
  const amt = e.amount ? `$${e.amount}` : "money";
  if (intent === "TRANSFER_MONEY") return e.recipient ? `send ${amt} to ${e.recipient}` : `send ${amt}`;
  if (intent === "BILL_PAY") return e.biller ? `pay ${amt} to ${e.biller}` : `pay a bill (${amt})`;
  return `withdraw ${amt}`;
}

const pct0 = (x: number) => `${Math.round(x * 100)}%`;

export function buildWorkflow(text: string, decision: Decision, ranked: { intent: string }[], menuSize = 3): Workflow {
  const final = decision.final_intent;
  const e = extractEntities(text);
  if (final === "UNKNOWN_INTENT") {
    return {
      kind: "clarify",
      screen: null,
      requires_confirmation: false,
      message: "I'm not sure what you'd like to do. Did you mean one of these?",
      options: [...ranked.slice(0, menuSize).map((r) => r.intent), "Something else / Help & Support"],
      entities: e,
      steps: ["Show clarification menu (top intents + Help & Support)", "Customer chooses an option", "Open the chosen screen"],
      note: `Confidence ${pct0(decision.confidence)} is below the ${pct0(decision.threshold)} threshold; no workflow is opened automatically.`,
    };
  }
  const screen = SCREENS[final] ?? final;
  if (FINANCIAL.has(final)) {
    const explicit = EXPLICIT.test(text);
    const missing = (
      [
        ["amount", Boolean(e.amount)],
        ["recipient", Boolean(e.recipient) || final !== "TRANSFER_MONEY"],
        ["biller", Boolean(e.biller) || final !== "BILL_PAY"],
      ] as [string, boolean][]
    )
      .filter(([, ok]) => !ok)
      .map(([k]) => k);
    const phrase = moneyPhrase(final, e);
    const message = !explicit
      ? `Would you like to ${phrase}?`
      : missing.length
        ? `Open ${screen} and ask for the missing details: ${missing.join(", ")}.`
        : `Review: ${phrase}. Confirm to continue.`;
    return {
      kind: "confirm",
      screen,
      requires_confirmation: true,
      message,
      options: [],
      entities: e,
      missing,
      explicit_request: explicit,
      steps: [
        `Open ${screen}`,
        e.amount || e.recipient || e.biller ? "Pre-fill the details the customer gave" : "Ask for amount and recipient",
        "Customer reviews and explicitly confirms",
        "Real app: authenticate inside the official Sole app (never in this demo)",
      ],
      note: "Simulation only. The demo never executes transactions or asks for a PIN.",
    };
  }
  if (final === "HELP_SUPPORT") {
    return {
      kind: "route",
      screen,
      requires_confirmation: false,
      message: "Opening Help & Support.",
      options: [],
      entities: e,
      steps: ["Open Help & Support", "Suggest relevant articles or a live agent"],
      note: "A genuine HELP_SUPPORT prediction, not the low-confidence fallback.",
    };
  }
  return {
    kind: "route",
    screen,
    requires_confirmation: false,
    message: `Opening ${screen}.`,
    options: [],
    entities: e,
    steps: [`Open ${screen}`, "Show read-only information"],
    note: "Read-only screen; no money moves.",
  };
}
