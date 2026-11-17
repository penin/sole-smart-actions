"""Illustrative Sole workflow for a decision. SIMULATION ONLY: nothing here moves money or contacts a wallet.

Safety rules:
* money-moving intents (transfer, bill payment, withdrawal) always end in an explicit confirmation step;
  the confirmation is simulated and never authorises anything;
* ambiguous or low-confidence requests get a clarification question instead of an assumed action;
* the demo never asks for, accepts or stores PINs, passwords or account numbers.
"""

from __future__ import annotations

import re

FINANCIAL = {"TRANSFER_MONEY", "BILL_PAY", "WITHDRAW_MONEY"}
SCREENS = {
    "TRANSFER_MONEY": "Send Money", "CHECK_BALANCE": "Wallet Balance", "VIEW_TRANSACTIONS": "Transaction History",
    "WITHDRAW_MONEY": "Cash Out", "BILL_PAY": "Pay Bills", "HELP_SUPPORT": "Help & Support",
    "MANAGE_CONTACTS": "Contacts", "VIEW_GROUP": "Groups", "CHECK_SPENDING": "Spending Insights",
}
VERBS = {"TRANSFER_MONEY": "send", "BILL_PAY": "pay", "WITHDRAW_MONEY": "withdraw"}

_AMOUNT = re.compile(r"(?:\$|fjd\s*)?\b(\d{1,6}(?:[.,]\d{1,2})?)\b\s*(?:dollars?|fjd|bucks)?", re.I)
_TO = re.compile(r"\b(?:to|for)\s+([A-Z][a-zA-Z'\-]+)")
_OWE = re.compile(r"\bowe\s+([A-Z][a-zA-Z'\-]+)")
_BILLER = re.compile(r"\b(EFL|Vodafone|Digicel|TFL|WAF|Water Authority|Fiji Water|electricity|water|internet|phone)\b", re.I)
_EXPLICIT = re.compile(r"\b(send|transfer|pay|withdraw|cash out|take out|give)\b", re.I)
# Words that do not count as a recipient name even when capitalised at the start of a clause.
_NOT_NAMES = {"My", "The", "Me", "Him", "Her", "Them", "Sole", "EFL", "Vodafone", "Digicel"}


def extract_entities(text: str) -> dict:
    """Very small, illustrative slot extraction (amount, recipient, biller). Not a production parser."""
    amount = None
    m = _AMOUNT.search(text)
    if m and ("$" in m.group(0) or re.search(r"dollar|fjd|bucks", m.group(0), re.I) or re.search(r"\$|send|transfer|pay|withdraw|owe|give", text, re.I)):
        amount = m.group(1).replace(",", ".")
    recipient = None
    for pat in (_OWE, _TO):
        r = pat.search(text)
        if r and r.group(1) not in _NOT_NAMES:
            recipient = r.group(1)
            break
    biller = _BILLER.search(text)
    return {"amount": amount, "recipient": recipient, "biller": biller.group(1) if biller else None}


def _money_phrase(intent: str, e: dict) -> str:
    amt = f"${e['amount']}" if e["amount"] else "money"
    if intent == "TRANSFER_MONEY":
        return f"send {amt} to {e['recipient']}" if e["recipient"] else f"send {amt}"
    if intent == "BILL_PAY":
        return f"pay {amt} to {e['biller']}" if e["biller"] else f"pay a bill ({amt})"
    return f"withdraw {amt}"


def build_workflow(text: str, decision: dict, ranked: list[dict], menu_size: int = 3) -> dict:
    final, raw = decision["final_intent"], decision["raw_prediction"]
    e = extract_entities(text)
    options = [r["intent"] for r in ranked[:menu_size]]
    if final == "UNKNOWN_INTENT":
        return {"kind": "clarify", "screen": None, "requires_confirmation": False,
                "message": "I'm not sure what you'd like to do. Did you mean one of these?",
                "options": options + ["Something else / Help & Support"], "entities": e,
                "steps": ["Show clarification menu (top intents + Help & Support)", "Customer chooses an option",
                          "Open the chosen screen"],
                "note": f"Confidence {decision['confidence']:.0%} is below the {decision['threshold']:.0%} threshold; "
                        "no workflow is opened automatically."}
    screen = SCREENS.get(final, final)
    if final in FINANCIAL:
        explicit = bool(_EXPLICIT.search(text))
        missing = [k for k, ok in (("amount", e["amount"]), ("recipient", e["recipient"] or final != "TRANSFER_MONEY"),
                                   ("biller", e["biller"] or final != "BILL_PAY")) if not ok]
        phrase = _money_phrase(final, e)
        if not explicit:
            message = f"Would you like to {phrase}?"  # e.g. "I owe Mary $50" -> "Would you like to send $50 to Mary?"
        elif missing:
            message = f"Open {screen} and ask for the missing details: {', '.join(missing)}."
        else:
            message = f"Review: {phrase}. Confirm to continue."
        return {"kind": "confirm", "screen": screen, "requires_confirmation": True, "message": message,
                "options": [], "entities": e, "missing": missing, "explicit_request": explicit,
                "steps": [f"Open {screen}", "Pre-fill the details the customer gave" if any(e.values()) else
                          "Ask for amount and recipient", "Customer reviews and explicitly confirms",
                          "Real app: authenticate inside the official Sole app (never in this demo)"],
                "note": "Simulation only. The demo never executes transactions or asks for a PIN."}
    if final == "HELP_SUPPORT":
        return {"kind": "route", "screen": screen, "requires_confirmation": False,
                "message": "Opening Help & Support.", "options": [], "entities": e,
                "steps": ["Open Help & Support", "Suggest relevant articles or a live agent"],
                "note": "A genuine HELP_SUPPORT prediction, not the low-confidence fallback."}
    return {"kind": "route", "screen": screen, "requires_confirmation": False, "message": f"Opening {screen}.",
            "options": [], "entities": e, "steps": [f"Open {screen}", "Show read-only information"],
            "note": "Read-only screen; no money moves."}
