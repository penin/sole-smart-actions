"""Optional real-browser smoke test of the running demo (presentation scenarios A–E).

Needs the demo running (python demo/backend/main.py, with the frontend built) and Playwright:
    pip install playwright          # uses your installed Microsoft Edge; no browser download needed
    python demo/e2e/browser_smoke.py [--url http://127.0.0.1:8000] [--shots <folder>]

Prints what the REAL model predicted for each scenario. It asserts behaviour (threshold logic, no re-inference,
layout), never specific model outputs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright


def text(page: Page, testid: str) -> str:
    return page.get_by_test_id(testid).inner_text().strip()


def run_sample(page: Page, category: str, command: str) -> dict:
    page.get_by_role("tab", name=category, exact=True).click()
    page.get_by_role("button", name=f"“{command}”", exact=True).click()
    results = page.get_by_role("region", name="Prediction Results")
    # The first request to a neural model loads it (lazy loading), which can take several seconds.
    expect(results.get_by_text(f"“{command}”", exact=True)).to_be_visible(timeout=90000)
    expect(page.get_by_role("button", name="Analyze Command", exact=True)).to_be_enabled(timeout=90000)
    return {"raw": text(page, "raw-intent"), "confidence": text(page, "confidence"),
            "final": text(page, "final-intent"), "status": " / ".join(text(page, "status").split("\n"))}


def set_threshold(page: Page, percent: int) -> None:
    slider = page.get_by_role("slider", name=re.compile("confidence threshold", re.I))
    slider.focus()
    page.keyboard.press("Home")
    for _ in range(percent):
        page.keyboard.press("ArrowRight")
    expect(page.locator("output#threshold, output[for=threshold]")).to_have_text(f"{percent}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--shots", default=None)
    args = ap.parse_args()
    shots = Path(args.shots) if args.shots else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        predict_requests: list[str] = []
        page.on("request", lambda r: predict_requests.append(r.post_data or "") if r.url.endswith("/api/predict") else None)
        page.goto(args.url)
        expect(page.get_by_role("heading", name="Sole Smart Actions", exact=True)).to_be_visible()
        expect(page.get_by_text("MIT AI Edge | Sole Smart Actions", exact=False)).to_be_visible()
        default = page.locator("output[for=threshold]").inner_text()
        print(f"Default threshold shown: {default}")
        options = page.locator("#model-select option")
        print("Model dropdown:", [f"{options.nth(i).inner_text()}{' (disabled)' if options.nth(i).is_disabled() else ''}"
                                  for i in range(options.count())])

        # Blank input: no request
        page.get_by_role("button", name="Analyze Command", exact=True).click()
        expect(page.get_by_text("Type a command first", exact=False)).to_be_visible()
        assert not predict_requests, "blank command was sent"

        # Scenario A
        a = run_sample(page, "Transfer Money", "Send $50 to Jone")
        print(f"A  'Send $50 to Jone' @ {default}: {a}")
        expect(page.get_by_text("Decision confirmed by backend")).to_be_visible()
        if shots:
            page.screenshot(path=str(shots / "A_send_desktop.png"), full_page=True)

        # Scenario B: raise threshold step by step; no new inference, raw/confidence fixed
        n_before = len(predict_requests)
        conf = float(a["confidence"].rstrip("%"))
        flips = None
        for t in range(0, 101, 5):
            set_threshold(page, t)
            assert text(page, "raw-intent") == a["raw"] and text(page, "confidence") == a["confidence"]
            final = text(page, "final-intent")
            expected_fallback = conf / 100 < t / 100
            assert (final == "UNKNOWN_INTENT") == expected_fallback, (t, final, conf)
            if final == "UNKNOWN_INTENT" and flips is None:
                flips = t
        assert len(predict_requests) == n_before, "slider triggered re-inference"
        print(f"B  confidence {a['confidence']}: final intent switches to UNKNOWN_INTENT at {flips}% "
              f"(no new /api/predict calls)")
        set_threshold(page, 90)
        if shots:
            page.screenshot(path=str(shots / "B_send_threshold90.png"), full_page=True)
        page.get_by_role("button", name="Reset to Default", exact=True).click()
        expect(page.locator("output[for=threshold]")).to_have_text(default)

        # Scenarios C, D, E (report the model's actual behaviour)
        c = run_sample(page, "Unknown / Out-of-Scope", "What's the weather tomorrow?")
        print(f"C  weather @ {default}: {c}")
        d_default = run_sample(page, "Ambiguous", "Payment")
        set_threshold(page, 40)
        print(f"D  'Payment' @ {default}: {d_default}; @ 40%: final={text(page, 'final-intent')}")
        page.get_by_role("button", name="Reset to Default", exact=True).click()
        e1 = run_sample(page, "Transfer Money", "Transfer 50 dollars to Jone")
        print(f"E  'Transfer 50 dollars to Jone': {e1}  vs 'Send $50 to Jone': {a['raw']} {a['confidence']}")

        # History restore without inference
        n = len(predict_requests)
        page.get_by_role("list", name="Recent predictions").get_by_text("“Send $50 to Jone”").click()
        expect(page.get_by_label("Customer command")).to_have_value("Send $50 to Jone")
        assert len(predict_requests) == n
        print("History: restore works without re-inference;",
              page.get_by_role("list", name="Recent predictions").get_by_role("button").count(), "entries")

        # Every primary model answers through the same UI (real inference; first neural call loads the model)
        for mid in ("tfidf", "transformer", "distilbert", "bow"):
            page.locator("#model-select").select_option(mid)
            expect(page.get_by_role("button", name="Analyze Command", exact=True)).to_be_enabled(timeout=60000)
            r = run_sample(page, "Transfer Money", "Send $50 to Jone")
            print(f"M  {mid:12s} 'Send $50 to Jone': {r}")

        # Ambiguous money request -> clarification question + simulated confirmation (nothing executed)
        page.get_by_label("Customer command").fill("I owe Mary $50")
        page.get_by_role("button", name="Analyze Command", exact=True).click()
        wf = page.get_by_test_id("workflow")
        expect(wf).to_be_visible()
        print("W  'I owe Mary $50' (bow):", wf.get_by_test_id("workflow-message").inner_text())
        if wf.get_by_role("button", name="Confirm (simulated)").count():
            wf.get_by_role("button", name="Confirm (simulated)").click()
            expect(wf.get_by_text("No money was moved", exact=False)).to_be_visible()

        # Compare All Models: four cards + dashboard from generated results
        page.locator("#model-select").select_option("__compare__")
        expect(page.get_by_role("heading", name="Model Comparison Dashboard")).to_be_visible()
        page.get_by_label("Customer command").fill("I owe Mary $50")
        page.get_by_role("button", name="Analyze Command", exact=True).click()
        for mid in ("bow", "tfidf", "transformer", "distilbert"):
            card = page.get_by_test_id(f"compare-{mid}")
            expect(card).to_be_visible(timeout=60000)
            print(f"C* {mid:12s}", " | ".join(card.inner_text().splitlines()[:5]))
        if shots:
            page.screenshot(path=str(shots / "compare_all_desktop.png"), full_page=True)

        # Mobile layout: true 390px viewport, no horizontal overflow
        m = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True)
        m.goto(args.url)
        run_sample(m, "Ambiguous", "Payment")
        overflow = m.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        assert overflow <= 0, f"horizontal overflow of {overflow}px on mobile"
        print(f"Mobile 390px: horizontal overflow {overflow}px")
        if shots:
            m.screenshot(path=str(shots / "mobile_payment.png"), full_page=True)
        m.locator("#model-select").select_option("__compare__")
        m.get_by_label("Customer command").fill("Pay my EFL bill")
        m.get_by_role("button", name="Analyze Command", exact=True).click()
        expect(m.get_by_test_id("compare-distilbert")).to_be_visible(timeout=60000)
        overflow = m.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        assert overflow <= 0, f"horizontal overflow of {overflow}px on mobile compare view"
        print(f"Mobile 390px compare view: horizontal overflow {overflow}px")
        if shots:
            m.screenshot(path=str(shots / "mobile_compare.png"), full_page=True)
        t = browser.new_page(viewport={"width": 768, "height": 1024})
        t.goto(args.url)
        run_sample(t, "Transfer Money", "Give Ana 20 dollars")
        assert t.evaluate("document.documentElement.scrollWidth - window.innerWidth") <= 0
        if shots:
            t.screenshot(path=str(shots / "tablet_give_ana.png"), full_page=True)
        browser.close()
    print("Browser smoke test passed.")


if __name__ == "__main__":
    main()
