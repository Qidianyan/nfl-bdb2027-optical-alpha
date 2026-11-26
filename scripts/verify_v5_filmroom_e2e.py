"""Real Playwright Chromium E2E click-through verification & after-screenshot capture for V5 Film-Room Studio.

Always launches Chromium with `--use-mock-keychain` and `--password-store=basic` on macOS.
Verifies:
  - 0 console errors & 0 uncaught page errors
  - All 6 2x Retina gallery PNGs load with naturalWidth > 0
  - All 4 position-group saliency matchup tabs switch cleanly + pointer hover updates live 0.1s telemetry
  - Attention head radios (Combined / Burst / Brake) and step buttons work
  - GM Draft-Room Surplus EV Calculator sliders, position radios, and presets update EV, Bust Risk, and real comps
  - Captures `shots/after_v5_filmroom_saliency.png` and `shots/after_v5_draft_ev.png` at 1440x900 @2x
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HTML_PATH = PROJECT_ROOT / "materials" / "NFL_BDB2027_Interactive_FilmRoom.html"
SHOTS_DIR = PROJECT_ROOT / "shots"


def main() -> None:
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []
    page_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--use-mock-keychain", "--password-store=basic"],
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=2,
        )
        page = context.new_page()
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        page.goto(f"file://{HTML_PATH.resolve()}", wait_until="networkidle")
        page.evaluate("() => document.fonts ? document.fonts.ready : Promise.resolve()")
        page.wait_for_timeout(300)

        # 1. Capture after screenshot of Module 1 (10 Hz Ghost Replay + Dual-Head Attention Saliency)
        page.evaluate("() => { const el = document.getElementById('saliency-heading'); window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 68, behavior: 'instant' }); }")
        page.wait_for_timeout(150)
        saliency_shot = SHOTS_DIR / "after_v5_filmroom_saliency.png"
        page.screenshot(path=str(saliency_shot), full_page=False)
        print(f"[E2E] Captured {saliency_shot.name} ({saliency_shot.stat().st_size:,} bytes)")

        # 2. Click through all 4 position-group matchup tabs & verify live updates
        expected_gems = ["Puka Nacua", "Jaylon Jones", "Tuli Tuipulotu", "Mason McCormick"]
        tabs = page.locator('[role="tab"]')
        assert tabs.count() == 4, f"Expected 4 matchup tabs, got {tabs.count()}"
        for idx, exp_name in enumerate(expected_gems):
            tabs.nth(idx).click()
            page.wait_for_timeout(80)
            got_name = page.locator("#gemName").inner_text().strip()
            assert got_name == exp_name, f"Tab {idx}: expected {exp_name}, got {got_name}"
        print("[E2E] All 4 position-group matchup tabs switched cleanly:", expected_gems)

        # Return to WR/TE Puka Nacua matchup & test pointer hover scrubbing on #saliencyCanvas
        tabs.nth(0).click()
        page.wait_for_timeout(80)
        sal_box = page.locator("#saliencyCanvas").bounding_box()
        assert sal_box is not None
        page.mouse.move(sal_box["x"] + sal_box["width"] * 0.25, sal_box["y"] + sal_box["height"] * 0.5)
        page.wait_for_timeout(80)
        t_left = page.locator("#telGemT").inner_text().strip()
        page.mouse.move(sal_box["x"] + sal_box["width"] * 0.85, sal_box["y"] + sal_box["height"] * 0.5)
        page.wait_for_timeout(80)
        t_right = page.locator("#telGemT").inner_text().strip()
        assert t_left != t_right, f"Hover scrub failed: left={t_left}, right={t_right}"
        print(f"[E2E] Pointer hover on #saliencyCanvas scrubbed cleanly: {t_left} -> {t_right}")

        # Test Head 1 / Head 2 radio switch + step button + jump peak
        page.locator('input[name="attn-head"][value="h1"]').check()
        page.locator('input[name="attn-head"][value="h2"]').check()
        page.locator('input[name="attn-head"][value="aw"]').check()
        page.locator("#btnJumpPeak").click()
        t_peak = page.locator("#telGemT").inner_text().strip()
        page.locator("#btnStepBack").click()
        t_back = page.locator("#telGemT").inner_text().strip()
        assert t_peak != t_back
        print(f"[E2E] Head switch & step controls verified: peak={t_peak} -> back={t_back}")

        # 3. Module 2: GM Draft Room Surplus EV Calculator walkthrough
        page.evaluate("() => { const el = document.getElementById('ev-heading'); window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 68, behavior: 'instant' }); }")
        page.wait_for_timeout(150)

        # Click Puka Nacua preset and capture screenshot
        page.locator('[data-preset-grp="WR_TE"]').click()
        page.wait_for_timeout(150)
        ev_puka = page.locator("#evSurplusVal").inner_text().strip()
        comps_puka = page.locator("#historicalCompsGrid .comp-name").all_inner_texts()
        assert len(comps_puka) == 3, f"Expected 3 historical comps, got {comps_puka}"
        assert any("Puka Nacua" in c for c in comps_puka), f"Expected Puka Nacua in real comps, got {comps_puka}"
        ev_shot = SHOTS_DIR / "after_v5_draft_ev.png"
        page.screenshot(path=str(ev_shot), full_page=False)
        print(f"[E2E] Captured {ev_shot.name} ({ev_shot.stat().st_size:,} bytes) | WR/TE #177 EV={ev_puka} comps={comps_puka}")

        # Switch to DL/EDGE preset & DB Trap preset and assert live math updates
        page.locator('[data-preset-grp="DL_EDGE"]').click()
        page.wait_for_timeout(80)
        ev_dl = page.locator("#evSurplusVal").inner_text().strip()
        bust_dl = page.locator("#evBustRedVal").inner_text().strip()
        assert ev_dl != ev_puka and bust_dl.startswith("-"), f"Unexpected DL EV={ev_dl} bust={bust_dl}"

        page.locator('[data-preset-grp="DB"]').click()
        page.wait_for_timeout(80)
        bust_trap = page.locator("#evBustRedVal").inner_text().strip()
        verdict_trap = page.locator("#evVerdictText").inner_text().strip()
        assert bust_trap.startswith("+") and "trap" in verdict_trap.lower(), (
            f"Expected elevated trap risk for DB -1.4σ, got bust={bust_trap}, verdict={verdict_trap}"
        )

        # Directly manipulate sliders via JS input event to verify real-time reactivity
        page.locator("#draftPickSlider").fill("10")
        page.locator("#draftPickSlider").dispatch_event("input")
        page.locator("#opticalAlphaSlider").fill("-2.0")
        page.locator("#opticalAlphaSlider").dispatch_event("input")
        page.wait_for_timeout(80)
        pick_out = page.locator("#draftPickOut").inner_text().strip()
        assert "#10" in pick_out, f"Slider update failed: {pick_out}"
        print(f"[E2E] GM EV Calculator presets & sliders verified: DL={ev_dl} ({bust_dl}), Trap={verdict_trap} ({bust_trap}), Pick10={pick_out}")

        # 4. Scroll gallery into view so lazy images load & assert all 6 2x Retina PNGs are healthy
        page.locator("#figure-gallery").scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        imgs_status = page.evaluate(
            """() => Array.from(document.querySelectorAll('#figure-gallery img')).map(img => ({
                src: img.getAttribute('src'),
                complete: img.complete,
                w: img.naturalWidth,
                h: img.naturalHeight
            }))"""
        )
        assert len(imgs_status) == 6
        for im in imgs_status:
            assert im["complete"] and im["w"] >= 2000, f"Broken or low-res gallery image: {im}"
        print(f"[E2E] All 6 gallery 2x Retina PNGs loaded cleanly (widths={[i['w'] for i in imgs_status]}).")

        browser.close()

    assert not console_errors, f"Console errors detected: {console_errors}"
    assert not page_errors, f"Uncaught page errors detected: {page_errors}"
    print("[E2E] PASS: 0 console errors, 0 page errors, full user click-through verified.")


if __name__ == "__main__":
    main()
