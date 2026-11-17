"""Capture README screenshots of the running Streamlit app (docs/screenshots/).

Start the app first (`streamlit run app/main.py`) and Ollama for the AI steps, then:
    python scripts/capture_screenshots.py [--url http://localhost:8501] [--skip-ai]
Needs `pip install playwright && python -m playwright install chromium`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def shot_between(page: Page, top_sel, bottom_sel, path: Path, pad: int = 12) -> None:
    """Screenshot the page from the top of one element to the bottom of another."""
    a = top_sel.bounding_box()
    b = bottom_sel.bounding_box()
    x0 = page.locator('[data-testid="stMain"]').bounding_box()
    page.screenshot(
        path=str(path),
        full_page=True,
        clip={
            "x": x0["x"],
            "y": max(a["y"] - pad, 0),
            "width": x0["width"],
            "height": b["y"] + b["height"] - a["y"] + 2 * pad,
        },
    )
    print("saved", path.name)


def shot_box(page: Page, locator, path: Path, pad: int = 6) -> None:
    """Screenshot an element's region; works while the element is being re-rendered (streaming)."""
    for _ in range(20):
        box = locator.bounding_box()
        if box:
            page.screenshot(
                path=str(path),
                full_page=True,
                clip={
                    "x": box["x"] - pad,
                    "y": box["y"] - pad,
                    "width": box["width"] + 2 * pad,
                    "height": box["height"] + 2 * pad,
                },
            )
            print("saved", path.name)
            return
        time.sleep(0.3)
    raise RuntimeError(f"no element for {path.name}")


def tab(page: Page, name: str) -> None:
    page.get_by_role("tab", name=name).click()
    time.sleep(1.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--skip-ai", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 3600}, device_scale_factor=1.5)
        page.goto(a.url)
        page.get_by_text("Space Weather Event Analyst").filter(visible=True).first.wait_for(
            timeout=60_000
        )
        time.sleep(4)
        for label in ("Don't show again", "Close"):  # Streamlit's own pop-ups, if any
            btn = page.get_by_role("button", name=label)
            if btn.count():
                btn.first.click()
                time.sleep(1)

        # 1 live tab (fresh NOAA data)
        page.get_by_role("button", name="Fetch latest NOAA data").click()
        page.get_by_text("Live data fetched from NOAA SWPC").filter(visible=True).first.wait_for(
            timeout=60_000
        )
        time.sleep(4)
        shot_between(
            page,
            page.locator(".hero"),
            page.get_by_text("Shaded band").filter(visible=True).first,
            OUT / "01_live.png",
        )

        # 2 what-if simulator
        tab(page, "🎛️ What-if simulator")
        page.get_by_role("button", name="Moderate storm").click()
        time.sleep(2)
        shot_between(
            page,
            page.get_by_text("What if…? Build your own space weather").filter(visible=True).first,
            page.get_by_role("button", name="Ask the AI what this would mean"),
            OUT / "02_whatif.png",
        )

        if not a.skip_ai:
            # 3 terminal chat while the model writes, then 4 when it is done
            page.get_by_role("button", name="Ask the AI what this would mean").click()
            term = page.locator("pre").filter(has_text="user@spacewx")
            term.wait_for(timeout=60_000)
            page.get_by_text("genai@spacewx:~$ Dst minimum").filter(visible=True).first.wait_for(
                timeout=300_000
            )
            time.sleep(6)
            shot_box(page, term.first, OUT / "03_terminal_chat_streaming.png")
            page.get_by_text("now checking every claim").filter(visible=True).first.wait_for(
                timeout=300_000
            )
            shot_box(page, term.first, OUT / "04_terminal_chat_done.png")
            # 5 the polished answer and the claim check, after the judge has finished
            page.wait_for_function(
                "document.querySelectorAll('.claim').length > 0 && "
                "document.querySelectorAll('.claim.wait').length === 0 && "
                "!document.body.innerText.includes('Verifying the cited sources')",
                timeout=300_000,
            )
            time.sleep(3)
            shot_between(
                page,
                page.get_by_text("What the AI says about this scenario").filter(visible=True).first,
                page.locator(".claim:visible").last,
                OUT / "05_ai_result_and_claim_check.png",
            )

        # 6 historical event (E216) and 7 its saved AI description with the claim check
        tab(page, "📚 Historical events")
        shot_between(
            page,
            page.get_by_text("E216 · super geomagnetic storm").filter(visible=True).first,
            page.locator(".js-plotly-plot:visible").first,
            OUT / "06_historical_event.png",
        )
        page.get_by_role("tab", name="🤖 AI description & check").click()
        time.sleep(2)
        shot_between(
            page,
            page.locator(".summary:visible").first,
            page.locator(".claim:visible").last,
            OUT / "07_historical_claims.png",
        )

        # 8 results, 9 how it works
        tab(page, "📊 Results")
        shot_between(
            page,
            page.get_by_text("Results on 15 evaluation events").filter(visible=True).first,
            page.locator(".card", has_text="What it shows").filter(visible=True).first,
            OUT / "08_results.png",
        )
        tab(page, "ℹ️ How it works")
        shot_between(
            page,
            page.locator(".steps:visible"),
            page.locator(".card", has_text="Explains past and current").filter(visible=True).first,
            OUT / "09_how_it_works.png",
            pad=24,
        )
        browser.close()


if __name__ == "__main__":
    main()
