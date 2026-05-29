"""Headless capture of Streamlit dashboard screenshots.

Prereq: dashboard is running on http://127.0.0.1:8501. The capture script
visits the page, waits for the network to idle (which Streamlit signals via
its app-ready state), then writes two PNGs into ``docs/screenshots``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

OUT = Path("docs/screenshots")


async def _capture(url: str = "http://127.0.0.1:8501/", preferred_pid: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            ctx = await browser.new_context(viewport={"width": 1440, "height": 900})
            page = await ctx.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=60000)

            if preferred_pid:
                # Streamlit selectbox: click to open, then click the option text.
                try:
                    await page.locator("div[data-baseweb='select']").first.click()
                    await page.wait_for_timeout(500)
                    await page.locator(
                        f"li[role='option']:has-text('{preferred_pid}')"
                    ).first.click()
                    await page.wait_for_timeout(1500)
                except Exception as exc:
                    print(f"WARN: could not select preferred PID {preferred_pid}: {exc}")

            try:
                await page.wait_for_selector("div.js-plotly-plot", timeout=45000)
            except Exception:
                print("WARN: plotly chart did not appear; capturing whatever rendered")
            await page.wait_for_timeout(2500)

            await page.screenshot(path=str(OUT / "dashboard_overview.png"), full_page=True)
            print(f"wrote {OUT / 'dashboard_overview.png'}")

            viewport_shot = OUT / "dashboard_topfold.png"
            await page.screenshot(path=str(viewport_shot), full_page=False)
            print(f"wrote {viewport_shot}")
        finally:
            await browser.close()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", default=None, help="Patient ID to pre-select in the dashboard")
    parser.add_argument("--url", default="http://127.0.0.1:8501/")
    args = parser.parse_args()
    try:
        asyncio.run(_capture(args.url, args.pid))
        return 0
    except Exception as exc:
        print(f"screenshot capture failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
