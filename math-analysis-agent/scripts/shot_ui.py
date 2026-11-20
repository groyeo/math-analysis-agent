"""用系统 Edge（channel=msedge）给 Streamlit 界面截图，不需要额外下载 Chromium。

Streamlit 是 WebSocket 驱动的，networkidle 之后骨架屏可能还在，
所以要显式等真实元素出现再截，否则拿到的是 loading 占位框。
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "outputs" / "shots"
OUT.mkdir(parents=True, exist_ok=True)
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501/"


def shot(page, name: str, wait_selector: str | None = None) -> None:
    if wait_selector:
        page.wait_for_selector(wait_selector, timeout=60000)
    page.wait_for_timeout(2500)
    path = OUT / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    print("saved", path, path.stat().st_size, "bytes")


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 980}, device_scale_factor=1)
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector(".stTextArea textarea", timeout=60000)
        page.wait_for_timeout(3000)
        shot(page, "01_workbench")

        # 知识库
        page.get_by_text("知识库", exact=True).first.click()
        page.wait_for_selector(".stSelectbox", timeout=30000)
        shot(page, "02_knowledge")

        # 设置
        page.get_by_text("设置", exact=True).first.click()
        page.wait_for_timeout(2000)
        shot(page, "03_settings")

        browser.close()


if __name__ == "__main__":
    main()
