"""验证按钮悬停动效：只允许「一次轻轻上浮」，并确认页面其它元素不位移。

量三个数：
- 按钮矩形在 hover 前后的 y 偏移（应约为 -1px）
- computed transform（应是 matrix(1,0,0,1,0,-1) 之类，不是动画）
- 相邻「试试这些」那一行的 y 偏移（应为 0，说明整页是稳的）
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "outputs" / "shots"
OUT.mkdir(parents=True, exist_ok=True)
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501/"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(URL, wait_until="domcontentloaded")
        btn = page.get_by_role("button", name="开始解题").first
        btn.wait_for(state="visible", timeout=60000)
        page.wait_for_timeout(2000)

        before = btn.bounding_box()
        sample = page.get_by_role("button", name="极限计算").first
        ref_before = sample.bounding_box()

        btn.hover()
        page.wait_for_timeout(400)
        after = btn.bounding_box()
        ref_after = sample.bounding_box()
        tf = btn.evaluate("el => getComputedStyle(el).transform")
        sh = btn.evaluate("el => getComputedStyle(el).boxShadow")

        print(f"button dy = {after['y'] - before['y']:.2f}px  transform = {tf}")
        print(f"shadow = {sh}")
        print(f"nearby row dy = {ref_after['y'] - ref_before['y']:.2f}px (should be 0)")

        clip = {"x": max(0, before["x"] - 24), "y": max(0, before["y"] - 24),
                "width": before["width"] + 220, "height": before["height"] + 48}
        page.screenshot(path=str(OUT / "70_hover.png"), clip=clip)
        print("saved", OUT / "70_hover.png")
        browser.close()


if __name__ == "__main__":
    main()
