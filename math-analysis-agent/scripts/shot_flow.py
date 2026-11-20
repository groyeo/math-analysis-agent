"""跑一道题，截「思考中」与「最终报告」两张图（验证流式思考面板 + 新报告版式）。"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "outputs" / "shots"
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8501/"
Q = "求极限 lim_{x->0} (e^x - 1 - x)/x^2"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector(".stTextArea textarea", timeout=60000)
        page.wait_for_timeout(2500)
        page.fill(".stTextArea textarea", Q)
        page.get_by_text("开始解题", exact=True).click()
        # 思考面板应很快出现（首字节 1~2s + 检索）
        page.wait_for_selector(".think-card", timeout=60000)
        page.wait_for_timeout(14000)
        page.screenshot(path=str(OUT / "10_thinking.png"), full_page=False)
        print("saved 10_thinking", flush=True)
        # 等报告
        page.wait_for_selector("text=推导过程", timeout=240000)
        page.wait_for_timeout(3000)
        page.screenshot(path=str(OUT / "11_report.png"), full_page=True)
        print("saved 11_report", flush=True)
        # 展开思考过程
        try:
            page.get_by_text("思考过程 ·", exact=False).first.click()
            page.wait_for_timeout(1200)
            page.screenshot(path=str(OUT / "12_trace.png"), full_page=True)
            print("saved 12_trace", flush=True)
        except Exception as exc:  # noqa: BLE001
            print("trace shot skipped:", exc, flush=True)
        browser.close()


if __name__ == "__main__":
    main()
