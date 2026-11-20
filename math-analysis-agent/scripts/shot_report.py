"""报告页截图：填题 → 点「开始解题」→ 等推导过程出现后整页截图。

auto 模式：题目从 outputs/last_result.json 里取，保证截图里的内容是真实一次求解。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "shots"
OUT.mkdir(parents=True, exist_ok=True)


def last_question() -> str:
    f = ROOT / "outputs" / "last_result.json"
    if f.exists():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            q = (d.get("report", {}) or {}).get("question") or ""
            if q.strip():
                return q.strip()
        except Exception:  # noqa: BLE001
            pass
    return "求极限 lim_{x->0} (e^x - 1 - x)/x^2"


def main() -> None:
    q = last_question()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 2400}, device_scale_factor=1)
        page.goto("http://localhost:8501/", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector(".stTextArea textarea", timeout=60000)
        # 把 last_result 里的题目填回去就够：题目一字不差时直接复用这份结果，
        # 不重复调模型（这正是新版「同一题目 = 同一份结果」的行为）。
        page.fill(".stTextArea textarea", q)
        page.keyboard.press("Tab")
        page.wait_for_selector("text=推导过程", timeout=90000)
        page.wait_for_timeout(3000)
        page.screenshot(path=str(OUT / "20_report_full.png"))
        print("saved 20_report_full", flush=True)
        try:
            page.get_by_text("思考过程 ·", exact=False).first.click()
            page.wait_for_timeout(1200)
            page.screenshot(path=str(OUT / "21_trace_full.png"))
            print("saved 21_trace_full", flush=True)
        except Exception as exc:  # noqa: BLE001
            print("trace skipped:", exc, flush=True)
        browser.close()


if __name__ == "__main__":
    sys.exit(main())
