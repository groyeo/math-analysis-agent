"""端到端验证「题目照片 → 自动识别 → 填入题目框」。

用法：
    python scripts/verify_photo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "shots"
IMG = ROOT / "outputs" / "_probe_question.png"
URL = "http://localhost:8501/"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not IMG.exists():
        print(f"缺少测试图 {IMG}，先跑 scripts/probe_vl2.py 生成")
        sys.exit(1)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1500})
        page.goto(URL, wait_until="domcontentloaded", timeout=90000)
        page.wait_for_selector("text=用题目照片输入", timeout=60000)
        print("页面已加载")

        page.get_by_text("用题目照片输入", exact=True).first.click()
        page.wait_for_timeout(1200)
        page.screenshot(path=str(OUT / "20_upload_empty.png"), full_page=True)
        print("已展开上传区")

        page.locator("input[type=file]").first.set_input_files(str(IMG))
        print("文件已传入，等待识别…")
        # 识别走网络（实测 1~3 秒），留足余量
        page.wait_for_selector("text=识别结果", timeout=120000)
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / "21_recognized.png"), full_page=True)

        # 识别成功后如果题目框是空的会自动填入（并 rerun），确认一次
        page.wait_for_timeout(1500)
        filled = page.locator("textarea").first.input_value()
        print("题目框内容：")
        print(filled[:400] if filled else "（空）")
        print("自动填入：", "成功" if filled.strip() else "失败")

        page.screenshot(path=str(OUT / "22_autofilled.png"), full_page=True)
        browser.close()
        print("截图已保存：20_upload_empty / 21_recognized / 22_autofilled")


if __name__ == "__main__":
    main()
