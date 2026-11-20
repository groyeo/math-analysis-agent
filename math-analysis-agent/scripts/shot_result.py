"""截一张真实解题结果页：点示例题 → 等完整链路跑完 → 截图。

这条链路要真调三次模型（约 30~60s），所以等待判据是「结论」标题出现，而不是固定 sleep。
"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "outputs" / "shots"
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8501/"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        # 视口给高一点：Streamlit 的滚动容器不是 body，full_page 截不到首屏之外的内容，
        # 想要完整报告只能把视口撑到足够高。
        page = browser.new_page(viewport={"width": 1440, "height": 2600}, device_scale_factor=1)
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector(".stTextArea textarea", timeout=60000)
        page.wait_for_timeout(2500)

        # 点「极限计算」示例，题目会写进输入框
        page.get_by_role("button", name="极限计算").first.click()
        page.wait_for_timeout(2000)

        page.get_by_role("button", name="开始解题").first.click()
        # 解析 → 结构化/检索（并行）→ 解题 → 验证 → 审核，真跑约 30~60s
        page.wait_for_selector("text=推导过程", timeout=180000)
        page.wait_for_timeout(2500)

        path = OUT / "04_result.png"
        page.screenshot(path=str(path), full_page=True)
        print("saved", path, path.stat().st_size, "bytes")
        browser.close()


if __name__ == "__main__":
    main()
