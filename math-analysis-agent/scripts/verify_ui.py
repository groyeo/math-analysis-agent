"""端到端验证：解题 → 切页签 → 切回，结果是否还在。

对应报障「输入题目再返回，进程就没了」。除了截图，还把关键判据打印出来：
- 侧栏向量召回是否显示正常；
- 切回工作台后「推导过程」卡片是否仍然存在（旧代码这里会整块消失）。
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
        page = browser.new_page(viewport={"width": 1440, "height": 2400}, device_scale_factor=1)
        page.goto(URL, wait_until="domcontentloaded", timeout=90000)
        page.wait_for_selector(".stTextArea textarea", timeout=90000)
        page.wait_for_timeout(3500)

        sidebar = page.locator('[data-testid="stSidebar"]').inner_text()
        print("[侧栏]\n" + sidebar[:400])

        page.get_by_role("button", name="极限计算").first.click()
        page.wait_for_timeout(2000)
        page.get_by_role("button", name="开始解题").first.click()

        # 关键：不能只等「推导过程」——页面上的「上次结果」里也有这几个字，
        # 会立刻命中，让人误以为任务已完成。要等进度条出现再消失。
        page.wait_for_selector("text=解题中…", timeout=60000)
        print("进度条已出现，等待解题完成…")
        page.wait_for_selector("text=解题中…", state="detached", timeout=300000)
        page.wait_for_selector("text=推导过程", timeout=60000)
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / "10_result.png"), full_page=True)
        print("结果页已截图")

        page.get_by_text("知识库", exact=True).first.click()
        page.wait_for_timeout(5000)
        page.screenshot(path=str(OUT / "11_kb.png"), full_page=True)

        page.get_by_text("解题工作台", exact=True).first.click()
        page.wait_for_timeout(6000)
        kept = page.locator("text=推导过程").count()
        print("切回后「推导过程」卡片数量：", kept, "→", "结果保留" if kept else "结果丢失")
        page.screenshot(path=str(OUT / "12_back.png"), full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
