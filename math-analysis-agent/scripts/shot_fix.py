"""验证两件事：审查/局限区的公式是否渲染成 KaTeX、知识来源能否点开跳转。

判据不靠肉眼：
1) 页面上不应再出现裸 LaTeX 片段（如 "\\in C^2"、"\\overline"）；
2) 点第一条来源后，知识库页出现 "来源定位" 卡片，且卡片里能看到该片段正文。
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
        page = browser.new_page(viewport={"width": 1440, "height": 2800}, device_scale_factor=1)
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector("text=局限与待核查", timeout=60000)
        page.wait_for_timeout(2500)

        # 报告区应有的 KaTeX 节点（结论、步骤、局限各有一批）
        katex = page.locator(".katex").count()
        print("katex nodes in report:", katex)

        body = page.inner_text("body")
        bad = [tok for tok in ("\\in C^2", "\\overline", "\\cap", "\\to 1", "\\(") if tok in body]
        print("raw latex leftovers:", bad or "none")

        path = OUT / "05_report_fix.png"
        page.screenshot(path=str(path), full_page=True)
        print("saved", path)

        # 点第一条知识来源
        btn = page.locator("[class*='st-key-src-'] button").first
        label = btn.inner_text()
        print("first source button:", label.replace("\n", " "))
        btn.click()
        page.wait_for_selector("text=来源定位", timeout=30000)
        page.wait_for_timeout(2000)
        path2 = OUT / "06_jump_focus.png"
        page.screenshot(path=str(path2), full_page=True)
        print("saved", path2)
        focus_text = page.locator("[class*='st-key-kb-focus']").inner_text()
        print("focus card head:", focus_text[:120].replace("\n", " | "))
        browser.close()


if __name__ == "__main__":
    main()
