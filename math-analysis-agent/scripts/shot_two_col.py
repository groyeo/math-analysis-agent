"""截图验证：左题右文的双栏布局 + 步骤/原文高亮联动。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright

OUT = ROOT / "outputs" / "shots"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 1600, "height": 2200})
        pg.goto("http://localhost:8501/", wait_until="domcontentloaded")
        pg.wait_for_selector("text=定理 · 教材原文", timeout=90000)
        pg.wait_for_timeout(3500)
        print("左右两栏:", pg.locator("[data-testid='stColumn']").count())
        print("步骤卡:", pg.locator("[class*='st-key-stepcard-']").count())
        print("原文卡:", pg.locator("[class*='st-key-srcdoc-']").count())
        print("右栏高亮:", pg.locator("[class*='st-key-srcdoc-']:has(.src-flag.on)").count())
        pg.screenshot(path=str(OUT / "09_two_col.png"))
        # 点第 3 步的「查看依据」，看高亮是否切到第 3 条
        btns = pg.locator("[class*='st-key-focus-'] button")
        print("依据按钮数:", btns.count())
        if btns.count() >= 3:
            btns.nth(2).click()
            pg.wait_for_timeout(2500)
            on = pg.locator("[class*='st-key-srcdoc-']:has(.src-flag.on)")
            print("点击后高亮条数:", on.count())
            try:
                print("高亮的是:", on.first.inner_text()[:40].replace("\n", " "))
            except Exception:
                pass
            pg.screenshot(path=str(OUT / "10_highlight_step3.png"))
        b.close()


if __name__ == "__main__":
    main()
