"""移动端检查：390×844，看是否横向溢出、按钮是否可点。"""
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "outputs" / "shots"

with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    page = b.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2,
                      is_mobile=True, has_touch=True)
    page.goto("http://localhost:8501/", wait_until="domcontentloaded", timeout=60000)
    page.wait_for_selector(".stTextArea textarea", timeout=60000)
    page.wait_for_timeout(4000)
    info = page.evaluate("""() => ({
        sw: document.documentElement.scrollWidth,
        cw: document.documentElement.clientWidth,
        bw: document.body.scrollWidth,
    })""")
    print("overflow", info)
    page.screenshot(path=str(OUT / "50_mobile_top.png"), full_page=False)
    page.screenshot(path=str(OUT / "51_mobile_full.png"), full_page=True)
    b.close()
