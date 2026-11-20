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
    page.mouse.move(195, 500)
    for i in range(3):
        page.mouse.wheel(0, 1400)
        page.wait_for_timeout(900)
        page.screenshot(path=str(OUT / f"53_mobile_{i}.png"))
        print("shot", i)
    b.close()
