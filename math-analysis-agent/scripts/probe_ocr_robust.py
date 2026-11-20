"""模拟手机拍照的劣化场景，验证识别鲁棒性。

真实用户拍出来的图几乎不会是干净的正视图：会倾斜、有阴影、有压缩噪点、
纸张底色发灰。这个脚本把这些退化逐个加上，看识别还能不能还原题目。

用法：
    python scripts/probe_ocr_robust.py
"""
from __future__ import annotations

import io
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
# 从 scripts/ 目录启动时 sys.path[0] 是 scripts/，import src 会失败
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def base_page() -> Image.Image:
    """一张干净题图：两道不同的题，避免和之前那张混。"""
    img = Image.new("RGB", (860, 300), (252, 252, 250))
    d = ImageDraw.Draw(img)
    from PIL import ImageFont

    f = "/Windows/Fonts/simsun.ttc"
    big = ImageFont.truetype(f, 28)
    mid = ImageFont.truetype(f, 24)
    d.text((44, 30), "习题 7.4", font=big, fill=(20, 20, 20))
    d.text((44, 84), "判断下列级数是否收敛，若收敛求其和：", font=mid, fill=(20, 20, 20))
    d.text((44, 130), "∑ (n=1→∞)  1 / (n(n+1))", font=big, fill=(20, 20, 20))
    d.text((44, 190), "并说明所用的判别法。", font=mid, fill=(20, 20, 20))
    return img


def degrade(img: Image.Image, *, angle: float, blur: float, shadow: bool) -> Image.Image:
    # 1) 倾斜：先扩画布再旋转，边缘填纸张色，模拟手持不正
    big = Image.new("RGB", (img.width + 220, img.height + 220), (250, 250, 248))
    big.paste(img, (110, 110))
    out = big.rotate(angle, resample=Image.BICUBIC, fillcolor=(248, 248, 246))

    # 2) 光照不均：一侧压暗，模拟窗边阴影
    if shadow:
        mask = Image.new("L", out.size, 0)
        md = ImageDraw.Draw(mask)
        for x in range(out.width):
            md.line([(x, 0), (x, out.height)], fill=int(70 * (x / out.width)))
        dark = Image.new("RGB", out.size, (0, 0, 0))
        out = Image.composite(dark, out, mask.point(lambda v: int(v * 0.55)))

    # 3) 模糊 + 对比度下降 + 噪点，模拟手机镜头与 JPEG 压缩
    out = out.filter(ImageFilter.GaussianBlur(blur))
    out = ImageEnhance.Contrast(out).enhance(0.86)
    px = out.load()
    for _ in range(out.width * out.height // 220):
        x, y = random.randint(0, out.width - 1), random.randint(0, out.height - 1)
        v = random.randint(-16, 16)
        r, g, b = px[x, y]
        px[x, y] = (max(0, min(255, r + v)), max(0, min(255, g + v)), max(0, min(255, b + v)))
    return out


def main() -> None:
    from src.ingestion.document import parse_image
    from src.ingestion.ocr import OcrError, recognize_question

    out_dir = ROOT / "outputs" / "ocr_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    page = base_page()
    page.save(out_dir / "clean.png")

    cases = [
        ("轻微倾斜 + 轻微模糊", dict(angle=1.8, blur=0.5, shadow=False)),
        ("明显倾斜 + 阴影 + 模糊", dict(angle=4.2, blur=0.9, shadow=True)),
        ("手抖 + 重噪点", dict(angle=-2.6, blur=1.3, shadow=True)),
    ]

    for name, kw in cases:
        img = degrade(page, **kw)
        fp = out_dir / f"degraded_{abs(kw['angle'])}.jpg"
        img.save(fp, quality=68)
        print(f"\n{'=' * 60}\n场景：{name}　（{fp.name}, {fp.stat().st_size // 1024}KB）")
        try:
            res = recognize_question(fp.read_bytes())
            print(f"耗时 {res.elapsed:.1f}s / {res.completion_tokens} tokens")
            print("识别：")
            print(res.text)
        except OcrError as exc:
            print(f"失败：{exc}")

    # 顺带验证 document.parse_image 这条对外接口
    text, warns = parse_image(out_dir / "clean.png")
    print(f"\n{'=' * 60}\nparse_image 干净图：{len(text)} 字，警告 {warns}")


if __name__ == "__main__":
    main()
