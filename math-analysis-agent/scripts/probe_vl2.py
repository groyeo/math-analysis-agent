"""探测 ModelScope 上哪些多模态模型能读数学题图片并输出 LaTeX。

为什么另起一个脚本：scripts/probe_vl.py 的候选清单是旧的（dashscope 那批
qwen-vl-* 已全部 401，ModelScope 也没有 Qwen2.5-VL / Qwen3-VL），
而 ModelScope 的可用模型表每季度都在变，必须实测。

用法：
    python scripts/probe_vl2.py
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

MODELSCOPE = "https://api-inference.modelscope.cn/v1"

# 候选：只挑模型名里带 VL / Intern 的（纯文本模型不可能读图，不必浪费调用）
CANDIDATES = [
    "Shanghai_AI_Laboratory/Intern-S1-mini",
    "Shanghai_AI_Laboratory/Intern-S1",
    "Shanghai_AI_Laboratory/Intern-S2-Preview",
    "OpenGVLab/InternVL3_5-241B-A28B",
    "PaddlePaddle/ERNIE-4.5-VL-28B-A3B-PT",
]

FONT = "/Windows/Fonts/simsun.ttc"

PROMPT = """你是数学题识别助手。把图片里的题目原样转写成文本：
1. 所有数学公式用 LaTeX，行内用 $...$；
2. 只输出题目本身，不要解答、不要解释、不要加标题；
3. 看不清的符号用 [?] 标记。"""


def make_question_image() -> tuple[str, str]:
    """造一张仿真教材题图：中文题干 + 分式结构的公式。"""
    img = Image.new("RGB", (900, 340), "white")
    d = ImageDraw.Draw(img)
    big = ImageFont.truetype(FONT, 30)
    mid = ImageFont.truetype(FONT, 26)

    d.text((50, 32), "例 3.2", font=big, fill="black")
    d.text((50, 86), "求下列极限，并讨论 f(x) 在 x = 0 处是否连续：", font=mid, fill="black")

    # 分式：分子 / 横线 / 分母，模拟印刷体的上下结构
    d.text((150, 150), "e", font=big, fill="black")
    d.text((172, 138), "x", font=mid, fill="black")
    d.text((190, 150), " - 1 - x", font=big, fill="black")
    d.line((140, 205, 420, 205), fill="black", width=2)
    d.text((150, 215), "x", font=big, fill="black")
    d.text((172, 203), "2", font=mid, fill="black")

    d.text((60, 165), "lim", font=big, fill="black")
    d.text((68, 200), "x→0", font=mid, fill="black")

    d.text((50, 285), "其中 f(x) = [ (e^x - 1 - x) / x^2 , x≠0 ; 1/2 , x=0 ]", font=mid, fill="black")

    out = ROOT / "outputs" / "_probe_question.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return base64.b64encode(out.read_bytes()).decode(), "png"


def probe(model: str, key: str, b64: str, ext: str) -> tuple[bool, str]:
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=MODELSCOPE, timeout=120, max_retries=0)
    try:
        t0 = time.time()
        r = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=2000,
        )
        dt = time.time() - t0
        msg = r.choices[0].message
        content = (msg.content or "").strip()
        reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
        usage = getattr(r, "usage", None)
        ct = getattr(usage, "completion_tokens", 0) or 0
        if not content:
            return False, f"content 为空（reasoning {len(reasoning)} 字 / {ct} tokens / {dt:.1f}s）"
        return True, f"{dt:.1f}s | {ct}tok | {content[:150]}"
    except Exception as e:  # noqa: BLE001
        return False, str(e).replace("\n", " ")[:150]


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = os.getenv("SOLVER_API_KEY", "")
    if not key:
        print("缺少 SOLVER_API_KEY")
        return

    b64, ext = make_question_image()
    print(f"测试图已生成：outputs/_probe_question.png（{len(b64) // 1024} KB base64）\n")

    ok_list = []
    for model in CANDIDATES:
        ok, info = probe(model, key, b64, ext)
        print(f"[{'可用' if ok else '不可用'}] {model}")
        print(f"    {info}\n")
        if ok:
            ok_list.append(model)

    print("可用清单：")
    for m in ok_list:
        print(" ", m)
    if not ok_list:
        print("  （无）")


if __name__ == "__main__":
    main()
