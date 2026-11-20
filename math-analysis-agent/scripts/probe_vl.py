"""探测可用的视觉 OCR 模型：逐个试一遍候选（含跨平台），输出可用清单。

用法：
    python scripts/probe_vl.py
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DASHSCOPE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODELSCOPE = "https://api-inference.modelscope.cn/v1"

# (平台, base_url, key_env, 模型)
CANDIDATES = [
    ("dashscope", DASHSCOPE, "REVIEWER_API_KEY", "qwen-vl-ocr"),
    ("dashscope", DASHSCOPE, "REVIEWER_API_KEY", "qwen-vl-plus"),
    ("dashscope", DASHSCOPE, "REVIEWER_API_KEY", "qwen-vl-max"),
    ("dashscope", DASHSCOPE, "REVIEWER_API_KEY", "qwen3-vl-plus"),
    ("dashscope", DASHSCOPE, "REVIEWER_API_KEY", "qwen3-vl-flash"),
    ("modelscope", MODELSCOPE, "SOLVER_API_KEY", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ("modelscope", MODELSCOPE, "SOLVER_API_KEY", "Qwen/Qwen2.5-VL-72B-Instruct"),
    ("modelscope", MODELSCOPE, "SOLVER_API_KEY", "Qwen/Qwen3-VL-8B-Instruct"),
    ("modelscope", MODELSCOPE, "SOLVER_API_KEY", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
]


def make_probe_image() -> tuple[str, str]:
    """画一张极小的测试图：一行中文 + 一个公式。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (420, 90), "white")
    d = ImageDraw.Draw(img)
    d.text((14, 30), "定理 1.1 (测试)  x^2 + 1 = 0", fill="black")
    tmp = ROOT / "outputs" / "_probe.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    img.save(tmp)
    return base64.b64encode(tmp.read_bytes()).decode(), "png"


def probe(base: str, key: str, model: str, b64: str, ext: str) -> tuple[bool, str]:
    client = OpenAI(api_key=key, base_url=base, timeout=60, max_retries=0)
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{ext};base64,{b64}"},
                        },
                        {"type": "text", "text": "把图里的文字原样输出，公式用 LaTeX。"},
                    ],
                }
            ],
            max_tokens=120,
        )
        out = (r.choices[0].message.content or "").strip().replace("\n", " ")
        return True, out[:80]
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")
        if "403" in msg and "quota" in msg.lower():
            return False, "额度耗尽(403)"
        if "404" in msg or "not found" in msg.lower() or "not exist" in msg.lower():
            return False, "模型不存在(404)"
        if "401" in msg or "unauthor" in msg.lower():
            return False, "鉴权失败(401)"
        return False, msg[:110]


def main() -> None:
    b64, ext = make_probe_image()
    ok_models = []
    for plat, base, key_env, model in CANDIDATES:
        key = os.getenv(key_env, "")
        if not key:
            print(f"[跳过] {plat:10s} {model:38s} 未配置 {key_env}")
            continue
        ok, info = probe(base, key, model, b64, ext)
        flag = "✅ 可用" if ok else "❌ 不可用"
        print(f"[{flag}] {plat:10s} {model:38s} {info}")
        if ok:
            ok_models.append((plat, base, key_env, model))
    print("\n可用清单：")
    for plat, _, key_env, model in ok_models:
        print(f"  {plat} | {key_env} | {model}")
    if not ok_models:
        print("  （无）")


if __name__ == "__main__":
    main()
