"""探测 Embedding 通路：找出当前 Key 能真正跑通的 base_url + model 组合。

用法：python scripts/probe_embed.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)

BASES = [
    ("modelscope", os.getenv("SOLVER_BASE_URL", "https://api-inference.modelscope.cn/v1")),
    ("dashscope", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
]

MODELS = [
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
    "qwen3.7-text-embedding-flash",
    "text-embedding-v3",
    "text-embedding-v4",
]

KEYS = {
    "solver": os.getenv("SOLVER_API_KEY", ""),
    "embed": os.getenv("EMBED_API_KEY", ""),
}


def main() -> None:
    from openai import OpenAI

    for base_name, base in BASES:
        for key_name, key in KEYS.items():
            if not key:
                continue
            for model in MODELS:
                try:
                    c = OpenAI(api_key=key, base_url=base, timeout=25, max_retries=0)
                    r = c.embeddings.create(model=model, input=["极限的定义"])
                    v = r.data[0].embedding
                    print(f"[OK]   {base_name:10} {key_name:5} {model:34} dim={len(v)}")
                except Exception as exc:  # noqa: BLE001
                    msg = " ".join(str(exc).split())[:110]
                    print(f"[FAIL] {base_name:10} {key_name:5} {model:34} {msg}")


if __name__ == "__main__":
    main()
