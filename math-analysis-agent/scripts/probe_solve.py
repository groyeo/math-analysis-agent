"""探测解题模型的原始输出：定位「无法解析为 JSON」的真实原因。

用法：python scripts/probe_solve.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.prompts import SOLVE_FAST_SYSTEM
from src.config import get_settings

QUESTION = r"""已知函数 \(f(x)=\begin{cases} A, & 0\leq x\leq \pi; \\ -A, & -\pi\leq x<0. \end{cases}\)
其中 \(A\neq0\) 为常数，求该函数在区间 \([-\pi,\pi]\) 上的傅里叶级数，并证明
\(\sum_{n=1}^{\infty}\frac{1}{(2n-1)^2}=\frac{\pi^2}{8}.\)
"""


def main() -> None:
    s = get_settings()
    spec = s.models.solver
    from openai import OpenAI

    client = OpenAI(api_key=spec.api_key, base_url=spec.base_url or None, timeout=180, max_retries=0)
    user = (
        f"题目结构：{json.dumps({'question': QUESTION, 'goal': '求傅里叶级数并证明级数求和', 'problem_type': '计算与证明题'}, ensure_ascii=False)}\n\n"
        "检索知识：\n（无可用检索结果，请基于题目本身作答并标注不确定性）"
    )
    for max_tokens in (4096, 8192):
        print("=" * 70)
        print(f"max_tokens={max_tokens}")
        try:
            resp = client.chat.completions.create(
                model=spec.model,
                messages=[{"role": "system", "content": SOLVE_FAST_SYSTEM}, {"role": "user", "content": user}],
                temperature=spec.temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            ch = resp.choices[0]
            msg = ch.message
            content = msg.content or ""
            extra = {k: v for k, v in msg.model_dump().items() if k not in ("content", "role") and v}
            print("finish_reason:", ch.finish_reason)
            print("usage:", resp.usage.model_dump() if resp.usage else None)
            print("content 长度:", len(content))
            print("额外字段 keys:", list(extra.keys()))
            for k, v in extra.items():
                print(f"  {k}: 长度={len(str(v))} 尾={str(v)[-160:]!r}")
            print("--- 头 500 ---")
            print(content[:500])
            print("--- 尾 400 ---")
            print(content[-400:])
            try:
                d = json.loads(content)
                print(">>> 直接 json.loads 成功, keys:", list(d.keys()))
            except Exception as e:
                print(">>> 直接解析失败:", e)
        except Exception as exc:  # noqa: BLE001
            print("调用失败:", str(exc)[:300])


if __name__ == "__main__":
    main()
