"""命令行解题入口：python scripts/solve.py "计算 lim(x->0)(e^x-1-x)/x^2" """
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.pipeline import MathAgentPipeline


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or "计算 lim(x→0) (e^x - 1 - x)/x^2"
    print(f"题目：{question}\n" + "=" * 60)

    def emit(node: str, message: str, data: dict | None = None) -> None:
        print(f"[{node:8}] {message}")

    result = MathAgentPipeline(emit=emit).run(question)
    print("=" * 60)
    print(json.dumps(result.report, ensure_ascii=False, indent=2))
    print("=" * 60)
    print("预算：", result.budget, "| mock:", result.mock)


if __name__ == "__main__":
    main()
