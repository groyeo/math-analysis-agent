"""对比不同融合策略的检索指标，用于给 configs/retrieval.yaml 定参数。

用法：python scripts/tune_fusion.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402

EVAL = ROOT / "data" / "eval" / "eval_set.jsonl"
K = 5


def hit(chunk, row: dict) -> bool:
    rel = row.get("relevant") or []
    if rel:
        return chunk.chunk_id in rel
    chapter = row.get("expect_chapter") or ""
    kws = row.get("expect_keywords") or []
    ch_ok = (not chapter) or (chapter in chunk.chapter) or (not chunk.chapter)
    kw_ok = (not kws) or any(k in chunk.content for k in kws)
    return ch_ok and kw_ok


def evaluate(r: HybridRetriever, rows: list[dict]) -> tuple[float, float, float]:
    ok = top1 = 0
    mrr = 0.0
    for row in rows:
        res = r.retrieve(row["question"], top_k=K)
        flags = [hit(c, row) for c in res]
        if any(flags):
            ok += 1
        if flags and flags[0]:
            top1 += 1
        for i, f in enumerate(flags):
            if f:
                mrr += 1.0 / (i + 1)
                break
    n = len(rows)
    return ok / n, top1 / n, mrr / n


def main() -> None:
    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    base = get_settings().retrieval

    variants = []

    def add(name, fn):
        variants.append((name, fn))

    add("bm25 单路", lambda c: (setattr(c.bm25, "enabled", True), setattr(c.vector, "enabled", False)))
    add("vector 单路", lambda c: (setattr(c.bm25, "enabled", False), setattr(c.vector, "enabled", True)))
    add("rrf k=60", lambda c: setattr(c.fusion, "method", "rrf"))

    def w(vw, bw):
        def _f(c):
            c.fusion.method = "weighted"
            c.fusion.vector_weight = vw
            c.fusion.bm25_weight = bw
        return _f

    add("weighted 0.5/0.5", w(0.5, 0.5))
    add("weighted 0.4/0.6", w(0.4, 0.6))
    add("weighted 0.3/0.7", w(0.3, 0.7))
    add("weighted 0.6/0.4", w(0.6, 0.4))

    print(f"{'策略':<20} {'Recall@5':>9} {'Top-1':>7} {'MRR':>7}")
    print("-" * 48)
    for name, fn in variants:
        cfg = base.model_copy(deep=True)
        fn(cfg)
        r = HybridRetriever(cfg)
        rec, t1, mrr = evaluate(r, rows)
        print(f"{name:<20} {rec:>8.1%} {t1:>6.1%} {mrr:>7.3f}")


if __name__ == "__main__":
    main()
