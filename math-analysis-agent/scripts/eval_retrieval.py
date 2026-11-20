"""检索质量评估：Recall@k（方案 7.4 / 13.1 的验收指标）。

评估集格式（data/eval/eval_set.jsonl），每行支持两种写法：
  1) {"id":"q1","question":"...","relevant":["c01","c04"]}          # 按 chunk_id（旧示例库）
  2) {"id":"q1","question":"...","expect_chapter":"函数极限","expect_keywords":["洛必达","0/0"]}

判定：Top-k 中任一块满足「章节匹配 且 命中任一关键词」即算召回成功。

用法：
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --k 5 --file data/eval/eval_set.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.hybrid import HybridRetriever


def hit(chunk, row: dict) -> bool:
    rel = row.get("relevant") or []
    if rel:
        return chunk.chunk_id in rel
    chapter = row.get("expect_chapter") or ""
    kws = row.get("expect_keywords") or []
    ch_ok = (not chapter) or (chapter in chunk.chapter) or (not chunk.chapter)
    kw_ok = (not kws) or any(k in chunk.content for k in kws)
    return ch_ok and kw_ok


def run_once(rows, r, k, label, force_rerank=None) -> tuple[int, int, float, list]:
    """跑一轮评估，返回 (命中数, top1数, mrr分子, 漏召回id列表)。"""
    if force_rerank is not None and getattr(r.cfg, "rerank", None):
        r.cfg.rerank.enabled = force_rerank
    ok = top1 = 0
    mrr = 0.0
    miss = []
    for row in rows:
        res = r.retrieve(row["question"], top_k=k)
        flags = [hit(c, row) for c in res]
        got = any(flags)
        ok += int(got)
        if flags and flags[0]:
            top1 += 1
        for i, f in enumerate(flags):
            if f:
                mrr += 1.0 / (i + 1)
                break
        else:
            miss.append(row["id"])
    return ok, top1, mrr, miss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--file", default=str(ROOT / "data" / "eval" / "eval_set.jsonl"))
    ap.add_argument("--rerank", choices=["on", "off", "both"], default="both",
                    help="on=强制开精排 off=强制关精排 both=按配置（默认）")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    r = HybridRetriever()
    print(f"知识库：{len(r.chunks)} 块 | 检索状态：{r.status()}")
    print("=" * 70)

    n = len(rows)

    def report(label, ok, top1, mrr, miss):
        print(f"\n[{label}]")
        print(f"  Recall@{args.k} = {ok}/{n} = {ok / n:.1%}" if n else "  评估集为空")
        print(f"  Top-1  命中  = {top1}/{n} = {top1 / n:.1%}")
        print(f"  MRR          = {mrr / n:.3f}" if n else "")
        if miss:
            print(f"  完全未召回：{miss}")

    if args.rerank == "both":
        # 先关精排跑一遍，再开精排跑一遍，对比收益
        ok0, t0, m0, miss0 = run_once(rows, r, args.k, "off", force_rerank=False)
        report("无精排（baseline）", ok0, t0, m0, miss0)
        ok1, t1, m1, miss1 = run_once(rows, r, args.k, "on", force_rerank=True)
        report("精排（rerank）", ok1, t1, m1, miss1)
        print("\n" + "=" * 70)
        print(f"Recall@{args.k}: {ok0/n:.1%} → {ok1/n:.1%}  ({'+' if ok1>=ok0 else ''}{ok1-ok0} 题)")
        print(f"Top-1      : {t0/n:.1%} → {t1/n:.1%}  ({'+' if t1>=t0 else ''}{t1-t0} 题)")
        print(f"MRR        : {m0/n:.3f} → {m1/n:.3f}")
        print(f"\n阶段目标 Recall@{args.k} ≥80% → {'达标' if ok1 / n >= 0.80 else '未达标'}")
        print(f"建议目标 Top-1 ≥60% → {'达标' if t1 / n >= 0.60 else '未达标'}")
        return

    force = True if args.rerank == "on" else False
    ok, top1, mrr, miss = run_once(rows, r, args.k, args.rerank, force_rerank=force)
    report(f"精排={args.rerank}", ok, top1, mrr, miss)
    print("=" * 70)
    print(f"阶段目标 Recall@{args.k} ≥80% → {'达标' if n and ok / n >= 0.80 else '未达标'}")
    print(f"建议目标 Top-1 ≥60% → {'达标' if n and top1 / n >= 0.60 else '未达标'}")


if __name__ == "__main__":
    main()
