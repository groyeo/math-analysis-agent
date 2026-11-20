"""把 last_result.json 的步骤重新做一次「定理 → 教材原文」匹配（不重跑 LLM）。

用途：验证匹配策略的覆盖率，顺便让 UI 立刻能看到效果。
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_STRIP = re.compile(r"[\s\(\)（）\[\]{}，。；、：:“”''《》<>\-—_]")


def norm(s: str) -> str:
    return _STRIP.sub("", str(s or ""))


def bigrams(s: str) -> set[str]:
    s = norm(s)
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else set()


def lcs(a: str, b: str) -> int:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0
    best, prev = 0, [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def thm_score(thm: str, c) -> float:
    key = norm(thm)
    need = min(4, len(key))
    if need < 2:
        return 0.0
    if lcs(thm, c.topic) >= need:
        return 1.0
    if lcs(thm, c.content) >= need:
        return 0.9
    kg = bigrams(thm)
    if not kg:
        return 0.0
    return max(len(kg & bigrams(c.topic)) / len(kg) * 0.5,
               len(kg & bigrams(c.content)) / len(kg) * 0.4)


def main() -> None:
    from src.retrieval.hybrid import HybridRetriever

    r = HybridRetriever()
    f = ROOT / "outputs" / "last_result.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    rep = d["report"]
    by_id = {c.chunk_id: c for c in r.chunks}
    ctx_ids = [s["chunk_id"] for s in rep.get("sources", [])]
    ctx = [by_id[i] for i in ctx_ids if i in by_id]

    def match_in(chunks, thm):
        key = norm(thm)
        for field in ("topic", "content"):
            for c in chunks:
                if key and key in norm(getattr(c, field, "")):
                    return c.chunk_id, 1.0, "exact"
        best, bs = None, 0.0
        for c in chunks:
            sc = thm_score(thm, c)
            if sc > bs:
                best, bs = c.chunk_id, sc
        return (best, bs, "lcs") if bs >= 0.95 else (None, bs, "low")

    def lookup(thm):
        best, bs = None, 0.0
        for c, _ in r.bm25.search(thm, 8):
            sc = thm_score(thm, c)
            if getattr(c, "content_type", "") in ("定理", "推论", "性质", "定义"):
                sc += 0.05
            if sc > bs:
                best, bs = c.chunk_id, sc
        return (best, bs) if bs >= 0.95 else (None, bs)

    extra = []
    for i, st in enumerate(rep.get("steps", []), 1):
        thm = st.get("theorem") or ""
        sid, sc, how = match_in(ctx, thm)
        if not sid:
            sid, sc2 = lookup(thm)
            how = f"corpus({sc2:.2f})"
            if sid:
                extra.append(sid)
        st["source_id"] = sid
        print(f"{i}. {thm or '(空)':22} → {sid or '未匹配'}  [{how}]")

    have = {s["chunk_id"] for s in rep.get("sources", [])}
    for cid in dict.fromkeys(extra):
        if cid in have or cid not in by_id:
            continue
        c = by_id[cid]
        rep["sources"].append({"citation": f"{c.source} · {c.chapter}",
                               "chunk_id": cid, "score": None})

    f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    n = sum(1 for s in rep["steps"] if s.get("source_id"))
    print(f"\n匹配 {n}/{len(rep['steps'])} 步；已写回 {f}")


if __name__ == "__main__":
    main()
