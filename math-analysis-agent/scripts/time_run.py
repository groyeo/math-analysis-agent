"""测一次真实解题：打印每个节点的耗时，定位时间大头。"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

t0 = time.time()
last = [t0]


def emit(node: str, msg: str, data=None) -> None:
    now = time.time()
    print(f"[{now - t0:6.1f}s] (+{now - last[0]:5.1f}s) {node:8} {msg}", flush=True)
    last[0] = now


if __name__ == "__main__":
    from src.agents.pipeline import MathAgentPipeline

    q = sys.argv[1] if len(sys.argv) > 1 else "计算积分 ∫_0^1 x e^x dx"
    import src.agents.nodes as N

    print("加载 pipeline...", flush=True)
    t_load = time.time()
    pipe = MathAgentPipeline()
    print(f"加载完成 {time.time() - t_load:.1f}s", flush=True)

    # pipeline.run 若支持 emit 回调就直接用，否则 fallback
    try:
        res = pipe.run(q, emit=emit)
    except TypeError:
        res = pipe.run(q)
    print(f"\n总耗时 {time.time() - t0:.1f}s", flush=True)
    rep = getattr(res, "report", None) or {}
    print("budget:", rep.get("budget"), flush=True)
    print("steps:", len(rep.get("steps") or []), flush=True)
    print("degraded:", rep.get("degraded"), flush=True)
    # 顺带写一份给 UI 用，便于截图验证双栏布局
    if rep and not rep.get("degraded"):
        out = ROOT / "outputs" / "last_result.json"
        out.write_text(
            __import__("json").dumps(
                {"report": rep, "elapsed": time.time() - t0}, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        print("已写 outputs/last_result.json", flush=True)
