"""为知识块批量生成向量，存成 npz（供 VectorIndex 的 local / api 通路使用）。

用法：
    python scripts/build_embeddings.py                      # 本地 BGE 全量重建（默认，离线免费）
    python scripts/build_embeddings.py --incremental         # 只补新增块（id 未变则复用旧向量）
    python scripts/build_embeddings.py --backend api         # 改用云端 Embedding 接口
    python scripts/build_embeddings.py --model BAAI/bge-base-zh-v1.5   # 换向量模型

两条通路写到不同文件，互不覆盖：local → chunks_local.npz，api → chunks.npz。
npz 里带 meta（backend/model/dim），运行时用它校验「库和配置是否对得上」。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_settings  # noqa: E402
from src.retrieval.embed import APIEmbedder, LocalEmbedder  # noqa: E402

CHUNK_FILE = ROOT / "knowledge_base" / "chunks" / "chunks.jsonl"


def load_chunks() -> list[dict]:
    rows = []
    with open(CHUNK_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cid = str(d.get("chunk_id", len(rows)))
            # 只嵌「主题 + 正文」：主题里带定理名，对语义匹配很有用
            text = f"{d.get('topic', '')}\n{d.get('content', '')}".strip()
            rows.append({"chunk_id": cid, "text": text})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="local", choices=["local", "api"])
    ap.add_argument("--model", default="", help="覆盖 embedding 模型名")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--incremental", action="store_true", help="复用已存在且 id 相同的向量")
    args = ap.parse_args()

    cfg = get_settings().retrieval.vector
    if args.backend == "local":
        out_path = ROOT / cfg.local_embed_file
        embedder = LocalEmbedder(args.model or cfg.local_model, batch_size=args.batch)
        model_name = embedder.model_name
        origin = "本地 ONNX（fastembed）"
    else:
        out_path = ROOT / cfg.embed_file
        embedder = APIEmbedder(cfg, batch_size=args.batch)
        if args.model:
            embedder.model = args.model
        model_name = embedder.model
        origin = f"云端接口 @ {embedder.base_url}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()
    if not chunks:
        sys.exit(f"没有找到知识块：{CHUNK_FILE}")
    print(f"知识块 {len(chunks)} 条，输出 {out_path}")
    print(f"向量模型：{model_name}（{origin}）")

    old: dict[str, np.ndarray] = {}
    if args.incremental and out_path.exists():
        d = np.load(out_path, allow_pickle=False)
        old = {str(i): v for i, v in zip(d["ids"].tolist(), d["vectors"])}
        print(f"复用旧向量：{len(old)} 条")

    todo = [c for c in chunks if c["chunk_id"] not in old]
    print(f"需要新算：{len(todo)} 条")

    vec_by_id = dict(old)
    if todo:
        t0 = time.time()

        def _progress(done: int, total: int) -> None:
            if done % 160 == 0 or done == total:
                dt = time.time() - t0
                rate = done / dt if dt else 0
                print(f"  进度 {done}/{total}  {dt:.0f}s  ({rate:.0f} 块/秒)")

        mat = embedder.embed([c["text"] for c in todo], progress=_progress)
        for c, v in zip(todo, mat):
            vec_by_id[c["chunk_id"]] = v

    ids = [c["chunk_id"] for c in chunks]
    vectors = np.stack([vec_by_id[i] for i in ids]).astype("float32")
    meta = json.dumps(
        {
            "backend": args.backend,
            "model": model_name,
            "dim": int(vectors.shape[1]),
            "chunks": len(ids),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        ensure_ascii=False,
    )
    np.savez_compressed(out_path, ids=np.array(ids), vectors=vectors, meta=np.array(meta))
    print(f"已写入 {out_path.name}：{vectors.shape[0]} 条 × {vectors.shape[1]} 维")


if __name__ == "__main__":
    main()
