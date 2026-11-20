"""混合检索：向量 + BM25 + 元数据过滤 + RRF 融合（见方案 7.4）。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from src.config import RetrievalConfig, get_settings
from src.retrieval.rerank import Reranker

_ZH = re.compile(r"[\u4e00-\u9fff]+")
_ASCII = re.compile(r"[a-z]+|\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    """英文/数字 token（长度≥2）+ 中文二元组。

    丢弃单个字母（x、e 等）与单个汉字，避免公式符号带来的检索噪声。
    """
    text = (text or "").lower()
    tokens = [t for t in _ASCII.findall(text) if len(t) >= 2]
    for seg in _ZH.findall(text):
        if len(seg) == 1:
            tokens.append(seg)
            continue
        tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))
    return tokens


@dataclass
class Chunk:
    chunk_id: str
    content: str
    chapter: str = ""
    section: str = ""
    content_type: str = ""
    topic: str = ""
    source: str = ""
    page: str = ""
    score: float = 0.0
    matched_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chapter": self.chapter,
            "section": self.section,
            "content_type": self.content_type,
            "topic": self.topic,
            "source": self.source,
            "page": self.page,
            "score": round(self.score, 4),
            "matched_by": list(self.matched_by),
        }

    @property
    def citation(self) -> str:
        loc = self.source or "知识库"
        if self.chapter:
            loc += f" · {self.chapter}"
        if self.page:
            loc += f" · p.{self.page}"
        return loc


def _load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    if not path.exists():
        return chunks
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunks.append(
                Chunk(
                    chunk_id=str(d.get("chunk_id", len(chunks))),
                    content=d.get("content", ""),
                    chapter=d.get("chapter", ""),
                    section=d.get("section", ""),
                    content_type=d.get("content_type", ""),
                    topic=d.get("topic", ""),
                    source=d.get("source", ""),
                    page=str(d.get("page", "")),
                )
            )
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._index = None
        self._simple = False
        self._tokens: list[list[str]] = []
        if not chunks:
            return
        # 字段加权：topic 是「定理 5.1.3 Lagrange 中值定理」这类标题，
        # 是学生提问时最直接命中的信号；但 content 常有几百字，标题只占十几个字，
        # 不做加权就会被正文淹没（表现为「问 Lagrange 却先返回 Darboux」）。
        # 这里把 topic 重复计入并额外带上章节标题，模拟字段权重。
        self._tokens = [
            tokenize(f"{c.content} {c.topic} {c.topic} {c.topic} {c.chapter} {c.chapter}") for c in chunks
        ]
        try:
            # BM25Plus 的 IDF 恒为正；Okapi 版对「定理」「函数」这类高频词会给负 IDF，
            # 一过滤 score>0 就只剩一两条结果（教材大库上尤其明显）。
            from rank_bm25 import BM25Plus

            # 试过把 b 从 0.75 调到 0.9（加强长度归一化），Top-1 无改善、MRR 反降，
            # 说明首位错并非长块占优所致，而是词法检索对「Darboux → 达布」这类
            # 术语变体无感知。保留默认值，语义缺口留给向量通路解决。
            self._index = BM25Plus(self._tokens)
        except Exception:
            try:
                from rank_bm25 import BM25Okapi

                self._index = BM25Okapi(self._tokens)
            except Exception:
                # 依赖缺失时用词项重合度兜底，保证检索不失效
                self._simple = True

    def search(self, query: str, top_n: int) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q_tokens = tokenize(query)
        if self._index is not None:
            scores = self._index.get_scores(q_tokens)
            ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
            return [(c, float(s)) for c, s in ranked[:top_n] if s > 0]
        if self._simple:
            q = set(q_tokens)
            denom = max(1, len(q))
            scored = []
            for c, toks in zip(self.chunks, self._tokens):
                s = len(q & set(toks)) / denom
                if s > 0:
                    scored.append((c, s))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_n]
        return []


class VectorIndex:
    """向量通路：local(ONNX) → api → chroma 依次尝试，谁先就绪用谁。

    任何一步失败都把原因记在 reason 上，设置页原样展示。之前只显示
    「向量召回未启用」，实际原因是 embedding 免费额度耗尽 / 云端没有 embedding
    模型，界面上完全看不出来 —— 这种「静默降级」比直接报错更难排查。
    """

    def __init__(self, chunks: list[Chunk], cfg):
        self.chunks = chunks
        self.cfg = cfg
        self.available = False
        self.reason = ""
        self.model_name = ""
        self._backend = ""
        self._embedder = None
        self._matrix = None
        self._ids: list[str] = []
        self._row_of: dict[str, int] = {}
        # chroma 通路
        self._model = None
        self._collection = None
        self._st = None
        self._chromadb = None
        if not cfg.enabled:
            self.reason = "向量检索已在配置中关闭"
            return

        order = [cfg.backend] if cfg.backend != "auto" else ["local", "api", "chroma"]
        reasons: list[str] = []
        for backend in order:
            ok = False
            if backend == "local":
                ok = self._init_npz(
                    "local",
                    getattr(cfg, "local_embed_file", "") or cfg.embed_file,
                    lambda: self._make_local_embedder(),
                )
            elif backend == "api":
                ok = self._init_npz("api", cfg.embed_file, lambda: self._make_api_embedder())
            elif backend == "chroma":
                ok = self._init_chroma()
            if ok:
                return
            if self.reason:
                reasons.append(self.reason)
        self.reason = "；".join(dict.fromkeys(reasons)) or "向量通路不可用"

    # ---------- embedder 构造 ----------
    def _make_local_embedder(self):
        from src.retrieval.embed import LocalEmbedder

        return LocalEmbedder(getattr(self.cfg, "local_model", "") or "BAAI/bge-small-zh-v1.5")

    def _make_api_embedder(self):
        from src.retrieval.embed import APIEmbedder

        return APIEmbedder(self.cfg)

    # ---------- npz 通路（local / api 共用） ----------
    def _init_npz(self, backend: str, rel_path: str, make_embedder) -> bool:
        path = get_settings().root / rel_path
        if not path.exists():
            self.reason = (
                f"缺少向量库 {rel_path}，请先运行 "
                f"python scripts/build_embeddings.py --backend {backend}"
            )
            return False
        try:
            import numpy as np

            data = np.load(path, allow_pickle=False)
            ids = [str(x) for x in data["ids"].tolist()]
            matrix = data["vectors"].astype("float32")
            meta: dict = {}
            if "meta" in data.files:
                try:
                    meta = json.loads(str(data["meta"].item()))
                except Exception:
                    meta = {}
        except Exception as exc:  # noqa: BLE001
            self.reason = f"向量库文件损坏，已降级：{exc}"
            return False

        # 不同后端的向量空间不能混用：拿 512 维本地向量去撞 1024 维云端库，
        # 维度对不上会崩，对上了则是「检索到完全无关的内容」，更隐蔽
        if meta.get("backend") and meta["backend"] != backend:
            self.reason = (
                f"向量库由 {meta['backend']} 通路生成，与当前 {backend} 配置不符，请重建"
            )
            return False

        # 知识块变了（重建过库）就必须重建向量，否则 id 对不上会导致「检索到别的内容」
        chunk_ids = [c.chunk_id for c in self.chunks]
        if set(ids) != set(chunk_ids):
            self.reason = (
                f"向量库与知识块不同步（向量 {len(ids)} 条 / 知识块 {len(chunk_ids)} 块），"
                f"请重新运行 scripts/build_embeddings.py --backend {backend}"
            )
            return False
        try:
            embedder = make_embedder()
        except Exception as exc:  # noqa: BLE001
            self.reason = f"{backend} 向量化组件不可用，已降级为 BM25：{exc}"
            return False

        self._embedder = embedder
        self._matrix = matrix
        self._ids = ids
        self._row_of = {cid: i for i, cid in enumerate(ids)}
        self._backend = backend
        self.model_name = meta.get("model") or getattr(embedder, "model_name", "") or ""
        self.available = True
        self.reason = ""
        return True

    # ---------- chroma 通路 ----------
    def _init_chroma(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb

            self._st = SentenceTransformer
            self._chromadb = chromadb
            self.available = True
            self._backend = "chroma"
            self.model_name = self.cfg.embedding_model
            return True
        except Exception as exc:  # noqa: BLE001
            self.reason = f"未安装向量依赖（sentence-transformers/chromadb），已降级为 BM25：{exc}"
            self.available = False
            return False

    def build(self) -> None:
        # local/api 通路的向量由 scripts/build_embeddings.py 离线建好，运行时无需构建
        if not self.available or not self.chunks or self._backend != "chroma":
            return
        try:
            persist = get_settings().root / self.cfg.persist_dir
            persist.mkdir(parents=True, exist_ok=True)
            self._model = self._st(self.cfg.embedding_model)
            client = self._chromadb.PersistentClient(path=str(persist))
            self._collection = client.get_or_create_collection(self.cfg.collection)
            if self._collection.count() >= len(self.chunks):
                return
            texts = [c.content for c in self.chunks]
            embeddings = self._model.encode(texts, normalize_embeddings=True).tolist()
            self._collection.upsert(
                ids=[c.chunk_id for c in self.chunks],
                documents=texts,
                embeddings=embeddings,
                metadatas=[
                    {"chapter": c.chapter, "content_type": c.content_type, "topic": c.topic} for c in self.chunks
                ],
            )
        except Exception as exc:
            self.available = False
            self.reason = f"向量索引构建失败，已降级：{exc}"

    def search(self, query: str, top_n: int) -> list[tuple[Chunk, float]]:
        if not self.available:
            return []
        if self._backend in ("api", "local"):
            return self._search_npz(query, top_n)
        return self._search_chroma(query, top_n)

    def _search_npz(self, query: str, top_n: int) -> list[tuple[Chunk, float]]:
        import numpy as np

        if self._matrix is None or self._embedder is None:
            return []
        try:
            q = np.asarray(self._embedder.embed_query(query), dtype="float32")
        except Exception as exc:  # noqa: BLE001
            # 接口抖动不应让整题失败：本次退化为纯 BM25，并记下原因
            self.available = False
            self.reason = f"查询向量化失败，本次降级为 BM25：{exc}"
            return []
        if q.ndim != 1 or q.shape[0] != self._matrix.shape[1]:
            self.available = False
            self.reason = (
                f"查询向量维度 {q.shape} 与向量库 {self._matrix.shape[1]} 不一致，请重建向量库"
            )
            return []
        sims = self._matrix @ q
        n = min(top_n, len(sims))
        idx = np.argpartition(-sims, n - 1)[:n]
        idx = idx[np.argsort(-sims[idx])]
        by_id = {c.chunk_id: c for c in self.chunks}
        out: list[tuple[Chunk, float]] = []
        for i in idx:
            cid = self._ids[int(i)]
            if cid in by_id:
                out.append((by_id[cid], float(sims[int(i)])))
        return out

    def _search_chroma(self, query: str, top_n: int) -> list[tuple[Chunk, float]]:
        if self._collection is None or self._model is None:
            return []
        try:
            emb = self._model.encode([query], normalize_embeddings=True).tolist()
            res = self._collection.query(query_embeddings=emb, n_results=min(top_n, max(1, len(self.chunks))))
            ids = res.get("ids", [[]])[0]
            dists = res.get("distances", [[]])[0]
            by_id = {c.chunk_id: c for c in self.chunks}
            out = []
            for cid, dist in zip(ids, dists):
                if cid in by_id:
                    out.append((by_id[cid], 1.0 - float(dist)))
            return out
        except Exception:
            return []


class HybridRetriever:
    def __init__(self, cfg: RetrievalConfig | None = None):
        self.cfg = cfg or get_settings().retrieval
        root = get_settings().root
        self.chunks = _load_chunks(root / self.cfg.bm25.chunk_file)
        self.bm25 = BM25Index(self.chunks) if self.cfg.bm25.enabled else None
        self.vector = VectorIndex(self.chunks, self.cfg.vector)
        if self.cfg.vector.enabled and self.vector.available:
            self.vector.build()
        # 精排层：懒加载，模型缺失时静默降级为「不精排」
        self.reranker = Reranker(getattr(self.cfg, "rerank", None))

    def retrieve(
        self,
        query: str,
        *,
        filters: dict | None = None,
        candidate_theorems: list[str] | None = None,
        top_k: int | None = None,
        rerank: bool = True,
    ) -> list[Chunk]:
        filters = {k: v for k, v in (filters or {}).items() if v}
        top_k = top_k or self.cfg.top_k
        cand_n = self.cfg.candidate_chunks
        query_all = f"{query} {' '.join(candidate_theorems or [])}".strip()

        pools: list[list[tuple[Chunk, float]]] = []
        tags: list[str] = []
        if self.vector and self.vector.available:
            pools.append(self.vector.search(query_all, cand_n))
            tags.append("vector")
        if self.bm25:
            pools.append(self.bm25.search(query_all, cand_n))
            tags.append("bm25")

        # 仅当 strict_chapter 时做章节硬过滤；默认按软加权（避免误杀跨章知识）
        if self.cfg.filter.strict_chapter and filters.get("chapter"):
            ch = filters["chapter"]
            pools = [[(c, s) for c, s in p if (not c.chapter) or ch in c.chapter or c.chapter in ch] for p in pools]

        # 注意：元数据加权不能加在原始分数上——RRF 只按排名融合，会把分数丢掉。
        # 因此统一在融合完成后对 chunk 得分做倍数调整，两种融合方式都生效。
        fused = self._fuse(pools, tags)
        for c in fused:
            boost = 0.0
            if filters.get("chapter") and filters["chapter"] and filters["chapter"] in c.chapter:
                boost += 0.30
            if filters.get("topic") and filters["topic"] and (
                filters["topic"] in c.topic or c.topic in filters["topic"]
            ):
                boost += 0.20
            if candidate_theorems and any(t and t in c.content for t in candidate_theorems):
                boost += 0.50
            boost += self._type_bias(c)
            if boost:
                c.score *= max(0.1, 1.0 + boost)
        fused.sort(key=lambda x: x.score, reverse=True)

        # —— 精排（rerank）：粗召回排序后再用交叉编码器重打分 ——
        # 按定理名检索时（rerank=False）跳过：query 本身就是强关键词，
        # 词法通路已经能把它顶到第一，交叉编码器每题多花 1~2s 换不来排序收益。
        if rerank:
            fused = self._rerank(query_all, fused)

        fused = fused[:top_k]
        # 加权是倍数调整，可能把分数推过 1；统一缩放到 ≤1，
        # 让 score 始终表示「0~1 的相关度」，下游做阈值判断时不会踩坑。
        if fused and fused[0].score > 1.0:
            top = fused[0].score
            for c in fused:
                c.score = c.score / top
        return fused

    def _rerank(self, query: str, fused: list[Chunk]) -> list[Chunk]:
        """交叉编码器精排。返回重新排序后的列表（原顺序保持分数语义）。

        只在精排可用且候选足够多时才生效；模型缺失/打分失败时原样返回，
        不影响主链路。融合时精排分与粗召回分按 rerank_weight 加权，避免
        完全丢掉「强关键词精确匹配」这一 BM25 的独有信号。
        """
        cfg = getattr(self.cfg, "rerank", None)
        if not fused or cfg is None or not cfg.enabled:
            return fused
        if len(fused) <= 1:
            return fused
        n = min(cfg.candidate_top_n, len(fused))
        cand = fused[:n]
        # 只对取出的候选精排，其余保持原相对顺序垫底
        # 正文截断到 max_doc_chars：交叉编码器耗时与长度近似线性，块级相关度
        # 判断看开头（定理陈述/定义）就够，截断后单题省 5~7s。
        limit = int(getattr(cfg, "max_doc_chars", 320) or 320)
        docs = [c.topic + "\n" + c.content[:limit] for c in cand]
        scores = self.reranker.rerank(query, docs)
        if not self.reranker.available:
            return fused
        # 归一化两路分数到 0~1，再加权
        mx = max(scores) or 1.0
        w = cfg.rerank_weight
        for c, s in zip(cand, scores):
            rs = (s / mx) if mx > 0 else 0.0
            # 粗召回分已 ≤1（融合后缩放前可能 >1，这里截到 [0,1] 区间比较）
            old = min(1.0, max(0.0, c.score))
            c.score = w * rs + (1.0 - w) * old
            if "rerank" not in c.matched_by:
                c.matched_by.append("rerank")
        cand.sort(key=lambda x: x.score, reverse=True)
        return cand + fused[n:]

    @staticmethod
    def _type_bias(c: Chunk) -> float:
        """按知识块类型做轻微偏向，修正「例题挤掉定理」的排序问题。

        教材库里「例」通常占六成以上，且正文长、词项多，BM25 天然占优。
        但解题报告要引用的是定理/定义本身（学生问「Rolle 定理成立条件」时
        要的是定理原文，不是某道例题）。这里只做小幅调整，不硬性过滤，
        避免把真正相关的例题（如「怎么算这个积分」）误压下去。
        """
        t = (c.content_type or "").strip()
        if t in ("定义", "定理", "推论", "引理", "命题", "性质", "法则"):
            return 0.18
        if t in ("例", "例题"):
            return -0.06
        return 0.0

    def _fuse(self, pools: list[list[tuple[Chunk, float]]], tags: list[str]) -> list[Chunk]:
        live = [p for p in pools if p]
        # 单通路时 RRF 会把分数压成常数（1/(k+rank)），既不可读也没区分度，
        # 直接改用归一化分数；只有两路以上才用 RRF。
        if len(live) <= 1:
            return self._weighted(live, [t for t, p in zip(tags, pools) if p], single=True)
        if self.cfg.fusion.method == "weighted":
            return self._weighted(live, [t for t, p in zip(tags, pools) if p])
        return self._rrf(live, [t for t, p in zip(tags, pools) if p])

    @staticmethod
    def _empty(c: Chunk) -> Chunk:
        return Chunk(
            chunk_id=c.chunk_id,
            content=c.content,
            chapter=c.chapter,
            section=c.section,
            content_type=c.content_type,
            topic=c.topic,
            source=c.source,
            page=c.page,
        )

    def _rrf(self, pools: list[list[tuple[Chunk, float]]], tags: list[str]) -> list[Chunk]:
        k = self.cfg.fusion.rrf_k
        agg: dict[str, Chunk] = {}
        score: dict[str, float] = {}
        for tag, pool in zip(tags, pools):
            for rank, (c, _s) in enumerate(pool):
                key = c.chunk_id
                if key not in agg:
                    agg[key] = self._empty(c)
                    score[key] = 0.0
                score[key] += 1.0 / (k + rank + 1)
                if tag not in agg[key].matched_by:
                    agg[key].matched_by.append(tag)
        out = []
        for key, c in agg.items():
            c.score = score[key]
            out.append(c)
        return out

    def _weighted(self, pools: list[list[tuple[Chunk, float]]], tags: list[str], single: bool = False) -> list[Chunk]:
        wmap = {"vector": self.cfg.fusion.vector_weight, "bm25": self.cfg.fusion.bm25_weight}
        agg: dict[str, Chunk] = {}
        score: dict[str, float] = {}
        for tag, pool in zip(tags, pools):
            if not pool:
                continue
            w = 1.0 if single else wmap.get(tag, 0.5)
            mx = max(s for _, s in pool) or 1.0
            for c, s in pool:
                key = c.chunk_id
                if key not in agg:
                    agg[key] = self._empty(c)
                    score[key] = 0.0
                score[key] += w * (s / mx)
                if tag not in agg[key].matched_by:
                    agg[key].matched_by.append(tag)
        out = []
        for key, c in agg.items():
            c.score = score[key]
            out.append(c)
        return out

    def status(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "bm25": bool(self.bm25 and (self.bm25._index is not None or self.bm25._simple)),
            "vector": bool(self.vector.available),
            "vector_backend": self.vector._backend or "none",
            "vector_model": self.vector.model_name,
            "vector_reason": self.vector.reason,
            "fusion": self.cfg.fusion.method,
            "rerank": bool(getattr(self.cfg, "rerank", None) and self.reranker.available),
            "rerank_model": self.reranker.model_name or "none",
            "rerank_reason": self.reranker.reason,
        }


def recall_at_k(retrieved_ids: list[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    hit = len(set(retrieved_ids[:k]) & rel)
    return hit / len(rel)
