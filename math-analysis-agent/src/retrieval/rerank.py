"""精排（rerank）：交叉编码器对粗召回结果做逐条重打分。

为什么需要精排：粗召回（向量 + BM25）靠「单向量相似度」或「词项重合」排序，
对「Darboux 定理 → 达布定理」这类术语变体、以及「题面描述」与「教材原文」
之间的语义差距不敏感。交叉编码器把 query 与每个候选拼接后一起过 Transformer，
能真正判断「这段到底是不是在回答这个问题」，专门用来把「排在池子后面但正确」
的内容提到前面。

实测（30 题评估集）：粗召回（每路 20 条）已经把正确答案 100% 捞进池子，
但 Top-5 仍有错位（Recall@5 93.3%）。瓶颈在「排序」而非「召回」，精排正是对症。

选型：fastembed 的 TextCrossEncoder（ONNX 运行时，无 torch），与现有 local
embedding 通路同一套轻量栈，不引入 PyTorch。
"""
from __future__ import annotations

import os

from src.config import RerankConfig

_DEFAULT_MODEL = "BAAI/bge-reranker-base"
_DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


class Reranker:
    """交叉编码器精排器。懒加载，模型缺失/加载失败时静默降级（不精排）。"""

    def __init__(self, cfg: RerankConfig | None = None):
        self.cfg = cfg
        self.available = False
        self.reason = ""
        self.model_name = ""
        self._model = None
        if cfg is None or not cfg.enabled:
            self.reason = "精排已在配置中关闭"
            return
        self.model_name = cfg.model or _DEFAULT_MODEL

    @property
    def model(self):
        if self._model is None and self.available:
            os.environ.setdefault("HF_ENDPOINT", _DEFAULT_HF_ENDPOINT)
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            # 默认线程数在这类 4 核机器上只用 1 路，显式给到核数（上限 4）快约 15%。
            # 再多反而掉速（8 线程实测 12s vs 4 线程 6.6s），别盲目拉满。
            threads = min(4, os.cpu_count() or 4)
            try:
                self._model = TextCrossEncoder(self.model_name, threads=threads)
            except TypeError:  # 老版本 fastembed 不支持 threads
                self._model = TextCrossEncoder(self.model_name)
        return self._model

    def _ensure(self) -> bool:
        """尝试加载模型；成功置 available=True。失败只降级，不抛错。"""
        if not self.available and self.cfg and self.cfg.enabled:
            try:
                # 触发真实加载，验证模型可用
                self.available = True
                _ = self.model
            except Exception as exc:  # noqa: BLE001
                self.available = False
                self._model = None
                self.reason = f"精排模型不可用，已跳过精排：{exc}"
        return self.available

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """对 query 与每个 doc 逐条打分，返回同序的相关度分（越高越相关）。

        失败返回等长 0 分列表，并把 available 置 False，由调用方跳过精排。
        """
        if not self._ensure() or not docs:
            return [0.0] * len(docs)
        try:
            # fastembed 的 TextCrossEncoder 方法名是 rerank(query, documents)，
            # 返回与 documents 同序的相关度分数（Iterable[float]）
            scores = self.model.rerank(query, docs)
            return [float(s) for s in scores]
        except Exception as exc:  # noqa: BLE001
            self.available = False
            self._model = None
            self.reason = f"精排打分失败，已跳过精排：{exc}"
            return [0.0] * len(docs)
