"""Embedding 通路：本地 ONNX 模型（默认）或 OpenAI 兼容的 /embeddings 接口。

两条通路的取舍：
- local：fastembed + BAAI/bge-small-zh-v1.5（ONNX，512 维）。不联网、不计费、
  不受免费额度影响，单条查询 10~30ms，是现在的默认通路。
- api：调云端 /embeddings。需要可用额度；曾因「免费额度耗尽」在运行时静默
  降级，界面上只显示「向量召回未启用」，很难查出原因，故退为备选。
"""
from __future__ import annotations

import os
import time

import numpy as np

_DEFAULT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen3.7-text-embedding-flash"
# HuggingFace 直连在境内多数网络下不通，镜像可用；fastembed 走 huggingface_hub，
# 认这个环境变量。已经缓存过模型时设不设都一样。
_DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def _normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class LocalEmbedder:
    """fastembed 本地 ONNX 向量化；模型首次使用时才加载（约 1~2s）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", batch_size: int = 64):
        self.model_name = model_name or "BAAI/bge-small-zh-v1.5"
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            os.environ.setdefault("HF_ENDPOINT", _DEFAULT_HF_ENDPOINT)
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return len(next(iter(self.model.embed(["dim"]))))

    def embed(self, texts: list[str], progress=None) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        vecs: list[np.ndarray] = []
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs.extend(np.asarray(v, dtype="float32") for v in self.model.embed(batch))
            if progress:
                progress(min(i + len(batch), total), total)
        return _normalize(np.asarray(vecs, dtype="float32"))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def resolve_credentials(cfg=None) -> tuple[str, str, str]:
    """按 专用 > 解析模型 > 解题模型 的顺序找 key，返回 (key, base_url, model)。"""
    key_env = getattr(cfg, "api_key_env", "") or "EMBED_API_KEY"
    key = os.getenv(key_env, "")
    if not key:
        key = os.getenv("LIGHT_API_KEY", "") or os.getenv("REVIEWER_API_KEY", "") or os.getenv("SOLVER_API_KEY", "")
    base = (getattr(cfg, "api_base_url", "") or "").strip() or os.getenv("EMBED_BASE_URL", "") or _DEFAULT_BASE
    model = (getattr(cfg, "api_model", "") or "").strip() or os.getenv("EMBED_MODEL", "") or _DEFAULT_MODEL
    return key, base, model


class APIEmbedder:
    """批量向量化；返回 L2 归一化后的 float32 矩阵，点积即余弦相似度。"""

    def __init__(self, cfg=None, batch_size: int = 16):
        self.key, self.base_url, self.model = resolve_credentials(cfg)
        self.batch_size = batch_size
        self._client = None
        if not self.key:
            raise RuntimeError("未找到 embedding 可用的 API Key（可配 EMBED_API_KEY 或复用 REVIEWER_API_KEY）")

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.key, base_url=self.base_url, timeout=90, max_retries=2)
        return self._client

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        last = None
        for attempt in range(4):
            try:
                r = self.client.embeddings.create(model=self.model, input=texts)
                # 接口按 index 返回时顺序可能乱，显式排序
                data = sorted(r.data, key=lambda d: getattr(d, "index", 0))
                return [d.embedding for d in data]
            except Exception as exc:  # noqa: BLE001
                last = exc
                msg = str(exc).lower()
                if "quota" in msg or "401" in msg or "invalid" in msg:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise last  # type: ignore[misc]

    def embed(self, texts: list[str], progress=None) -> np.ndarray:
        vecs: list[list[float]] = []
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs.extend(self._embed_batch(batch))
            if progress:
                progress(min(i + len(batch), total), total)
        arr = np.asarray(vecs, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
