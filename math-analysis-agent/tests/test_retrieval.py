"""混合检索的单元测试。

注意：这里用 tests/fixtures/chunks.jsonl 这个固定样本，**不依赖**生产知识库
（knowledge_base/chunks/chunks.jsonl 会随教材导入而变，测试不该跟着变）。
"""
import pytest

from src.config import BM25Config, RerankConfig, RetrievalConfig, VectorConfig
from src.retrieval.hybrid import HybridRetriever, recall_at_k, tokenize

FIXTURE = "tests/fixtures/chunks.jsonl"


@pytest.fixture()
def retriever() -> HybridRetriever:
    cfg = RetrievalConfig(
        bm25=BM25Config(chunk_file=FIXTURE),
        vector=VectorConfig(enabled=False),  # 单测不依赖向量依赖与模型下载
        rerank=RerankConfig(enabled=False),  # 单测不触发精排模型下载
    )
    return HybridRetriever(cfg)


def test_tokenize_chinese_math():
    toks = tokenize("洛必达法则 limit x^2")
    assert "洛必达法则" in toks or "洛必" in toks
    assert "limit" in toks


def test_chunks_loaded(retriever):
    assert len(retriever.chunks) == 20


def test_bm25_hits_lhopital(retriever):
    res = retriever.retrieve("用洛必达法则求极限", top_k=5)
    ids = [c.chunk_id for c in res]
    assert "c05" in ids


def test_metadata_soft_boost(retriever):
    res = retriever.retrieve("中值定理的应用", filters={"chapter": "中值定理"}, top_k=5)
    assert res
    # 章节匹配应作为软加权（融合后乘系数），使同章知识排在前列
    assert any("中值定理" in c.chapter for c in res[:3])


def test_single_pool_scores_are_normalized(retriever):
    """单通路时不能用 RRF 常数分（1/(k+rank)），否则分数没有区分度。"""
    res = retriever.retrieve("极限", top_k=5)
    assert len(res) >= 2
    scores = [c.score for c in res]
    assert max(scores) <= 1.0001
    assert len({round(s, 6) for s in scores}) > 1


def test_recall_at_k():
    assert recall_at_k(["c05", "c01"], ["c05"], 5) == 1.0
    assert recall_at_k(["c01"], ["c05"], 5) == 0.0
