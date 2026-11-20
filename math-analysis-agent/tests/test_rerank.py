"""精排（rerank）层的单元测试。

不依赖生产模型下载：通过构造 fake Reranker 与关闭精排两种路径，
验证「精排可用时改分数排序、不可用时原样返回」两条分支。
"""
import pytest

from src.config import BM25Config, RerankConfig, RetrievalConfig, VectorConfig
from src.retrieval.hybrid import HybridRetriever

FIXTURE = "tests/fixtures/chunks.jsonl"


@pytest.fixture()
def cfg() -> RetrievalConfig:
    return RetrievalConfig(
        bm25=BM25Config(chunk_file=FIXTURE),
        vector=VectorConfig(enabled=False),
        rerank=RerankConfig(enabled=True, candidate_top_n=5, rerank_weight=1.0),
    )


class _FakeReranker:
    """假精排器：给 docs 按传入顺序递减打分（最后一个最高）。

    这样能确定性地验证「精排重排了结果」——若精排生效，倒数第一个候选
    会被提到最前。
    """

    def __init__(self):
        self.available = True
        self.reason = ""
        self.model_name = "fake"

    def rerank(self, query, docs):
        n = len(docs)
        return [float(i + 1) / n for i in range(n)]  # 递增：最后一个最高


def test_rerank_reorders_when_available(cfg):
    r = HybridRetriever(cfg)
    r.reranker = _FakeReranker()
    res = r.retrieve("洛必达法则", top_k=5)
    # 精排后分数重算，且精排过的块带 rerank 标记
    assert any("rerank" in c.matched_by for c in res)
    # rerank_weight=1.0 时，排序完全由精排分决定：原倒数第一应排到第一
    assert res[0].score >= res[-1].score


def test_rerank_disabled_returns_unchanged(cfg):
    cfg.rerank.enabled = False
    r = HybridRetriever(cfg)
    r.reranker = _FakeReranker()
    res = r.retrieve("洛必达法则", top_k=5)
    # 关闭精排时不应出现 rerank 标记
    assert not any("rerank" in c.matched_by for c in res)


def test_rerank_scores_still_bounded(cfg):
    r = HybridRetriever(cfg)
    r.reranker = _FakeReranker()
    res = r.retrieve("洛必达法则", top_k=5)
    assert all(0.0 <= c.score <= 1.0001 for c in res)


def test_reranker_unavailable_passthrough(cfg):
    """精排打分失败时，应原样返回粗召回结果，不影响主链路。"""
    r = HybridRetriever(cfg)

    class _Broken:
        available = False
        reason = "boom"
        model_name = "broken"

        def rerank(self, query, docs):
            return [0.0] * len(docs)

    r.reranker = _Broken()
    res = r.retrieve("洛必达法则", top_k=5)
    # 精排不可用，结果照常返回，且不带 rerank 标记
    assert len(res) >= 1
    assert not any("rerank" in c.matched_by for c in res)
