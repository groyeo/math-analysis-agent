# -*- coding: utf-8 -*-
"""依据只放定理/公式，例题（例/习题块）一律排除。"""
from src.agents.nodes import _is_example, _pick_theorem_chunk


def _c(**kw):
    base = {"chunk_id": "x", "topic": "定理 7.3.2", "content": "内容",
            "content_type": "定理"}
    base.update(kw)
    return base


def test_example_by_content_type():
    assert _is_example(_c(content_type="例", topic="例 6.2.19"))
    assert _is_example(_c(content_type="例题", topic="例 1.1.1"))
    assert _is_example(_c(content_type="习题", topic="习题 3.1"))


def test_example_by_topic_prefix():
    assert _is_example(_c(content_type="", topic="例 1.1.1 求极限"))
    assert _is_example(_c(content_type=None, topic="习题 2.3"))
    assert not _is_example(_c(topic="定理 7.3.2 微积分基本定理"))
    assert not _is_example(_c(topic="推论 1 闭区间上连续函数可积"))


def test_pick_theorem_skips_examples():
    # 例题正文里常出现定理名（「由洛必达法则可得」），不能因此指到例题
    chunks = [
        _c(chunk_id="ex", content_type="例", topic="例 4.1.2",
           content="由洛必达法则可得该极限不存在"),
        _c(chunk_id="thm", content_type="定理", topic="定理 4.1.1 洛必达法则"),
    ]
    assert _pick_theorem_chunk("洛必达法则", chunks) == "thm"


def test_pick_theorem_returns_none_if_only_examples():
    chunks = [_c(chunk_id="ex", content_type="例", topic="例 4.1.2",
                 content="由洛必达法则可得")]
    assert _pick_theorem_chunk("洛必达法则", chunks) is None
