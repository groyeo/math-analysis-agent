"""LangGraph 工作流编排：节点串联 + 审核条件分支 + 最大重试。"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.nodes import build_nodes
from src.agents.schema import AgentState


def route_after_review(state: dict) -> str:
    if state.get("degraded"):
        return "format"
    review = state.get("review", {}) or {}
    if review.get("will_retry"):
        return "plan"
    return "format"


def route_after_verify(state: dict) -> str:
    """极速模式直接跳过交叉审核（省 8~25s），其余情况照常送审。"""
    return "format" if state.get("skip_review") else "review"


def build_graph(clients: dict, retriever, emit=None):
    nodes = build_nodes(clients, retriever, emit)
    graph = StateGraph(AgentState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.set_entry_point("parse")
    # 顺序：解析 → 结构化 → 规划 → 解题 → 按定理检索 → 复算 → 审核 → 报告。
    #
    # 检索放在解题之后是刻意的：解题模型先把题做出来、并在每一步标注所用的
    # theorem，再拿这些定理名去教材里找原文。好处有两个 ——
    #   1) 检索目标确定（就是这条定理的原文），不依赖题面能否猜中章节；
    #   2) 解题时不给检索片段，模型不会被跑偏的召回结果带偏，右栏的
    #      「定理 ↔ 原文」也天然一一对应，而不是事后硬凑。
    # 审核打回重解时，也会走一遍新的 solve → retrieve，用新解答的定理重查。
    graph.add_edge("parse", "analyze")
    graph.add_edge("analyze", "plan")
    graph.add_edge("plan", "solve")
    graph.add_edge("solve", "retrieve")
    graph.add_edge("retrieve", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {"review": "review", "format": "format"})
    graph.add_conditional_edges("review", route_after_review, {"plan": "plan", "format": "format"})
    graph.add_edge("format", END)
    return graph.compile()
