"""高层 Pipeline：供前端/脚本一站式调用，负责预算、客户端装配与状态初始化。"""
from __future__ import annotations

import os
from dataclasses import dataclass

from src.agents.graph import build_graph
from src.config import ModelSpec, get_settings
from src.llm.client import Budget, LLMClient
from src.retrieval.hybrid import HybridRetriever


@dataclass
class PipelineResult:
    report: dict
    state: dict
    budget: dict
    mock: bool


def build_clients(budget: Budget) -> dict[str, LLMClient]:
    s = get_settings()
    mock = s.mock_llm

    def mk(spec: ModelSpec) -> LLMClient:
        return LLMClient(spec, budget, mock)

    light_spec = s.models.light
    if light_spec.fallback_to and not light_spec.api_key:
        light_spec = getattr(s.models, light_spec.fallback_to, light_spec)
    return {"solver": mk(s.models.solver), "reviewer": mk(s.models.reviewer), "light": mk(light_spec)}


class MathAgentPipeline:
    def __init__(self, emit=None, retriever: HybridRetriever | None = None, budget: Budget | None = None):
        self.settings = get_settings()
        # 预算参数从这里统一取，别在实例化时写死数字：
        # 之前 pipeline 里硬编码 24000，改了 Budget 默认值也不生效（踩过）。
        self.budget = budget or Budget(
            max_calls=int(os.getenv("BUDGET_MAX_CALLS", "12")),
            max_tokens=int(os.getenv("BUDGET_MAX_TOKENS", "36000")),
            max_cost=float(os.getenv("BUDGET_MAX_COST", "1.0")),
        )
        self.clients = build_clients(self.budget)
        self.retriever = retriever or HybridRetriever()
        self._emit = emit
        self.graph = build_graph(self.clients, self.retriever, emit)

    @property
    def mock(self) -> bool:
        return all(c.mock for c in self.clients.values())

    def run(self, question: str, input_type: str = "text", ocr_text: str | None = None,
            emit=None, skip_review: bool = False) -> PipelineResult:
        self.budget.reset()
        if emit is not None and emit is not self._emit:
            self.graph = build_graph(self.clients, self.retriever, emit)
            self._emit = emit
        init = {
            "question_raw": question,
            "input_type": input_type,
            "ocr_text": ocr_text or "",
            "retry_count": 0,
            "max_retries": self.settings.app.max_retries,
            "skip_review": skip_review,
            "degraded": False,
            "error": None,
        }
        state = self.graph.invoke(init)
        snap = self.budget.snapshot()
        state["budget"] = snap
        if isinstance(state.get("final_report"), dict):
            state["final_report"]["budget"] = snap
        return PipelineResult(state.get("final_report", {}), state, snap, self.mock)
