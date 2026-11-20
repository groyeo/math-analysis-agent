"""端到端流程测试：无 API Key 时自动走 mock，验证 LangGraph 全链路连通。"""
import pytest


@pytest.fixture()
def pipeline(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")
    from src.config import get_settings

    get_settings.cache_clear()
    from src.agents.pipeline import MathAgentPipeline

    return MathAgentPipeline()


def test_end_to_end_mock(pipeline):
    result = pipeline.run("计算 lim(x→0) (e^x - 1 - x)/x^2")
    report = result.report
    assert report
    for key in ("question", "structure", "method", "steps", "verification", "conclusion", "limitations"):
        assert key in report
    assert result.budget["calls"] > 0


def test_events_stream(pipeline):
    events = []
    pipeline.run("判断级数 Σ 1/n^2 是否收敛", emit=lambda n, m, d=None: events.append(n))
    assert {"parse", "analyze", "retrieve", "plan", "solve", "verify", "review", "format"} <= set(events)


def test_route_after_review():
    from src.agents.graph import route_after_review

    assert route_after_review({"review": {"will_retry": True}}) == "plan"
    assert route_after_review({"review": {"will_retry": False}}) == "format"
    assert route_after_review({"degraded": True, "review": {"will_retry": True}}) == "format"


def test_budget_circuit_breaker(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")
    from src.config import get_settings

    get_settings.cache_clear()
    from src.agents.pipeline import MathAgentPipeline
    from src.llm.client import Budget

    # 结构化已改为本地规则（不消耗调用），关键路径上只剩 solve 与 review 两次调用。
    # 因此把上限压到 1 才能真正触发熔断（以前是 analyze+solve 两次，现在只剩 solve）。
    p = MathAgentPipeline(budget=Budget(max_calls=1, max_tokens=100000, max_cost=99))
    res = p.run("计算 lim(x→0) (e^x - 1 - x)/x^2")
    assert res.report.get("degraded") is True
    assert res.budget["calls"] == 1
