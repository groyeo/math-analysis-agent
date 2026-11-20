"""结构化数据模型：约束 LLM 输出，保证节点间数据契约稳定。"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator


def _as_text_list(v: Any) -> Any:
    """模型常把 list[对象] 写成 list[str]（如 steps: ["第1步 泰勒展开", ...]）。

    严格校验会直接失败，而失败后的兜底曾把 mock 样例当成真实解答返回
    （出现过「结论 1/2 来自内置演示」的假答案），所以这里先做宽松归一化，
    能救回来的就不算失败。
    """
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [x.strip() for x in v if isinstance(x, str) and x.strip()] if all(
            isinstance(x, str) for x in v
        ) else v
    return v


def _as_step_list(v: Any) -> Any:
    """把 ["先泰勒展开", "代回原式"] 这类字符串步骤补成 SolutionStep 结构。"""
    if not isinstance(v, list):
        return v
    out = []
    for i, item in enumerate(v, 1):
        if isinstance(item, str):
            out.append({"title": f"第{i}步", "content": item})
        elif isinstance(item, dict):
            out.append(item)
        else:
            out.append(str(item))
    return out


class ProblemStruct(BaseModel):
    question: str = ""
    conditions: list[str] = Field(default_factory=list)
    goal: str = ""
    problem_type: str = ""
    chapter: str = ""
    topic: str = ""
    difficulty: str = "基础"
    variables: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    # 题目字面点名的定理/公式/方法名，供最后一步「按题目提到的定理和公式查依据」。
    # 可选字段：旧缓存/降级结果没有它也不影响解析。
    mentioned: list[str] = Field(default_factory=list)

    @field_validator("conditions", "variables", "ambiguities", "mentioned", mode="before")
    @classmethod
    def _coerce_lists(cls, v: Any) -> Any:
        return _as_text_list(v)


class SolutionPlan(BaseModel):
    method: str = ""
    alternatives: list[str] = Field(default_factory=list)
    required_theorems: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)

    @field_validator("alternatives", "required_theorems", "preconditions", "steps", "risks", mode="before")
    @classmethod
    def _coerce_lists(cls, v: Any) -> Any:
        return _as_text_list(v)


class VerifiableStep(BaseModel):
    claim: str = ""
    tool: str = ""
    expr: str = ""
    var: str = "x"
    point: str | None = None
    lower: str | None = None
    upper: str | None = None
    order: int | None = None
    expected: str | None = None

    @field_validator("point", "lower", "upper", "expected", mode="before")
    @classmethod
    def _blank_to_none(cls, v: Any) -> Any:
        # 模型对「不适用的参数」常写 "" 而不是省略，空串留着会让下游拼成 expr.lower("") 之类
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("order", mode="before")
    @classmethod
    def _coerce_order(cls, v: Any) -> Any:
        # 实测拿过 order=""（不限阶）和 "2"（字符串数字），严格校验会整题降级 —— 
        # 只是「展开阶数」这一个可选参数，没必要因此丢掉整份解答。
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            if not s or s.lower() in ("none", "null", "n/a", "-"):
                return None
            try:
                return int(float(s))
            except ValueError:
                return None
        if isinstance(v, float):
            return int(v)
        return v


class SolutionStep(BaseModel):
    title: str = ""
    content: str = ""
    latex: str = ""
    # 本步用到的定理/依据名称（如「狄利克雷判别法」「夹逼定理」），
    # 供前端把这一步箭头指向教材里对应定理的原文。模型解题时天然会写，无需它记 chunk_id。
    theorem: str = ""


class DraftSolution(BaseModel):
    # method / alternatives 只在快速模式（规划合并进解题）时由解题模型一并给出，
    # 两段式流程里仍然以 SolutionPlan 为准，这里留空即可。
    method: str = ""
    alternatives: list[str] = Field(default_factory=list)
    answer: str = ""
    steps: list[SolutionStep] = Field(default_factory=list)
    verifiable_steps: list[VerifiableStep] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)

    @field_validator("verifiable_steps", mode="before")
    @classmethod
    def _coerce_verifiable(cls, v: Any) -> Any:
        if v is None:
            return []
        if not isinstance(v, list):
            return v
        out = []
        for item in v:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, str) and item.strip():
                out.append({"claim": item.strip()})
        return out

    @field_validator("steps", mode="before")
    @classmethod
    def _coerce_steps(cls, v: Any) -> Any:
        return _as_step_list(v)

    @field_validator("caveats", "alternatives", mode="before")
    @classmethod
    def _coerce_caveats(cls, v: Any) -> Any:
        return _as_text_list(v)


class ReviewItem(BaseModel):
    step: int | str = 0
    claim: str = ""
    issue: str = ""
    severity: Literal["blocker", "warning", "info"] = "warning"

    @field_validator("severity", mode="before")
    @classmethod
    def _coerce_severity(cls, v: Any) -> Any:
        # 审核模型常写中文或大写（「严重」「Warning」），统一落到三个枚举值
        s = str(v or "").strip().lower()
        if s in ("blocker", "warning", "info"):
            return s
        if any(k in s for k in ("block", "error", "critical", "严重", "错误", "阻断")):
            return "blocker"
        if any(k in s for k in ("info", "hint", "minor", "提示", "轻微", "建议")):
            return "info"
        return "warning"


class ReviewResult(BaseModel):
    # 注意：不输出「通过/不通过」，只输出断言与质疑点（方案 4.2）
    assertions: list[str] = Field(default_factory=list)
    issues: list[ReviewItem] = Field(default_factory=list)
    needs_revision: bool = False
    handcheck: list[str] = Field(default_factory=list)

    @field_validator("assertions", "handcheck", mode="before")
    @classmethod
    def _coerce_text(cls, v: Any) -> Any:
        return _as_text_list(v)

    @field_validator("issues", mode="before")
    @classmethod
    def _coerce_issues(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, list):
            return [{"issue": x} if isinstance(x, str) else x for x in v]
        return v


class AgentState(TypedDict, total=False):
    question_raw: str
    input_type: str
    ocr_text: str
    ocr_warnings: list[str]
    problem_struct: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]]
    retrieval_meta: dict[str, Any]
    candidate_theorems: list[str]
    # 检索阶段为「题解里声明的每一条定理」找到的教材原文：{定理名: chunk_id}。
    # 由 retrieve 节点（解题之后运行）产出，format 阶段优先采用，
    # 保证右栏的「定理 ↔ 原文」是这次按定理查出来的，而不是事后在混合池里猜。
    theorem_hits: dict[str, str]
    # 「题目字面点名的定理/公式」回查到的教材块 id（不挂到具体推导步骤，
    # 而是与题目平行展示）。检索节点产出，format 阶段透传到报告。
    question_basis: list[str]
    solution_plan: dict[str, Any]
    draft_solution: dict[str, Any]
    verification: dict[str, Any]
    verification_status: str
    review: dict[str, Any]
    review_issues: list[dict[str, Any]]
    retry_count: int
    max_retries: int
    skip_review: bool
    final_report: dict[str, Any]
    budget: dict[str, Any]
    degraded: bool
    error: str | None
    events: list[dict[str, Any]]
