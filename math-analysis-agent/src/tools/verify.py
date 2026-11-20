"""验证分级：把解答中的可计算步骤交给 SymPy 复算，并给出四档状态（见方案 8.2）。"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.tools.sympy_tools import call_tool, equal_result

VERIFIED = "verified"
PARTIAL = "partial"
UNVERIFIED = "unverified"
MISMATCH = "mismatch"

LABELS = {
    VERIFIED: "已验证",
    PARTIAL: "部分验证",
    UNVERIFIED: "未验证",
    MISMATCH: "发现不一致",
}

_PARAM_KEYS = ("var", "point", "lower", "upper", "order")


@dataclass
class Verification:
    status: str = UNVERIFIED
    tool_calls: list[dict] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return LABELS.get(self.status, self.status)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "label": self.label,
            "tool_calls": self.tool_calls,
            "checked": self.checked,
            "unchecked": self.unchecked,
            "mismatches": self.mismatches,
            "notes": self.notes,
        }


def verify_solution(verifiable_steps: list[dict] | None) -> Verification:
    v = Verification()
    steps = verifiable_steps or []
    if not steps:
        v.notes.append("该题未提供可由 SymPy 复算的步骤（常见于纯证明题），不作自动验证。")
        v.status = UNVERIFIED
        return v

    for step in steps:
        claim = step.get("claim") or step.get("expr") or ""
        tool = step.get("tool")
        expr = step.get("expr", "")
        if not tool or not expr:
            v.unchecked.append(claim or "（缺少可计算表达式）")
            continue
        params = {k: step[k] for k in _PARAM_KEYS if k in step and step[k] not in (None, "")}
        if "order" in params:
            try:
                params["order"] = int(params["order"])
            except (TypeError, ValueError):
                params.pop("order")
        res = call_tool(tool, expr, **params)
        v.tool_calls.append(res.to_dict())
        if res.extra.get("ignored_params"):
            v.notes.append(
                f"`{tool}` 忽略了不支持的参数：{', '.join(res.extra['ignored_params'])}"
            )
        if not res.ok:
            # 工具不支持 ≠ 数学错误
            v.unchecked.append(claim or expr)
            v.notes.append(f"`{tool}({expr})` 未能执行：{res.note}")
            continue
        expected = step.get("expected")
        if expected and not equal_result(res.output, str(expected)):
            # 阶数给少了（要 1+x+x²/2，只展开到 x² 就只剩 1+x+O(x²)）会被判不一致，
            # 但这是「参数不足」而非「数学错误」。多展开一阶再比一次，仍不符才算真差异。
            retried = False
            if tool == "series" and "order" in params:
                for extra in (1, 2):
                    alt = call_tool(tool, expr, **{**params, "order": params["order"] + extra})
                    if alt.ok and equal_result(alt.output, str(expected)):
                        v.tool_calls.append(alt.to_dict())
                        v.checked.append(claim or f"{tool}({expr}) = {alt.output}")
                        retried = True
                        break
            if not retried:
                v.mismatches.append(f"{claim or expr}：解答给出 {expected}，工具复算为 {res.output}")
        else:
            v.checked.append(claim or f"{tool}({expr}) = {res.output}")

    if v.mismatches:
        v.status = MISMATCH
    elif not v.checked and not v.unchecked:
        v.status = UNVERIFIED
    elif not v.checked:
        v.status = UNVERIFIED
        v.notes.append("工具均无法执行，未完成任何自动验证。")
    elif v.unchecked:
        v.status = PARTIAL
        v.notes.append("部分步骤已验证，其余步骤需要人工检查。")
    else:
        v.status = VERIFIED
    return v
