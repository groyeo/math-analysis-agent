"""SymPy 工具：安全封装 limit / diff / integrate / series / simplify / solveset / summation。"""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_DANGEROUS = re.compile(r"(__|\bimport\b|\beval\b|\bexec\b|\bopen\b|\bos\.|\bsys\.|\blambda\b|\bclass\b)")
_SYMBOLS = {name: sp.Symbol(name) for name in ["x", "y", "z", "n", "t", "a", "b", "c", "k", "m"]}
_LOCAL = {**_SYMBOLS, "E": sp.E, "pi": sp.pi, "oo": sp.oo, "inf": sp.oo}


@dataclass
class ToolResult:
    tool: str
    ok: bool
    input: str
    output: str = ""
    note: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "input": self.input,
            "output": self.output,
            "note": self.note,
            **self.extra,
        }


def normalize(expr: str) -> str:
    s = (expr or "").strip()
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-").replace("π", "pi")
    s = s.replace("^", "**")
    s = re.sub(r"(?<![A-Za-z])e\*\*", "E**", s)
    s = re.sub(r"(?<![A-Za-z])e(?![A-Za-z0-9])", "E", s)
    return s


def safe_parse(expr: str) -> Any:
    if not expr or not expr.strip():
        raise ValueError("空表达式")
    if _DANGEROUS.search(expr):
        raise ValueError("表达式包含不允许的语法")
    return parse_expr(normalize(expr), local_dict=_LOCAL, transformations=_TRANSFORMS, evaluate=True)


def _sym(var: str):
    return _SYMBOLS.get(var, sp.Symbol(var))


def _run(tool: str, expr: str, fn) -> ToolResult:
    try:
        parsed = safe_parse(expr)
        out = fn(parsed)
        return ToolResult(tool=tool, ok=True, input=expr, output=sp.sstr(out), extra={"expected_type": type(out).__name__})
    except Exception as exc:
        return ToolResult(tool=tool, ok=False, input=expr, note=f"工具不支持或执行失败：{exc}")


def limit(expr: str, var: str = "x", point: str = "0") -> ToolResult:
    return _run("limit", expr, lambda e: sp.limit(e, _sym(var), safe_parse(point)))


def diff(expr: str, var: str = "x", order: int = 1) -> ToolResult:
    return _run("diff", expr, lambda e: sp.diff(e, _sym(var), order))


def integrate(expr: str, var: str = "x", lower: str | None = None, upper: str | None = None) -> ToolResult:
    def _do(e):
        if lower is not None and upper is not None:
            return sp.integrate(e, (_sym(var), safe_parse(lower), safe_parse(upper)))
        return sp.integrate(e, _sym(var))

    return _run("integrate", expr, _do)


def series(expr: str, var: str = "x", point: str = "0", order: int = 6) -> ToolResult:
    return _run("series", expr, lambda e: sp.series(e, _sym(var), safe_parse(point), order))


def simplify(expr: str) -> ToolResult:
    return _run("simplify", expr, lambda e: sp.simplify(e))


def solveset(expr: str, var: str = "x") -> ToolResult:
    return _run("solveset", expr, lambda e: sp.solveset(e, _sym(var)))


def summation(expr: str, var: str = "n", lower: str = "1", upper: str = "oo") -> ToolResult:
    def _do(e):
        return sp.summation(e, (_sym(var), safe_parse(lower), safe_parse(upper)))

    return _run("summation", expr, _do)


TOOL_MAP = {
    "limit": limit,
    "diff": diff,
    "integrate": integrate,
    "series": series,
    "simplify": simplify,
    "solveset": solveset,
    "summation": summation,
}


def call_tool(name: str, expr: str, **kwargs) -> ToolResult:
    fn = TOOL_MAP.get(name)
    if not fn:
        return ToolResult(tool=name, ok=False, input=expr, note=f"未知工具 {name}")
    return fn(expr, **kwargs)


_ORDER_TERM = re.compile(r"\s*[+\-–]\s*O\([^)]*\)")


def _drop_order(s: str) -> str:
    """去掉级数展开里的 O(...) 余项项。

    series 工具的输出天然带余项（`1 + x + x**2/2 + O(x**3)`），而解答里写的是
    主部。不剥掉就永远「符号解析失败 → 退化为字符串比较 → 判不一致」，
    会把正确的解答标成红色「发现不一致」。
    """
    return _ORDER_TERM.sub("", str(s or "")).strip()


def equal_result(actual: str, expected: str) -> bool:
    """比较工具输出与解答声称的结果（符号化简 + 数值容差）。"""
    actual, expected = _drop_order(actual), _drop_order(expected)
    try:
        a = safe_parse(actual)
        b = safe_parse(expected)
        if sp.simplify(a - b) == 0:
            return True
        return bool(abs(complex(sp.N(a)) - complex(sp.N(b))) < 1e-8)
    except Exception:
        return (actual or "").strip().replace(" ", "") == (expected or "").strip().replace(" ", "")


def describe_capabilities() -> dict:
    return {
        "limit": "常见函数极限；复杂极限可能无法求解，需检查条件",
        "diff": "导数、高阶导数；不能代替证明可导条件",
        "integrate": "不定/定积分；可能返回特殊函数",
        "series": "Taylor/Laurent 展开；需核对展开点与阶数",
        "simplify": "代数化简；不能证明一般命题",
        "solveset": "方程求解；注意定义域与分支",
        "summation": "级数求和；不能替代敛散性证明",
    }
