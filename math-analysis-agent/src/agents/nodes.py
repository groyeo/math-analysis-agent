"""LangGraph 节点实现：解析 → 结构化 → 检索 → 规划 → 解题 → 验证 → 审核 → 报告。"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from pydantic import BaseModel, Field

from src.agents.prompts import (
    ANALYZE_SYSTEM,
    CANDIDATE_THEOREM_PROMPT,
    FORMAT_SYSTEM,
    PLAN_SYSTEM,
    REVIEW_SYSTEM,
    SOLVE_FAST_SYSTEM,
    SOLVE_SYSTEM,
)
from src.agents.schema import (
    DraftSolution,
    ProblemStruct,
    ReviewResult,
    SolutionPlan,
)
from src.config import get_settings
from src.llm.client import BudgetExceeded, LLMClient
from src.retrieval.hybrid import HybridRetriever
from src.tools.verify import LABELS, MISMATCH, UNVERIFIED, Verification, verify_solution

Emit = Callable[[str, str, dict | None], None]


class _Theorems(BaseModel):
    theorems: list[str] = Field(default_factory=list)


def _emit(emit: Emit | None, node: str, message: str, data: dict | None = None) -> None:
    if emit:
        emit(node, message, data)


def _strip_fence(text: str) -> str:
    """去掉 ```json ... ``` 包裹。模型在 json_mode 下偶尔仍会包一层。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        low = text.lstrip().lower()
        if low.startswith("json"):
            text = text.lstrip()[4:]
    return text.strip()


def _balanced_json(text: str) -> str | None:
    """截出第一个括号配平的 JSON 对象；顺带修掉被 max_tokens 砍断的尾巴。

    为什么不能只用 ``re.search(r"\\{.*\\}", text, re.S)``：那是贪婪匹配，正文里
    出现两个对象时会把中间的解释文字一起吃进去；而输出被截断时根本没有收尾的
    大括号，正则直接匹配不到 —— 两种情况都会白丢一份完整解答。
    """
    start = text.find("{")
    if start < 0:
        return None
    stack: list[str] = []
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
                if not stack:
                    return text[start : i + 1]
            else:
                return None
    if not stack:
        return None
    # 走到结尾仍未配平 = 被截断：补上没闭合的引号和括号，交给 json.loads 判生死
    tail = '"' if in_str else ""
    return text[start:] + tail + "".join(reversed(stack))


def _try_loads(text: str) -> dict | None:
    if not text:
        return None
    # 截断补全后常见 `...,]` 这种尾逗号，JSON 不允许
    candidate = re.sub(r",\s*([}\]])", r"\1", text)
    for c in (text, candidate):
        try:
            d = json.loads(c)
        except Exception:
            continue
        if isinstance(d, dict):
            return d
    return None


def _parse_payload(content: str, reasoning: str = "") -> tuple[dict | None, str]:
    """从 content（必要时退到 reasoning_content）里取出 JSON 对象。"""
    body = _strip_fence(content)
    if body:
        got = _try_loads(body) or _try_loads(_balanced_json(body) or "")
        if got is not None:
            return got, ""
    # 极端情况：模型的正文全空、JSON 只写进了思维链。
    # reasoning 一般是草稿，可靠性差，但总比整题没有解答强，返回来源让上层标注。
    reason = _strip_fence(reasoning)
    if reason:
        got = _try_loads(_balanced_json(reason) or "")
        if got is not None:
            return got, "JSON 取自思维链（正文为空）"
    return None, ""


def _json_failure_hint(res) -> str:
    """把「解析失败」翻译成用户能据此行动的原因。"""
    content = (res.content or "").strip()
    finish = getattr(res, "finish_reason", "") or ""
    reasoning = getattr(res, "reasoning", "") or ""
    if not content:
        if reasoning:
            return (
                "模型把全部输出预算用在了思考过程上，正文为空"
                f"（finish_reason={finish or '未知'}，思考 {len(reasoning)} 字）；"
                "已在 model.yaml 提高 max_tokens 并支持自动翻倍，仍失败请继续调大"
            )
        return f"模型返回空内容（finish_reason={finish or '未知'}）"
    if finish == "length":
        return "模型输出被 max_tokens 截断，JSON 不完整"
    return "模型输出不符合结构要求：无法解析为 JSON"


def _llm_json(client: LLMClient, messages: list[dict], model_cls, mock: dict, on_delta=None):
    """调用模型并解析为 Pydantic。

    返回 (对象, 错误信息)：
    - 接口层失败（额度/鉴权/网络）→ (None, 原因)，由上层标记 degraded，绝不伪装成正常结果；
    - 输出结构解析失败 → 同样是 (None, 原因)。
      这里曾经返回「兜底结构」让流程继续，结果 mock 样例被当成真实解答写进报告
      （出现过内置演示答案被当作模型答案），属于必须堵掉的假阳性，故改为降级。
    - 真正 mock 模式（无 Key）走 client.chat 内部，与本函数无关。
    """
    try:
        res = client.chat(messages, json_mode=True, mock=mock, on_delta=on_delta)
    except BudgetExceeded as exc:
        return None, f"预算熔断：{exc}"
    except Exception as exc:  # 额度不足、Key 失效、网络超时等
        return None, f"模型调用失败：{exc}"

    # 调用层若已标记降级，内容就不可信（可能是兜底样例），不能拿去当解答
    if getattr(res, "degraded", False):
        return None, f"模型调用失败：{res.error}"

    payload, note = _parse_payload(res.content, getattr(res, "reasoning", ""))
    if payload is None:
        detail = _json_failure_hint(res)
        # 只有真的拿到了文本，再花一次调用让它修正才有意义（正文全空时修不了）
        if not (res.content or "").strip():
            return None, detail
        try:
            fix = client.chat(
                [
                    *messages,
                    {"role": "assistant", "content": (res.content or "")[:4000]},
                    {
                        "role": "user",
                        "content": "上一段输出不符合要求的结构，请只输出修正后的合法 JSON，字段严格按系统提示，不要任何解释。",
                    },
                ],
                json_mode=True,
                mock=None,  # 修正失败要如实暴露，别用样例顶替
            )
            payload, _ = _parse_payload(fix.content, getattr(fix, "reasoning", ""))
            note = f"结构问题已自动修复一次：{detail[:90]}"
        except Exception:  # noqa: BLE001
            return None, detail
        if payload is None:
            return None, detail

    try:
        return model_cls(**payload), note
    except Exception as exc:  # noqa: BLE001
        # 只留第一行会丢掉 pydantic 的具体字段错误（「1 validation error for X」没有信息量），
        # 这里压成一行带上字段名，排查时不用再现跑一遍。
        detail = " ".join(str(exc).split())[:240]
        return None, f"模型输出字段不合规：{detail}"


def _demo_solution(ps: dict) -> dict:
    """离线 mock 模式下的内置演示样例：用于在无 API Key 时演示 SymPy 验证闭环。"""
    q = (ps.get("question") or "").lower().replace(" ", "")
    if ("lim" in q or "极限" in q) and ("e^x" in q or "exp" in q):
        return {
            "answer": "1/2",
            "steps": [
                {
                    "title": "第1步 泰勒展开",
                    "content": "由 e^x = 1 + x + x^2/2 + o(x^2)，得 (e^x-1-x)/x^2 = 1/2 + o(1)。",
                    "latex": "e^x = 1 + x + \\frac{x^2}{2} + o(x^2)",
                },
                {
                    "title": "第2步 取极限",
                    "content": "令 x→0，高阶无穷小趋于 0，故原极限为 1/2。",
                    "latex": "\\lim_{x\\to 0}\\frac{e^x-1-x}{x^2}=\\frac{1}{2}",
                },
            ],
            "verifiable_steps": [
                {
                    "claim": "该极限等于 1/2",
                    "tool": "limit",
                    "expr": "(exp(x)-1-x)/x**2",
                    "var": "x",
                    "point": "0",
                    "expected": "1/2",
                }
            ],
            "caveats": ["离线演示样例：解答由内置示例给出，仅用于验证工具链，不代表真实模型输出。"],
        }
    if ("∫" in q or "积分" in q) and ("xe^x" in q or "x*e^x" in q):
        return {
            "answer": "(x-1)e^x + C",
            "steps": [
                {
                    "title": "第1步 分部积分",
                    "content": "∫x e^x dx = x e^x - ∫e^x dx = (x-1)e^x + C。",
                    "latex": "\\int x e^x\\,dx = (x-1)e^x + C",
                }
            ],
            "verifiable_steps": [
                {
                    "claim": "原函数为 (x-1)e^x",
                    "tool": "integrate",
                    "expr": "x*exp(x)",
                    "var": "x",
                    "expected": "(x-1)*exp(x)",
                }
            ],
            "caveats": ["离线演示样例：解答由内置示例给出，仅用于验证工具链，不代表真实模型输出。"],
        }
    return {
        "answer": "（mock 模式）请配置模型 API Key 以获得真实解答。",
        "steps": [{"title": "第1步 整理题目", "content": ps.get("question", ""), "latex": ""}],
        "verifiable_steps": [],
        "caveats": ["当前为离线 mock 输出，仅用于验证流程连通性。"],
    }


# 解题模型经常只顾着写推导、把 method 字段漏掉（schema 有默认值不会报错，
# 报告里就成了空白）。这里按步骤文本里的关键词兜一个方法名出来，
# 比显示「未声明」对用户有用，也不需要为此多花一次模型调用。
_METHOD_HINTS = [
    (("泰勒", "Taylor", "麦克劳林", "Maclaurin"), "泰勒展开"),
    (("洛必达", "L'Hospital", "L'Hôpital", "L'Hopital"), "洛必达法则"),
    (("等价无穷小", "等价代换"), "等价无穷小替换"),
    (("夹逼", "夹挤"), "夹逼定理"),
    (("分部积分",), "分部积分"),
    (("换元", "变量代换"), "换元积分"),
    (("Weierstrass", "魏尔斯特拉斯"), "Weierstrass 判别法"),
    (("Dirichlet", "狄利克雷"), "Dirichlet 判别法"),
    (("Abel", "阿贝尔"), "Abel 判别法"),
    (("比值", "比式", "d'Alembert"), "比值判别法"),
    (("根值", "Cauchy 判别"), "根值判别法"),
    (("比较判别", "放缩"), "比较判别法"),
    (("拉格朗日", "Lagrange"), "拉格朗日中值定理"),
    (("罗尔", "Rolle"), "罗尔定理"),
    (("柯西中值", "Cauchy 中值"), "柯西中值定理"),
    (("积分中值",), "积分中值定理"),
    (("牛顿-莱布尼茨", "Newton-Leibniz"), "牛顿-莱布尼茨公式"),
    (("归纳",), "数学归纳法"),
    (("反证",), "反证法"),
]


def _infer_method(steps: list[dict], fallback_text: str = "") -> str:
    text = " ".join(
        [(s.get("title") or "") + " " + (s.get("content") or "") for s in steps] + [fallback_text]
    )
    for keys, name in _METHOD_HINTS:
        if any(k in text for k in keys):
            return name
    return ""


# 定理名的同义写法。教材（陈纪修）里常用外文人名或另一套译名，而模型习惯写
# 中文俗称 —— 「洛必达法则」在原文里是「L'Hospital 法则」，「泰勒展开」在原文里
# 是「泰勒公式 / Taylor」。不做这层映射，报告右栏就会大面积显示「未匹配到原文」。
_THM_ALIASES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("洛必达", "l'hospital", "l'hopital", "lhospital"), ("L'Hospital", "洛必达")),
    (("泰勒", "taylor", "麦克劳林", "maclaurin"), ("Taylor", "泰勒", "Maclaurin")),
    (("夹逼", "夹挤", "迫敛"), ("夹逼", "夹挤")),
    (("牛顿-莱布尼茨", "牛顿－莱布尼茨", "newton-leibniz", "微积分基本定理"),
     ("Newton-Leibniz", "牛顿-莱布尼茨", "微积分基本定理")),
    (("拉格朗日", "lagrange"), ("Lagrange", "拉格朗日")),
    (("柯西中值", "cauchy"), ("Cauchy", "柯西")),
    (("罗尔", "rolle"), ("Rolle", "罗尔")),
    (("weierstrass", "魏尔斯特拉斯"), ("Weierstrass", "魏尔斯特拉斯")),
    (("dirichlet", "狄利克雷"), ("Dirichlet", "狄利克雷")),
    (("abel", "阿贝尔"), ("Abel", "阿贝尔")),
    (("darboux", "达布"), ("Darboux", "达布")),
    (("分部积分",), ("分部积分",)),
    (("等价无穷小", "等价代换"), ("等价无穷小", "等价代换")),
    (("连续函数极限", "连续函数的极限", "直接代入"), ("连续函数的极限", "连续函数")),
    (("未定式",), ("不定式", "未定式")),
    (("单调有界",), ("单调有界",)),
]

# 模型常把定理名写成「洛必达法则适用条件」，尾巴是它自己加的限定语，
# 留着会稀释匹配（相似度被拉低），先在匹配前剥掉。
_THM_TAILS = ("的适用条件", "适用条件", "的应用条件", "应用条件", "的前提条件",
              "前提条件", "的条件", "的应用", "条件")


# 题目「字面点名」的定理/公式 → 规范检索名。
# 只收题目里真的写出来的（「用…证明」「由…可得」），不做推断——
# 这是最后一步「按题目提到的定理和公式查依据」的输入，混入猜测会把无关章节塞进题面依据。
_MENTION_PATTERNS: list[tuple[str, str]] = [
    ("weierstrass", "Weierstrass 判别法"), ("魏尔斯特拉斯", "Weierstrass 判别法"),
    ("dirichlet", "Dirichlet 判别法"), ("狄利克雷", "Dirichlet 判别法"),
    ("abel", "Abel 判别法"), ("阿贝尔", "Abel 判别法"),
    ("洛必达", "洛必达法则"), ("l'hospital", "洛必达法则"), ("l'hopital", "洛必达法则"),
    ("泰勒", "泰勒公式"), ("taylor", "泰勒公式"),
    ("麦克劳林", "麦克劳林公式"), ("maclaurin", "麦克劳林公式"),
    ("牛顿-莱布尼茨", "牛顿-莱布尼茨公式"), ("牛顿－莱布尼茨", "牛顿-莱布尼茨公式"),
    ("牛顿—莱布尼茨", "牛顿-莱布尼茨公式"), ("微积分基本定理", "牛顿-莱布尼茨公式"),
    ("拉格朗日", "拉格朗日中值定理"), ("lagrange", "拉格朗日中值定理"),
    ("柯西中值", "柯西中值定理"), ("罗尔", "罗尔定理"), ("rolle", "罗尔定理"),
    ("夹逼", "夹逼定理"), ("迫敛", "夹逼定理"), ("单调有界", "单调有界定理"),
    ("分部积分", "分部积分法"), ("等价无穷小", "等价无穷小"),
    ("莱布尼茨", "莱布尼茨判别法"), ("比值判别", "比值判别法"),
    ("根值判别", "根值判别法"), ("比较判别", "比较判别法"),
]


def _mentioned_in(text: str) -> list[str]:
    """从题面抽出「字面点名」的定理/公式规范名，去重、最多 5 个。"""
    low = (text or "").lower()
    out: list[str] = []
    for key, name in _MENTION_PATTERNS:
        if key in low and name not in out:
            out.append(name)
        if len(out) >= 5:
            break
    return out


def _thm_terms(thm: str) -> list[str]:
    """把一步声明的定理名摊成若干可检索词面（原词 + 别名），按可信度排序。"""
    base = str(thm or "").strip()
    for tail in _THM_TAILS:
        if base.endswith(tail) and len(base) > len(tail):
            base = base[: -len(tail)].strip()
            break
    terms = [base] if base else []
    low = base.lower()
    for keys, aliases in _THM_ALIASES:
        if any(k in low for k in keys):
            terms.extend(aliases)
    out: list[str] = []
    seen: set[str] = set()
    for t in terms:
        t = re.sub(r"\s+", " ", t).strip()
        if len(re.sub(r"[\s()（）]", "", t)) < 2:
            continue
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


_STRIP = re.compile(r"[\s\(\)（）\[\]{}，。；、：:“”‘’《》<>\-—_]")


def _norm(s) -> str:
    return _STRIP.sub("", str(s or ""))


def _bigrams(s: str) -> set[str]:
    s = _norm(s)
    if len(s) < 2:
        return set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _lcs(a: str, b: str) -> int:
    """最长公共子串长度。2-gram 重叠会把「可积定理」和「积分定理」判成相似，
    导致「连续函数可积定理」误配到 Fourier 逐项积分定理。改成看有没有
    连续 4 个字以上完全相同，噪声立刻消失。"""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _field(c, name: str) -> str:
    """兼容两种形态：检索上下文是 dict，BM25 全库补查返回的是 Chunk 对象。"""
    return str(c.get(name, "") if isinstance(c, dict) else getattr(c, name, "") or "")


_EXAMPLE_TYPES = ("例", "例题", "习题", "例题解答")


def _is_example(c) -> bool:
    """教材块是不是例题/习题。

    依据栏只放定理、公式这类「规则原文」；例题是别人的解答过程，
    放进依据既占版面又容易让用户误以为是本题要引的定理。
    """
    if _field(c, "content_type") in _EXAMPLE_TYPES:
        return True
    return bool(re.match(r"^\s*(例|习题)\s*\d", _field(c, "topic")))


def _thm_score(thm: str, c) -> float:
    """定理名与一条教材块的吻合度：连续公共片段为主，2-gram 为辅。"""
    key = _norm(thm)
    need = min(4, len(key))
    if need < 2:
        return 0.0
    topic, content = _field(c, "topic"), _field(c, "content")
    if _lcs(thm, topic) >= need:
        return 1.0
    if _lcs(thm, content) >= need:
        return 0.9
    kg = _bigrams(thm)
    if not kg:
        return 0.0
    return max(
        len(kg & _bigrams(topic)) / len(kg) * 0.5,
        len(kg & _bigrams(content)) / len(kg) * 0.4,
    )


def _pick_theorem_chunk(thm: str, chunks: list) -> str | None:
    """从候选块里挑出「这条定理名」最对应的原文。

    先按词面（含别名）整串包含，再按 _thm_score 高阈值兜底；
    都不够就返回 None —— 宁可右栏空着，也不要指错原文。
    例题块直接跳过：定理名出现在例题正文里（「由洛必达法则可得」）不算定理原文。
    """
    if not thm:
        return None
    chunks = [c for c in chunks if not _is_example(c)]
    for term in _thm_terms(thm):
        k = _norm(term).lower()
        if len(k) < 2:
            continue
        # 标题命中比正文命中更可靠：正文里常只是「由泰勒公式可得」
        for c in chunks:
            if k and k in _norm(_field(c, "topic")).lower():
                return _field(c, "chunk_id")
        for c in chunks:
            if k and k in _norm(_field(c, "content")).lower():
                return _field(c, "chunk_id")
    best, best_score = None, 0.0
    for c in chunks:
        score = _thm_score(thm, c)
        if _field(c, "content_type") in ("定理", "推论", "性质", "定义", "法则"):
            score += 0.02
        if score > best_score:
            best, best_score = _field(c, "chunk_id"), score
    return best if best_score >= 0.95 else None


def _lookup_in_corpus(bm25, thm: str, chapter: str = "") -> str | None:
    """用 BM25 单路到全库找一条定理的原文（不碰向量、不碰精排，毫秒级）。

    阈值是硬的：定理名与原文标题/正文必须有 ≥4 字的连续公共片段（或整串包含），
    宁缺勿错 —— 右栏指错原文比右栏空着更糟。

    章节一致性单独处理：「连续函数的极限性质」这种泛名在上下册都可能出现，
    拿下册多元章节的原文当依据是错的。章节一致的块只需 0.95，不一致的要 0.995
    （几乎逐字相同）才认。
    """
    if not thm or bm25 is None or not _bigrams(thm):
        return None
    cands: list[tuple[float, bool, str]] = []
    for term in _thm_terms(thm):
        try:
            hits = bm25.search(term, 8)
        except Exception:
            continue
        for c, _s in hits:
            raw = _thm_score(term, c)
            if _norm(term).lower() in _norm(_field(c, "content")).lower():
                raw = max(raw, 0.97)
            if _field(c, "content_type") in ("定理", "推论", "性质", "定义", "法则"):
                raw += 0.02
            ch = _field(c, "chapter")
            same = (not chapter) or (not ch) or (chapter in ch)
            cands.append((raw, same, _field(c, "chunk_id")))
    for raw, same, cid in sorted(cands, key=lambda x: (x[1], x[0]), reverse=True):
        if raw >= (0.95 if same else 0.995):
            return cid
    return None


def _guess_chapter(text: str) -> tuple[str, str]:
    """启发式章节/知识点推断：用于检索软加权与 mock 模式的结构化结果。"""
    t = text.lower()
    rules = [
        (["lim", "极限", "趋近", "→"], "函数极限", "极限计算"),
        (["连续", "间断"], "函数连续", "连续性与间断点"),
        (["导数", "求导", "d/dx", "微分"], "导数与微分", "导数计算"),
        (["中值定理", "罗尔", "拉格朗日", "柯西中值"], "中值定理", "微分中值定理"),
        (["不定积分", "原函数"], "不定积分", "不定积分计算"),
        (["定积分", "∫", "积分上限"], "定积分", "定积分计算"),
        (["级数", "收敛", "∑", "sum"], "级数", "级数敛散性"),
    ]
    for keys, chapter, topic in rules:
        if any(k in t for k in keys):
            return chapter, topic
    return "函数极限", "综合"


def _local_struct(text: str) -> dict:
    """本地规则做题目结构化（不调模型）。

    为什么要有这条路：结构化结果只喂给「解题提示词」和报告顶部的小标签，
    而 analyze 是一次完整的 LLM 调用，实测 8~11s，正好卡在关键路径上
    （retrieve 与它并行，但 solve 必须等它）。改用规则后单题端到端省掉这段，
    题目本身仍原样交给解题模型，不影响推导质量。要更细的语义切分时把
    configs/app.yaml 的 analyze_llm 打开即可切回模型路径。
    """
    q = (text or "").strip()
    chapter, topic = _guess_chapter(q)
    is_proof = any(k in q for k in ("证明", "求证", "试证"))
    goal = "证明命题" if is_proof else ("求极限" if "lim" in q.lower() or "极限" in q
                                    else ("计算积分" if "∫" in q or "积分" in q else "求解题目"))
    # 条件：按分句切，只收「设/已知/若/当」开头的短句，不臆造
    conds = [s.strip(" ，,。;；") for s in re.split(r"[，,；;。\n]", q)
             if len(s.strip()) <= 40 and re.match(r"^(设|已知|若|当|令)", s.strip())]
    variables = sorted({v for v in re.findall(r"(?<![A-Za-z])([a-zA-Z])(?![A-Za-z])", q)
                        if v in "xyzntkab"})
    difficulty = "较难" if any(k in q for k in ("证明", "一致收敛", "可积", "Fourier", "傅里叶")) \
        else ("中等" if len(q) > 40 else "基础")
    return {
        "question": q,
        "conditions": conds[:3],
        "goal": goal,
        "problem_type": "证明题" if is_proof else "计算题",
        "chapter": chapter,
        "topic": topic,
        "difficulty": difficulty,
        "variables": variables[:4] or ["x"],
        "ambiguities": [],
        "mentioned": _mentioned_in(q),
    }


def build_nodes(
    clients: dict[str, LLMClient],
    retriever: HybridRetriever,
    emit: Emit | None = None,
) -> dict[str, Callable]:
    cfg = get_settings()
    solver, reviewer, light = clients["solver"], clients["reviewer"], clients["light"]

    # ---------- 1. 解析 ----------
    def parse_node(state: dict) -> dict:
        raw = state.get("question_raw", "")
        input_type = state.get("input_type", "text")
        warnings: list[str] = []
        if input_type in ("image", "pdf"):
            text = state.get("ocr_text") or ""
            if not text:
                warnings.append("尚未接入 OCR/PDF 解析，请改用文字输入，或先粘贴识别出的题目文本。")
        else:
            text = raw
        text = re.sub(r"[ \t]+", " ", text).strip()
        _emit(emit, "parse", "题目解析完成", {"warnings": warnings})
        return {"ocr_text": text, "ocr_warnings": warnings}

    # ---------- 2. 结构化 ----------
    def analyze_node(state: dict) -> dict:
        text = state.get("ocr_text") or state.get("question_raw", "")
        chapter, topic = _guess_chapter(text)
        mock = {
            "question": text,
            "conditions": [],
            "goal": "求解题目",
            "problem_type": "计算题" if any(k in text for k in ("lim", "求", "计算")) else "证明题",
            "chapter": chapter,
            "topic": topic,
            "difficulty": "基础",
            "variables": ["x"],
            "ambiguities": [],
            "mentioned": _mentioned_in(text),
        }
        # 默认走本地规则（省掉一次 8~11s 的模型调用，见 _local_struct 注释）
        if not cfg.app.analyze_llm:
            local = _local_struct(text)
            _emit(emit, "analyze", f"结构化（本地规则）：{local['chapter']} · {local['topic']}",
                  local)
            return {"problem_struct": local, "struct_source": "local"}

        obj, err = _llm_json(light, [{"role": "system", "content": ANALYZE_SYSTEM}, {"role": "user", "content": text}], ProblemStruct, mock)
        if obj is None:
            # 模型结构化失败不该把整题标成降级：本地规则给的结构足够解题用
            local = _local_struct(text)
            _emit(emit, "analyze", f"结构化失败，改用本地规则：{err}", local)
            return {"problem_struct": local, "struct_source": "local"}
        if err:
            _emit(emit, "analyze", f"结构化降级：{err}")
        _emit(emit, "analyze", "题目结构化完成", obj.model_dump())
        return {"problem_struct": obj.model_dump(), "struct_source": "llm"}

    # ---------- 3. 按题解里的定理回查教材 ----------
    def retrieve_node(state: dict) -> dict:
        """先解题、后检索：query 来自题解每一步声明的定理，而不是题面。

        旧流程是「按题面检索 → 把 5 个片段塞进解题提示词」。问题是题面里往往
        没有定理名（「求 lim (e^x-1-x)/x^2」不会出现「泰勒公式」），检索命中与否
        全看向量通路猜得准不准，而且一旦检索跑偏，解题模型也会被带偏。

        现在反过来：解题模型先凭题目推出解答并逐步标注 theorem，再用这些定理名
        逐条回查教材。检索目标是确定的（就找这条定理的原文），做法也更省 ——
        定理名是强关键词，BM25 直接命中，不需要交叉编码器精排。
        """
        text = state.get("ocr_text") or state.get("question_raw", "")
        ps = state.get("problem_struct") or {}
        plan = state.get("solution_plan") or {}
        draft = state.get("draft_solution") or {}
        steps = draft.get("steps") or []

        # ① 题解里声明的定理，按出现顺序去重（挂到具体推导步骤）
        thms: list[str] = []
        for st in steps:
            t = (st.get("theorem") or "").strip()
            if t and t not in thms:
                thms.append(t)

        # ② 题目字面点名的定理/公式 + 规划声明的所需定理（挂到题目，作为题面依据）。
        #    这是「先规划、再解题、最后按题解与题目提到的定理和公式查依据」里
        #    「题目」这一路：题面点名的依据不依附某一步，应与题目平行展示。
        #    已在题解定理里的不重复收录，避免同一条依据在题目和步骤里出现两遍。
        q_terms: list[str] = []
        for t in list(ps.get("mentioned") or []) + list(plan.get("required_theorems") or []):
            t = re.split(r"[（(]", str(t or ""))[0].strip()  # 规划常带「（0/0 型）」尾巴
            if t and t not in q_terms and t not in thms:
                q_terms.append(t)

        chapter, topic = _guess_chapter(text)
        filters = {"chapter": chapter, "topic": topic}
        hits: dict[str, str] = {}
        per_thm: list[tuple[str, list]] = []
        for thm in thms[:6]:
            terms = _thm_terms(thm) or [thm]
            # 用「定理名 + 别名」拼 query：教材写「L'Hospital 法则」、题解写
            # 「洛必达法则」，只查原词会漏；别名展开后 BM25 才能命中。
            chunks = retriever.retrieve(
                " ".join(terms),
                filters=filters,
                candidate_theorems=terms,
                top_k=3,
                # 定理名是强关键词，词法通路足够准；跳过精排每次省 1~2s，
                # 而一道题通常有 3~5 条定理，这一项直接决定「解题后检索」能不能
                # 控制在 1s 以内。
                rerank=False,
            )
            per_thm.append((thm, chunks))
            cid = _pick_theorem_chunk(thm, chunks)
            if not cid:
                # top-3 里没有就单路全库补查一次：定理名是强关键词，
                # BM25 单查往往比「混合池里挑」更直接（毫秒级）。
                cid = _lookup_in_corpus(getattr(retriever, "bm25", None), thm, chapter)
            if cid:
                hits[thm] = cid

        # ③ 题目点名 / 规划声明的定理与公式：同样逐条回查教材，但归为「题面依据」，
        #    不挂到任何推导步骤（前端与题目平行展示）。用同一套强关键词 + 词法通路，
        #    不引入精排，和步骤级检索保持同一套口径。
        q_hits: list[str] = []
        for term in q_terms[:6]:
            terms = _thm_terms(term) or [term]
            chunks = retriever.retrieve(
                " ".join(terms), filters=filters, candidate_theorems=terms,
                top_k=3, rerank=False,
            )
            cid = _pick_theorem_chunk(term, chunks)
            if not cid:
                cid = _lookup_in_corpus(getattr(retriever, "bm25", None), term, chapter)
            if cid and cid not in q_hits and cid not in hits.values():
                q_hits.append(cid)

        # 合并去重。只收「对得上题解某一步」或「题面点名」的块，各留 1 条：
        #   先放每条已定位定理的原文，再放题面依据，最后给没定位上的定理留兜底
        #   —— 把整个 top-3 都塞进去，会出现「第九章数项级数」混在洛必达的题里，
        #   比空着更容易误导。
        merged: list[dict] = []
        seen: set[str] = set()
        by_id = {c.chunk_id: c for c in getattr(retriever, "chunks", [])}
        for cid in dict.fromkeys([*hits.values(), *q_hits]):
            c = by_id.get(cid)
            # 例题块不进依据（_pick_theorem_chunk 已跳过，这里兜住 BM25 补查那条路）
            if c is not None and cid not in seen and not _is_example(c):
                seen.add(cid)
                merged.append(c.to_dict())
        for thm, chunks in per_thm:
            if thm in hits or not chunks:
                continue
            top = chunks[0]
            if _is_example(top):
                continue
            # 兜底候选也要过最低相关度：没定位上的定理（如「未定式判定」这种
            # 模型自造的方法名）检索结果往往是噪声，塞进右栏比空着更误导。
            if max(_lcs(thm, _field(top, "topic")), _lcs(thm, _field(top, "content"))) < 3:
                continue
            if top.chunk_id not in seen:
                seen.add(top.chunk_id)
                merged.append(top.to_dict())
        merged = merged[:8]

        # 兜底：题解没写定理（或全都没匹配上）时，不能一条教材都不给，
        # 这时退回按题面检索一次（带精排，尽量找相关章节）。
        if not merged:
            _emit(emit, "retrieve", "题解未标注定理，改按题面检索")
            merged = [
                c.to_dict()
                for c in retriever.retrieve(" ".join(filter(None, [text, topic, chapter])), filters=filters)
                if not _is_example(c)
            ]

        _emit(
            emit,
            "retrieve",
            f"按题解 {len(thms)} 条定理 + 题面 {len(q_terms)} 条点名依据检索到 {len(merged)} 个教材片段"
            f"（{len(hits)} 条步骤定理 / {len(q_hits)} 条题面依据已定位原文）",
            {"theorems": thms, "hits": hits, "question_terms": q_terms,
             "question_basis": q_hits, "status": retriever.status()},
        )
        return {
            "retrieved_chunks": merged,
            "retrieval_meta": retriever.status(),
            "candidate_theorems": thms,
            "theorem_hits": hits,
            "question_basis": q_hits,
        }

    def _context(chunks: list[dict], limit: int = 5, max_chars: int = 700) -> str:
        # 单块截断：教材块本身可能上千字（例题带完整解答），5 块全量塞进去光上下文就上万 token，
        # 实测会把单题预算直接顶爆（预算熔断后拿不到解答）。截断保头部即可，
        # 定理陈述和定义都在块首，尾部多是推导细节。
        lines = []
        for c in (chunks or [])[:limit]:
            loc = f"{c.get('source','知识库')} · {c.get('chapter','')}"
            body = (c.get("content", "") or "").strip()
            if len(body) > max_chars:
                body = body[:max_chars].rstrip() + " …"
            lines.append(f"[{c.get('chunk_id')}] ({loc})\n{body}")
        # 先解题后检索的流程里，解题节点执行时还没有任何教材片段 ——
        # 返回空串而不是「无可用检索结果」，避免模型以为检索失败而拒答。
        return "\n\n".join(lines)

    # ---------- 4. 规划 ----------
    def plan_node(state: dict) -> dict:
        ps = state.get("problem_struct", {})
        retry = state.get("retry_count", 0)
        mock = {
            "method": "",
            "alternatives": [],
            "required_theorems": [ps.get("topic", "相关定理")],
            "preconditions": ["检查函数在相关点连续/可导"],
            "steps": ["整理表达式", "选择方法", "逐步推导", "核对结果"],
            "risks": ["需确认定理前提"],
        }
        # 快速模式：不单独发一次 solver 请求做规划（这是整条链上第二长的一段，
        # 带思维链时常超过 20s），直接把「选方法」交给解题节点一并产出。
        # 规划节点保留为本地骨架，保证 method 字段与「审核→重试」的条件边仍然可用。
        if cfg.app.fast_mode and retry == 0:
            _emit(emit, "plan", "快速模式：跳过独立规划，与解题合并")
            return {"solution_plan": mock, "plan_fused": True}
        issues = state.get("review_issues") or state.get("review", {}).get("issues") or []
        user = f"题目结构：{json.dumps(ps, ensure_ascii=False)}"
        ctx = _context(state.get("retrieved_chunks", []))
        if ctx:
            user += f"\n\n教材片段（上一轮按定理检索）：\n{ctx}"
        if issues:
            user += f"\n\n上一轮审核质疑（请更换方法或修正前提，勿重复原思路）：{json.dumps(issues, ensure_ascii=False)}"
        obj, err = _llm_json(solver, [{"role": "system", "content": PLAN_SYSTEM}, {"role": "user", "content": user}], SolutionPlan, mock)
        if obj is None:
            _emit(emit, "plan", f"⚠️ 降级：{err}")
            return {"degraded": True, "error": err, "solution_plan": mock}
        _emit(emit, "plan", f"解题规划完成：{obj.method}", {"retry": retry})
        return {"solution_plan": obj.model_dump(), "plan_fused": False}

    # ---------- 5. 解题 ----------
    def solve_node(state: dict) -> dict:
        ps = state.get("problem_struct", {})
        plan = state.get("solution_plan", {})
        fused = bool(state.get("plan_fused"))
        mock = _demo_solution(ps)
        ctx = _context(state.get("retrieved_chunks", []))
        # 教材片段的两种来路：重试轮（审核打回重解）拿的是上一轮按定理检回的原文，
        # 这时给参考是有价值的；首轮没有片段，明确告诉模型「本流程先解题后检索」，
        # 否则它会因为看不到教材而在 caveats 里写一堆「缺少教材依据」。
        ctx_block = f"\n\n教材片段（上一轮按定理检索，可参考但不必依赖）：\n{ctx}" if ctx else (
            "\n\n（本流程为先解题、后按定理检索教材，此处不提供教材片段；"
            "请凭题目独立完整作答，并在每一步的 theorem 字段写明所依据的定理/方法规范名称。）"
        )
        if fused:
            # 快速模式：解题模型自己选方法，提示词换成合并版
            system = SOLVE_FAST_SYSTEM
            user = f"题目结构：{json.dumps(ps, ensure_ascii=False)}{ctx_block}"
            issues = state.get("review_issues") or []
            if issues:
                user += f"\n\n上一轮审核质疑（请更换方法或修正前提）：{json.dumps(issues, ensure_ascii=False)}"
            mock = {**mock, "method": mock.get("method") or "泰勒展开 / 等价无穷小"}
        else:
            system = SOLVE_SYSTEM
            user = (
                f"题目结构：{json.dumps(ps, ensure_ascii=False)}\n\n"
                f"解题计划：{json.dumps(plan, ensure_ascii=False)}{ctx_block}"
            )
        # 只有前端声明能接收流式增量时才开流（脚本里逐条打印会把终端刷爆）
        _emit(emit, "solve", "推理中…")
        delta = None
        if getattr(emit, "supports_delta", False):
            delta = lambda kind, chunk: _emit(  # noqa: E731
                emit, "solve", "", {"delta": kind, "text": chunk}
            )
        obj, err = _llm_json(
            solver,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            DraftSolution,
            mock,
            # 解题是整条链上最慢的一段（30~60s）。把思维链/正文增量实时抛给界面，
            # 用户 1~2s 内就能看到「它在想什么」，而不是盯着一根进度条。
            on_delta=delta,
        )
        if obj is None:
            # 不能把 mock 样例当成解答交出去：用户会以为「只是没配 Key」，
            # 实际原因是预算熔断/接口失败，必须如实说明并让报告标注「未生成解答」。
            _emit(emit, "solve", f"⚠️ 降级：{err}")
            return {
                "degraded": True,
                "error": err,
                "draft_solution": {
                    "method": "",
                    "answer": f"（未生成完整解答：{err}）",
                    "steps": [],
                    "verifiable_steps": [],
                    "caveats": [f"解题模型调用未成功，未产出解答。原因：{err}"],
                },
            }
        _emit(emit, "solve", "候选解答生成完成", {"steps": len(obj.steps)})
        draft = obj.model_dump()
        if err:
            # 自动修复过一次 / JSON 取自思维链：解答可用，但要在报告里如实标注
            _emit(emit, "solve", f"提示：{err}")
            draft["caveats"] = [*draft.get("caveats", []), err]
        return {"draft_solution": draft}

    # ---------- 6. SymPy 验证 ----------
    def verify_node(state: dict) -> dict:
        draft = state.get("draft_solution", {})
        try:
            v = verify_solution(draft.get("verifiable_steps"))
        except Exception as exc:  # 工具层异常不应打断整条链路
            v = Verification(status=UNVERIFIED)
            v.notes.append(f"验证环节内部异常，已跳过自动验证：{exc}")
        _emit(emit, "verify", f"工具验证：{v.label}", v.to_dict())
        return {"verification": v.to_dict(), "verification_status": v.status}

    # ---------- 7. 异构审核 ----------
    def review_node(state: dict) -> dict:
        ps = state.get("problem_struct", {})
        plan = state.get("solution_plan", {})
        draft = state.get("draft_solution", {})
        verification = state.get("verification", {})
        vstatus = state.get("verification_status", "unverified")
        mock = {
            "assertions": [
                f"结论为「{draft.get('answer','')}」，需核对每一步推导是否可逆。",
                f"工具验证状态为「{LABELS.get(vstatus, vstatus)}」，未覆盖的证明部分需人工检查。",
            ],
            "issues": [],
            "needs_revision": vstatus == MISMATCH,
            "handcheck": list(draft.get("caveats", [])),
        }
        if state.get("plan_fused"):
            # 快速模式下 solution_plan 只是本地骨架，把「方法」从解答里补回去，
            # 否则审稿人看到空计划会以为解题者没选方法（会白给一堆 issue）。
            plan_block = f"解题所用方法：{draft.get('method') or '（未声明）'}"
        else:
            plan_block = f"解题计划：{json.dumps(plan, ensure_ascii=False)}"
        user = (
            f"题目结构：{json.dumps(ps, ensure_ascii=False)}\n\n"
            f"{plan_block}\n\n"
            f"候选解答：{json.dumps(draft, ensure_ascii=False)}\n\n"
            f"工具验证结果：{json.dumps(verification, ensure_ascii=False)}"
        )
        obj, err = _llm_json(reviewer, [{"role": "system", "content": REVIEW_SYSTEM}, {"role": "user", "content": user}], ReviewResult, mock)
        if obj is None:
            return {"degraded": True, "error": err, "review": mock, "review_issues": []}
        review = obj.model_dump()
        needs_revision = bool(review.get("needs_revision")) or vstatus == MISMATCH
        retry = int(state.get("retry_count", 0))
        max_retry = int(state.get("max_retries", cfg.app.max_retries))
        will_retry = needs_revision and retry < max_retry
        if will_retry:
            retry += 1
        _emit(
            emit,
            "review",
            f"审核完成：{len(obj.issues)} 个质疑点" + ("，返回规划换方法" if will_retry else ""),
            review,
        )
        return {
            "review": {**review, "needs_revision": needs_revision, "will_retry": will_retry},
            "review_issues": [i.model_dump() for i in obj.issues],
            "retry_count": retry,
        }

    # ---------- 8. 报告 ----------
    def format_node(state: dict) -> dict:
        ps = state.get("problem_struct", {})
        plan = state.get("solution_plan", {})
        draft = state.get("draft_solution", {})
        verification = state.get("verification", {})
        review = state.get("review", {})
        chunks = state.get("retrieved_chunks", [])
        degraded = bool(state.get("degraded"))

        sources = [
            {"citation": f"{c.get('source','知识库')} · {c.get('chapter','')}", "chunk_id": c.get("chunk_id"), "score": c.get("score")}
            for c in chunks
        ]
        # 题目级依据：与题目平行展示的教材块 id。retrieve 节点已经把这些块
        # 并进 retrieved_chunks，这里只透传 id 顺序，前端按 chunk_index 取原文。
        question_basis = list(state.get("question_basis") or [])

        # 把每一步声明的定理名，映射回检索到的教材原文，补出 source_id。
        # 让前端能把「这一步」与「教材里对应定理的原文」一一对应。
        #
        # 优先级（从高到低）：
        #   0) retrieve 节点按定理检索时已经确认过的命中（theorem_hits）——
        #      那是「拿这条定理名单独去查」得到的最好结果，比在混合池里再挑一次准
        #   1) 在本次检索结果里按定理名（含别名）整串包含 / 高阈值 LCS 匹配
        #   2) BM25 全库补查（毫秒级），仍要求高阈值，宁缺勿错
        thm_hits = state.get("theorem_hits") or {}
        chunk_ids = {_field(c, "chunk_id") for c in chunks}

        def _match_theorem_source(step: dict, chunks: list[dict]) -> str | None:
            return _pick_theorem_chunk((step.get("theorem") or "").strip(), chunks)

        # 本次检索已按题解的每条定理查过，仍有步骤没命中（题解换了说法），
        # 这里再用 BM25 全库补查一次兜底（毫秒级，不碰 LLM 也不碰 reranker）。
        def _lookup_theorem_in_corpus(thm: str) -> str | None:
            return _lookup_in_corpus(getattr(retriever, "bm25", None), thm, ps.get("chapter") or "")

        steps = draft.get("steps", []) or []
        extra_ids: list[str] = []
        for st in steps:
            thm = (st.get("theorem") or "").strip()
            sid = thm_hits.get(thm) if thm else None
            if sid and sid not in chunk_ids:
                sid = None
            if not sid:
                sid = _match_theorem_source(st, chunks)
            if not sid:
                sid = _lookup_theorem_in_corpus(thm)
                if sid:
                    extra_ids.append(sid)
            if sid:
                st["source_id"] = sid
        # 全库补查到的定理原文也挂进 sources，保证「步骤 ↔ 来源」两边自洽
        if extra_ids:
            have = {c.get("chunk_id") for c in chunks}
            for cid in dict.fromkeys(extra_ids):
                if cid in have:
                    continue
                for c in getattr(retriever, "chunks", []):
                    if c.chunk_id == cid:
                        sources.append({
                            "citation": f"{c.source or '知识库'} · {c.chapter}",
                            "chunk_id": c.chunk_id,
                            "score": None,
                        })
                        break
        limitations = list(draft.get("caveats", []))
        limitations += list(verification.get("notes", []))
        limitations += list(review.get("handcheck", []))
        if state.get("skip_review"):
            limitations.append("本次为极速模式，跳过了交叉审核；定理前提与严谨性需自行核对。")
        if degraded:
            limitations.append(f"本次运行发生降级：{state.get('error') or '模型/预算限制'}，结论仅供参考。")
        if verification.get("status") == "unverified":
            limitations.append("本题未通过 SymPy 自动验证，正确性需人工确认。")
        limitations = list(dict.fromkeys([x for x in limitations if x]))

        # 快速模式下方法由解题模型给，两段式由规划节点给；两边都空时按推导文本兜一个，
        # 否则报告里「方法」一栏是空白，看起来像渲染坏了。
        method = draft.get("method") or plan.get("method", "")
        if not method.strip():
            method = _infer_method(draft.get("steps") or [], draft.get("answer", "")) or \
                "（模型未单独声明方法，见下方推导）"

        report = {
            "title": f"解题报告：{ps.get('problem_type') or '数学分析题目'}",
            "question": state.get("ocr_text") or state.get("question_raw", ""),
            "structure": ps,
            "knowledge_points": sorted({c.get("topic") or c.get("chapter") or "" for c in chunks if c.get("topic") or c.get("chapter")}),
            "sources": sources,
            "question_basis": question_basis,
            "method": method,
            "alternatives": draft.get("alternatives") or plan.get("alternatives", []),
            "steps": steps,
            "verification": verification,
            "review_assertions": review.get("assertions", []),
            "review_issues": review.get("issues", []),
            "conclusion": draft.get("answer", ""),
            "limitations": limitations,
            "budget": state.get("budget", {}),
            "degraded": degraded,
        }
        _emit(emit, "format", "报告生成完成", {"degraded": degraded})
        return {"final_report": report}

    return {
        "parse": parse_node,
        "analyze": analyze_node,
        "retrieve": retrieve_node,
        "plan": plan_node,
        "solve": solve_node,
        "verify": verify_node,
        "review": review_node,
        "format": format_node,
    }
