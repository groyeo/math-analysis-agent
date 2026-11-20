"""LLM 客户端：兼容 OpenAI 协议的模型调用、异构审核、mock 降级、token 计量。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from src.config import ModelSpec

try:  # tiktoken 可选，用于更精确计数
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


class BudgetExceeded(RuntimeError):
    """单题预算（调用次数 / token / 成本）超限。"""


class LLMError(RuntimeError):
    """模型调用失败。"""


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数；中文约 1 字 ≈ 0.6~1 token。"""
    if not text:
        return 0
    return max(1, int(len(text) / 1.6))


# 一张图片在视觉模型里的消耗。OpenAI 兼容接口按图片分块计费，实际在 800~1600
# token 之间浮动；这里取上界，宁可把预算估紧一点也不要中途熔断。
IMAGE_TOKEN_COST = 1600


def flatten_messages(messages: Iterable[dict]) -> tuple[str, int]:
    """把 messages 摊成纯文本，并返回图片张数。

    多模态消息的 content 是 list（[{"type":"image_url",...}, {"type":"text",...}]），
    直接 "".join 会 TypeError。图片本身没有可数文本，按 IMAGE_TOKEN_COST 单算。
    """
    parts: list[str] = []
    images = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    images += 1
    return "".join(parts), images


@dataclass
class Budget:
    """单题预算（方案 15）。

    max_tokens 从 24000 提到 36000：接入上下两册（1693 块）后，检索上下文变长，
    证明题在「规划 + 解题 + 审核 + 返修」这条链上很容易顶到 24k 就被熔断，
    结果连解答都没生成出来。成本上界仍由 max_cost 兜着（¥1/题）。
    """

    max_calls: int = 12
    max_tokens: int = 36000
    max_cost: float = 1.0
    calls: int = 0
    tokens: int = 0
    cost: float = 0.0
    history: list[dict] = field(default_factory=list)

    def check(self, est_tokens: int = 0) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"已达到单题最大调用次数 {self.max_calls}")
        if self.tokens + est_tokens > self.max_tokens:
            raise BudgetExceeded(f"预计超出单题 token 上限 {self.max_tokens}")
        if self.cost > self.max_cost:
            raise BudgetExceeded(f"已超出单题成本上限 {self.max_cost}")

    def reset(self) -> None:
        self.calls = 0
        self.tokens = 0
        self.cost = 0.0
        self.history.clear()

    def record(self, prompt_tokens: int, completion_tokens: int, cost: float, model: str) -> None:
        self.calls += 1
        self.tokens += prompt_tokens + completion_tokens
        self.cost += cost
        self.history.append(
            {
                "ts": time.time(),
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": round(cost, 6),
            }
        )

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "tokens": self.tokens,
            "cost": round(self.cost, 6),
            "max_calls": self.max_calls,
        }


@dataclass
class ChatResult:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    degraded: bool = False
    error: str | None = None
    # 下面三个字段是排查「正文为空」的关键：
    # 现在的解题模型（DeepSeek-V4.1-Flash 等）会把思维链写进 reasoning_content，
    # 且 reasoning 也算 completion 预算。预算不够时 finish_reason=length、content=""，
    # 表现为「模型输出不符合结构要求」，光看 content 完全看不出是被截断了。
    finish_reason: str = ""
    reasoning: str = ""
    escalated_tokens: int = 0

    def json(self) -> Any:
        return json.loads(self._strip_fence(self.content))

    @staticmethod
    def _strip_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
        return text.strip()


class LLMClient:
    """单个角色（solver / reviewer / light）的模型客户端。"""

    def __init__(self, spec: ModelSpec, budget: Budget, mock: bool = False):
        self.spec = spec
        self.budget = budget
        self.mock = mock or not bool(spec.api_key)
        self._client = None

    # -------- 内部：真实调用 --------
    def _ensure_client(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI  # 延迟导入，避免 mock 模式依赖网络

        # 超时取 configs/app.yaml 的 request_timeout：以前这里硬编码 60s，
        # 改 yaml 完全不生效，带思维链的模型（单次 40~80s）必然超时重试。
        try:
            from src.config import get_settings

            timeout = float(get_settings().app.request_timeout or 120)
        except Exception:  # pragma: no cover
            timeout = 120.0

        self._client = OpenAI(
            api_key=self.spec.api_key,
            base_url=self.spec.base_url or None,
            timeout=timeout,
            max_retries=0,  # 重试由上层按「截断/降级」语义决定，这里静默重试只会拖长耗时
        )
        return self._client

    def chat(
        self,
        messages: Iterable[dict],
        *,
        json_mode: bool = False,
        mock: Any = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        on_delta: Callable[[str, str], None] | None = None,
    ) -> ChatResult:
        """调用模型；输出被 max_tokens 截断时自动放大预算重试一次。

        on_delta(kind, text) 用于流式展示思考过程：kind 取 "reasoning"（思维链）
        或 "content"（正文增量）。传入它会走 stream=True 的分支 —— 首字节通常
        1~2s 就回来，用户不必盯着空白页等 40s。不传则维持原来的整段返回。

        推理模型（reasoning_content）的思考过程与正文共用 completion 预算。
        固定 4096 时出现过「思考写了 11000 字、正文一个字没有、finish_reason=length」，
        整题因此降级为「无法解析为 JSON」。这里不再假设一个固定值够用：
        一旦发现被截断就把预算翻倍重试，直到 spec.max_tokens_ceiling 为止。
        """
        messages = list(messages)
        prompt_text, images = flatten_messages(messages)
        est = estimate_tokens(prompt_text) + images * IMAGE_TOKEN_COST

        if self.mock:
            self.budget.check(est)
            return self._mock_result(mock, prompt_text)

        base = int(max_tokens or self.spec.max_tokens)
        ceiling = max(int(getattr(self.spec, "max_tokens_ceiling", 0) or 0), base)
        attempt_tokens = base
        while True:
            self.budget.check(est)
            try:
                client = self._ensure_client()
                kwargs: dict[str, Any] = {
                    "model": self.spec.model,
                    "messages": messages,
                    "temperature": self.spec.temperature if temperature is None else temperature,
                    "max_tokens": attempt_tokens,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if on_delta is not None:
                    # 上一轮若因截断重试，先让界面把半截内容清掉再重新流式输出
                    on_delta("reset", "")
                    content, reasoning, finish_reason, pt, ct = self._stream(client, kwargs, on_delta)
                else:
                    resp = client.chat.completions.create(**kwargs)
                    choice = resp.choices[0]
                    msg = choice.message
                    content = msg.content or ""
                    reasoning = getattr(msg, "reasoning_content", "") or ""
                    finish_reason = choice.finish_reason or ""
                    usage = getattr(resp, "usage", None)
                    pt = getattr(usage, "prompt_tokens", 0) or 0
                    ct = getattr(usage, "completion_tokens", 0) or 0
            except BudgetExceeded:
                raise
            except Exception as exc:  # 网络/额度/鉴权
                # 真实调用失败一律上抛：早期这里会退回 mock 样例，而样例是「看起来很像
                # 真答案」的演示内容，上层一解析成功就会被当成模型解答写进报告（踩过）。
                raise LLMError(str(exc)) from exc

            # 部分兼容接口不返回 usage，用估算值兜底，成本核算不至于全成 0
            pt = pt or estimate_tokens(prompt_text) + images * IMAGE_TOKEN_COST
            ct = ct or estimate_tokens(content) + estimate_tokens(reasoning)
            cost = pt / 1000 * self.spec.price_in_per_1k + ct / 1000 * self.spec.price_out_per_1k
            self.budget.record(pt, ct, cost, self.spec.model)

            if finish_reason == "length" and attempt_tokens < ceiling:
                # 只有明确「被截断」才升级预算；正常 stop 的短回答不会多花钱
                attempt_tokens = min(attempt_tokens * 2, ceiling)
                continue
            return ChatResult(
                content,
                self.spec.model,
                pt,
                ct,
                cost,
                finish_reason=finish_reason,
                reasoning=reasoning,
                escalated_tokens=attempt_tokens if attempt_tokens != base else 0,
            )

    def _stream(self, client, kwargs: dict, on_delta: Callable[[str, str], None]):
        """流式调用：边收边回调，返回 (content, reasoning, finish_reason, pt, ct)。

        stream_options 只有部分服务端支持（ModelScope 早期版本会 400），
        失败时去掉它重开一次；两条路都失败就照常上抛，由上层降级。
        """
        stream = None
        try:
            stream = client.chat.completions.create(
                **kwargs, stream=True, stream_options={"include_usage": True}
            )
        except Exception:
            stream = client.chat.completions.create(**kwargs, stream=True)

        content_parts: list[str] = []
        reason_parts: list[str] = []
        finish_reason = ""
        usage = None
        try:
            for chunk in stream:
                usage = getattr(chunk, "usage", None) or usage
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    rc = getattr(delta, "reasoning_content", "") or ""
                    if rc:
                        reason_parts.append(rc)
                        on_delta("reasoning", rc)
                    cc = getattr(delta, "content", "") or ""
                    if cc:
                        content_parts.append(cc)
                        on_delta("content", cc)
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

        return (
            "".join(content_parts),
            "".join(reason_parts),
            finish_reason,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )


    def _mock_result(self, mock: Any, prompt_text: str, degraded: bool = False, error: str | None = None) -> ChatResult:
        if mock is None:
            payload: Any = {"note": "mock 模式：未提供结构化兜底内容"}
        else:
            payload = mock
        content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        pt = estimate_tokens(prompt_text)
        ct = estimate_tokens(content)
        cost = pt / 1000 * self.spec.price_in_per_1k + ct / 1000 * self.spec.price_out_per_1k
        self.budget.record(pt, ct, cost, self.spec.model or "mock")
        return ChatResult(content, self.spec.model or "mock", pt, ct, cost, degraded=degraded, error=error)
