"""API Key 自检：逐个角色发一条最小请求，确认 Key / base_url / 模型名都正确。

用法：
    python scripts/check_env.py

常见报错的含义已在输出里给出中文提示。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import active_env_file, get_settings
from src.llm.client import Budget, LLMClient

ROLES = [
    ("solver", "解题链路"),
    ("reviewer", "审核（异构）"),
    ("light", "解析/整理"),
]

TIPS = {
    # 顺序有意义：先匹配具体短语，再落到通用状态码
    "overdue": "账号欠费 —— 到百炼/DeepSeek 控制台充值后即可恢复（错误码含 overdue-payment）",
    "good standing": "账号状态异常（多为欠费或免费额度用尽）—— 控制台充值/实名后恢复",
    "access denied": "无权限或账号异常 —— 确认已开通该模型，且账号未欠费",
    "free quota": "免费额度已用尽 —— 该模型转为付费，需充值或换模型",
    "insufficient": "余额不足 —— 需要到平台充值",
    "401": "Key 无效或未生效 —— 确认复制完整、没有多余空格，且与 base_url 属于同一平台",
    "403": "无权限 —— 该 Key 可能未开通此模型，或账号未实名认证",
    "404": "模型名或路径不对 —— 检查 MODEL 名称拼写、base_url 是否少了 /v1 或多了斜杠",
    "400": "请求被拒绝 —— 多为模型名不存在，或该模型不支持 response_format",
    "429": "限流 —— 免费模型有并发/额度限制，稍后重试或充值",
    "Connection": "网络不通 —— 检查代理设置，或 base_url 域名拼写",
    "timeout": "请求超时 —— 网络慢或模型响应太慢",
}


def hint(err: str) -> str:
    low = err.lower()
    for key, msg in TIPS.items():
        if key.lower() in low:
            return msg
    return "未识别的错误，请把上面的原始信息发给开发者"


def main() -> int:
    s = get_settings()
    print("=" * 66)
    print(f"配置文件：{ROOT / 'configs' / 'model.yaml'}")
    print(f"环境文件：{ROOT / '.env'}" + ("（存在）" if (ROOT / ".env").exists() else "（不存在！请先复制 .env.example）"))
    print(f"离线模式 MOCK_LLM：{s.mock_llm}")
    print("=" * 66)

    if s.mock_llm:
        print("\n当前处于 mock 模式（MOCK_LLM=1），不会发起真实请求。")
        print("要看真实效果，请把 .env 里的 MOCK_LLM 改成 0。\n")

    ok_count = 0
    checked = 0

    for role, label in ROLES:
        spec = getattr(s.models, role)
        key = spec.api_key
        print(f"\n[{role}] {label}")
        print(f"  base_url : {spec.base_url or '（空）'}")
        print(f"  model    : {spec.model or '（空）'}")

        if not key:
            if role == "light":
                print("  api_key  : 未配置 → 自动复用 solver，跳过")
            else:
                print("  api_key  : 未配置 ✗  请到 .env 里填写")
            continue

        checked += 1
        print(f"  api_key  : {key[:6]}...{key[-4:]}（长度 {len(key)}）")

        budget = Budget(max_calls=3, max_tokens=4000, max_cost=1.0)
        client = LLMClient(spec, budget, mock=s.mock_llm)
        t0 = time.time()
        try:
            res = client.chat(
                [{"role": "user", "content": "只回复两个字：收到"}],
                max_tokens=16,
            )
            dt = time.time() - t0
            text = (res.content or "").strip().replace("\n", " ")[:20]
            print(f"  结果     : 成功 ✓  {dt:.1f}s  回复「{text}」  tokens={res.prompt_tokens + res.completion_tokens}")
            if res.degraded:
                print(f"  注意     : 走了降级兜底，原始错误：{res.error}")
            else:
                ok_count += 1
        except Exception as exc:  # noqa: BLE001
            dt = time.time() - t0
            msg = str(exc)
            print(f"  结果     : 失败 ✗  {dt:.1f}s")
            print(f"  原始错误 : {msg[:300]}")
            print(f"  可能原因 : {hint(msg)}")

    print("\n" + "=" * 66)
    if checked == 0:
        print("没有任何真实 Key 被配置。")
        print("最少需要两个：SOLVER_API_KEY（解题）+ REVIEWER_API_KEY（审核，必须不同厂商）")
        print("注册入口见 .env.example 内的注释。")
    else:
        print(f"通过 {ok_count} / {checked} 个角色。")
        if ok_count == checked:
            print("全部可用，直接运行： streamlit run src/app/streamlit_app.py")
        else:
            print("有角色未通过，按上面的提示修正 .env 后重跑本脚本。")
    print("=" * 66)
    return 0 if ok_count == checked else 1


if __name__ == "__main__":
    raise SystemExit(main())
