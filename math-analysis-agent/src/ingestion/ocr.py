"""单题图片 → LaTeX 题目文本（视觉模型）。

为什么单独一个模块而不是塞进 document.py：
- 图片预处理（EXIF 转正、缩放、压缩）是独立逻辑，和文本规范化无关；
- 视觉调用有自己的一套失败模式（模型无权限、图片过大、输出被截断），
  需要单独把原因报给界面，而不是笼统抛 "OCR 失败"。

走的是 OpenAI 兼容的 chat.completions 多模态接口，不依赖任何厂商 SDK。
模型可用性会变，换模型先跑 scripts/probe_vl2.py 实测。
"""
from __future__ import annotations

import base64
import io
import os
import re
import time
from dataclasses import dataclass, field

from src.config import ModelSpec, get_settings
from src.llm.client import Budget, LLMClient, LLMError

# 视觉模型能看清的最大边长。手机直出照片常见 4000×3000，base64 后接近 10MB，
# 既慢又容易被接口拒；缩到 1568 长边对 A4 纸张的题目完全够（实测公式仍清晰）。
MAX_SIDE = 1568
JPEG_QUALITY = 88
# 小于这个字节数基本不可能是一张有效题目图
MIN_IMAGE_BYTES = 1024

VISION_PROMPT = """你是数学题识别助手。把图片里的题目转写成可以直接计算的文本。

要求：
1. 数学公式一律用 LaTeX，行内公式用 $...$，独立成行的公式用 $$...$$；
2. 求和、积分、极限、分数必须用 \\sum \\int \\lim \\frac，不要写成 a/b 这种线性形式；
3. 只输出题目本身：不要解答、不要解释、不要重复原文标题、不要加"识别结果："之类前缀；
4. 忽略页码、页眉、手写批注和旁注；
5. 图中有多个小问时按 (1)(2)(3) 保留；
6. 实在看不清的符号写成 [?]，不要猜。"""

# 模型时不时会把"识别结果："或 markdown 围栏一起吐出来
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?", re.M)
_PREFIX_RE = re.compile(r"^\s*(识别结果|识别出的题目|题目|原题|题目如下)\s*[:：]\s*")
_FAIL_HINT = ("抱歉", "无法识别", "看不清", "无法读取", "没有看到题目", "图片中没")

# ---------------------------------------------------------------- 图片中转兜底
# 2026-11 起，ModelScope 推理网关在收到 base64 图片时会先把图写进它自己的 OSS，
# 那一步稳定返回 500 / oss 403 bucket acl —— 换任何模型（Intern、ERNIE、Qwen3.5、
# GLM）都是同一条报错，属于平台侧故障，不是我们传错了格式。
# 同一张图改成「公网 URL」送进去却是好的，所以这里补一条兜底：图先传到临时图床，
# 再把 URL 交给模型。图床链接 1 小时后自动失效，且只在上面的报错出现时才走。
RELAY_UPLOAD = "https://litterbox.catbox.moe/resources/internals/api.php"
RELAY_TTL = "1h"
RELAY_ENABLED = os.getenv("VISION_RELAY", "1").strip() not in ("0", "false", "False", "")
_RELAY_HINT = ("failed to upload object", "bucket acl", "StatusCode=403")
# 一次进程内只要确认过 base64 入口坏了，后面就别再白等一次失败的往返
_base64_broken = False


def _is_upload_broken(exc: BaseException) -> bool:
    msg = str(exc)
    return any(h in msg for h in _RELAY_HINT)


def _relay_image_url(img_bytes: bytes, mime: str) -> str:
    """把图片丢到临时图床，换回一个模型能抓取的 URL。失败抛 OcrError。"""
    try:
        import requests
    except Exception as exc:  # pragma: no cover - requests 是已有依赖
        raise OcrError(f"图片中转需要 requests：{exc}") from exc

    ext = "jpg" if mime in ("jpeg", "jpg") else mime
    # 图床会挡裸 requests 的 UA（直接 412/403），必须带一个正常浏览器的
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    last = ""
    for attempt in range(2):
        try:
            r = requests.post(
                RELAY_UPLOAD,
                data={"reqtype": "fileupload", "time": RELAY_TTL},
                files={"fileToUpload": (f"question.{ext}", img_bytes, f"image/{mime}")},
                headers=headers,
                timeout=30,
            )
        except Exception as exc:  # noqa: BLE001
            last = f"网络异常：{exc}"
            continue

        url = (r.text or "").strip()
        if r.status_code == 200 and url.startswith("http"):
            return url
        last = f"HTTP {r.status_code}：{url[:60].replace(chr(10), ' ')}"
    raise OcrError(f"图片中转失败（{last}）")


class OcrError(RuntimeError):
    """图片识别失败，消息直接展示给用户。"""


@dataclass
class OcrResult:
    text: str
    model: str
    elapsed: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    warning: str = ""
    # 预处理后的实际尺寸，用于告诉用户"你的图被缩放过"
    size: tuple[int, int] = (0, 0)
    # 是否走了「临时图床中转」这条兜底路径（图在公网短暂可见，界面需要如实说明）
    relayed: bool = False
    raw: str = field(default="", repr=False)


def prepare_image(data: bytes) -> tuple[bytes, str, tuple[int, int]]:
    """EXIF 转正 + 等比缩放 + 转 JPEG。返回 (字节, mime, 最终尺寸)。

    手机竖拍的题图如果不做 exif_transpose，到模型眼里就是躺倒 90 度的，
    识别出来全是乱码 —— 这是这类功能最常见的坑，必须在这里处理。
    """
    if not data or len(data) < MIN_IMAGE_BYTES:
        raise OcrError("图片内容为空或过小，可能没有选择到正确文件")

    try:
        from PIL import Image, ImageOps
    except Exception:  # pragma: no cover - PIL 是硬依赖，装不上就直接透传
        return data, "png", (0, 0)

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise OcrError(f"无法读取这张图片（可能不是标准图片格式）：{exc}") from exc

    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass  # EXIF 损坏不应阻断识别，按原方向继续

    # 透明底转白底：PNG 截图直接转 RGB 会得到黑底，公式黑字黑底等于不可读
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        canvas = Image.new("RGB", img.size, "white")
        canvas.paste(img, mask=img.split()[-1])
        img = canvas
    elif img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue(), "jpeg", img.size


def clean_output(text: str) -> str:
    """去掉围栏、前缀和多余空行，留下可以直接喂给解题流程的题目文本。"""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    t = _FENCE_RE.sub("", t)
    t = _PREFIX_RE.sub("", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _looks_like_failure(text: str) -> str:
    """返回失败原因；看起来正常则返回空串。"""
    if not text:
        return "模型没有返回任何文字"
    if len(text) < 4:
        return f"识别结果过短（{len(text)} 字）"
    head = text[:80]
    for hint in _FAIL_HINT:
        if hint in head:
            return f"模型表示读不出内容：{head[:40]}"
    return ""


def recognize_question(
    data: bytes,
    *,
    spec: ModelSpec | None = None,
    max_calls: int = 2,
) -> OcrResult:
    """把题目图片转成 LaTeX 文本。

    失败一律抛 OcrError 并带上可读原因：这个功能在界面上是显式入口，
    用户点了就必须知道为什么没成（模型没权限 / 图太大 / 输出被截断），
    而不是静默返回空字符串。
    """
    settings = get_settings()
    spec = spec or settings.models.vision
    if not spec or not spec.model:
        raise OcrError("未配置视觉模型：请在 configs/model.yaml 的 vision 段填写 model")
    if not spec.api_key:
        raise OcrError(
            f"视觉模型缺少 API Key：请在 .env 里设置 {spec.api_key_env}"
            + (f" 或 {spec.fallback_key_env}" if spec.fallback_key_env else "")
        )

    img_bytes, mime, size = prepare_image(data)
    b64 = base64.b64encode(img_bytes).decode()
    if len(b64) > 5_000_000:
        raise OcrError(
            f"图片压缩后仍过大（{len(b64) / 1_048_576:.1f}MB base64），请裁剪出题目区域再上传"
        )

    # 独立预算：识别属于前置步骤，不该占用解题那份额度
    budget = Budget(max_calls=max_calls, max_tokens=24000, max_cost=1.0)
    client = LLMClient(spec, budget)

    def _ask(image_url: str):
        return client.chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": VISION_PROMPT},
                    ],
                }
            ],
            temperature=0.0,
        )

    global _base64_broken
    t0 = time.time()
    relayed = False
    try:
        if _base64_broken and RELAY_ENABLED:
            res = _ask(_relay_image_url(img_bytes, mime))
            relayed = True
        else:
            try:
                res = _ask(f"data:image/{mime};base64,{b64}")
            except LLMError as exc:
                # 平台侧的图片上传故障：换成公网 URL 再试一次，别让整个入口哑掉
                if not (RELAY_ENABLED and _is_upload_broken(exc)):
                    raise
                _base64_broken = True
                res = _ask(_relay_image_url(img_bytes, mime))
                relayed = True
    except LLMError as exc:
        raise OcrError(f"视觉模型调用失败：{exc}") from exc

    elapsed = time.time() - t0
    text = clean_output(res.content)
    # 正文空但思维链有内容：推理型视觉模型偶发，退一步从 reasoning 里取
    if not text and res.reasoning:
        text = clean_output(res.reasoning)

    reason = _looks_like_failure(text)
    if reason:
        detail = f"（finish_reason={res.finish_reason}）" if res.finish_reason else ""
        raise OcrError(f"{reason}{detail}")

    warning = ""
    if res.finish_reason == "length":
        warning = "输出可能被截断，请核对题目最后一行是否完整"
    if size and max(size) < 400:
        warning = (warning + "；" if warning else "") + "图片分辨率偏低，公式容易认错，建议重拍"

    return OcrResult(
        text=text,
        model=res.model,
        elapsed=elapsed,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        warning=warning,
        size=size,
        relayed=relayed,
        raw=res.content,
    )
