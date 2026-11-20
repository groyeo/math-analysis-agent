"""扫描版教材 OCR：整页图 → 视觉模型 → 带 LaTeX 的 Markdown。

特点：
- 断点续跑：每页单独存盘，已存在则跳过（中断后重跑即可继续）
- 并发：--workers 控制，注意平台 QPS 限制
- 元数据：从页眉/页脚抽取 chapter 与页码，写入 index.json
- 失败可重试：失败的页记录在 failed.txt，重跑时自动重试

用法：
    python scripts/ocr_book.py --pdf "D:\\...\\上册.pdf" --book shangce --start 1 --end 20
    python scripts/ocr_book.py --pdf "D:\\...\\上册.pdf" --book shangce            # 全书
    python scripts/ocr_book.py --pdf "..." --book shangce --retry-failed          # 只重试失败页
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

OUT_ROOT = ROOT / "outputs" / "ocr"

PROMPT = (
    "请把这一页教材完整转写为 Markdown。要求：\n"
    "1. 所有数学公式用 LaTeX 写在行内 $...$ 或独立 $$...$$ 中，不要用图片或文字描述代替公式；\n"
    "2. 保留定理/定义/例的编号（如「定理 2.3.3 (Stolz 定理)」）、章节标题、正文分段；\n"
    "3. 页眉的章节名和页脚的页码照原样输出；\n"
    "4. 不要添加原文没有的内容，不要总结、不要解释、不要加代码块围栏。"
)


def make_client() -> OpenAI:
    key = os.getenv("REVIEWER_API_KEY") or os.getenv("SOLVER_API_KEY")
    base = os.getenv("REVIEWER_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not key:
        sys.exit("未找到 API Key，请先在 .env 中配置 REVIEWER_API_KEY（百炼的 Key 即可）")
    return OpenAI(api_key=key, base_url=base, timeout=180, max_retries=2)


_local = threading.local()


def get_doc(pdf_path: str) -> pymupdf.Document:
    """pymupdf 的 Document 不是线程安全的，每个线程各持一个实例。"""
    doc = getattr(_local, "doc", None)
    if doc is None:
        doc = pymupdf.open(pdf_path)
        _local.doc = doc
    return doc


def page_image_b64(doc: pymupdf.Document, idx: int) -> tuple[str, str]:
    """优先取扫描原图（保留最高清晰度）；取不到再渲染。返回 (base64, mime_ext)。"""
    page = doc[idx]
    imgs = page.get_images(full=True)
    if imgs:
        try:
            info = doc.extract_image(imgs[0][0])
            if info and info.get("image"):
                return base64.b64encode(info["image"]).decode(), info.get("ext", "jpeg")
        except Exception:
            pass
    pix = page.get_pixmap(dpi=200)
    return base64.b64encode(pix.tobytes("png")).decode(), "png"


# 视觉模型降级链的共享状态：记住哪个可用、哪个已额度耗尽，避免每页重复撞墙
_model_lock = threading.Lock()
_good_model: str | None = None
_bad_models: set[str] = set()


def _model_order(models: list[str]) -> list[str]:
    with _model_lock:
        good, bad = _good_model, set(_bad_models)
    order = [m for m in models if m not in bad]
    if good in order:
        order.remove(good)
        order.insert(0, good)
    return order or list(models)


def _mark_good(model: str) -> None:
    global _good_model
    with _model_lock:
        _good_model = model


def _mark_bad(model: str) -> None:
    with _model_lock:
        _bad_models.add(model)


def clean_markdown(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


CHAP_RE = re.compile(r"第\s*([一二三四五六七八九十百]+)\s*章\s*([^\n|｜]*)")
PAGENO_RE = re.compile(r"^\s*(\d{1,3})\s*$", re.M)


def extract_meta(md: str) -> dict:
    head = "\n".join(md.splitlines()[:3])
    tail = "\n".join(md.splitlines()[-4:])
    chapter = ""
    m = CHAP_RE.search(head) or CHAP_RE.search(md[:200])
    if m:
        chapter = f"第{m.group(1)}章 {m.group(2).strip()}".strip()
    page = ""
    nums = PAGENO_RE.findall(tail)
    if nums:
        page = nums[-1]
    return {"chapter": chapter, "page": page}


def ocr_one(
    client: OpenAI, pdf_path: str, idx: int, out_dir: Path, models: list[str], max_tokens: int
) -> tuple[int, str]:
    fp = out_dir / f"p{idx + 1:04d}.md"
    if fp.exists() and fp.stat().st_size > 30:
        return idx, "skip"
    doc = get_doc(pdf_path)
    b64, ext = page_image_b64(doc, idx)

    last_err: Exception | None = None
    # 依次尝试模型链：某个视觉模型额度用尽（403）或下线（404）时自动切下一个
    for model in _model_order(models):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            md = clean_markdown(resp.choices[0].message.content or "")
            if len(md) < 20:
                raise RuntimeError(f"返回内容过短（{len(md)} 字），可能不是正文页")
            fp.write_text(md, encoding="utf-8")
            _mark_good(model)
            return idx, f"ok:{model}"
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            msg = str(exc)
            if "403" in msg or "404" in msg or "quota" in msg.lower():
                _mark_bad(model)
                continue
            break
    raise last_err if last_err else RuntimeError("OCR 失败")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--book", required=True, help="输出子目录名，如 shangce / xiace")
    ap.add_argument("--start", type=int, default=1, help="起始页（1 起）")
    ap.add_argument("--end", type=int, default=0, help="结束页（含），0 表示到末尾")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--model",
        default="qwen3-vl-plus,qwen-vl-ocr,qwen-vl-max",
        help="视觉模型，可用逗号给出降级链：前面的额度用尽/下线后自动切后面的",
    )
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    out_dir = OUT_ROOT / args.book
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_fp = out_dir / "failed.txt"

    doc = pymupdf.open(args.pdf)
    total = doc.page_count
    start = max(0, args.start - 1)
    end = total if args.end in (0, None) else min(total, args.end)
    pages = list(range(start, end))

    if args.retry_failed and failed_fp.exists():
        nums = [int(x) for x in failed_fp.read_text().split() if x.strip().isdigit()]
        pages = [n - 1 for n in nums if 0 < n <= total]
        print(f"仅重试失败页：{len(pages)} 页")

    pending = [i for i in pages if not ((out_dir / f"p{i + 1:04d}.md").exists() and (out_dir / f"p{i + 1:04d}.md").stat().st_size > 30)]
    print(f"《{args.book}》共 {total} 页，本次范围 {start + 1}-{end}，待处理 {len(pending)} 页，并发 {args.workers}")

    client = make_client()
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    ok = skip = fail = 0
    failed_pages: list[int] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(ocr_one, client, args.pdf, i, out_dir, models, args.max_tokens): i for i in pending}
        for n, fut in enumerate(as_completed(futs), 1):
            idx = futs[fut]
            try:
                _, status = fut.result()
                if status.startswith("ok"):
                    ok += 1
                else:
                    skip += 1
            except Exception as exc:  # noqa: BLE001
                fail += 1
                failed_pages.append(idx + 1)
                print(f"  [失败] p{idx + 1}: {str(exc)[:120]}")
            if n % 10 == 0 or n == len(pending):
                dt = time.time() - t0
                speed = n / dt if dt else 0
                eta = (len(pending) - n) / speed if speed else 0
                print(f"  进度 {n}/{len(pending)}  ok={ok} skip={skip} fail={fail}  {speed:.2f}页/秒  ETA {eta/60:.1f}分")

    if failed_pages:
        failed_fp.write_text("\n".join(str(p) for p in sorted(failed_pages)), encoding="utf-8")
        print(f"失败页已记录：{failed_fp}（用 --retry-failed 重试）")

    # 重建索引
    index = []
    for fp in sorted(out_dir.glob("p*.md")):
        md = fp.read_text(encoding="utf-8")
        meta = extract_meta(md)
        index.append({"file": fp.name, "pdf_page": int(fp.stem[1:]), **meta, "chars": len(md)})
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")

    done = len(list(out_dir.glob("p*.md")))
    print(f"完成：本次新增 {ok} 页，累计 {done}/{total} 页，耗时 {(time.time()-t0)/60:.1f} 分钟")
    doc.close()


if __name__ == "__main__":
    main()
