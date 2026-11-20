"""输入解析（P1 阶段）：文本规范化；单题图片走视觉模型；PDF 仍待 MinerU。"""
from __future__ import annotations

import re
from pathlib import Path


class OcrNotAvailable(RuntimeError):
    """该输入类型尚未接入。"""


def normalize_text(raw: str) -> str:
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_image(path: str | Path) -> tuple[str, list[str]]:
    """手机拍摄的单题图片 → (题目文本, 警告列表)。

    批量教材 PDF 仍建议走 MinerU；这里只解决「一道题拍张照就能算」的场景，
    所以直接交给视觉模型输出 LaTeX，不做版面分析。
    """
    from src.ingestion.ocr import OcrError, recognize_question

    p = Path(path)
    if not p.exists():
        raise OcrError(f"图片不存在：{p}")
    try:
        result = recognize_question(p.read_bytes())
    except OcrError:
        raise
    warnings: list[str] = []
    if result.warning:
        warnings.append(result.warning)
    return normalize_text(result.text), warnings


def parse_pdf(path: str | Path) -> tuple[str, list[str]]:
    """教材 PDF：建议 MinerU 解析版面与公式，产出 Markdown/LaTeX。"""
    raise OcrNotAvailable(
        "PDF 解析尚未接入 MinerU。批量教材请用 scripts/ocr_book.py 按页 OCR，"
        "再用 scripts/build_knowledge_base.py 建库。"
    )
