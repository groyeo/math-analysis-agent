"""从 knowledge_base/documents/*.md 构建 chunks.jsonl；无文档时校验现有知识库。

Markdown 约定：
- 每个 "## " 二级标题作为一个知识单元；
- 可在标题下用 "分类: 定理" 形式标注 content_type；
- chapter 取文档首行 "章节: xxx" 或文件名。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "knowledge_base" / "documents"
OUT = ROOT / "knowledge_base" / "chunks" / "chunks.jsonl"

_CT = ("定义", "定理", "推论", "证明", "例题", "公式", "方法", "判别法", "性质", "分类", "法则")


def guess_type(text: str) -> str:
    for t in _CT:
        if t in text[:60]:
            return t
    return "知识"


def build() -> None:
    docs = sorted(DOCS.glob("*.md"))
    if not docs:
        validate()
        return
    chunks = []
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        m = re.search(r"章节[:：]\s*(.+)", text)
        chapter = m.group(1).strip() if m else doc.stem
        sections = re.split(r"\n##\s+", text)
        for i, sec in enumerate(sections):
            sec = sec.strip()
            if not sec or i == 0:
                continue
            title, _, body = sec.partition("\n")
            body = body.strip()
            if not body:
                continue
            chunks.append(
                {
                    "chunk_id": f"{doc.stem}_{i:03d}",
                    "chapter": chapter,
                    "section": "",
                    "content_type": guess_type(sec),
                    "topic": title.strip(),
                    "source": doc.stem,
                    "page": "",
                    "content": body,
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"已从 {len(docs)} 个文档生成 {len(chunks)} 个知识片段 → {OUT}")


def validate() -> None:
    if not OUT.exists():
        print("未找到 chunks.jsonl，且 documents/ 无 Markdown 文档。")
        sys.exit(1)
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    chapters = {}
    for r in rows:
        chapters[r.get("chapter", "未分类")] = chapters.get(r.get("chapter", "未分类"), 0) + 1
    print(f"现有知识库：{len(rows)} 个片段")
    for k, v in sorted(chapters.items()):
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    build()
