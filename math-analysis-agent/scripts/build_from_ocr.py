"""把 OCR 出来的教材页面切分成结构化知识块（chunks）。

设计要点（对应方案 7.4）：
- 按「定理 / 定义 / 推论 / 引理 / 命题 / 例」的**语义边界**切分，而不是固定窗口，
  否则定理会被页边界或字数上限拦腰截断，检索出来的内容不可用。
- 每个块带 chapter / section / content_type / topic / page，混合检索的元数据过滤才有效。
- 页码取书本印刷页码（来自页脚），便于报告里写「依据 定理 5.2（第 80 页）」。

用法：
    python scripts/build_from_ocr.py --book shangce --title "数学分析（陈纪修）上册"
    python scripts/build_from_ocr.py --book shangce --book xiace --merge   # 多本合并
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OCR_ROOT = ROOT / "outputs" / "ocr"
OUT = ROOT / "knowledge_base" / "chunks" / "chunks.jsonl"

# 语义锚点：定理类可无编号；「例」必须带编号（1.2.3 形式），否则会把正文里的
# 「例如…」当成例题切分点，把整段论述切碎（这是实测踩到的坑）。
ANCHOR_THEO = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?[ \t]*"
    r"(定理|定义|推论|引理|命题|性质|法则)[ \t]*"
    r"(\d+(?:[.．]\d+)*)?",
    re.M,
)
ANCHOR_EX = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?[ \t]*例(?!如|题|子)[ \t]*(\d+(?:[.．]\d+)+)",
    re.M,
)
# 章名必须是「第X章 + 空格 + 干净标题」：正文里「第一章 § 2 中，我们曾指出…」这类
# 引用句若被当成章名，会把后续所有块的章节归属带偏，所以标题里排除标点与公式符号。
CHAP_LINE = re.compile(r"第\s*([一二三四五六七八九十]+)\s*章[　 ]+([^\n|｜‖，。；：、（）()【】$…*]{2,20})")
# 册/页眉的多种写法：`# 第十三章`、`第十三章`、`$\text { 第十三章 } \mid$`。
# 下册页眉被 OCR 成 LaTeX 片段，靠 CHAP_LINE 抓不到标题，结果整章挂到上一章名下
# （实测第十一章直接消失）。这里改成「整行去噪后就是第X章」即判定章节边界，
# 章名再用固定的章目录表补全，避免依赖不稳定的页眉标题。
_TEX_CMD = re.compile(r"\\[A-Za-z]+")
_LINE_JUNK = re.compile(r"[\$#\{\}\[\]\|｜‖_\\\s]")
LINE_CHAP = re.compile(r"^第([一二三四五六七八九十]+)章(.*)$")

# 陈纪修《数学分析》上下册章目录（OCR 页眉常常只剩「第X章」，标题由此表补全）
CHAPTER_TITLES = {
    "一": "集合与映射",
    "二": "数列极限",
    "三": "函数极限与连续函数",
    "四": "微分",
    "五": "微分中值定理及其应用",
    "六": "不定积分",
    "七": "定积分",
    "八": "反常积分",
    "九": "数项级数",
    "十": "函数项级数",
    "十一": "Euclid 空间上的极限和连续",
    "十二": "多元函数的微分学",
    "十三": "重积分",
    "十四": "曲线积分、曲面积分与场论",
    "十五": "含参变量积分",
    "十六": "Fourier 级数",
}
# 节标题两种常见排版：「§1 集合」与「2.1 数列极限的概念」
SEC_SEC = re.compile(r"^[ \t]*§\s*(\d{1,2})[ \t]+([^\n|｜‖]{1,28})[ \t]*[|｜‖]*[ \t]*$", re.M)
SEC_NUM = re.compile(r"^[ \t]*(\d{1,2}[.．]\d{1,2})[ \t]+([^\n|｜‖]{1,28})[ \t]*[|｜‖]*[ \t]*$", re.M)

# 噪声：目录页（大量省略号/点线）、封面、版权页
NOISE_HINT = ("……", "....", "…")


def clean_name(name: str) -> str:
    """清除 OCR 残留的花括号与首尾空白，如「第八章 反常积分}」→「第八章 反常积分」。

    数学章节页标题常夹带 LaTeX 花括号，若不清洗会让同章出现多种写法，
    导致混合检索的 chapter 元数据软加权无法命中。
    """
    s = re.sub(r"[{}\[\]￥\s]+$", "", (name or "").strip())
    return re.sub(r"\s+", " ", s).strip()


def is_noise_page(text: str) -> bool:
    if len(text.strip()) < 80:
        return True
    dots = sum(text.count(h) for h in NOISE_HINT)
    return dots >= 5


def load_book(book: str) -> list[dict]:
    d = OCR_ROOT / book
    index_fp = d / "index.json"
    meta = {}
    if index_fp.exists():
        for row in json.loads(index_fp.read_text(encoding="utf-8")):
            meta[row["file"]] = row
    pages = []
    for fp in sorted(d.glob("p*.md")):
        text = fp.read_text(encoding="utf-8")
        if is_noise_page(text):
            continue
        row = meta.get(fp.name, {})
        pages.append(
            {
                "pdf_page": int(fp.stem[1:]),
                "print_page": str(row.get("page", "") or ""),
                "text": text,
            }
        )
    return pages


def build_chunks(pages: list[dict], source: str, book: str) -> list[dict]:
    # 拼成一条字符流，并记录每页的起始偏移，用于把字符位置映射回页码
    buf: list[str] = []
    spans: list[tuple[int, int, dict]] = []  # (start, end, page)
    pos = 0
    for p in pages:
        t = p["text"].strip() + "\n"
        buf.append(t)
        spans.append((pos, pos + len(t), p))
        pos += len(t)
    full = "".join(buf)

    def locate(offset: int) -> dict:
        for s, e, p in spans:
            if s <= offset < e:
                return p
        return spans[-1][2] if spans else {"pdf_page": 0, "print_page": ""}

    anchors = [(m.start(), m.group(1), (m.group(2) or "").strip()) for m in ANCHOR_THEO.finditer(full)]
    anchors += [(m.start(), "例", m.group(1).strip()) for m in ANCHOR_EX.finditer(full)]
    anchors.sort(key=lambda x: x[0])
    # 去重：同一位置只保留一个（正则可重叠匹配）
    seen: set[int] = set()
    uniq = []
    for a in anchors:
        if a[0] in seen:
            continue
        seen.add(a[0])
        uniq.append(a)
    first_anchor = uniq[0][0] if uniq else 0

    # 扫描章节边界。判定方式：逐行去掉 $ # { } | \text \mid 等噪声后，
    # 若整行恰好是「第X章 [标题]」且很短，就认为是一处章节边界；
    # 章名优先用 CHAPTER_TITLES 补全（下册页眉常只剩「第十三章」+ LaTeX 残渣）。
    #
    # 目录页也会命中「第九章 数项级数」这类行（没有点线时无法靠正则区分），
    # 它出现在正文之前，一旦当成边界，那一章就只覆盖「目录→下一章标题」这一小段，
    # 整章内容会被算到下一章头上（实测上下册都踩过）。
    # 判据：同一页出现 ≥2 个不同章号 ⇒ 目录页；正文某页不可能同时讲两章。
    # 只在前 40 页里找，因为正文里也有「见第七章」这类跨章引用，
    # 若把后面的页也算进来，toc_end 会被推到书尾，所有章节归属全部清空（踩过）。
    toc_end = 0
    for s, e, pg in spans:
        if pg["pdf_page"] > 40:
            continue
        nums = set(re.findall(r"第([一二三四五六七八九十]+)章", full[s:e]))
        if len(nums) >= 2:
            toc_end = max(toc_end, e)

    chapters: list[tuple[int, str]] = []
    see: set[tuple[int, str]] = set()
    for m in re.finditer(r"(?m)^.*$", full):
        if m.start() < toc_end:
            continue
        line = m.group(0)
        if "章" not in line or len(line) > 80:
            continue
        flat = _LINE_JUNK.sub("", _TEX_CMD.sub("", line))
        lm = LINE_CHAP.match(flat)
        if not lm:
            continue
        num, rest = lm.group(1), lm.group(2)
        # 目录条目会带点线或页码（「第十二章 多元函数的微分学…120」）
        if len(rest) > 16 or re.match(r"^[§\d.．]", rest) or "..." in rest or "…" in rest:
            continue
        title = CHAPTER_TITLES.get(num, "")
        name = f"第{num}章 {title}".strip() if title else clean_name(f"第{num}章 {rest}")
        # 同一章名只在首次出现处记一次，避免页眉每页重复画边界
        key = (num, name)
        if key in see:
            continue
        see.add(key)
        chapters.append((m.start(), name))
    chapters.sort(key=lambda x: x[0])
    # 章号必须单调递增：页眉偶发误识别（如「第十二章」排在第十一章之前）
    # 会让中间好几章内容整体错挂，这里强制按章序过滤。
    order = {c: i + 1 for i, c in enumerate("一二三四五六七八九十")}
    order.update({"十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15, "十六": 16})
    kept: list[tuple[int, str]] = []
    last = 0
    for pos, name in chapters:
        n = order.get(name[1 : name.index("章")], 0)
        if n <= last:
            continue
        last = n
        kept.append((pos, name))
    chapters = kept
    sections = [(m.start(), clean_name(f"§{m.group(1)} {m.group(2).strip()}")) for m in SEC_SEC.finditer(full)]
    sections += [(m.start(), clean_name(f"{m.group(1)} {m.group(2).strip()}")) for m in SEC_NUM.finditer(full)]
    sections.sort(key=lambda x: x[0])

    def ctx(offset: int) -> tuple[str, str]:
        ch = ""
        for s, v in chapters:
            if s <= offset:
                ch = v
            else:
                break
        sec = ""
        for s, v in sections:
            if s <= offset:
                sec = v
            else:
                break
        return ch, sec

    # 正文起点：第一个章名出现之前的内容（前言、序、目录）不进知识库
    body_start = chapters[0][0] if chapters else 0
    uniq = [a for a in uniq if a[0] >= body_start]

    chunks = []
    for i, (start, kind, num) in enumerate(uniq):
        end = uniq[i + 1][0] if i + 1 < len(uniq) else len(full)
        body = full[start:end].strip()
        # 掐掉跨章溢出的尾部（下一个块已经包含）
        if len(body) < 25:
            continue
        # 过长的块按段落再切，避免单块塞进几页；同时保留块内偏移，让页码精确
        pieces = [(body, 0)] if len(body) <= 1800 else _split_long(body, 1200)
        for j, (piece, off) in enumerate(pieces):
            page = locate(start + off)
            ch, sec = ctx(start + off)
            head_line = piece.splitlines()[0][:60].replace("**", "").strip()
            topic = clean_name(
                f"{kind} {num}".strip() + (" " + _title_of(head_line) if _title_of(head_line) else "")
            )
            chunks.append(
                {
                    "chunk_id": f"{book}_{len(chunks) + 1:04d}",
                    "chapter": ch or "",
                    "section": sec,
                    "content_type": kind,
                    "topic": topic[:60],
                    "source": source,
                    "page": page["print_page"] or str(page["pdf_page"]),
                    "pdf_page": page["pdf_page"],
                    "part": j + 1 if len(pieces) > 1 else 0,
                    "content": piece.strip(),
                }
            )
    return chunks


def _title_of(head_line: str) -> str:
    """从「定理 2.3.3 (Stolz 定理) 设 …」里抠出定理名。首行是公式时返回空。"""
    if head_line.lstrip().startswith(("$", "$$", "#")):
        return ""
    m = re.search(r"[（(]([^）)]{2,20})[）)]", head_line)
    if m:
        return clean_topic_title(m.group(1))
    rest = re.sub(r"^(定理|定义|推论|引理|命题|性质|法则|例)\s*\d*[.．\d]*\s*", "", head_line)
    return clean_topic_title(rest[:24])


_HTML_TAG = re.compile(r"<[^>]{0,120}>")
_HTML_ATTR = re.compile(r"""(?:style|class|align|width|height)\s*=\s*["'][^"']*["']""", re.I)
_LATEX_CMD = re.compile(r"\\[a-zA-Z]+")
_SYMBOL_ONLY = re.compile(r"^[\W_]+$")


def clean_topic_title(t: str) -> str:
    """清洗 topic 中的 HTML / LaTeX / 纯符号残留。

    OCR 首行常夹带 `<div style="text-align:center">`、`$\\frac{a}{b}$` 等，
    若混进 topic 会成为 BM25 的噪声词项，直接拉低检索精度。
    清洗后若剩余内容不构成有意义的名称（如「a,b」「0,b-a」），返回空串。
    """
    s = _HTML_TAG.sub(" ", t or "")
    s = _HTML_ATTR.sub(" ", s)
    s = _LATEX_CMD.sub(" ", s)
    s = re.sub(r"[\${}·•|｜‖#*]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" ，,。.;；:：、-—")
    if len(s) < 2 or _SYMBOL_ONLY.match(s):
        return ""
    # 至少要有一段连续两个及以上的中文或字母，才当作名称
    if not re.search(r"[\u4e00-\u9fffA-Za-z]{2}", s):
        return ""
    return s[:28]


def _split_long(body: str, limit: int) -> list[tuple[str, int]]:
    """按空行切成不超过 limit 的段，返回 (文本, 在 body 内的起始偏移)。"""
    out: list[tuple[str, int]] = []
    start, n = 0, len(body)
    while start < n:
        end = min(n, start + limit)
        if end < n:
            cut = body.rfind("\n\n", start, end)
            if cut > start + limit // 3:
                end = cut
        seg = body[start:end].strip()
        if seg:
            out.append((seg, start))
        start = end
        while start < n and body[start : start + 2] == "\n\n":
            start += 2
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", action="append", required=True, help="ocr 输出子目录名，可多次指定")
    ap.add_argument("--title", action="append", default=None, help="与 --book 对应的书名")
    ap.add_argument("--merge", action="store_true", help="多本写入同一个 chunks.jsonl")
    args = ap.parse_args()

    titles = args.title or []
    all_chunks: list[dict] = []
    for i, book in enumerate(args.book):
        pages = load_book(book)
        if not pages:
            print(f"！{book} 没有可用的 OCR 页面，先跑 scripts/ocr_book.py")
            continue
        title = titles[i] if i < len(titles) else book
        cs = build_chunks(pages, title, book)
        print(f"{book}: {len(pages)} 页 → {len(cs)} 个知识块")
        by_type: dict[str, int] = {}
        for c in cs:
            by_type[c["content_type"]] = by_type.get(c["content_type"], 0) + 1
        print("   类型分布:", dict(sorted(by_type.items(), key=lambda x: -x[1])))
        all_chunks.extend(cs)

    if not all_chunks:
        print("没有生成任何块。")
        return

    if not args.merge:
        backup = OUT.with_name("chunks.jsonl.bak")
        if OUT.exists() and not backup.exists():
            backup.write_text(OUT.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"原知识库已备份 → {backup.name}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"写入 {len(all_chunks)} 个知识块 → {OUT}")


if __name__ == "__main__":
    main()
