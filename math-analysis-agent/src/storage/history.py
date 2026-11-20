"""历史记录存储：JSONL 追加，便于后续「历史记录 / 错题本」页面扩展。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.config import get_settings


def _path() -> Path:
    p = get_settings().root / get_settings().app.history_file
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_record(question: str, report: dict, budget: dict | None = None) -> None:
    record = {
        "ts": time.time(),
        "question": question,
        "title": report.get("title", ""),
        "conclusion": report.get("conclusion", ""),
        "verification": (report.get("verification") or {}).get("status", ""),
        "degraded": bool(report.get("degraded")),
        "budget": budget or report.get("budget", {}),
    }
    with open(_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_records(limit: int = 50) -> list[dict]:
    p = _path()
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[-limit:][::-1]
