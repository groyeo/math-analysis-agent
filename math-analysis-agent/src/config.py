"""统一配置加载：读取 configs/*.yaml，展开 ${ENV}，并用 Pydantic 校验。"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

try:  # python-dotenv 为可选但推荐
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent
# 支持 ${VAR} 与 ${VAR:-默认值} 两种写法，后者在环境变量为空时回退到默认值
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")


# 常见坑：Windows 记事本保存 `.env` 时会自动加后缀变成 `.env.txt`，
# 导致改了配置却不生效。这里把三种写法都收进来，并以修改时间最新的一份为准。
_ENV_CANDIDATES = (".env", ".env.txt", ".env.local")
_active_env: Path | None = None


def _load_env() -> None:
    global _active_env
    if load_dotenv is None:
        return
    existing = [ROOT / n for n in _ENV_CANDIDATES if (ROOT / n).exists()]
    if not existing:
        return
    # 新的先加载：先到先得，因此最新修改的文件拥有最高优先级
    existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    _active_env = existing[0]
    for path in existing:
        load_dotenv(path, override=False)


def active_env_file() -> Path | None:
    """返回实际生效的 env 文件，供自检脚本提示。"""
    if _active_env is None:
        _load_env()
    return _active_env


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def _sub(m: re.Match) -> str:
            env_val = os.getenv(m.group(1), "")
            if env_val:
                return env_val
            return m.group(2) if m.group(2) is not None else ""

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _read_yaml(name: str) -> dict:
    path = ROOT / "configs" / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return _expand(yaml.safe_load(f) or {})


# ---------------- Pydantic 模型 ----------------
class AppConfig(BaseModel):
    name: str = "MathAnalysis Agent"
    version: str = "2.2"
    # 审核发现问题后回解题节点重试的次数。默认 0：一次重试等于 plan+solve 再跑一遍，
    # 端到端耗时直接翻倍，而实测绝大多数 issue 属于「表述不够严谨」而非「结论错误」，
    # 交给用户在报告里看 issue 列表比自动重算更划算。需要时在界面上开。
    max_retries: int = 0
    # 快速模式：不单独调用模型做「解题规划」，把方法与步骤合并到解题节点一次产出。
    # 规划本身是一次完整的 solver 调用（带思维链，20s 上下），而它对最终答案的边际
    # 贡献很小 —— 合并后 solver 只需调 1 次，端到端省掉三分之一左右。
    fast_mode: bool = True
    # 题目结构化是否走模型。默认 False：本地规则足够（只影响报告顶部标签与解题提示词），
    # 却能在关键路径上省掉一次 8~11s 的 LLM 调用。需要更细的语义切分时置 True。
    analyze_llm: bool = False
    request_timeout: int = 60
    pipeline_timeout: int = 240
    log_dir: str = "outputs/logs"
    history_file: str = "outputs/history.jsonl"
    default_top_k: int = 5


class ModelSpec(BaseModel):
    provider: str = "deepseek"
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048
    # 被截断（finish_reason=length）时允许自动翻倍到的上限。
    # 推理模型的思维链和正文共用这份预算，固定值必然有题目会被砍断。
    # 0 表示不允许升级。
    max_tokens_ceiling: int = 0
    price_in_per_1k: float = 0.001
    price_out_per_1k: float = 0.002
    fallback_to: str | None = None
    # 专用 key 没配时退回这个环境变量。vision 等附属角色通常和 solver 同平台，
    # 单独再申请一个 key 没有必要，但也要允许用 VISION_API_KEY 覆盖。
    fallback_key_env: str | None = None

    @property
    def api_key(self) -> str:
        if self.api_key_env:
            key = os.getenv(self.api_key_env, "")
            if key:
                return key
        if self.fallback_key_env:
            return os.getenv(self.fallback_key_env, "")
        return ""


class ModelsConfig(BaseModel):
    solver: ModelSpec = Field(default_factory=ModelSpec)
    reviewer: ModelSpec = Field(default_factory=ModelSpec)
    light: ModelSpec = Field(default_factory=ModelSpec)
    # 视觉角色：把题目照片转成 LaTeX 文本，只在「上传图片」这条入口用到
    vision: ModelSpec = Field(default_factory=ModelSpec)


class FusionConfig(BaseModel):
    method: str = "rrf"
    rrf_k: int = 60
    vector_weight: float = 0.5
    bm25_weight: float = 0.5


class VectorConfig(BaseModel):
    enabled: bool = True
    # local  = fastembed 本地 ONNX（BGE-small-zh，512 维，离线、免费）—— 默认
    # api    = 云端 Embedding 接口 + 本地 npz（需要可用额度）
    # chroma = 本地 sentence-transformers + Chroma（需 torch 与 GB 级模型）
    # auto   = local → api → chroma 依次尝试，谁先就绪用谁
    backend: str = "local"
    persist_dir: str = "knowledge_base/chroma_db"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    collection: str = "math_analysis"
    # --- local 通路 ---
    local_model: str = "BAAI/bge-small-zh-v1.5"
    local_embed_file: str = "knowledge_base/embeddings/chunks_local.npz"
    # --- api 通路 ---
    embed_file: str = "knowledge_base/embeddings/chunks.npz"
    api_model: str = "qwen3.7-text-embedding-flash"
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "EMBED_API_KEY"


class BM25Config(BaseModel):
    enabled: bool = True
    chunk_file: str = "knowledge_base/chunks/chunks.jsonl"


class RerankConfig(BaseModel):
    """交叉编码器精排层：粗召回后对候选做逐条重打分（见 retrieval/rerank.py）。"""
    enabled: bool = True
    model: str = "BAAI/bge-reranker-base"
    # 精排前保留的候选数（从粗召回池取前 N 送精排）
    candidate_top_n: int = 8
    # 精排分与粗召回分的融合权重：rerank 0.7 / 原分 0.3
    # 精排分更可靠，但完全丢弃粗召回会丢掉「强关键词精确匹配」的信号，
    # 故折中。0.7/0.3 与纯精排（1.0/0.0）在评估集上指标接近，取更稳健的一侧。
    rerank_weight: float = 0.7
    # 送进交叉编码器的正文长度上限。bge-reranker 的耗时几乎与文本长度成正比：
    # 本机 10 条 × 1000 字要 13.4s，截到 320 字只要 5s 左右，而块级相关度判断
    # 只看开头（定理陈述、定义）就够，几乎不影响排序。
    max_doc_chars: int = 320


class FilterConfig(BaseModel):
    strict_chapter: bool = False
    metadata_fields: list[str] = Field(default_factory=lambda: ["chapter", "section", "content_type", "topic"])


class RetrievalConfig(BaseModel):
    vector: VectorConfig = Field(default_factory=VectorConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    top_k: int = 5
    candidate_chunks: int = 20
    candidate_theorems: bool = True
    recall_target: float = 0.80


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    mock_llm: bool = False

    @property
    def root(self) -> Path:
        return ROOT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_env()
    app_raw = _read_yaml("app.yaml")
    model_raw = _read_yaml("model.yaml")
    retrieval_raw = _read_yaml("retrieval.yaml")
    return Settings(
        app=AppConfig(**{k: v for k, v in app_raw.get("app", {}).items() if v not in ("", None)}),
        models=ModelsConfig(
            solver=ModelSpec(**model_raw.get("solver", {})),
            reviewer=ModelSpec(**model_raw.get("reviewer", {})),
            light=ModelSpec(**model_raw.get("light", {})),
            vision=ModelSpec(**model_raw.get("vision", {})),
        ),
        retrieval=RetrievalConfig(**retrieval_raw.get("retrieval", {})),
        mock_llm=os.getenv("MOCK_LLM", "0").strip() in ("1", "true", "True", "yes"),
    )
