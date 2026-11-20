# MathAnalysis Agent

面向大学数学分析课程的智能解题系统（对应技术方案 **V2.2**）。
纯 Agent 架构：不训练、不微调模型，通过 LLM API + 混合检索 + SymPy 验证 + 异构模型审核完成可解释的解题。

## 特性

- **多节点工作流（LangGraph）**：解析 → 结构化 → **规划** → **解题** → **按题解与题目点名的定理/公式检索教材** → SymPy 验证 → 异构审核 → 报告
- **三阶段解题策略**：先规划解题路径，再逐步求解，最后才查依据——检索 query 由「题解各步声明的定理/公式 + 题目点名的定理 + 规划所需定理」共同构成，而不是题面原文
- **题目照片输入**：手机拍照/截图 → 视觉模型直接输出 LaTeX 题目 → 自动填入输入框，公式无需手打
- **混合检索**（方案 7.4）：向量（本地 BGE-ONNX，512 维） + BM25 关键词 + 元数据加权 + 交叉编码器精排，weighted 融合
- **异构审核**（方案 4.2）：审核节点强制使用与解题链路不同厂商的模型，且只输出**逐条可核验断言**，不输出"通过/不通过"
- **来源可回溯（平行布局）**：报告左栏是题目与推导步骤，右栏依据与对应步骤**逐条平行**——题面点名的定理/公式对应题面引用行，每一步的定理与教材原文（含公式原式与页码）对应该步卡片；
  点某一步的「查看依据」会高亮右侧对应原文，也能一键跳到知识库里那一段（章节/页码/全文，公式照常渲染）
- **速度**：单题约 27s（embedding + reranker 在页面打开时后台预热，不占用解题时间）
- **验证分级**（方案 8.2）：已验证 / 部分验证 / 未验证 / 发现不一致，工具不支持不当成数学错误
- **Token 与成本控制**（方案 15）：节点模型分级、单题预算熔断、调用日志、降级透明
- **可降级**：未安装向量依赖时自动降级为 BM25；无 API Key 时进入 mock 模式，流程仍可跑通

## 目录结构

```
math_analysis_agent/
├── configs/            # app.yaml / model.yaml / retrieval.yaml
├── data/               # raw / processed / questions / eval
├── knowledge_base/     # documents / chunks / chroma_db
├── src/
│   ├── ingestion/      # 输入解析（文本/图片/PDF 预留位）
│   ├── retrieval/      # 混合检索
│   ├── agents/         # 状态、提示词、节点、LangGraph 编排、Pipeline
│   ├── tools/          # SymPy 工具与验证分级
│   ├── llm/            # 模型客户端与预算
│   ├── storage/        # 历史记录
│   └── app/            # Streamlit 页面
├── scripts/            # 解题 CLI、知识库构建
├── tests/              # 单元与端到端测试
└── outputs/            # 报告与日志
```

## 快速开始

```bash
cd math_analysis_agent
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # 填入 SOLVER / REVIEWER 的 API Key

python scripts/solve.py "计算 lim(x→0) (e^x - 1 - x)/x^2"
streamlit run src/app/streamlit_app.py
```

不配置 Key 也能启动：此时为 **mock 模式**，用于验证流程连通性，解答为占位内容。

## 获取 API Key（两个，建议来自不同厂商）

| 角色 | 推荐平台 | 注册地址 | 取 Key | 默认模型 | 费用 |
|---|---|---|---|---|---|
| solver 解题 | DeepSeek | https://platform.deepseek.com | `/api_keys` | `deepseek-flash` | 新号送额度，之后约 ¥0.001/千输入 |
| solver 备选 | 阿里云百炼 | https://bailian.console.aliyun.com | `?apiKey=1` | `qwen-max` | 新号 90 天免费额度，之后约 ¥0.0024/千输入 |
| reviewer 审核 | 智谱 GLM | https://open.bigmodel.cn | `/usercenter/apikeys` | `glm-4-flash` | 免费（限 1 并发） |

**关键约束：模型名必须和 base_url 属于同一平台。** 把 `SOLVER_BASE_URL` 换成百炼却留着 `SOLVER_MODEL=deepseek-flash`，会直接 404 —— 百炼上只有 `qwen-max / qwen-plus / qwen-flash` 可用。

填完 `.env` 后先自检，脚本会逐个角色发一条最小请求并给出中文错误提示：

```bash
python scripts/check_env.py
```

> 注意：`deepseek-chat` / `deepseek-reasoner` 已于 2026-07 退役，请使用 `deepseek-flash` 或 `deepseek-v4-pro`。
>
> `.env` 用记事本保存时可能变成 `.env.txt`。配置层已兼容：`.env` / `.env.txt` / `.env.local` 都会读取，**以修改时间最新的一份为准**，自检输出里会打印"实际生效"的文件名。

## 知识库：从扫描版教材构建

> **仓库不包含教材原文与切块结果。** 陈纪修《数学分析》为受版权保护的出版物，
> `knowledge_base/chunks/*.jsonl`（1693 块全文 OCR）与 `embeddings/*.npz` 已在 `.gitignore` 中排除。
> 克隆后需自备 PDF，按下面三步在本地重建；没有知识库时系统仍能启动，只是检索结果为空、
> 报告标注「无引用来源」（此时更适合用 `MOCK_LLM=1` 看界面）。

陈纪修《数学分析》上下册是**纯扫描件**（每页一张 300dpi 图，无文字层），必须走视觉 OCR。流程分三步：

```bash
# 1) 整页图 → 视觉模型 → 带 LaTeX 的 Markdown（断点续跑，中断后重跑即可继续）
python scripts/ocr_book.py --pdf "D:\...\上册.pdf" --book shangce --workers 5

# 某些页失败时（额度不足、单页超时）只重试失败页
python scripts/ocr_book.py --pdf "D:\...\上册.pdf" --book shangce --retry-failed

# 2) 按「定理/定义/推论/例」语义边界切分，补 chapter/section/page 元数据
python scripts/build_from_ocr.py --book shangce --title "数学分析（陈纪修）上册"

# 上下册合并成一个库（--merge 让多本写入同一个 chunks.jsonl）
python scripts/build_from_ocr.py \
  --book shangce --book xiace \
  --title "数学分析（陈纪修）上册" --title "数学分析（陈纪修）下册" \
  --merge

# 3) 量化召回（方案 13.1 的 Recall@5，同时输出 Top-1 与 MRR）
python scripts/eval_retrieval.py --k 5
```

OCR 的视觉模型支持**降级链**：`--model "qwen3-vl-plus,qwen-vl-ocr"`，前者额度耗尽（403/quota）或下线（404）时自动切下一个，不会整批中断。

关键设计：

- **按语义切，不按字数切**：固定窗口会把定理从中间截断，检索出来的内容不可用。锚点见 `build_from_ocr.py` 的 `ANCHOR_THEO` / `ANCHOR_EX`。
- **「例」必须带编号**才切（`例 1.2.3`），否则正文里的「例如…」会把整段论述切碎。
- **页码取书页脚印刷页码**，报告里能写「依据 定理 5.2（第 80 页）」。
- 视频页/目录页/前言自动过滤（`is_noise_page` + 正文起点判定）。
- **章名/题目要清洗 OCR 残留**：页眉常夹带 LaTeX 花括号（`第八章 反常积分}`）或 HTML（`<div style=...>`），不清洗会让同章出现多种写法、元数据软加权失效。
- **字段加权**：`topic`（标题）在 BM25 索引里重复计入，否则会被几百字的正文淹没。

### 当前检索实测（陈纪修上下册，1693 块，30 题评估集）

| 指标 | 无精排（baseline） | 精排（rerank） | 说明 |
|---|---|---|---|
| Recall@5 | 93.3% (28/30) | **100%** (30/30) | 方案 13.1 门槛 ≥80% |
| Top-1 命中 | 73.3% (22/30) | 73.3% (22/30) | 报告主引用是否指向正确定理 |
| MRR | 0.805 | — | 首个正确结果的排名倒数均值 |

检索分两段：**粗召回**（BM25 + 向量，weighted 0.4/0.6 融合）先把候选捞进池子，
**精排**（`BAAI/bge-reranker-base` 交叉编码器，ONNX 无 torch）再对候选逐条重打分、
取前 5。曾用 RRF，但它只看排名不看分数，把 BM25 的强匹配压平成常数分，Recall@5
掉到与单路持平。不同策略的对比可以直接跑 `python scripts/tune_fusion.py`。

精排的价值：粗召回（每路 20 条）已把正确答案 100% 捞进池子，瓶颈在「排序」而非
「召回」——`t14`「复合函数的极限」、`t28`「Darboux 定理」这类术语变体题，正确块
排在池子靠后位置，交叉编码器能真正判断「这段是不是在回答这个问题」把它提到前 5。
精排跑两轮对比：`python scripts/eval_retrieval.py --rerank both`（模型缺失自动跳过精排）。

评估集 `data/eval/eval_set.jsonl` 按 8 章知识点构造，每题都验证过「库中存在可答块」，避免用无解题误判。

**必须知道的短板**：
1. 少量块的**章节归属有偏移**（如第四章内容被标成第三章），因为 OCR 页眉与正文边界不完全对齐。评估集的 `hit()` 按「章节 + 关键词」判定，个别题的误判其实是章节字段错位导致，非检索本身的问题。

已知限制：教材中大量「无编号的性质罗列」（如集合运算性质 1.~4.）不会被单独切块，会并入前一个块的后续分段。

## 配置说明

- `configs/model.yaml`：`solver`（规划/解题，强模型）、`reviewer`（审核，**必须与 solver 不同厂商**）、`light`（解析/整理，可回退 solver）、`vision`（题目照片转 LaTeX，缺 `VISION_API_KEY` 时复用 `SOLVER_API_KEY`）
- `configs/retrieval.yaml`：向量后端、BM25、融合方式与权重、元数据过滤、精排（rerank）、Recall@5 目标

### 题目照片输入

工作台 →「用题目照片输入」→「用题目照片输入」展开上传，识别后自动填入题目框（框里已有内容时不覆盖，改为提供「用作题目」按钮）。

- 预处理：`ImageOps.exif_transpose` 转正（手机竖拍的图不做这步到模型眼里是躺倒的）、长边缩到 1568、透明底转白底、存 JPEG q88
- 识别：`src/ingestion/ocr.py`，prompt 要求只输出题目、公式一律 `\sum \int \lim \frac`、看不清写 `[?]` 不许猜
- 实测：正视图 1.6s；倾斜 4.2° + 阴影 + 模糊 + 噪点 的劣化图仍能正确还原（`scripts/probe_ocr_robust.py`）
- **图片中转兜底**：2026-11 起 ModelScope 推理网关收到 base64 图片时会先写自家 OSS，那一步稳定返回
  `500 / oss 403 bucket acl`（Intern、ERNIE、Qwen3.5、GLM 逐个体测，报错完全一致，属平台故障）。
  同一张图改用公网 URL 送进去就正常，所以 `ocr.py` 检测到这类报错后会自动把图传一次临时图床
  （1 小时失效）再把 URL 交给模型，并在界面上写明「图片经临时图床中转」。设 `VISION_RELAY=0` 可关闭此兜底。

模型可用性会变，换模型前先实测，别照抄文档：

```bash
python scripts/probe_vl2.py          # 逐个试 ModelScope 上的多模态模型
python scripts/probe_ocr_robust.py   # 用劣化图测识别精度
```

### 公式渲染的几条硬规则（改前端前先读）

1. **任何正文都不能用 `unsafe_allow_html` 拼 HTML 再塞进去**。一旦走这条路，KaTeX 就不再处理这段内容，
   `$u \in C^2(B)$` 会原样露出反斜杠（审查区、局限区都这么翻过车）。要加圆点/分隔线，用
   `st.container(key="vrow-…")` + CSS 挂在 key 类上，正文交给 `st.markdown`。
2. 模型爱写 `\[ … \]` 且**经常跨行**；跨行的 `$$` 会被 markdown 劈成两半，KaTeX 解析失败后把源码当红字吐出来。
   `md()` 先把跨行公式压成单行独占一段，再加 `\displaystyle` —— Streamlit 把 `$$` 当行内渲染，
   不加这句积分上下限会缩到右边，跟教材排版对不上。
3. 步骤标题同样要过 `md()`（`第 4 步 证明 \(\int_B u^2\,dx>0\)` 不过就是一条转义后的乱码）。
4. 模型的 `step.content` 里常已写了一遍公式，`step.latex` 再画一遍就重复；用 `_norm_tex` 判重。


> 2026-11 实测：dashscope 的 `qwen-vl-ocr`/`qwen3-vl-plus` 已 401，ModelScope 上
> `Qwen2.5-VL`/`Qwen3-VL` 全部 no provider，只有 `Shanghai_AI_Laboratory/Intern-S2-Preview` 可用。

### 向量通路

默认走**本地** `fastembed` + `BAAI/bge-small-zh-v1.5`（ONNX，512 维）：离线、免费、
单条查询 10~30ms，不受云端额度影响。

```bash
pip install fastembed
python scripts/build_embeddings.py --backend local     # 约 6 分钟，1693 块
```

云端接口（`--backend api`）保留为备选，需要可用的 Embedding 额度；注意 ModelScope
的推理服务里没有任何 embedding 模型，百炼的 `qwen3.7-text-embedding-flash` 免费额度
用完后会直接 401，这时设置页会明确提示原因，而不是只显示「未启用」。

## 界面与速度（2026-11 改版）

### 视觉规范（深梅紫 · Plum & Rose）
- 配色取自参考站点 `exploremehere.com/s/sample-vertex`：
  深梅紫底 `#2C1333`、面板 `#3A1B42`、玫瑰粉 `#C47A92`（主按钮 / 链接 / 强调）、丁香紫 `#B69AC6`（次强调）。
  `.streamlit/config.toml` 里 `base=dark` + `primaryColor=#C47A92`，否则开关打开时是默认红色，会跟整套配色打架。
- **深色底不靠阴影分层**，靠「页面底最暗 → 卡片亮一档 → 卡内嵌块再压暗」三档明度 + 一道发丝线；
  阴影只留极淡一层。主按钮是玫瑰粉药丸（实底 + 深紫字、不描边、不加渐变）。
- 字体：一套现代无衬线（中文优先 HarmonyOS Sans / MiSans / PingFang），字号 5 档、字重 3 档；
  数字与页码用等宽，只作辅助信息。小节标签用「大写小字 + 0.18em 字距 + 发丝线」。
- **页面上不写标语、不写操作提示、不放次要标签**：首屏只有一行「解题工作台」，
  下面直接是输入卡（输入框 → 主按钮 + 两个开关 → 示例胶囊 → 拍照入口）。耗时挪进开关 tooltip。
- **按钮是悬浮，不是有厚度**：极细边框 + 一层朝下的软投影，悬停上浮 2px、投影散开，按下回位、投影收紧。
  没有下沿厚边、没有渐变、没有玻璃拟态、没有入场动画。
- 版式：内容宽 1160px 封顶；空态首屏把面板放在视口上三分之一，下面留白就是「等解答的位置」。
- 响应式：≤960px 所有列纵向堆叠、控件行重排成「按钮整行 + 两个开关并排」；≤620px 公式降一号，
  页面无横向溢出（390px 实测 `scrollWidth == clientWidth`）。
- **没有「上次结果」区块**：结果只跟着输入框里的题目走。题目一字不差且磁盘上有同一份结果时
  直接渲染（不重复计费），否则该区域就是空的。

### 流程：先规划 → 解题 → 最后按题解与题目点名的定理/公式查依据
工作流 `parse → analyze → plan → solve → retrieve → verify → review → format`，检索刻意放在解题之后：
- **prompt 明确三段职责**：`PLAN` 只规划、`SOLVE` 只解题且逐步标注 `theorem`（本阶段不给教材片段）、
  检索在最后统一回查。解题时不被可能跑偏的召回带偏，检索目标也是确定的（已经用上的定理/公式）。
- **依据来源扩为两路**：① 题解每一步的 `theorem`（挂到具体步骤）；② 题目字面点名的定理/公式
  （`analyze` 抽 `mentioned`，本地规则也有别名表兜底）+ 规划 `required_theorems`（题目级依据，与题目平行）。
  每路强关键词 + BM25 词法通路、跳过精排，各留 1 条，避免把无关章节塞进来。

### 报告版式：依据与题解逐步平行
- 顶部：题目 / 结论卡；题目点名的依据以紧凑引用行与题目平行（只留定理名 + 页码，可点开原文）。
- **每一步一行两栏**：左 = 这一步题解（序号、定理签、公式出处签、正文、核心公式），
  右 = 这一步的教材依据（定理原文 + 复写公式的出处卡）。左右顶部天然对齐，不用再靠点击高亮
  去找对应关系；窄屏两列自动上下紧邻。这是「每个依据和相应题解平行」的落地方式。
- 底部：符号验证 / 独立模型审核 / 知识来源三块并排 + 思考过程。字号两档：推导正文 13.5px、结论 19px。
- `theorem` 映射不到时先剥「适用条件」尾巴、展开同义词（`洛必达 → L'Hospital`）再查，仍不行
  BM25 单路全库补查（毫秒级）。宁缺勿错：匹配不上就留空，兜底候选过最低相关度。

### 公式级引用：题解里复写的式子也指回书页
- 定理级映射只回答「这一步用哪条定理」，答不了「这个式子书上是怎么写、在哪一页」。
  题解里大量复写教材公式（洛必达的结论式、Taylor 展开、Green 公式），这些式子也做了出处映射。
- 做法：把知识库里近三万个公式（`$...$` / `$$...$$`）按 `chunk` 建一份归一化索引——
  `\dfrac→\frac`、去掉 `\limits/\displaystyle/\left/\right`、`\leqslant→\le`、抹掉空格与句末标点；
  再存一份去花括号键（`\frac{1}{2}` 与 `\frac12`）和前 6 字符分桶（供子式匹配）。
  解题时按归一化字面命中；对不上再在同一桶里找唯一的包含关系，候选不唯一就放弃。
- 精度优先：长度 < 12 的片段、纯数字比值（`\frac{0}{0}`、`\frac{1}{2}`）、无变量的式子一律不参与匹配——
  指错页码比不指更糟。实测「拿教材公式截断 70% 当查询」的 40 个用例中命中 19 个，全部落在正确 chunk。
- 落位：右栏「教材依据」里多一张奶油黄书脊的公式卡（`式 · 定理 5.3.1` + 教材原式 + 章节页码），
  左栏对应步骤头多一枚鼠尾草绿的「式 p.163」小签；两边同号联动，切步骤时一起高亮。
  同一步的定理出处和公式出处若落在同一个教材块上，只留定理卡，不摆两张内容重叠的卡片。

### 思考过程：边跑边看
- `LLMClient.chat(on_delta=...)` 走 `stream=True`，把 `reasoning_content`（思维链）与
  `content`（正文）增量推给前端；首字节通常 1~3s 就回来，不必等整段 40s。
- 前端每秒刷新一块 `<div class="think-card">`：阶段时间线（解析 → 结构化 → 规划 → 推导 → 查教材 →
  复算 → 审核 → 报告，带各段耗时）+ 流式文本尾巴（思维链，或把半截 JSON 折成「第 N 步 · 标题」）。
- 这些增量只进内存环形缓冲（各留 6000 字），不写时间线、不落盘全量，避免每秒钟几十行日志。
- 跑完后同一个渲染器放进「思考过程」折叠区，过程与结果是同一套叙事。

### 耗时账（单题，4 核 CPU，DeepSeek-V4.1-Flash + Qwen3.8-Flash-Next）

| 环节 | 改版前 | 现在 | 做法 |
|---|---|---|---|
| 题目结构化 | 8~11s（LLM） | **0s** | 改本地规则 `_local_struct`，`app.analyze_llm` 可切回模型 |
| 教材检索 | 13.2s | **1.0s** | 改到解题之后，query 换成题解里的定理名（强关键词，BM25 直接命中）；跳过精排，每条定理独立检索 3 条候选 |
| 解题 | 30~60s | 10~30s | 流式输出，首字 1~3s；提示词限制 steps ≤6、每步 ≤150 字；且不再背 5 条检索片段，输入更短 |
| 交叉审核 | 8~25s | 可跳过 | 「极速模式」直接 `verify → format`，省掉这一段 |
| 端到端 | ~52s | **~27s** | 实测 4 步题 27.3s（解题 10.4s + 检索 1.0s + 审核 10.8s） |

> 端到端波动主要来自模型侧排队。检索与结构化这两段原来的固定开销（约 20s）已经压掉。

### 相关开关
```yaml
app:
  fast_mode: true        # 规划合并进解题（solver 只调 1 次）
  analyze_llm: false     # 结构化走本地规则
retrieval:
  rerank:
    candidate_top_n: 8
    max_doc_chars: 320
```
界面上还有两个勾选：**极速模式**（跳过交叉审核）、**质疑后重做**（审核有质疑时回炉，耗时翻倍）。

## 测试

```bash
pytest -q
```

覆盖：SymPy 工具（含危险表达式拦截）、混合检索与 Recall@5、验证分级、mock 端到端全链路。

## 当前进度

- [x] P3 文本解题闭环（主流程可运行）
- [x] 混合检索（BM25 + 本地向量，weighted 融合）
- [x] 异构审核 + 断言清单
- [x] SymPy 验证分级
- [x] Token 预算熔断
- [x] Streamlit 工作台
- [x] 题目照片输入（图片 → LaTeX 题目，含 EXIF 转正与降采样）
- [ ] P1 PDF 解析（MinerU 版面分析）
- [ ] P8 100 题验收与 Recall@5 实测报告
