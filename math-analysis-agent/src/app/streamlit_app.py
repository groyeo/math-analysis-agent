"""Streamlit 前端：解题工作台 · 实时思考过程 · 双栏报告 · 知识库 · 设置。

视觉规范（改版时请保持）：
- 页面里不写标语、不写操作提示、不放次要标签。看不懂就改布局，不要加说明文字；
  例外只有数据本身（条数、耗时、页码）和必要的分节标题；
- 配色取自参考站点 exploremehere.com/s/sample-vertex（Plum & Rose）：
  深梅紫底 #2C1333 + 面板 #3A1B42 + 玫瑰粉 #C47A92（主/按钮/强调）+ 丁香紫 #B69AC6（次强调）；
  深色底不靠阴影分层，靠「页面底最暗 → 卡片亮一档 → 卡内嵌块再压暗」三档明度 + 一道发丝线；
- 现代无衬线一套字体（中文优先 HarmonyOS/PingFang），字号 5 档、字重 3 档；
  数字与页码用等宽，只作辅助信息；标签用大写小字 + 字距，不用彩色底；
- 按钮是「悬浮」而不是「有厚度」：一层朝下的软投影 + 极细边框，悬停上浮 2px、投影散开，
  按下回到原位、投影收紧；不用下沿厚边、不用渐变、不用玻璃拟态；
- 动效只保留按钮的 hover/active，不做入场动画、脉冲、旋转、闪烁；
- 留白优先：内容宽 1160px 封顶，模块间距 24~40px，卡片内 padding 16~18px；
- 结构化组件一律用 st.container(key=...) + CSS key 类装饰，绝不用 HTML 包住正文
  —— 一开 unsafe_allow_html，KaTeX 就跳过那段，公式会原样露出反斜杠。
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import html
import json
import re
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.agents.pipeline import MathAgentPipeline
from src.config import get_settings
from src.ingestion.ocr import OcrError, recognize_question
from src.retrieval.hybrid import HybridRetriever
from src.storage.history import save_record

st.set_page_config(page_title="数学分析 Agent", page_icon="∫", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------- 设计系统
CSS = """
<style>
/* ============================================================
   设计系统 · Plum & Rose
   参考站点（exploremehere.com/s/sample-vertex）的色板：
   深梅底 #2C1333 / 面板 #3A1B42 / 玫瑰粉 #C47A92 / 丁香紫 #B69AC6。
   深色底上不靠阴影分层，靠「三档明度 + 一道发丝线」拉开距离：
   页面底最暗 → 卡片亮一档 → 卡内嵌块再压暗。
   动效克制到只剩「按钮悬停一次轻浮」，整页不做入场动画。
   ============================================================ */
:root{
  --bg:#2C1333; --bg-deep:#231028;
  --surface:#3A1B42; --soft:#331739; --soft-2:#2A1230;
  --line:rgba(255,255,255,.16); --line-soft:rgba(255,255,255,.09);
  --ink:#FBF5FA; --ink-2:rgba(255,255,255,.76); --muted:rgba(255,255,255,.50);
  /* 主色：参考站点按钮与标题强调用的那支玫瑰粉 */
  --rose:#C47A92; --rose-deep:#D48FA7;
  --orchid:#B69AC6; --orchid-soft:rgba(182,154,198,.15); --orchid-line:rgba(182,154,198,.42);
  --primary:#C47A92; --primary-hover:#D48FA7; --primary-ink:#2A1230;
  --primary-soft:rgba(196,122,146,.17); --primary-line:rgba(196,122,146,.46);
  --mauve:#8A6E96; --mauve-soft:rgba(138,110,150,.16); --mauve-line:rgba(138,110,150,.50);
  --mauve-text:#C9B5D4;
  --ok:#8FC7A8; --ok-soft:rgba(143,199,168,.14); --ok-line:rgba(143,199,168,.34);
  --warn:#E3BB77; --warn-soft:rgba(227,187,119,.14); --warn-line:rgba(227,187,119,.34);
  --bad:#E58C81; --bad-soft:rgba(229,140,129,.14); --bad-line:rgba(229,140,129,.36);
  --r-xs:6px; --r-sm:9px; --r-md:12px; --r-lg:14px; --r-xl:20px;
  /* 深色底没有真实阴影可看，投影只做极淡的一层，靠边框承担分层 */
  --sh-1:0 1px 2px rgba(0,0,0,.28);
  --sh-2:0 24px 48px -28px rgba(0,0,0,.75);
  --lift:0 1px 2px rgba(0,0,0,.30), 0 10px 22px -14px rgba(0,0,0,.70);
  --lift-hi:0 2px 6px rgba(0,0,0,.34), 0 20px 30px -18px rgba(0,0,0,.80);
  --ease:cubic-bezier(.22,.61,.36,1);
  --sans:"Inter","HarmonyOS Sans SC","MiSans","PingFang SC","Source Han Sans SC",
         "Noto Sans CJK SC","Microsoft YaHei",-apple-system,"Segoe UI",sans-serif;
  --num:"SFMono-Regular",ui-monospace,Menlo,Consolas,monospace;
}
html, body, [class*="st-"]{
  font-family:var(--sans);
  -webkit-font-smoothing:antialiased;
  text-rendering:optimizeLegibility;
  letter-spacing:0;
}
/* 绝不要加 font-feature-settings:"palt"：数学字体的 palt 表会把斜体变量的
   advance width 归零，公式里的 x、e 会整体消失（踩过一次）。 */
.katex, .katex *{font-feature-settings:normal !important;}
.katex{color:var(--ink);}
/* 块级公式统一口径：无论来自 markdown 的 $$...$$ 还是 st.latex()，
   都给同一种「浅底 + 左细线」的待遇，避免同一列里公式深浅不一。 */
.katex-display{max-width:100%; overflow-x:auto; overflow-y:hidden; padding-bottom:.08rem;}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"]{background:var(--bg); color:var(--ink);}
/* 长页面滚动到视口之外时，body 也要是纸色，否则底部会露出一条白 */
html, body{background:var(--bg);}
[data-testid="stHeader"]{background:transparent; height:0;}
/* 顶部工具条只留「展开侧栏」这一个按钮（它挂在 toolbar 里，整条藏掉会连按钮一起消失）。
   Deploy / 主菜单这些一律不显示。 */
[data-testid="stToolbar"]{background:transparent; padding:0;}
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"],
[data-testid="stMainMenu"], [data-testid="stHeaderActionElements"],
[data-testid="stDecoration"], #MainMenu, footer{display:none !important;}
/* 侧栏被收起时（窄屏会自动收起）左上角会出现一个展开按钮。header 被压成 0 高，
   它会跟着一起看不见 —— 所以这边把它钉在视口左上角，别让人找不到入口。 */
[data-testid="stExpandSidebarButton"]{
  position:fixed !important; left:.55rem; top:.55rem; z-index:1000;
  width:34px !important; height:34px !important; padding:0 !important;
  border-radius:10px !important; background:var(--surface) !important;
  border:1px solid var(--line) !important; color:var(--ink) !important;
  box-shadow:0 1px 2px rgba(0,0,0,.06), 0 6px 14px -10px rgba(0,0,0,.5) !important;
}
[data-testid="stExpandSidebarButton"]:hover{border-color:var(--rose-deep) !important;}
[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] *{color:var(--ink) !important; fill:var(--ink) !important;}
/* Streamlit 的图标是 Material Symbols 字体，本机不一定加载得到（加载失败就是一个白框）。
   这两个开关太关键，改用 CSS 画箭头，不依赖字体。 */
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] > span,
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]{display:none !important;}
/* 外层 span 即使内容被藏了也还占位，会把箭头挤到边缘，一起清掉 */
[data-testid="stSidebarCollapseButton"] button > span{display:none !important;}
[data-testid="stExpandSidebarButton"]{display:flex !important; align-items:center; justify-content:center;}
[data-testid="stExpandSidebarButton"]::after{
  content:""; display:block; width:14px; height:14px;
  background:center/14px 14px no-repeat
    url("data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FBF5FA' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9 5l7 7-7 7'/%3E%3C/svg%3E");
}
[data-testid="stAppViewContainer"]{overflow-x:hidden;}
.block-container{max-width:1160px; padding-top:3.1rem; padding-bottom:7rem;}
p, li, .stMarkdown{color:var(--ink-2);}
p{text-wrap:pretty; line-break:strict;}
a{color:var(--primary); text-underline-offset:3px;}
::selection{background:var(--primary-soft); color:var(--ink);}

/* ---------- 侧栏 ---------- */
section[data-testid="stSidebar"]{background:var(--bg-deep); border-right:1px solid var(--line);}
section[data-testid="stSidebar"] .block-container{padding:1.6rem 1rem 2rem; max-width:100%;}
.brand{display:flex; align-items:center; gap:.6rem; margin:.15rem 0 1.5rem;}
.brand-mark{
  width:30px; height:30px; border-radius:9px; flex:0 0 30px;
  background:var(--rose); border:1px solid var(--rose); color:var(--primary-ink);
  display:flex; align-items:center; justify-content:center;
  font-size:15px; line-height:1;
}
.brand-name{font-size:13.5px; font-weight:600; letter-spacing:-.1px; color:var(--ink); line-height:1.2;}
.brand-sub{font-size:11px; color:var(--muted); margin-top:2px;}
section[data-testid="stSidebar"] .stRadio > div{gap:.15rem;}
section[data-testid="stSidebar"] .stRadio label{
  padding:.5rem .62rem; border-radius:var(--r-sm); font-size:13px; color:var(--ink-2);
  transition:background .18s var(--ease), color .18s var(--ease);
}
section[data-testid="stSidebar"] .stRadio label p{
  font-size:13px !important; font-weight:500 !important; color:inherit !important;
}
section[data-testid="stSidebar"] .stRadio label:hover{background:rgba(255,255,255,.06);}
section[data-testid="stSidebar"] .stRadio label:has(input:checked){
  background:var(--surface); color:var(--primary);
  /* 左侧一道 2px 鼠尾草短线：比整块填色克制，像目录里的当前页 */
  box-shadow:inset 2px 0 0 var(--rose-deep), var(--sh-1);
}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) p{
  color:var(--primary) !important; font-weight:600 !important;
}
[data-testid="stRadioOption"] > div > div:first-child{display:none !important;}
.srow{
  display:flex; justify-content:space-between; align-items:baseline; gap:.75rem;
  font-size:12px; padding:.42rem 0; color:var(--muted);
  border-bottom:1px solid var(--line-soft);
}
.srow:last-child{border-bottom:none;}
.srow b{
  color:var(--ink-2); font-weight:600; font-variant-numeric:tabular-nums;
  text-align:right; overflow-wrap:anywhere; min-width:0;
}
.srow span{flex:0 1 auto;}

/* ---------- 首屏 ---------- */
/* 只有一行功能名。没有眉标、没有副标题、没有数据条——
   识别「这是什么、能做什么」交给输入框本身。 */
.hero-wrap{padding:.1rem 0 1.5rem;}
/* 空态把面板放到视口上三分之一处，下面留白就是「等解答的位置」 */
.hero-wrap.solo{padding-top:6vh;}
.hero-wrap.compact{padding:.1rem 0 1.1rem;}
.hero-wrap.compact .hero{font-size:22px; letter-spacing:-.4px; margin:0;}
.hero{
  font-size:34px; font-weight:600; letter-spacing:-1.4px; line-height:1.3;
  color:var(--ink); margin:0;
}

/* ---------- 卡片 ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--surface); border:1px solid var(--line) !important;
  border-radius:var(--r-lg); box-shadow:var(--sh-1); padding:1.05rem 1.15rem;
}
.sec{
  display:flex; align-items:center; gap:.55rem;
  font-size:10px; font-weight:600; letter-spacing:.18em; color:var(--ink-2);
  text-transform:uppercase; margin:.2rem 0 .95rem;
}
.sec::after{content:""; flex:1; height:1px; background:var(--line-soft);}

/* ---------- 输入 ---------- */
.stTextArea textarea{
  background:var(--soft-2); border:1px solid var(--line-soft); border-radius:var(--r-md);
  font-size:15px; line-height:1.78; color:var(--ink); padding:.85rem 1rem;
}
.stTextArea textarea::placeholder{color:rgba(255,255,255,.30);}
.stTextArea textarea:focus{
  background:var(--surface); border-color:var(--primary-line);
  box-shadow:0 0 0 3px rgba(196,122,146,.10);
}
.stTextInput input{
  background:var(--soft-2); border:1px solid var(--line-soft); border-radius:var(--r-sm);
  font-size:14px; padding:.55rem .8rem; color:var(--ink);
}
.stTextInput input:focus{
  background:var(--surface); border-color:var(--primary-line);
  box-shadow:0 0 0 3px rgba(196,122,146,.10);
}
.stSelectbox [data-baseweb="select"],
.stSelectbox [data-baseweb="select"] > div,
.stSelectbox [data-baseweb="select"] div{
  background:var(--surface) !important;
}
.stSelectbox [data-baseweb="select"]{
  border:1px solid var(--line-soft); border-radius:var(--r-sm); min-height:38px;
  font-size:13.5px;
}

/* 示例题紧贴主按钮下方，靠 4px 上间距挂在一起，不占一整行页脚 */
[class*="st-key-samples"]{margin:.5rem 0 0;}
/* 拍照入口是卡里最后一行，弱化成一条灰色小字，不再画分隔线 */
[class*="st-key-inputfoot"]{margin:.55rem 0 0;}
[class*="st-key-inputfoot"] [data-testid="stExpander"] summary{padding-left:1rem !important;}

/* ---------- 按钮 ---------- */
/* 悬浮，不靠「厚度」：边框极细、投影朝下散开，像一张纸浮在纸面上。
   悬停上浮 2px 并让投影变大，按下回到原位、投影收紧——这是唯一的交互动效。
   不做下沿厚边、不做渐变、不做玻璃。 */
.stButton button{
  border-radius:var(--r-sm); border:1px solid var(--line);
  font-weight:600; padding:.55rem 1rem; font-size:13px; color:var(--ink);
  background:var(--surface); box-shadow:var(--lift);
  transform:translateY(0);
  transition:transform .18s var(--ease), box-shadow .18s var(--ease),
             background .18s var(--ease), border-color .18s var(--ease), color .18s var(--ease);
}
.stButton button:hover{
  background:#45224E; border-color:var(--rose-deep); color:var(--ink);
  transform:translateY(-2px); box-shadow:var(--lift-hi);
}
.stButton button:active{
  transform:translateY(0);
  box-shadow:0 1px 2px rgba(0,0,0,.10), 0 4px 8px -6px rgba(0,0,0,.30);
}
/* 主按钮就是参考站点那颗玫瑰粉药丸：实底 + 深紫字，不描边、不加渐变 */
.stButton button[kind="primary"]{
  background:var(--primary); border-color:var(--primary); color:var(--primary-ink);
  padding:.62rem 1.35rem; font-size:13.5px; font-weight:600; letter-spacing:.02em;
  border-radius:999px;
  box-shadow:0 2px 4px rgba(0,0,0,.22), 0 16px 26px -14px rgba(196,122,146,.55);
}
.stButton button[kind="primary"] p,
.stButton button[kind="primary"] div,
.stButton button[kind="primary"] span{color:var(--primary-ink) !important;}
.stButton button[kind="primary"]:hover{
  background:var(--primary-hover); border-color:var(--primary-hover);
  transform:translateY(-2px);
  box-shadow:0 4px 8px rgba(0,0,0,.24), 0 22px 32px -16px rgba(196,122,146,.62);
}
.stButton button[kind="primary"]:active{
  transform:translateY(0);
  box-shadow:0 1px 3px rgba(122,48,78,.28), 0 6px 12px -8px rgba(122,48,78,.45);
}
.stButton button:disabled, .stButton button:disabled:hover{
  transform:none; background:var(--soft-2); color:var(--muted);
  border-color:var(--line-soft); box-shadow:none;
}
.stButton button:disabled{opacity:.55;}
.stButton button p{color:inherit !important; font-weight:inherit !important;}
.stDownloadButton button{
  border-radius:var(--r-sm); font-size:12.5px; box-shadow:var(--lift);
}
/* 示例题：奶油黄实体小胶囊，和灰底、和主按钮都分得开，
   不靠「试试这些」四个字也已经看得出可点 */
[class*="st-key-sample_"] button{
  background:var(--orchid-soft) !important; border:1px solid var(--orchid-line) !important;
  border-radius:999px !important; color:#EADFF1 !important;
  font-weight:600 !important; font-size:12.5px !important;
  min-height:34px; padding:.32rem 1rem !important; letter-spacing:.02em;
  box-shadow:0 1px 2px rgba(120,80,140,.10), 0 9px 16px -12px rgba(120,80,140,.5) !important;
}
[class*="st-key-sample_"] button:hover{
  background:var(--orchid) !important; border-color:#CBB2DA !important;
  color:#FBF1FF !important; transform:translateY(-2px);
  box-shadow:0 3px 6px rgba(120,80,140,.14), 0 16px 22px -14px rgba(120,80,140,.6) !important;
}
[class*="st-key-sample_"] button:active{
  transform:translateY(0);
  box-shadow:0 1px 2px rgba(120,80,140,.14) !important;
}
/* 三枚胶囊贴在一起，靠 flex 收边，而不是三列平分把间距拉成 160px */
[class*="st-key-samples"] [data-testid="stHorizontalBlock"]{
  flex-wrap:wrap !important; gap:.45rem !important; align-items:center;
}
[class*="st-key-samples"] [data-testid="stColumn"]{
  flex:0 0 auto !important; width:auto !important; min-width:0 !important;
}
[class*="st-key-samples"] [data-testid="stColumn"]:last-child{display:none !important;}

/* ---------- 标签 ---------- */
.pill{
  display:inline-flex; align-items:center; gap:.32rem;
  padding:.18rem .5rem; border-radius:var(--r-xs); font-size:11px; font-weight:500;
  margin:0 .3rem .3rem 0; background:var(--soft-2); color:var(--ink-2);
  border:1px solid var(--line-soft); letter-spacing:.01em;
}
.pill.ok{background:var(--ok-soft); color:var(--ok); border-color:var(--ok-line);}
.pill.warn{background:var(--warn-soft); color:var(--warn); border-color:var(--warn-line);}
.pill.bad{background:var(--bad-soft); color:var(--bad); border-color:var(--bad-line);}
.pill.mute{background:var(--soft); color:var(--muted);}
.pill.blue{background:var(--primary-soft); color:var(--primary); border-color:var(--primary-line);}
.pill .dot{width:5px; height:5px; border-radius:50%; background:currentColor;}

/* ---------- 结论 ---------- */
/* 注意：这里不能用 overflow:auto —— 设了任一轴的 auto，另一轴也会变成 auto，
   而 Streamlit 量出来的容器高度比 KaTeX 排版后的实际高度小，第二行会被裁掉。
   宁可让公式自然撑开，也不要裁字。 */
[class*="st-key-conclusion"]{
  overflow:visible;
}
[class*="st-key-conclusion"] p{
  font-size:19px !important; font-weight:600; color:var(--ink);
  line-height:1.62; margin:0 0 .18rem; letter-spacing:-.15px;
  font-variant-numeric:tabular-nums;
  word-break:keep-all;   /* 中文不在两字之间断行，「成立」不会被拆成两行 */
}
[class*="st-key-conclusion"] .katex{font-size:.95em;}
[class*="st-key-conclusion"] .frac-line{border-bottom-width:1.5px !important;}
[class*="st-key-conclusion"] .katex-display{
  margin:.55rem 0 .2rem; overflow-x:auto; overflow-y:hidden; padding-bottom:.1rem;
}
[class*="st-key-qtext"] p{
  font-size:14px; color:var(--ink); line-height:1.95; margin:0;
  overflow-wrap:anywhere;
}
/* 题目直接写在卡片里，外面不再套一层浅色盒子——双层框会让信息显得廉价 */
[class*="st-key-qtext"]{
  background:transparent; border:none; border-left:2px solid var(--rose-deep);
  border-radius:0; padding:.1rem 0 .1rem .9rem; margin-bottom:.35rem; overflow:visible;
}
[class*="st-key-qtext"] .katex{font-size:.96em;}

/* ---------- 思考过程 ---------- */
.think-card{
  border:1px solid var(--line); border-radius:var(--r-lg); padding:1.05rem 1.15rem 1.1rem;
  background:var(--surface); margin:.2rem 0 1rem; box-shadow:var(--sh-1);
}
.think-head{display:flex; align-items:center; gap:.5rem; margin-bottom:.9rem;}
.think-title{font-size:13px; font-weight:600; color:var(--ink);}
.think-time{
  font-size:11.5px; color:var(--muted); margin-left:auto;
  font-variant-numeric:tabular-nums; font-family:var(--num);
}
.dot-live{
  width:6px; height:6px; border-radius:50%; background:var(--primary);
  box-shadow:0 0 0 4px var(--primary-soft);
}
.dot-done{width:6px; height:6px; border-radius:50%; background:var(--rose-deep);}
.stage{display:flex; gap:.65rem; align-items:flex-start; padding:.34rem 0; font-size:12.5px;}
.stage-dot{
  flex:0 0 14px; width:14px; height:14px; border-radius:50%; margin-top:.2rem;
  border:1px solid var(--line); background:var(--surface); color:#fff;
  display:inline-flex; align-items:center; justify-content:center; font-size:8px;
}
.stage.done .stage-dot{background:var(--rose-deep); border-color:var(--rose-deep); color:var(--primary-ink);}
.stage.done .stage-dot::after{content:"✓";}
.stage.active .stage-dot{border:1.5px solid var(--orchid-line); background:var(--orchid-soft);}
.stage.active .stage-dot::after{content:""; width:4px; height:4px; border-radius:50%; background:var(--warn); opacity:.75;}
@media (prefers-reduced-motion:reduce){*{animation:none !important; transition:none !important;}}
.stage-name{font-weight:600; color:var(--ink); margin-right:.45rem;}
.stage-msg{color:var(--muted);}
.stage-ms{
  margin-left:auto; color:rgba(255,255,255,.36); font-size:11px;
  font-variant-numeric:tabular-nums; font-family:var(--num);
}
.stream{
  margin-top:.75rem; background:var(--soft-2); border:1px solid var(--line-soft);
  border-radius:var(--r-md); padding:.75rem .9rem; max-height:230px; overflow:auto;
  font-size:12.5px; line-height:1.95; color:var(--ink-2); white-space:pre-wrap;
}
.stream-label{font-size:10.5px; color:var(--muted); letter-spacing:.08em; text-transform:uppercase; margin-bottom:.4rem;}
.caret{display:inline-block; width:1px; height:13px; background:var(--primary);
  vertical-align:-2px; opacity:.55;}

/* ---------- 推导步骤 ---------- */
[class*="st-key-stepcard-"]{
  position:relative; padding:.72rem .95rem .82rem 1.9rem;
  background:var(--surface); border:1px solid var(--line-soft);
  border-radius:var(--r-md); margin-bottom:.72rem;
}
[class*="st-key-stepcard-"]:last-child{margin-bottom:0;}
[class*="st-key-stepcard-"]::before{
  content:""; position:absolute; left:0; top:.66rem; bottom:.5rem; width:1px; background:var(--line);
}
[class*="st-key-stepcard-"]::after{
  content:""; position:absolute; left:-3px; top:.72rem; width:7px; height:7px;
  border-radius:2px; background:var(--bg); border:1px solid var(--mauve);
  transition:border-color .18s var(--ease), background .18s var(--ease);
}
[class*="st-key-stepcard-"]:has(.step-flag.on)::after{
  background:var(--primary); border-color:var(--primary);
  box-shadow:0 0 0 4px rgba(196,122,146,.14);
}
.step-flag{display:none;}
.step-head{display:flex; align-items:center; gap:.5rem; flex-wrap:wrap; margin-bottom:.3rem;}
.stepno{
  display:inline-flex; align-items:center; justify-content:center;
  min-width:21px; height:21px; border-radius:var(--r-xs); padding:0 6px;
  background:var(--surface); border:1px solid var(--line); color:var(--ink-2);
  font-size:11.5px; font-weight:600; font-family:var(--num);
  font-variant-numeric:tabular-nums;
}
[class*="st-key-stepcard-"]:has(.step-flag.on) .stepno{
  background:var(--primary); border-color:var(--primary); color:var(--primary-ink);
}
.thm-tag{
  display:inline-flex; align-items:center; gap:.32rem;
  background:var(--orchid-soft); color:#E6D9EF; border:1px solid var(--orchid-line);
  border-radius:var(--r-xs); padding:.12rem .45rem;
  font-size:11px; font-weight:500; letter-spacing:.01em;
}
.thm-tag::before{content:"§"; opacity:.55; font-family:var(--num);}
[class*="st-key-stepbody-"] p{
  font-size:13.5px; line-height:1.95; color:var(--ink-2);
  margin:.02rem 0 .3rem; max-width:64ch;
}
[class*="st-key-stepbody-"] p:last-child{margin-bottom:.1rem;}
[class*="st-key-stepbody-"] .katex-display,
[class*="st-key-stepbody-"] p > .katex-display{
  margin:.85rem 0 .95rem;
  overflow-x:auto; overflow-y:hidden;
}
[class*="st-key-stepbody-"] p:has(.katex-display){margin:.2rem 0;}
[class*="st-key-steptitle-"] p{
  font-size:14px !important; font-weight:600; color:var(--ink);
  margin:.05rem 0 .22rem; letter-spacing:-.1px; line-height:1.6;
}
/* 卡内「查看第 N 步依据」是一条链接，不是一颗按钮：
   做成按钮会和右侧真正的主操作抢注意力，只保留颜色 + 下划线作为可点提示。 */
[class*="st-key-focus-"] button{
  background:transparent !important; border:none !important; box-shadow:none !important;
  padding:.1rem 0 !important; min-height:0 !important; font-size:12px !important;
  font-weight:600 !important; color:var(--primary) !important;
  text-decoration:underline; text-decoration-thickness:1px;
  text-underline-offset:4px; text-decoration-color:var(--primary-line);
}
[class*="st-key-focus-"] button p{font-size:12px !important; color:inherit !important;}
[class*="st-key-focus-"] button:hover{color:var(--primary-hover) !important;}
[class*="st-key-focus-"] button:hover p{color:var(--primary-hover) !important;}

/* ---------- 右栏：教材依据（书页式，陶土书脊） ---------- */
[class*="st-key-srcdoc-"]{
  background:var(--surface); border:1px solid var(--line-soft);
  border-left:2px solid var(--mauve-line);
  border-radius:0 var(--r-md) var(--r-md) 0; padding:.85rem .95rem .7rem; margin:.55rem 0;
  transition:background .18s ease, border-color .18s ease;
}
[class*="st-key-srcdoc-"]:hover{border-left-color:var(--mauve);}
[class*="st-key-srcdoc-"]:has(.src-flag.on){
  background:var(--primary-soft); border-color:var(--primary-line);
  border-left-color:var(--primary);
}
.src-flag{display:none;}
.src-head{display:flex; align-items:flex-start; gap:.5rem; margin-bottom:.45rem;}
.src-no{
  flex:0 0 auto; min-width:20px; height:20px; border-radius:var(--r-xs); padding:0 6px;
  background:var(--surface); border:1px solid var(--line); color:var(--muted);
  font-family:var(--num); font-size:10.5px; font-weight:600; display:inline-flex;
  align-items:center; justify-content:center; font-variant-numeric:tabular-nums;
  white-space:nowrap;
}
[class*="st-key-srcdoc-"]:has(.src-flag.on) .src-no{
  background:var(--primary); border-color:var(--primary); color:var(--primary-ink);
}
.src-topic{
  font-size:12.5px; font-weight:600; color:var(--ink); line-height:1.55;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
  overflow:hidden; letter-spacing:-.05px; text-wrap:balance;
}
[class*="st-key-srcdoc-"]:has(.src-flag.on) .src-topic{color:var(--primary);}
[class*="st-key-srcdoc-"] .katex-display{overflow-x:auto; overflow-y:hidden;}
[class*="st-key-srcdoc-"] p{
  font-size:12.5px; line-height:1.8; color:var(--ink-2); margin:0 0 .3rem;
  text-align:left; hyphens:auto; word-break:break-word;
}
[class*="st-key-srcdoc-"] .katex{font-size:1em;}
[class*="st-key-srcdoc-"] .katex-display{margin:.4rem 0;}
/* 公式引用卡：定理卡讲「这一步用哪条定理」，公式卡讲「这个式子在书里哪一页」。
   同一套书页样式，只把书脊换成奶油黄，用来和定理卡区分。 */
[class*="st-key-fmref-"]{
  background:var(--surface); border:1px solid var(--line-soft);
  border-left:2px solid var(--orchid-line);
  border-radius:0 var(--r-md) var(--r-md) 0; padding:.62rem .9rem .5rem; margin:.45rem 0;
  transition:background .18s var(--ease), border-color .18s var(--ease);
}
[class*="st-key-fmref-"]:hover{border-left-color:var(--orchid);}
[class*="st-key-fmref-"]:has(.src-flag.on){
  background:var(--orchid-soft); border-color:var(--orchid-line);
  border-left-color:var(--orchid);
}
[class*="st-key-fmref-"] .src-no{
  background:var(--orchid-soft); border-color:var(--orchid-line); color:var(--warn);
}
[class*="st-key-fmref-"]:has(.src-flag.on) .src-no{
  background:var(--orchid); border-color:var(--orchid-line); color:#2A1230;
}
[class*="st-key-fmref-"] .katex{font-size:.94em;}
[class*="st-key-fmref-"] .katex-display{
  margin:.1rem 0 .3rem; overflow-x:auto; overflow-y:hidden;
  /* 右边缘一道极淡的渐隐：教材里的长公式一定超出栏宽，
     渐隐是「后面还有」的唯一提示，比一根滚动条安静 */
  mask-image:linear-gradient(to right, #000 calc(100% - 16px), rgba(0,0,0,.25));
  -webkit-mask-image:linear-gradient(to right, #000 calc(100% - 16px), rgba(0,0,0,.25));
}
/* 长公式超出栏宽时给一根 4px 的细滚动条：比没有提示强，也比默认滚动条安静 */
[class*="st-key-fmref-"] .katex-display::-webkit-scrollbar{height:4px;}
[class*="st-key-fmref-"] .katex-display::-webkit-scrollbar-track{background:transparent;}
[class*="st-key-fmref-"] .katex-display::-webkit-scrollbar-thumb{
  background:var(--orchid-line); border-radius:2px;
}
/* 推导步骤头上的「式 p.57」小签：和定理签（奶油黄 §）区分开，用鼠尾草绿 */
.fm-tag{
  display:inline-flex; align-items:center; gap:.32rem;
  background:var(--primary-soft); color:var(--primary); border:1px solid var(--primary-line);
  border-radius:var(--r-xs); padding:.12rem .45rem;
  font-size:10.5px; font-weight:600; letter-spacing:.01em;
  font-variant-numeric:tabular-nums;
}
.fm-tag::before{content:"式"; opacity:.6; font-weight:500;}

.src-meta{
  font-size:10.5px; color:var(--muted); margin-top:.42rem;
  font-family:var(--num); letter-spacing:-.1px;
}
[class*="st-key-srcbtn-"] button,
[class*="st-key-fmbtn-"] button{
  background:transparent !important; border:none !important; box-shadow:none !important;
  padding:.05rem 0 !important; min-height:0 !important;
  font-size:11.5px !important; font-weight:500 !important; color:var(--primary) !important;
}
[class*="st-key-srcbtn-"] button p,
[class*="st-key-fmbtn-"] button p{font-size:11.5px !important; color:var(--primary) !important;}
/* 题面依据：题目点名的定理/公式，压成扁引用行，不铺教材正文，
   避免顶部这一行被两张长卡撑高、在题目卡下方留出大片空白。 */
[class*="st-key-srcdoc-q"]{padding:.5rem .8rem .38rem; margin:.35rem 0;}
[class*="st-key-srcdoc-q"] .src-head{margin-bottom:.15rem;}
[class*="st-key-srcdoc-q"] .src-topic{-webkit-line-clamp:1;}
[class*="st-key-srcdoc-q"] .src-meta{margin-top:.2rem;}
[class*="st-key-srcdoc-q"] [class*="st-key-srcbtn-"] button{padding:.02rem 0 !important;}

/* ---------- 逐条行（验证 / 审核 / 局限） ---------- */
[class*="st-key-vrow-"]{
  position:relative; padding:.28rem 0 .3rem 1rem;
  border-bottom:1px solid var(--line-soft);
}
[class*="st-key-vrow-"]:last-child{border-bottom:none;}
[class*="st-key-vrow-"]::before{
  content:""; position:absolute; left:0; top:1rem;
  width:4px; height:4px; border-radius:50%; background:var(--mauve);
}
[class*="st-key-vrow-"] p{margin:0 0 .25rem; font-size:13px; line-height:1.75; color:var(--ink-2);}
[class*="st-key-vrow-"] p:last-child{margin-bottom:0;}
[class*="st-key-vrow-ok"]::before{background:var(--ok);}
[class*="st-key-vrow-bad"]::before{background:var(--bad);}
[class*="st-key-vrow-warn"]::before{background:var(--warn);}

/* ---------- 知识来源行 ---------- */
[class*="st-key-src-"] button{
  justify-content:flex-start !important; text-align:left !important;
  width:100% !important; background:transparent !important; border:none !important;
  border-bottom:1px solid var(--line-soft) !important; border-radius:0 !important;
  box-shadow:none !important; padding:.5rem 1.2rem .5rem 0 !important;
  font-weight:500 !important; font-size:12.5px !important; color:var(--ink-2) !important;
}
[class*="st-key-src-"] button:hover{color:var(--primary) !important;}
[class*="st-key-src-"] button > div,
[class*="st-key-src-"] button span,
[class*="st-key-src-"] button div[data-testid="stMarkdownContainer"]{
  width:100% !important; justify-content:flex-start !important; text-align:left !important;
}
[class*="st-key-src-"] button p{
  font-size:12.5px !important; font-weight:500 !important; color:inherit !important; text-align:left !important;
}

/* ---------- 知识库 ---------- */
[class*="st-key-kb-focus"]{
  background:var(--surface); border:1px solid var(--primary-line); border-radius:var(--r-lg);
  padding:.9rem 1.05rem 1rem; box-shadow:var(--sh-1);
}
[class*="st-key-kbcard-"] p{
  font-size:13px; line-height:1.8; color:var(--ink-2); margin:0 0 .2rem; max-width:82ch;
}
[class*="st-key-kbcard-"] p:last-child{margin-bottom:0;}

/* ---------- 其它 ---------- */
[data-testid="stIconMaterial"]{display:none !important;}
[data-testid="stExpander"], [data-testid="stExpander"] details,
[data-testid="stExpander"] summary{
  background:transparent !important; border:none !important; box-shadow:none !important;
}
[data-testid="stExpander"] summary{
  position:relative; padding-left:1rem !important; border-radius:var(--r-sm);
}
[data-testid="stExpander"] summary:hover{background:var(--soft-2) !important;}
[data-testid="stExpander"] summary::before{
  content:""; position:absolute; left:.15rem; top:50%; width:5px; height:5px;
  margin-top:-3px; border-right:1.4px solid var(--muted); border-bottom:1.4px solid var(--muted);
  transform:rotate(-45deg);
}
[data-testid="stExpander"] details[open] > summary::before{transform:rotate(45deg); margin-top:-4px;}
[data-testid="stExpander"] summary p{font-size:12.5px !important; color:var(--ink-2) !important;}
/* 桌面端侧栏常驻，不需要「收起」这个动作，把箭头藏掉；
   窄屏侧栏是覆盖式抽屉，必须留一个关闭入口，否则会挡住正文。 */
@media (min-width:961px){
  [data-testid="stSidebarCollapseButton"]{display:none !important;}
}
hr{border:none; border-top:1px solid var(--line-soft); margin:1.3rem 0;}
.stAlert{border-radius:var(--r-md); font-size:12.5px; border:1px solid var(--line);}
.stProgress > div > div{background:var(--primary);}
.stCaption, [data-testid="stCaptionContainer"] p{color:var(--muted) !important; font-size:11.5px; letter-spacing:.01em;}
.stImage img{border-radius:var(--r-md); border:1px solid var(--line);}
.stMarkdown{overflow-wrap:break-word;}

/* ---------- 深色底上的原生控件 ---------- */
/* 主题里已声明 base=dark，这里只补那些 Streamlit 自带深色仍不够「梅紫」的地方 */
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] ul{background:var(--surface) !important;}
[data-baseweb="menu"] li{color:var(--ink-2) !important; font-size:13px;}
[data-baseweb="menu"] li:hover{background:var(--primary-soft) !important;}
[data-testid="stFileUploaderDropzone"]{
  background:var(--soft-2) !important; border:1px dashed var(--line) !important;
  border-radius:var(--r-md) !important;
}
[data-testid="stFileUploaderDropzone"] *{color:var(--ink-2) !important;}
.stToggle label p{color:var(--ink-2) !important; font-size:12.5px !important;}
.stAlert{background:var(--soft-2); border:1px solid var(--line); color:var(--ink-2);}
.stCode, pre, code{background:var(--soft-2) !important; color:#EBD8F2 !important;}

/* ---------- 响应式 ---------- */
@media (max-width: 960px){
  .block-container{padding:1.6rem 1.15rem 4.5rem;}
  .hero{font-size:30px; letter-spacing:-.7px;}
  /* 列在小屏一律纵向堆叠，避免横向挤压/溢出 */
  [data-testid="stHorizontalBlock"]{flex-wrap:wrap !important;}
  [data-testid="stColumn"]{flex:1 1 100% !important; min-width:100% !important;}
  .katex-display{overflow-x:auto; overflow-y:hidden;}
  /* 控件行重排：主按钮整行，两个开关并排 */
  [class*="st-key-controls"] [data-testid="stHorizontalBlock"]{
    display:grid !important; grid-template-columns:1fr 1fr; gap:.55rem .8rem;
  }
  [class*="st-key-controls"] [data-testid="stColumn"]{
    flex:none !important; width:auto !important; min-width:0 !important;
  }
  [class*="st-key-controls"] [data-testid="stColumn"]:first-child{grid-column:1 / -1;}
  [class*="st-key-stepbody-"] p{font-size:13px; line-height:1.9;}
  /* 窄屏侧栏是覆盖式抽屉，关闭箭头必须看得出来，否则正文被挡住就没路走了 */
  [data-testid="stSidebarCollapseButton"] button{
    width:30px !important; height:30px !important; border-radius:8px !important;
    background:var(--surface) !important; border:1px solid var(--line) !important;
    box-shadow:0 1px 2px rgba(0,0,0,.06) !important;
    display:flex !important; align-items:center; justify-content:center;
  }
  /* 图标是字体图标，只给按钮上色不会跟过去，得连子节点一起染 */
  [data-testid="stSidebarCollapseButton"] button,
  [data-testid="stSidebarCollapseButton"] button *{color:var(--ink) !important; fill:var(--ink) !important;}
  /* 同上：字体图标可能没加载，自己画一个朝左的箭头 */
  [data-testid="stSidebarCollapseButton"] button::after{
    content:""; display:block; width:14px; height:14px;
    background:center/14px 14px no-repeat
      url("data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FBF5FA' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M15 5l-7 7 7 7'/%3E%3C/svg%3E");
  }
  [class*="st-key-conclusion"] p{font-size:18px !important;}
}
@media (max-width: 620px){
  .block-container{padding:1.1rem .85rem 4rem;}
  /* 示例胶囊在小屏也排成一行，不要各占一整行撑出大片空白 */
  [class*="st-key-samples"] [data-testid="stHorizontalBlock"]{flex-wrap:wrap !important; gap:.4rem !important;}
  [class*="st-key-samples"] [data-testid="stColumn"]{
    flex:0 0 auto !important; width:auto !important; min-width:0 !important;
  }
  [class*="st-key-samples"] [data-testid="stColumn"]:last-child{display:none !important;}
  [class*="st-key-sample_"] button{width:auto !important; min-width:0; padding:.3rem .8rem !important;}
  .hero{font-size:27px; letter-spacing:-.4px;}
  .src-topic{font-size:12px;}
  [class*="st-key-srcdoc-"] p{text-align:left;}
  .stButton button{min-height:42px;}
  /* 窄屏公式降一号，避免整行被裁掉 */
  .katex{font-size:.93em;}
  [class*="st-key-stepbody-"] .katex-display{padding-right:1rem;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

STATUS_PILL = {
    "verified": ("ok", "已验证"),
    "partial": ("warn", "部分验证"),
    "unverified": ("mute", "未验证"),
    "mismatch": ("bad", "发现不一致"),
}
SEV_PILL = {"blocker": ("bad", "阻断"), "warning": ("warn", "警告"), "info": ("mute", "提示")}
NODE_LABEL = {
    "parse": "解析题目",
    "analyze": "题目结构化",
    "plan": "解题规划",
    "solve": "逐步推导",
    "retrieve": "按定理查教材",
    "verify": "符号复算",
    "review": "交叉审核",
    "format": "生成报告",
}


@st.cache_resource(show_spinner=False)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


def warmup_models() -> None:
    """后台预热 embedding + reranker。

    这两个模型首次加载合计 20~30s（reranker 是 1GB ONNX）。若等到用户第一次
    点「开始解题」才加载，这段就全算在解题耗时里。页面起来后就扔后台线程预热。
    """
    if st.session_state.get("_warmup_started"):
        return
    st.session_state["_warmup_started"] = True

    def _run() -> None:
        try:
            r = get_retriever()
            if r.vector.available and r.cfg.vector.enabled:
                r.vector.build()
            if getattr(r, "reranker", None) is not None:
                r.reranker.rerank("预热", ["预热文本"])
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


@st.cache_data(show_spinner=False)
def load_chunks() -> list[dict]:
    path = ROOT / "knowledge_base" / "chunks" / "chunks.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


@st.cache_data(show_spinner=False)
def chunk_index() -> dict[str, dict]:
    return {c.get("chunk_id"): c for c in load_chunks() if c.get("chunk_id")}


def pill(text: str, kind: str = "") -> str:
    # 一切拼进 HTML 的文本都要转义：教材原文里有 $\{a,b\}$，浏览器会当标签解析
    text = html.escape(str(text))
    dot = '<span class="dot"></span>' if kind else ""
    return f'<span class="pill {kind}">{dot}{text}</span>'


_LATEX_CMD = re.compile(r"\\[a-zA-Z]{2,}")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_ZH_BOUNDARY = re.compile(r"[\u4e00-\u9fff，。；：、！？“”‘’（）《》\s]")
_ZH_STOP = re.compile(r"[\u4e00-\u9fff，。；：、！？“”‘’（）《》]")


def _wrap_inline_latex(text: str) -> str:
    """把「中文句子里裸写的公式」包成 $...$（模型经常写裸 LaTeX）。"""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        m = _LATEX_CMD.search(text, i)
        if not m:
            out.append(text[i:])
            break
        j = m.start()
        while j > i and not _ZH_BOUNDARY.match(text[j - 1]):
            j -= 1
        k = m.end()
        while k < n and not _ZH_STOP.match(text[k]):
            k += 1
        raw = text[j:k]
        seg = raw.strip()
        out.append(text[i:j])
        out.append(f"${seg}$" if seg else raw)
        if seg and raw.endswith((" ", "\t")):
            out.append(raw[len(seg):])
        i = k
    return "".join(out)


def _flat(x: str) -> str:
    """公式内部换行折成空格：markdown 会把跨行的 $$...$$ 拆开导致解析失败。"""
    return " ".join(x.split())


def _norm_tex(x: str) -> str:
    return re.sub(r"\s+", "", str(x or ""))


def _bare_latex(x: str) -> str:
    """去掉外层的 $ / $$ / \\[ \\] ，再交给 st.latex。

    st.latex 自己会补一对 $$，模型或提示词若已经带了定界符，就会变成 $$$$…$$$$，
    KaTeX 报 "Can't use function '$' in math mode"，页面上直接露出红色原文。
    """
    t = str(x or "").strip()
    pairs = (("$$", "$$"), ("$", "$"), ("\\[", "\\]"), ("\\(", "\\)"))
    for a, b in pairs:
        if t.startswith(a) and t.endswith(b) and len(t) > len(a) + len(b):
            return t[len(a):-len(b)].strip()
    return t


def _snippet(text: str, limit: int = 170) -> str:
    """截预览：先退回标点，再砍掉落单的 $，避免 KaTeX 吞掉后文。"""
    raw = " ".join(str(text or "").split())
    if len(raw) <= limit:
        return raw
    cut = raw[:limit]
    for sep in ("。", "；", "，", " "):
        pos = cut.rfind(sep)
        if pos >= int(limit * 0.6):
            cut = cut[: pos + 1]
            break
    if cut.count("$") % 2:
        pos = cut.rfind("$")
        if pos > 0:
            cut = cut[:pos]
    return cut.rstrip() + " …"


# ---------------------------------------------------------------- 公式引用索引
# 教材里近三万个公式，抹掉排版差异后大量重复。解题时模型复写教材公式很常见
# （洛必达的结论式、泰勒展开、中值定理的端点式），右栏若能把「这个式子在书里
# 哪一页」也指出来，「定理有出处」就升级成「每一步、每个式子都有出处」。
_MATH_SPAN = re.compile(r"\$\$(.+?)\$\$|\$([^$\n]+?)\$", re.S)
_FM_STRIP = re.compile(r"\\(?:left|right|limits|nolimits|displaystyle|,|;|!|quad|qquad| )")
_FM_TEXT = re.compile(r"\\text\{[^{}]*\}")


_FM_CMD = re.compile(r"\\[a-zA-Z]+")
_FM_GREEK = re.compile(r"\\(?:lambda|mu|nu|alpha|beta|gamma|delta|Delta|theta|omega|xi|eta|"
                       r"varepsilon|epsilon|pi|rho|sigma|tau|varphi|phi|psi|partial|nabla|infty)")


def _fm_meaningful(key: str) -> bool:
    """过滤掉 `\\frac{0}{0}`、`\\frac{1}{2}` 这种处处都有的比值。

    判据：去掉命令名后还剩字母（`f`、`I`、`x`），或者含希腊字母 / ∇ / ∂ 这类
    真正带数学对象的符号。纯数字、纯运算符一律不算。
    """
    rest = _FM_CMD.sub("", key)
    if re.search(r"[a-zA-Z]", rest):
        return True
    return bool(_FM_GREEK.search(key))


def _norm_formula(tex: str) -> str:
    """抹平排版差异：\\dfrac/\\frac、\\limits、\\leqslant/\\le、间距与空格。"""
    t = str(tex or "")
    t = t.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    t = t.replace("\\leqslant", "\\le").replace("\\geqslant", "\\ge")
    t = t.replace("\\mid", "|").replace("\\vert", "|")
    t = _FM_TEXT.sub("", t)
    t = _FM_STRIP.sub("", t)
    # a^{+} 与 a^+ 是同一种写法，只收短上/下标（长的留给去括号键处理）
    t = re.sub(r"([\^_])\{([^{}]{1,3})\}", r"\1\2", t)
    t = re.sub(r"\s+", "", t)
    # 教材的展示公式常带句末标点（`=1.`、`=0,`），模型复写时一般不带
    return t.strip("$").rstrip(".,;，。")


def _loose_formula(key: str) -> str:
    """再去掉花括号：教材写 \\frac{1}{2}、模型写 \\frac12 时仍能对上。"""
    return key.replace("{", "").replace("}", "")


@st.cache_data(show_spinner=False)
def formula_index() -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]], dict[str, list[str]]]:
    """归一化公式 → (chunk_id, 原始 TeX)。

    严格键、去括号键各一份；再按前 6 个字符分桶，供「子式」匹配用——
    模型常把书上一整条公式写成其中一个式子，字面不会完全相同。
    """
    strict: dict[str, tuple[str, str]] = {}
    loose: dict[str, tuple[str, str]] = {}
    for c in load_chunks():
        cid = c.get("chunk_id")
        if not cid:
            continue
        # 例题/习题块不参与公式引用：公式原文若只在例题里出现，就宁可不指，
        # 依据栏只保留定理、公式这类规则原文（与后端 retrieve 的例题过滤同一口径）。
        ct = str(c.get("content_type") or "")
        if ct in ("例", "例题", "习题") or re.match(r"^\s*(例|习题)\s*\d", str(c.get("topic") or "")):
            continue
        for m in _MATH_SPAN.finditer(c.get("content") or ""):
            raw = (m.group(1) or m.group(2) or "").strip()
            key = _norm_formula(raw)
            # 12 个字符以下基本是 $x$、$f(x)$、$n\to\infty$ 这类到处都有的碎片，
            # 拿它去指页码只会指错地方；纯数字比值（0/0、1/2）同理。
            if len(key) < 12 or not _fm_meaningful(key):
                continue
            strict.setdefault(key, (cid, raw))
            loose.setdefault(_loose_formula(key), (cid, raw))
    by_prefix: dict[str, list[str]] = {}
    for key in loose:
        by_prefix.setdefault(key[:6], []).append(key)
    return strict, loose, by_prefix


def _step_formulas(step: dict) -> list[str]:
    """这一步写到的公式原文：正文里的行内/独立公式 + 独立的 latex 字段。"""
    blob = " ".join([step.get("content") or "", step.get("latex") or ""])
    out: list[str] = []
    seen: set[str] = set()
    for m in _MATH_SPAN.finditer(blob):
        raw = (m.group(1) or m.group(2) or "").strip()
        key = _norm_formula(raw)
        if len(key) < 12 or not _fm_meaningful(key) or key in seen:
            continue
        seen.add(key)
        out.append(raw)
    latex = _bare_latex(step.get("latex") or "")
    if latex and _norm_formula(latex) not in seen:
        out.append(latex)
    return out


def match_formulas(step: dict, limit: int = 2) -> list[dict]:
    """把一步里的公式对回教材块。

    先按归一化后的字面完全对；对不上再在「开头相同」的桶里找唯一的包含关系
    （模型写书上一整条公式里的那一段）。候选不唯一就直接放弃——
    指错页码比不指更糟。
    """
    strict, loose, by_prefix = formula_index()
    hits: list[dict] = []
    for raw in _step_formulas(step):
        key = _norm_formula(raw)
        hit = strict.get(key) or loose.get(_loose_formula(key))
        if not hit:
            cand = _loose_formula(key)
            if len(cand) >= 14:
                pool = [x for x in by_prefix.get(cand[:6], [])
                        if x != cand and (cand in x or x in cand)]
                if len(pool) == 1:
                    hit = loose[pool[0]]
        if not hit:
            continue
        cid, tex = hit
        if any(h["chunk_id"] == cid and h["tex"] == tex for h in hits):
            continue
        hits.append({"chunk_id": cid, "tex": tex})
        if len(hits) >= limit:
            break
    return hits


def _short(text: str, limit: int = 16) -> str:
    """卡片标题只留「是哪一条」，不整句照搬教材目录。"""
    t = " ".join(str(text or "").split())
    return t if len(t) <= limit else t[:limit].rstrip("，,。；; ") + "…"


def _fm_title(topic: str) -> str:
    """教材块的 topic 有时是一整句话，标题里只取定理/例题编号。"""
    t = " ".join(str(topic or "").split())
    m = re.match(r"((?:定理|例|推论|引理|定义|公式)\s*[\d.]*\d)", t)
    if m:
        return m.group(1)
    return _short(t, 14) or "教材原文"


def md(s: str) -> str:
    """规范化模型正文里的公式定界符，再交给 markdown 渲染（Streamlit 只认 $）。"""
    t = str(s or "")
    if not t:
        return t
    t = re.sub(r"\$\$(.+?)\$\$",
               lambda m: "\n\n$$\\displaystyle " + _flat(m.group(1)) + "$$\n\n", t, flags=re.S)
    t = re.sub(r"\\\[(.+?)\\\]",
               lambda m: "\n\n$$\\displaystyle " + _flat(m.group(1)) + "$$\n\n", t, flags=re.S)
    t = re.sub(r"\\\((.+?)\\\)", lambda m: "$" + _flat(m.group(1)) + "$", t, flags=re.S)
    out = []
    for line in t.split("\n"):
        bare = line.strip()
        if len(bare) > 4 and bare.startswith("$$") and bare.endswith("$$") and bare.count("$$") == 2:
            out.extend(["", bare, ""])
            continue
        if bare.count("$") % 2 == 1:
            # 教材 OCR 常把公式的 $ 弄丢一半（`$\lim_{x \to 0^-} f(x)=...` 后面没有收尾）。
            # 不处理的话 KaTeX 整段放弃解析，界面上直接露出反斜杠 —— 比不渲染更难看。
            # 这里丢掉落单的 $，当成裸 LaTeX 重新包一层。
            bare = bare.replace("$", "").strip()
            if not bare:
                continue
            if _LATEX_CMD.search(bare) and len(bare) <= 320 and len(_CJK.findall(bare)) <= 2:
                out.extend(["", f"$${bare}$$", ""])
            else:
                out.append(_wrap_inline_latex(bare))
            continue
        if bare and "$" not in bare and _LATEX_CMD.search(bare):
            if len(_CJK.findall(bare)) <= 2 and len(bare) <= 320:
                out.extend(["", f"$${bare}$$", ""])
                continue
            out.append(_wrap_inline_latex(line))
            continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))


def md_rows(items, kind: str = "plain", key: str = "vrow") -> None:
    for i, raw in enumerate(items or []):
        text = str(raw or "").strip()
        if not text:
            continue
        with st.container(key=f"{key}-{kind}-{i}"):
            st.markdown(md(text))


# 句末标点（不含逗号/顿号：数学推导里逗号常连接同一判断的两个部分，
# 按逗号切会把「设 $f$ 在 $[a,b]$ 连续，在 $(a,b)$ 可导」拆成两行）
_SENT_END = re.compile(r"(?<=[。！？])")
_MATH_BLOCK = re.compile(r"(\$\$.+?\$\$|\$[^$\n]+?\$)", re.S)
_HOLE = re.compile(r"\u0000(\d+)\u0000")


def _split_sentences(text: str, min_chars: int = 12) -> str:
    """把正文按句切开，一句一段。

    模型即使被要求「一句一行」也常写成一大段；前端再兜一层，保证版面稳定。
    公式先打成占位符再切分 —— 否则 `$f(x)=1, x>0$` 里的标点会把公式劈成两半，
    KaTeX 解析失败后整段会露出反斜杠。过短的句子（< min_chars）并入下一句，
    避免出现「故。」这种只有两个字的孤行。
    """
    if not text:
        return text
    holes: list[str] = []

    def _stash(m: re.Match) -> str:
        holes.append(m.group(0))
        token = f"\u0000{len(holes) - 1}\u0000"
        # 独立公式前后强制断开，让它自占一段
        return f"\n\n{token}\n\n" if m.group(0).startswith("$$") else token

    masked = _MATH_BLOCK.sub(_stash, text)
    lines: list[str] = []
    for raw_line in masked.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        pieces = [p.strip() for p in _SENT_END.split(line) if p.strip()]
        for idx, piece in enumerate(pieces):
            ends = piece.endswith(("。", "！", "？"))
            # 独立公式的占位符不能被当成「残段」并回去（它只有 3 个字符）
            is_math_hole = piece.startswith("\u0000")
            # 只有「末尾没有句号、且极短」的残段（如「即」）才并回上一行；
            # 带句号的短句（如「故 $x=0$。」）是完整判断，必须独立成行。
            if (not ends and not is_math_hole and idx == len(pieces) - 1
                    and lines and len(piece) < 6):
                lines[-1] = f"{lines[-1]}{piece}"
            else:
                lines.append(piece)
    out = "\n\n".join(lines)
    return _HOLE.sub(lambda m: holes[int(m.group(1))], out)


def md_lines(s: str) -> str:
    """正文排版：先规范公式定界符，再按句分行。"""
    return md(_split_sentences(str(s or "")))


# 结论里常并列多条命题：「（1）…成立；（2）λ<0。」挤在一行会把序号甩到行尾。
_CLAIM_SPLIT = re.compile(r"[；;]\s*(?=[（(]\s*\d)")


def split_claims(text: str) -> str:
    """在「；」+ 序号处断行，让每条命题各占一段。"""
    return _CLAIM_SPLIT.sub("；\n\n", str(text or ""))


# ---------------------------------------------------------------- 报告渲染
NAV_OPTIONS = ["解题工作台", "知识库", "设置"]


def _open_source(chunk_id: str) -> None:
    st.session_state["nav"] = NAV_OPTIONS[1]
    st.session_state["kb_focus"] = chunk_id


def _source_doc(chunk_id: str | None, idx: dict) -> dict | None:
    if not chunk_id:
        return None
    c = idx.get(chunk_id)
    if not c:
        return None
    # 例题/习题不作为依据展示：依据栏只放定理、公式这类规则原文
    #（旧结果缓存里的 source_id 若指向例题，这里也会拦掉）。
    ct = str(c.get("content_type") or "")
    if ct in ("例", "例题", "习题") or re.match(r"^\s*(例|习题)\s*\d", str(c.get("topic") or "")):
        return None
    return {
        "chunk_id": c.get("chunk_id"),
        "topic": c.get("topic") or c.get("section") or "教材原文",
        "content_type": c.get("content_type"),
        "chapter": c.get("chapter"),
        "page": c.get("page"),
        "content": _snippet(c.get("content") or "", 240),
    }


def _focus_step(i: int) -> None:
    st.session_state["active_step"] = i


_JSON_TITLE = re.compile(r'"title"\s*:\s*"([^"]{0,80})')
_JSON_BODY = re.compile(r'"content"\s*:\s*"([^"]{0,600})')


def _content_digest(text: str) -> str:
    """把流式收到的半截 JSON 变成人话：正在写哪一步、这一步写了什么。

    直接铺原始 JSON 会满屏花括号和空行（白底上一片稀疏），用户看不出进展。
    这里抽 title / content 字段：没有思维链的模型（如 Qwen2.5-Math）也能
    让人看到「第 2 步 · 两次洛必达」这种明确的推进。
    """
    t = str(text or "").replace("\\n", "\n").replace('\\"', '"')
    titles = [x.strip() for x in _JSON_TITLE.findall(t) if x.strip()]
    bodies = _JSON_BODY.findall(t)
    lines: list[str] = []
    for i, title in enumerate(titles[-4:], start=max(1, len(titles) - 3)):
        lines.append(f"第 {i} 步 · {title}")
    if bodies:
        tail = bodies[-1].replace("$", "").replace("\\", "")
        tail = re.sub(r"\s+", " ", tail).strip()[-150:]
        if tail:
            lines.append("　" + tail + ("…" if len(tail) >= 150 else ""))
    if not lines:
        tail = re.sub(r"\s+", " ", t.replace("$", "").replace("\\", "")).strip()[-220:]
        lines = [tail]
    return "\n".join(x for x in lines if x)


def _tidy_reasoning(text: str, limit: int = 1400) -> str:
    """思维链里若模型直接往里写 JSON，花括号铺一屏会显得很乱。

    保留自然语言段落，把 JSON 片段丢掉后补一句「第 N 步 · 标题」的摘要；
    整段都是 JSON 时退回 _content_digest，至少让用户看到推进到第几步了。
    """
    tail = str(text or "")[-limit:]
    lines: list[str] = []
    for raw in tail.split("\n"):
        s = raw.strip()
        if not s:
            continue
        if '"title"' in s or '"content"' in s or '"theorem"' in s:
            continue
        if s.startswith(("{", "}", "[", "]", "```")):
            continue
        if s.count('"') >= 4:
            continue
        lines.append(s)
    if len(lines) < 2:
        return _content_digest(tail) or tail
    return "\n".join(lines[-10:])


def _render_thinking(trace: list[dict], elapsed: float | None = None,
                     reasoning: str = "", content: str = "", live: bool = False) -> None:
    """思考过程卡片：阶段时间线 + 流式文本尾巴。

    live=True 时用于解题中的实时面板；跑完后用同一个渲染器放进报告里，
    保证「过程」和「结果」是同一套叙事，而不是两张不相干的图。
    """
    total = f"{elapsed:.1f}s" if elapsed else ""
    head_right = total or ("进行中" if live else "")
    rows = ""
    for i, ev in enumerate(trace):
        node = ev.get("node", "")
        msg = ev.get("msg", "")
        ms = ev.get("ms")
        last = i == len(trace) - 1
        state = "active" if (live and last) else "done"
        ms_s = f"{ms:.1f}s" if isinstance(ms, (int, float)) else ""
        rows += (
            f'<div class="stage {state}"><span class="stage-dot"></span>'
            f'<span><span class="stage-name">{html.escape(NODE_LABEL.get(node, node))}</span>'
            f'<span class="stage-msg">{html.escape(msg)}</span></span>'
            f'<span class="stage-ms">{ms_s}</span></div>'
        )
    if not rows:
        rows = '<div class="stage active"><span class="stage-dot"></span><span class="stage-name">启动中</span></div>'

    stream = ""
    label = ""
    if reasoning:
        label = "模型思维链"
        stream = re.sub(r"\n{3,}", "\n\n", _tidy_reasoning(reasoning))
    elif content:
        label = "正在书写推导"
        stream = _content_digest(content)

    card = f'<div class="think-card"><div class="think-head">'
    card += '<span class="dot-live"></span>' if live else '<span class="dot-done"></span>'
    card += f'<span class="think-title">思考过程</span><span class="think-time">{head_right}</span></div>'
    card += rows
    if stream:
        caret = '<span class="caret"></span>' if live else ""
        card += (f'<div class="stream-label">{label}</div>'
                 f'<div class="stream">{html.escape(stream)}{caret}</div>')
    card += "</div>"
    st.markdown(card, unsafe_allow_html=True)


def render_report(report: dict, elapsed: float | None = None, trace: list[dict] | None = None,
                  reasoning: str = "", content: str = "") -> None:
    if not report:
        st.error("未生成报告，请检查输入或模型配置。")
        return
    if report.get("degraded"):
        st.warning("本次运行发生了降级（预算熔断或模型输出异常），结论仅供参考。")

    v = report.get("verification", {}) or {}
    v_kind, v_text = STATUS_PILL.get(v.get("status", ""), ("mute", v.get("label", "未验证")))
    s = report.get("structure", {}) or {}
    steps = report.get("steps", []) or []
    idx = chunk_index()

    step_docs = []
    for i, step in enumerate(steps, 1):
        step_docs.append((i, step, _source_doc(step.get("source_id"), idx)))
    active = st.session_state.get("active_step") or (step_docs[0][0] if step_docs else 0)

    # 题解里复写过教材的式子 → 指回它在书里的位置。
    # 同一步的定理出处和公式出处常常落在同一个教材块上，那种情况只留定理卡，
    # 免得右栏摆两张内容重叠的卡片。
    fm_hits: list[tuple[int, dict]] = []
    for i, step, doc in step_docs:
        cid = (doc or {}).get("chunk_id")
        for hit in match_formulas(step):
            if cid and hit["chunk_id"] == cid:
                continue
            fm_hits.append((i, hit))
    fm_by_step: dict[int, list[dict]] = {}
    for i, hit in fm_hits:
        fm_by_step.setdefault(i, []).append(hit)

    # ======================= 题目 / 结论 ‖ 题面点名的依据 =======================
    # 题目字面点名的定理/公式回查到的教材块：与「题目」平行，不挂到某一步。
    qb = [cid for cid in (report.get("question_basis") or []) if idx.get(cid)]

    def _question_card() -> None:
        st.markdown('<div class="sec">题目</div>', unsafe_allow_html=True)
        with st.container(key="qtext"):
            st.markdown(md(report.get("question") or ""))
        st.markdown('<div class="sec" style="margin-top:1rem">结论</div>',
                    unsafe_allow_html=True)
        with st.container(key="conclusion"):
            st.markdown(md_lines(split_claims(report.get("conclusion") or "（无结论）")))
        badges = pill(f"符号验证 · {v_text}", v_kind)
        for k, val in [("题型", s.get("problem_type")), ("章节", s.get("chapter")),
                       ("知识点", s.get("topic")), ("难度", s.get("difficulty"))]:
            if val:
                badges += pill(f"{k} · {val}")
        st.markdown(f"<div style='margin-top:.55rem'>{badges}</div>", unsafe_allow_html=True)
        if s.get("conditions"):
            st.caption(md("条件：" + "；".join(s["conditions"])))
        if s.get("ambiguities"):
            st.caption(md("歧义点：" + "；".join(s["ambiguities"])))

    def _basis_card(doc: dict, slot: str, tag: str, compact: bool = False) -> None:
        # 教材原文卡：题面依据与每一步的依据共用同一套书页样式。
        # 题面依据用 compact：只留定理名 + 一句极短摘要 + 页码，避免把题目那一行
        # 撑得过高、在左侧题目卡下方留一大片空白；逐步依据保留完整原文供对照。
        with st.container(key=f"srcdoc-{slot}"):
            meta = (f"{doc.get('content_type') or '教材'} · p.{doc.get('page')}"
                    if doc.get("page") else (doc.get("content_type") or "教材"))
            st.markdown(
                f'<div class="src-head"><span class="src-no">{tag}</span>'
                f'<span class="src-topic">{html.escape(str(doc.get("topic") or "教材原文"))}</span></div>',
                unsafe_allow_html=True)
            # 题面依据只做「引用行」：定理名 + 页码 + 可点跳转，不铺正文，
            # 免得题目那一行被两张长卡撑出空白（原文在逐步依据里已完整出现）。
            body = doc.get("content") or ""
            if not compact:
                st.markdown(md(body))
            st.markdown(
                f'<div class="src-meta">{html.escape(str(doc.get("chapter") or ""))}　'
                f'{html.escape(str(meta))}</div>', unsafe_allow_html=True)
            st.button("在知识库中打开 ›", key=f"srcbtn-{slot}",
                      on_click=_open_source, args=(doc["chunk_id"],))

    def _formula_card(hit: dict, d: dict, slot: str, tag: str) -> None:
        # 教材公式卡：题解里复写的式子，指回它在书里的原始写法与页码。
        with st.container(key=f"fmref-{slot}"):
            st.markdown(
                f'<div class="src-head"><span class="src-no">{tag}</span>'
                f'<span class="src-topic">式 · {html.escape(_fm_title(d.get("topic")))}</span></div>',
                unsafe_allow_html=True)
            try:
                st.latex(_bare_latex(hit["tex"]))
            except Exception:
                st.code(hit["tex"], language="latex")
            meta = (f"{d.get('content_type') or '教材'} · p.{d.get('page')}"
                    if d.get("page") else (d.get("content_type") or "教材"))
            st.markdown(
                f'<div class="src-meta">{html.escape(str(d.get("chapter") or ""))}　'
                f'{html.escape(str(meta))}</div>', unsafe_allow_html=True)
            st.button("在知识库中打开 ›", key=f"fmbtn-{slot}",
                      on_click=_open_source, args=(hit["chunk_id"],))

    if qb:
        top_l, top_r = st.columns([1.85, 1], gap="large")
        with top_l:
            with st.container(border=True):
                _question_card()
        with top_r:
            with st.container(border=True):
                st.markdown('<div class="sec">题面依据</div>', unsafe_allow_html=True)
                for j, cid in enumerate(qb[:3], 1):
                    d = _source_doc(cid, idx)
                    if d:
                        _basis_card(d, f"q{j}-{cid[:10]}", f"{j:02d}", compact=True)
    else:
        with st.container(border=True):
            _question_card()

    # ======================= 推导过程：逐步「题解 ‖ 依据」平行 =======================
    st.markdown('<div class="sec" style="margin-top:1.5rem">推导过程</div>',
                unsafe_allow_html=True)
    if report.get("method"):
        st.markdown(md(f"**{report['method']}**"))
    if report.get("alternatives"):
        st.caption(md("备选：" + "、".join(report["alternatives"])))

    for i, step, doc in step_docs:
        # 每一行就是「这一步题解 ‖ 这一步依据」：左右顶部天然对齐，
        # 不用再靠点击高亮去找对应关系。窄屏时两列自动堆叠成上下紧邻。
        cl, cr = st.columns([1.85, 1], gap="large")
        with cl:
            with st.container(key=f"stepcard-{i}"):
                title = step.get("title") or f"第 {i} 步"
                thm = (step.get("theorem") or "").strip()
                head = f'<div class="step-head"><span class="stepno">{i}</span>'
                if thm:
                    head += f'<span class="thm-tag">{html.escape(thm)}</span>'
                fm = fm_by_step.get(i) or []
                for hit in fm[:1]:
                    page = (idx.get(hit["chunk_id"]) or {}).get("page")
                    if page:
                        head += f'<span class="fm-tag">p.{html.escape(str(page))}</span>'
                head += "</div>"
                st.markdown(head, unsafe_allow_html=True)
                with st.container(key=f"steptitle-{i}"):
                    st.markdown(md(title))
                body = step.get("content") or ""
                with st.container(key=f"stepbody-{i}"):
                    if body:
                        st.markdown(md_lines(body))
                    latex = _bare_latex(step.get("latex") or "")
                    if latex and _norm_tex(latex) not in _norm_tex(body):
                        try:
                            st.latex(latex)
                        except Exception:
                            st.code(latex, language="latex")
        with cr:
            cid = (doc or {}).get("chunk_id") or "x"
            if doc:
                _basis_card(doc, f"{i}-{cid[:10]}", f"{i:02d}")
            for j, hit in enumerate(fm[:2], 1):
                d = _source_doc(hit["chunk_id"], idx) or {}
                _formula_card(hit, d, f"{i}-fm{j}-{hit['chunk_id'][:10]}", f"{i:02d}")

    if not steps:
        st.caption("未产出推导步骤。")

    # —— 符号验证 + 独立审核 + 知识来源：三块并排，收在题解流下方
    assertions = report.get("review_assertions", []) or []
    issues = report.get("review_issues", []) or []
    g1, g2, g3 = st.columns([1, 1, 1], gap="large")
    with g1:
        with st.container(border=True):
            st.markdown('<div class="sec">符号验证</div>', unsafe_allow_html=True)
            st.markdown(f"**SymPy 复算**　{pill(v_text, v_kind)}", unsafe_allow_html=True)
            checked = v.get("checked") or []
            mismatches = v.get("mismatches") or []
            unchecked = v.get("unchecked") or []
            if checked or mismatches or unchecked:
                md_rows(checked, "ok")
                md_rows(mismatches, "bad")
                md_rows(unchecked, "warn")
            else:
                st.caption("本题没有可机器复算的步骤（证明题属正常情况）。")
            for note in v.get("notes", []):
                st.caption(md(str(note)))
    with g2:
        with st.container(border=True):
            st.markdown('<div class="sec">独立模型审核</div>', unsafe_allow_html=True)
            st.markdown(pill(f"{len(assertions)} 断言 · {len(issues)} 质疑", "blue"),
                        unsafe_allow_html=True)
            for a in assertions:
                st.markdown(md(f"- {a}"))
            for k, it in enumerate(issues):
                kind, label = SEV_PILL.get(it.get("severity", "warning"), ("warn", "警告"))
                step_no = it.get("step")
                where = f"第 {step_no} 步 · " if step_no not in (0, "0", None, "") else ""
                claim = str(it.get("claim") or "").strip()
                head = f"**{label}**　{where}{it.get('issue') or ''}"
                if claim:
                    head += f"\n\n*{claim}*"
                with st.container(key=f"vrow-{kind}-issue-{k}"):
                    st.markdown(md(head))
            if not assertions and not issues:
                st.caption("未获得审核输出。")
    with g3:
        with st.container(border=True):
            st.markdown('<div class="sec">知识来源</div>', unsafe_allow_html=True)
            src = report.get("sources", []) or []
            if src:
                for k, x in enumerate(src):
                    cid = x.get("chunk_id")
                    score = x.get("score")
                    has_score = isinstance(score, (int, float)) and score > 0
                    score_s = f"{score:.3f}" if has_score else ""
                    hit = idx.get(cid) or {}
                    # 例题不进知识来源列表，与依据栏同一口径
                    hct = str(hit.get("content_type") or "")
                    if hct in ("例", "例题", "习题") or re.match(
                            r"^\s*(例|习题)\s*\d", str(hit.get("topic") or "")):
                        continue
                    name = hit.get("topic") or x.get("citation") or "教材片段"
                    tip = f"{x.get('citation', '')}" + (f"｜相似度 {score_s}" if has_score else "")
                    if hit.get("content_type") or hit.get("page"):
                        tip += f"｜{hit.get('content_type', '')} p.{hit.get('page', '')}"
                    if cid:
                        st.button(f"{name} · {score_s}" if has_score else name, key=f"src-{k}",
                                  help=tip + "\n点击查看教材原文",
                                  use_container_width=True, on_click=_open_source, args=(cid,))
                    else:
                        st.caption(f"{name} · {score_s}" if has_score else name)
            else:
                st.caption("无引用来源。")

            lim = report.get("limitations", []) or []
            st.markdown('<div class="sec" style="margin-top:1rem">局限与待核查</div>',
                        unsafe_allow_html=True)
            if lim:
                md_rows(lim, "warn")
            else:
                st.caption("无特别说明。")

    # —— 思考过程（完成后回放）
    if trace or reasoning:
        with st.expander(f"思考过程 · {len(trace)} 个阶段"
                         + (f" · {elapsed:.1f}s" if elapsed else ""), expanded=False):
            _render_thinking(trace or [], elapsed, reasoning, content)

    b = report.get("budget", {}) or {}
    if b:
        bits = [f"调用 {b.get('calls', '-')} 次", f"{b.get('tokens', '-')} tokens",
                f"¥{b.get('cost', '-')}"]
        if elapsed:
            bits.insert(0, f"耗时 {elapsed:.1f}s")
        st.caption(" · ".join(str(x) for x in bits))


SAMPLES = [
    ("极限计算", "求极限 lim_{x->0} (e^x - 1 - x)/x^2"),
    ("一致收敛", "用 Weierstrass 判别法证明：级数 sum_{n=1}^{oo} sin(nx)/n^2 在 R 上一致收敛"),
    ("定积分", "计算积分 ∫_0^1 x e^x dx"),
]

LAST_RESULT_FILE = ROOT / "outputs" / "last_result.json"


def remember_result(report: dict, elapsed: float | None, trace=None,
                    reasoning: str = "", content: str = "") -> None:
    """结果同时留在 session 和磁盘：浏览器刷新后仍能恢复。"""
    payload = {"report": report, "elapsed": elapsed,
               "trace": trace or [], "reasoning": reasoning, "content": content}
    st.session_state["last_result"] = payload
    try:
        LAST_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
        # 思考过程可能上万字，落盘留个尾巴就够回放
        slim = {**payload, "reasoning": (reasoning or "")[-6000:],
                "content": (content or "")[-6000:]}
        LAST_RESULT_FILE.write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def restore_result() -> dict | None:
    if "last_result" in st.session_state:
        return st.session_state["last_result"]
    if not LAST_RESULT_FILE.exists():
        return None
    try:
        data = json.loads(LAST_RESULT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("report"), dict):
        return None
    st.session_state["last_result"] = data
    return data


# ---------------------------------------------------------------- 后台任务
# 执行器必须是模块级单例：写在函数里用 `with ThreadPoolExecutor(...)` 的话，
# with 退出会 join，用户切页签时新脚本会卡在 join 上，页面像「进程没了」。
_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="math-agent")
_ORDER = ["parse", "analyze", "plan", "solve", "retrieve", "verify", "review", "format"]


def _start_job(question: str, retriever, auto_retry: bool, skip_review: bool = False,
               cache_key: str | None = None) -> None:
    log: list[dict] = []
    lock = threading.Lock()
    t0 = time.time()
    think = {"reasoning": "", "content": ""}
    last_ts = {"t": t0}

    def emit(node: str, message: str, data: dict | None = None) -> None:
        # 流式增量走这里：不写进时间线（否则每秒几十行），只累积到 think 缓冲。
        if data and data.get("delta"):
            with lock:
                key = "reasoning" if data["delta"] == "reasoning" else "content"
                if data["delta"] == "reset":
                    think["reasoning"] = ""
                    think["content"] = ""
                else:
                    buf = think[key] + str(data.get("text") or "")
                    think[key] = buf[-6000:]  # 只留尾巴，够界面滚动展示
            return
        now = time.time()
        with lock:
            ms = (now - last_ts["t"]) * 1000
            last_ts["t"] = now
            log.append({"node": node, "msg": message, "ms": ms / 1000, "t": now - t0})

    # 声明本回调支持流式增量：agent 侧据此决定是否开 stream（脚本入口保持整段打印）
    emit.supports_delta = True  # type: ignore[attr-defined]

    pipeline = MathAgentPipeline(emit=emit, retriever=retriever)
    if auto_retry:
        pipeline.settings.app.max_retries = 1
    future = _POOL.submit(pipeline.run, question, "text", None, None, skip_review)
    st.session_state["job"] = {
        "q": question.strip(), "key": cache_key or question.strip(),
        "future": future, "log": log, "lock": lock,
        "think": think, "t0": t0, "harvested": False,
    }


def _job_view(job: dict) -> tuple[list[dict], float, str, str]:
    with job["lock"]:
        trace = list(job["log"])
        reasoning = job["think"]["reasoning"]
        content = job["think"]["content"]
    return trace, time.time() - job["t0"], reasoning, content


def _progress_panel() -> None:
    if st.session_state.get("job"):
        _progress_tick()


@st.fragment(run_every=1.0)
def _progress_tick() -> None:
    job = st.session_state.get("job")
    if not job:
        return
    if job["future"].done():
        harvest_job()
        st.rerun(scope="app")
        return
    trace, elapsed, reasoning, content = _job_view(job)
    # 不再画独立进度条：阶段时间线本身就带「到哪一步了」，多一根横线只会让页面显得躁。
    _render_thinking(trace, elapsed, reasoning, content, live=True)


def harvest_job() -> bool:
    """后台任务跑完后收结果；任意页面调用都安全。"""
    job = st.session_state.get("job")
    if not job or job.get("harvested"):
        return False
    future = job["future"]
    if not future.done():
        return False
    job["harvested"] = True
    st.session_state.pop("job", None)
    exc = future.exception()
    if exc is not None:
        st.session_state["job_error"] = f"{type(exc).__name__}: {exc}"
        st.session_state["job_traceback"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__))
        return False
    result = future.result()
    elapsed = time.time() - job["t0"]
    trace, _, reasoning, content = _job_view(job)
    st.session_state.setdefault("cache", {})[job.get("key") or job["q"]] = {
        "report": result.report, "elapsed": elapsed,
        "trace": trace, "reasoning": reasoning, "content": content,
    }
    remember_result(result.report, elapsed, trace, reasoning, content)
    try:
        save_record(job["q"], result.report, result.budget)
    except Exception:
        pass
    return True


def _report_job_error() -> None:
    err = st.session_state.pop("job_error", None)
    if not err:
        return
    st.error(f"本次求解中断：{err}")
    tb = st.session_state.pop("job_traceback", None)
    if tb:
        with st.expander("查看错误详情"):
            st.code(tb)


# ---------------------------------------------------------------- 工作台
def _photo_input() -> None:
    """题目照片 → LaTeX 文本（按图片哈希缓存，避免每次 rerun 重复识别）。"""
    settings = get_settings()
    vis = settings.models.vision

    with st.expander("用题目照片输入", expanded=False):
        if not vis.model or not vis.api_key:
            st.warning("视觉模型未配置，无法识别图片。请在 configs/model.yaml 的 vision 段填写模型，"
                       "并在 .env 里配置 API Key。")
            return
        up = st.file_uploader("题目照片", type=["png", "jpg", "jpeg", "webp", "bmp"],
                              label_visibility="collapsed", key="qimg")
        if up is not None:
            raw = up.getvalue()
            digest = hashlib.md5(raw, usedforsecurity=False).hexdigest()[:16]
            if st.session_state.get("ocr_digest") != digest:
                with st.spinner("正在识别题目…"):
                    try:
                        res = recognize_question(raw)
                        st.session_state["ocr_result"] = {
                            "text": res.text, "model": res.model, "elapsed": res.elapsed,
                            "size": res.size, "warning": res.warning, "image": raw,
                            "relayed": res.relayed,
                        }
                        st.session_state["ocr_error"] = ""
                    except OcrError as exc:
                        st.session_state["ocr_result"] = None
                        st.session_state["ocr_error"] = str(exc)
                    except Exception as exc:  # noqa: BLE001
                        st.session_state["ocr_result"] = None
                        st.session_state["ocr_error"] = f"{type(exc).__name__}: {exc}"
                    st.session_state["ocr_digest"] = digest

        err = st.session_state.get("ocr_error")
        if err:
            st.error(f"识别失败：{err}")
            st.caption("可以换个角度重拍、把题目区域裁出来再试，或者直接手打。")

        res = st.session_state.get("ocr_result")
        if not res:
            return
        text = (res.get("text") or "").strip()
        c_img, c_txt = st.columns([1, 2.1], gap="medium")
        with c_img:
            if res.get("image"):
                st.image(res["image"], use_container_width=True)
        with c_txt:
            st.markdown('<div class="sec">识别结果</div>', unsafe_allow_html=True)
            st.markdown(md(re.sub(r"(?<!\n)\n(?!\n)", "\n\n", text)))
        meta = f"{res.get('model', '')} · {res.get('elapsed', 0):.1f}s"
        if res.get("size"):
            w, h = res["size"]
            meta += f" · {w}×{h}"
        if res.get("relayed"):
            meta += " · 图片经临时图床中转（1 小时后自动失效）"
        st.caption(meta)
        if res.get("warning"):
            st.warning(res["warning"])

        def _fill(value: str) -> None:
            st.session_state["ocr_pending"] = value

        current = (st.session_state.get("q") or "").strip()
        if not current:
            _fill(text)
            st.rerun()
        st.button("用作题目（覆盖当前内容）", key="ocr_fill", on_click=_fill, args=(text,))


def _hero(compact: bool = False) -> None:
    # 只有一行功能名。原先的英文眉标、宣传句、四格数据条都砍了：
    # 那些字只是在替布局解释「这个页面是干什么的」，输入框摆在哪里本来就说明了一切。
    cls = "hero-wrap compact" if compact else "hero-wrap solo"
    st.markdown(f'<div class="{cls}"><h1 class="hero">解题工作台</h1></div>',
                unsafe_allow_html=True)


def page_workbench() -> None:
    settings = get_settings()
    retriever = get_retriever()

    last = st.session_state.get("last_result")

    st.session_state.setdefault("q", "")
    # 图片识别/示例按钮都先写 pending，再在 text_area 实例化之前灌进去。
    # 一旦 widget 建立，再赋值就会抛 StreamlitWidgetAlreadyInstantiatedError。
    pending_q = st.session_state.pop("ocr_pending", None)
    if pending_q:
        st.session_state["q"] = pending_q
        st.session_state["last_question"] = pending_q
    elif not st.session_state.get("q") and st.session_state.get("last_question"):
        # 切页签时 Streamlit 会清理未渲染 widget 的状态，用独立键兜住题目
        st.session_state["q"] = st.session_state["last_question"]

    # 页头收成小字只看「框里有没有题」，跟磁盘上有没有旧结果无关
    has_q = bool(st.session_state.get("q", "").strip())
    _hero(compact=has_q)

    with st.container(border=True):
        short = has_q
        lines = st.session_state.get("q", "").count("\n") + 1
        auto_h = min(max(84, 23 * lines + 26), 260)
        if not st.session_state.get("q"):
            # 空态给一块像样的书写区：没有标语和数据条，输入框本身就是首屏主角
            auto_h = 132
        question = st.text_area(
            "题目", key="q",
            height=min(auto_h, 120) if short else auto_h,
            placeholder="输入题目",
            label_visibility="collapsed",
        )
        # key="controls"：小屏时靠它把「按钮整行 + 两个开关并排」摆成网格
        with st.container(key="controls"):
            c1, c2, c3 = st.columns([1.45, 1, 1], gap="small")
            with c1:
                run = st.button("开始解题", type="primary", use_container_width=True)
            with c2:
                speed = st.toggle("极速模式", value=False,
                                  help="跳过交叉审核，约 25s 出结果；SymPy 复算与教材依据照常保留。")
            with c3:
                auto_retry = st.toggle("质疑后重做", value=False,
                                       help="审核提出质疑时回炉重解，约 40s，默认关闭。")

        # 示例题挂在「开始解题」正下方：按钮决定动作，胶囊决定拿什么去试，
        # 两者是同一件事的上下半段，不该被一条页脚分隔线隔开。
        def _use_sample(text: str) -> None:
            st.session_state.q = text
            st.session_state["last_question"] = text

        if not question.strip():
            with st.container(key="samples"):
                cols = st.columns([1, 1, 1, 2.6], gap="small")
                for col, (label, text) in zip(cols, SAMPLES):
                    col.button(label, use_container_width=False, key=f"sample_{label}",
                               on_click=_use_sample, args=(text,))

        # 拍照入口留在卡片最后一行，弱化成一条灰字
        with st.container(key="inputfoot"):
            _photo_input()

    if question.strip():
        st.session_state["last_question"] = question.strip()

    if not question.strip():
        return

    # 缓存键带上模式：极速结果与标准结果不能互相命中（审核字段不一样）
    cache_key = question.strip() + ("|fast" if speed else "")
    cache = st.session_state.setdefault("cache", {})
    cached = cache.get(cache_key)
    # 磁盘上的旧结果只在「题目一字不差」时复用，且不加任何标题/说明——
    # 它就是这个输入框当前题目的结果，不是一份叫「上次结果」的存档。
    restored = None
    if last and (last.get("report", {}).get("question") or "").strip() == question.strip():
        restored = last

    job = st.session_state.get("job")
    if job:
        _progress_panel()
        if cached:
            render_report(cached["report"], cached.get("elapsed"), cached.get("trace"),
                          cached.get("reasoning", ""), cached.get("content", ""))
        return

    if not run:
        if cached:
            render_report(cached["report"], cached.get("elapsed"), cached.get("trace"),
                          cached.get("reasoning", ""), cached.get("content", ""))
        elif restored:
            render_report(restored["report"], restored.get("elapsed"), restored.get("trace"),
                          restored.get("reasoning", ""), restored.get("content", ""))
        return

    if cached and not auto_retry:
        render_report(cached["report"], cached.get("elapsed"), cached.get("trace"),
                      cached.get("reasoning", ""), cached.get("content", ""))
        return

    _start_job(question, retriever, auto_retry, skip_review=speed, cache_key=cache_key)
    _progress_panel()


# ---------------------------------------------------------------- 知识库
def _render_focus_card() -> None:
    cid = st.session_state.get("kb_focus")
    if not cid:
        return
    chunk = chunk_index().get(cid)
    if not chunk:
        st.session_state.pop("kb_focus", None)
        st.warning("没找到该知识片段（知识库可能已重建）。")
        return
    with st.container(key="kb-focus"):
        head = pill("来源定位", "blue")
        if chunk.get("chapter"):
            head += pill(chunk["chapter"])
        if chunk.get("content_type"):
            head += pill(chunk["content_type"])
        if chunk.get("page"):
            head += pill(f"p.{chunk['page']}", "mute")
        st.markdown(head, unsafe_allow_html=True)
        if chunk.get("topic"):
            st.markdown(md(f"**{chunk['topic']}**"))
        st.markdown(md(chunk.get("content") or ""))
        foot = " · ".join(x for x in (chunk.get("section"), f"`{chunk.get('chunk_id')}`",
                                      chunk.get("source", "")) if x)
        st.caption(foot)
        if st.button("收起定位", key="kbclose"):
            st.session_state.pop("kb_focus", None)
            st.rerun()


def page_knowledge() -> None:
    st.markdown('<div class="hero-wrap compact"><h1 class="hero">教材知识库</h1></div>',
                unsafe_allow_html=True)
    chunks = load_chunks()
    if not chunks:
        st.warning("尚未构建知识库，请先运行 scripts/build_from_ocr.py")
        return

    _render_focus_card()

    chapters = sorted({c.get("chapter", "") for c in chunks if c.get("chapter")})
    c1, c2 = st.columns([2.2, 1], gap="medium")
    with c1:
        q = st.text_input("搜索", placeholder="搜索定理或关键词",
                          label_visibility="collapsed")
    with c2:
        chap = st.selectbox("章节", ["全部章节"] + chapters, label_visibility="collapsed")

    pool = [c for c in chunks
            if (chap == "全部章节" or c.get("chapter") == chap)
            and (not q or q in (c.get("content") or "") or q in (c.get("topic") or ""))]

    st.caption(f"共 {len(pool)} 条结果")
    if not pool:
        return

    PER = 12
    total_pages = max(1, (len(pool) + PER - 1) // PER)
    key = f"kb_page_{hash((q, chap)) & 0xFFFFFF}"
    page = min(st.session_state.get(key, 1), total_pages)
    for c in pool[(page - 1) * PER: page * PER]:
        with st.container(border=True):
            head = pill(c.get("chapter") or "未标章节")
            if c.get("content_type"):
                head += pill(c["content_type"])
            if c.get("page"):
                head += pill(f"p.{c['page']}", "mute")
            st.markdown(head, unsafe_allow_html=True)
            if c.get("topic"):
                st.markdown(md(f"**{c['topic']}**"))
            with st.container(key=f"kbcard-{c.get('chunk_id') or 'x'}"):
                st.markdown(md(_snippet(c.get("content") or "")))
            with st.expander("查看全文"):
                st.markdown(md(c.get("content", "")))
                st.caption(f"`{c.get('chunk_id')}` · {c.get('source', '')}")

    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("上一页", disabled=page <= 1, use_container_width=True):
            st.session_state[key] = page - 1
            st.rerun()
    with nav2:
        st.markdown(
            f'<div style="text-align:center;color:var(--ink-2);font-size:12.5px;padding-top:.6rem">'
            f'第 {page} / {total_pages} 页</div>', unsafe_allow_html=True)
    with nav3:
        if st.button("下一页", disabled=page >= total_pages, use_container_width=True):
            st.session_state[key] = page + 1
            st.rerun()


# ---------------------------------------------------------------- 设置
def page_settings() -> None:
    st.markdown('<div class="hero-wrap compact"><h1 class="hero">设置与状态</h1></div>',
                unsafe_allow_html=True)
    settings = get_settings()
    retriever = get_retriever()
    status = retriever.status()

    with st.container(border=True):
        st.markdown('<div class="sec">模型分级</div>', unsafe_allow_html=True)
        vis = settings.models.vision
        # 只列角色与模型名。角色说明属于「替读者解释」，删掉后表格反而一眼能扫完。
        rows = "".join(
            f'<div class="srow"><span>{role}</span>'
            f'<b>{html.escape(spec.model or "未配置")}</b></div>'
            for role, spec in [
                ("解题", settings.models.solver),
                ("审核", settings.models.reviewer),
                ("轻量", settings.models.light),
                ("视觉", vis),
            ])
        st.markdown(rows, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="sec">检索通路</div>', unsafe_allow_html=True)
        vec_on = bool(status.get("vector"))
        rr_on = bool(status.get("rerank"))
        rows = (f'<div class="srow"><span>知识片段</span><b>{status.get("chunks", 0)}</b></div>'
                f'<div class="srow"><span>BM25 词项召回</span><b>{"开启" if status.get("bm25") else "关闭"}</b></div>'
                f'<div class="srow"><span>向量语义召回</span><b>{"开启" if vec_on else "关闭"}</b></div>'
                f'<div class="srow"><span>向量后端</span><b>{status.get("vector_backend") or "none"}</b></div>'
                f'<div class="srow"><span>融合方式</span><b>{status.get("fusion", "-")}</b></div>'
                f'<div class="srow"><span>精排（rerank）</span><b>{"开启" if rr_on else "未加载"}</b></div>'
                f'<div class="srow"><span>精排模型</span>'
                f'<b>{html.escape(status.get("rerank_model") or "—")}</b></div>')
        st.markdown(rows, unsafe_allow_html=True)
        if not vec_on and status.get("vector_reason"):
            st.caption(f"未启用原因：{status['vector_reason']}")

    with st.container(border=True):
        st.markdown('<div class="sec">耗时与预算</div>', unsafe_allow_html=True)
        rows = (f'<div class="srow"><span>快速模式（规划合并进解题）</span>'
                f'<b>{"开启" if settings.app.fast_mode else "关闭"}</b></div>'
                f'<div class="srow"><span>题目结构化</span>'
                f'<b>{"模型" if settings.app.analyze_llm else "本地规则（省 8~11s）"}</b></div>'
                f'<div class="srow"><span>单次请求超时</span><b>{settings.app.request_timeout}s</b></div>'
                f'<div class="srow"><span>审核质疑后自动重做</span><b>{settings.app.max_retries} 次</b></div>'
                f'<div class="srow"><span>Recall@5 目标</span>'
                f'<b>{settings.retrieval.recall_target:.0%}</b></div>')
        st.markdown(rows, unsafe_allow_html=True)


# ---------------------------------------------------------------- 主入口
def main() -> None:
    warmup_models()
    restore_result()
    harvest_job()
    _report_job_error()
    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="brand-mark">'
            '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"'
            ' stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M15.2 3.8c-2.1 0-3.2 1.2-3.2 3.3v9.8c0 2.1-1.1 3.3-3.2 3.3"/>'
            '<path d="M7.8 8.6h6.8"/></svg></span>'
            '<div><div class="brand-name">Math Agent</div>'
            '<div class="brand-sub">数学分析 · v2.3</div></div></div>',
            unsafe_allow_html=True)
        page = st.radio("导航", NAV_OPTIONS, label_visibility="collapsed", key="nav")
        st.write("")
        status = get_retriever().status()
        settings = get_settings()
        mock = all(not c.api_key for c in [settings.models.solver, settings.models.reviewer])
        vec_on = bool(status.get("vector"))
        rows = (f'<div class="srow"><span>知识片段</span><b>{status.get("chunks", 0)}</b></div>'
                f'<div class="srow"><span>向量召回</span>'
                f'<b>{"正常" if vec_on else "未启用"}</b></div>'
                f'<div class="srow"><span>模型</span>'
                f'<b>{"离线 mock" if mock or settings.mock_llm else "已连接"}</b></div>')
        st.markdown(rows, unsafe_allow_html=True)
        if not vec_on and status.get("vector_reason"):
            st.caption(f"向量召回未启用：{status['vector_reason'][:80]}")

    if mock or settings.mock_llm:
        st.warning("当前是离线 mock 模式：流程可跑通，但解答为占位内容。请检查 `.env` 中的 Key。")

    if page == "解题工作台":
        page_workbench()
    elif page == "知识库":
        page_knowledge()
    else:
        page_settings()


if __name__ == "__main__":
    main()
