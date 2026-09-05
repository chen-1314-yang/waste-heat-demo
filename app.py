# -*- coding: utf-8 -*-
"""工业余热回收智能决策演示平台（正式版 · 深色科技风）。

数据链路：CoolProp 物性仿真（ORC 5925 工况 + 蒸汽朗肯 1600 工况）→
sklearn 代理模型 → pymoo 多目标优化 → 动态 LCA 核算 → 两级智能决策。
数据口径：仿真/推算/示意/公开文献四类，详见《数据来源台账》。
"""
import json
import math
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# P3：两级决策统一内核（decision_core v2/P2 标定版）＋ 纯逻辑适配层
import decision_core as dc
import decision_v2_ui as v2u

st.set_page_config(page_title="工业余热回收智能决策演示平台", layout="wide",
                   initial_sidebar_state="auto")

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "data")
dc.set_data_dir(DATA)  # 内核读本地 orc/steam 仿真数据与 eval_weights（部署版同样适用）

# 排放因子口径（2021年度全国电网平均排放因子，生态环境部 2022-03 发布，
# 环办气候函〔2022〕111号；0.5810 不是华东区域因子，勿误标）
GRID_EF = 0.581          # tCO2/MWh
GRID_EF_LABEL = "2021年度全国电网平均排放因子 0.5810（生态环境部 2022-03 发布）"
HP_COP = 2.8             # 压缩式热泵演示 COP


def _mtime(fname):
    """数据文件修改时间：作为缓存键，数据更新后页面自动刷新（无需重启服务）。"""
    return os.path.getmtime(os.path.join(DATA, fname))


@st.cache_data
def load_sweep(_t=None):
    # 2026-08-13 审计 H5：网站统一读 orc_sweep_coolprop.csv
    # （原 data 目录 dwsim_sweep_full.csv 是 CoolProp 扫描换名，已归档至 data/_archived/；
    #  400 行旧 DWSIM 存档已改名 dwsim_sweep_400_deprecated.csv）
    return pd.read_csv(os.path.join(DATA, "orc_sweep_coolprop.csv"),
                       encoding="utf-8-sig")


@st.cache_data
def load_front_real(_t=None):
    return pd.read_csv(os.path.join(DATA, "pareto_front_real.csv"),
                       encoding="utf-8-sig")


@st.cache_data
def load_front_pymoo(_t=None):
    return pd.read_csv(os.path.join(DATA, "pymoo_pareto.csv"),
                       encoding="utf-8-sig")


@st.cache_data
def load_front_orc_verified(_t=None):
    return pd.read_csv(os.path.join(DATA, "front_orc_verified.csv"),
                       encoding="utf-8-sig")


@st.cache_data
def load_front_steam_verified(_t=None):
    return pd.read_csv(os.path.join(DATA, "front_steam_verified.csv"),
                       encoding="utf-8-sig")


@st.cache_data
def load_steam_sweep(_t=None):
    return pd.read_csv(os.path.join(DATA, "steam_sweep_coolprop.csv"),
                       encoding="utf-8-sig")


@st.cache_data
def load_steam_pareto(_t=None):
    return pd.read_csv(os.path.join(DATA, "steam_pareto.csv"),
                       encoding="utf-8-sig")


@st.cache_data
def load_verify(_t=None):
    return pd.read_csv(os.path.join(DATA, "dwsim_verify_pymoo.csv"),
                       encoding="utf-8-sig")


@st.cache_data
def load_lca_monthly(_t=None):
    return pd.read_csv(os.path.join(DATA, "lca_monthly_real.csv"),
                       encoding="utf-8-sig")


@st.cache_data
def load_weights(_t=None):
    with open(os.path.join(DATA, "eval_weights.json"), encoding="utf-8") as f:
        return json.load(f)


DEMAND_OPTIONS = ["发电", "工艺蒸汽", "供暖/热水", "储热调峰",
                  "供冷（制冷）", "干燥/烘干"]
FLOW_PROFILES = ["平稳波动", "班次阶跃", "随机游走"]
NAV_ITEMS = ["手动设定", "典型工况", "实时模拟"]


def load_conditions():
    """工况库（data/conditions_db.csv，可用 WPS/Excel 编辑）。

    每次读取（文件很小），编辑保存后刷新页面即可生效。
    字段约定见 README_使用说明.md；未知值回退到安全默认。
    """
    path = os.path.join(DATA, "conditions_db.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    if df.empty or "工况名称" not in df.columns:
        return pd.DataFrame()
    return df


COND_COLUMNS = [
    "工况名称", "行业", "热源类型", "热源温度_℃", "热源流量_kg_s",
    "波动模式", "波动幅度_℃", "波动速率", "年运行小时", "用能需求",
    "热源连续性", "数据来源", "说明",
]

COND_LIMITS = {
    "热源温度_℃": (40.0, 900.0),
    "热源流量_kg_s": (0.1, 200.0),
    "波动幅度_℃": (5.0, 150.0),
    "波动速率": (0.5, 3.0),
    "年运行小时": (4000, 8000),
}


def empty_conditions_template():
    return pd.DataFrame(
        [{c: ("" if c not in COND_LIMITS else 100.0) for c in COND_COLUMNS}])


def validate_conditions(df):
    """返回错误列表；空列表表示可以保存。"""
    errors = []
    if df is None or df.empty:
        return ["工况库不能为空，至少保留一行。"]
    missing = [c for c in COND_COLUMNS if c not in df.columns]
    if missing:
        return [f"缺少列：{', '.join(missing)}"]
    names = df["工况名称"].astype(str).str.strip()
    if names.isna().any() or (names == "").any() or (names == "nan").any():
        errors.append("存在空的“工况名称”。")
    if df["工况名称"].duplicated().any():
        dup = df.loc[df["工况名称"].duplicated(), "工况名称"].tolist()
        errors.append(f"“工况名称”重复：{', '.join(map(str, dup[:3]))}")
    for col, allowed in [("波动模式", FLOW_PROFILES),
                         ("用能需求", DEMAND_OPTIONS),
                         ("热源连续性", ["连续", "间歇"])]:
        bad = ~df[col].astype(str).isin(allowed)
        if bad.any():
            rows = df.index[bad].tolist()[:5]
            errors.append(f"“{col}”必须是 {allowed}，错误行：{rows}")
    for col, (lo, hi) in COND_LIMITS.items():
        num = pd.to_numeric(df[col], errors="coerce")
        bad = num.isna() | (num < lo) | (num > hi)
        if bad.any():
            rows = df.index[bad].tolist()[:5]
            errors.append(f"“{col}”需为 {lo}~{hi} 的数值，错误行：{rows}")
    return errors


def save_conditions(df):
    path = os.path.join(DATA, "conditions_db.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ---------------------------------------------------------------
# 深色科技风样式
# ---------------------------------------------------------------
st.markdown("""
<style>
:root {
  --bg: #0B1220; --card: #111C2E; --line: rgba(34,211,238,.16);
  --cyan: #22D3EE; --green: #34D399; --gold: #FBBF24;
  --text: #E2E8F0; --muted: #94A3B8;
  --radius-card: 14px; --radius-ctl: 8px;
}
html, body, [class*="css"], [data-testid="stAppViewContainer"] * {
  font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
}
.stApp {
  background:
    radial-gradient(1100px 520px at 18% -8%, rgba(34,211,238,.12), transparent 60%),
    radial-gradient(900px 460px at 92% 0%, rgba(52,211,153,.09), transparent 55%),
    var(--bg);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(13,22,38,.96), rgba(11,18,32,.96));
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] hr { border-color: var(--line); }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; }

.hero { padding: 6px 0 10px; }
.hero-title {
  font-size: 40px; font-weight: 800; letter-spacing: 1px; line-height: 1.2;
  background: linear-gradient(92deg, #67E8F9 0%, #34D399 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub { color: var(--muted); font-size: 15px; margin-top: 2px; }
.hero-badge {
  display: inline-block; font-size: 12px; font-weight: 700; color: #06121f;
  background: linear-gradient(90deg, var(--cyan), var(--green));
  border-radius: 999px; padding: 3px 12px; margin-right: 8px;
}

.sec-title {
  color: var(--text); font-size: 20px; font-weight: 700; margin: 10px 0 2px;
  display: flex; align-items: center; gap: 10px;
}
.sec-title .tag {
  font-size: 12px; font-weight: 700; color: #06121f;
  background: linear-gradient(90deg, var(--cyan), var(--green));
  border-radius: 6px; padding: 2px 9px;
}
.sec-note { color: var(--muted); font-size: 13px; margin-bottom: 6px; }

.side-sec {
  color: #7DD3FC; font-size: 12px; font-weight: 700; letter-spacing: 2px;
  margin: 10px 0 2px; text-transform: uppercase;
}
.rec-banner {
  margin: 8px 0 4px; padding: 12px 16px; border-radius: 12px;
  border: 1px solid rgba(52,211,153,.45);
  background: linear-gradient(92deg, rgba(52,211,153,.16), rgba(34,211,238,.10));
  color: #E2E8F0; font-size: 15px;
  box-shadow: 0 0 24px rgba(52,211,153,.12);
}
.rec-banner b { color: #6EE7B7; font-size: 18px; }
.rec-banner .muted { color: var(--muted); font-size: 13px; }
.footer {
  margin-top: 26px; padding-top: 14px;
  border-top: 1px solid var(--line); color: var(--muted); font-size: 12px;
  text-align: center; line-height: 1.8;
}
.mobile-swipe-hint {
  display: none;
}

[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(180deg, rgba(34,211,238,.05), rgba(17,28,46,.72));
  border: 1px solid var(--line) !important;
  border-radius: 14px;
  box-shadow: 0 8px 24px rgba(0,0,0,.25);
}
[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: rgba(34,211,238,.32) !important; }

[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(17,28,46,.92), rgba(11,18,32,.92));
  border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px;
  box-shadow: 0 6px 18px rgba(0,0,0,.22);
}
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-weight: 600; }
[data-testid="stMetricValue"] { color: var(--cyan) !important; font-weight: 800; }
[data-testid="stMetricDelta"] { font-weight: 600; }
[data-testid="stMetricValue"] { font-family: "Fira Code", Consolas, monospace; }

.stButton > button, .stDownloadButton > button {
  background: linear-gradient(92deg, #0EA5E9, #10B981);
  color: #06121f; font-weight: 700; border: none; border-radius: 8px;
  transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 18px rgba(34,211,238,.35);
}

[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
[data-testid="stCaptionContainer"] { color: var(--muted); }
div[data-testid="stInfo"] {
  background: rgba(34,211,238,.08); border: 1px solid rgba(34,211,238,.22);
  color: #A5F3FC; border-radius: 10px;
}
div[data-testid="stSuccess"] {
  background: rgba(52,211,153,.08); border: 1px solid rgba(52,211,153,.25);
  color: #A7F3D0; border-radius: 10px;
}
div[data-testid="stWarning"] {
  background: rgba(251,191,36,.08); border: 1px solid rgba(251,191,36,.25);
  color: #FDE68A; border-radius: 10px;
}
div[data-testid="stExpander"] {
  background: linear-gradient(180deg, rgba(17,28,46,.75), rgba(11,18,32,.75));
  border: 1px solid var(--line); border-radius: 12px;
}
div[data-testid="stExpander"] summary { color: var(--text); font-weight: 600; }

@keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"] {
  animation: fadeUp .5s ease both;
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}

/* ---- 布局密度优化（2026-08-13）---- */
[data-testid="stSidebar"] {
  width: 400px !important;
  min-width: 400px !important;
  background: linear-gradient(180deg, rgba(13,22,38,.97), rgba(11,18,32,.97));
}
[data-testid="stSidebarContent"] { padding: 1.3rem 1.4rem 2rem; }
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  font-size: 14px !important; color: #CBD5E1; font-weight: 600; letter-spacing: .2px;
}
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stSlider input { font-size: 15px !important; }
[data-testid="stSidebar"] .stSelectbox span,
[data-testid="stSidebar"] .stRadio label p { font-size: 14px !important; }
[data-testid="stSidebar"] .stToggle label { font-size: 14px !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  font-size: 12.5px !important; line-height: 1.55;
}
[data-testid="stSidebar"] .stExpander summary p { font-size: 13px !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] { margin-bottom: 6px; }

.block-container {
  max-width: 1380px; margin: 0 auto; padding: 1.4rem 2rem 3rem;
}
[data-testid="stMetric"] { padding: 16px 18px; }
[data-testid="stMetricValue"] { font-size: 22px !important; }
[data-testid="stMetricLabel"] p { font-size: 13px !important; }
.sec-title { font-size: 21px; margin: 14px 0 4px; }
[data-testid="stVerticalBlockBorderWrapper"] { padding: 4px 6px; }

/* ---- 顶部导航（多页面切换）---- */
.topnav-wrap { margin: 16px 0 8px; }
.nav-item {
  text-align: center; padding: 11px 0; border-radius: 10px;
  font-size: 15px; font-weight: 700; letter-spacing: 1px;
  border: 1px solid var(--line); color: var(--text);
  background: linear-gradient(180deg, rgba(17,28,46,.92), rgba(11,18,32,.92));
}
.nav-active {
  background: linear-gradient(92deg, #0EA5E9, #10B981) !important;
  color: #06121f !important; border-color: transparent;
  box-shadow: 0 4px 18px rgba(34,211,238,.30);
}
.topnav-wrap .stButton > button {
  background: linear-gradient(180deg, rgba(17,28,46,.92), rgba(11,18,32,.92));
  color: var(--text); border: 1px solid var(--line); font-weight: 700;
  letter-spacing: 1px;
}
.topnav-wrap .stButton > button:hover {
  background: linear-gradient(92deg, rgba(14,165,233,.25), rgba(16,185,129,.25));
  color: #A5F3FC; border-color: rgba(34,211,238,.45);
}

/* ---- 移动端适配（2026-09-05）：窄屏单列堆叠 + 可触摸横向滚动 ---- */
@media (max-width: 900px) {
  html, body {
    overflow-x: hidden;
  }
  .block-container {
    max-width: 100% !important;
    padding: 0.9rem 0.7rem 2.2rem !important;
  }
  .hero-title {
    font-size: 26px !important;
  }
  .hero-sub {
    font-size: 12.5px !important;
    line-height: 1.65;
  }
  .hero-badge {
    font-size: 11px !important;
    padding: 2px 9px !important;
    margin-right: 5px;
  }
  .topnav-wrap {
    display: flex;
    gap: 6px;
    flex-wrap: nowrap;
  }
  .topnav-wrap > div {
    flex: 1 1 0;
    min-width: 0;
  }
  .topnav-wrap .stButton > button {
    font-size: 13px !important;
    padding: 9px 2px !important;
    letter-spacing: 0 !important;
  }
  /* 侧栏：收起时按屏幕宽度适配，避免 400px 固定宽度溢出 */
  [data-testid="stSidebar"] {
    width: min(88vw, 360px) !important;
    min-width: min(88vw, 360px) !important;
  }
  [data-testid="stSidebarContent"] {
    padding: 1rem 1rem 2rem;
  }
  /* 多列一律堆叠为单列，杜绝挤压截断 */
  [data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
    row-gap: 0.6rem;
  }
  [data-testid="stHorizontalBlock"] > div {
    flex: 0 0 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
  }
  [data-testid="stMetric"] {
    padding: 12px 14px !important;
  }
  [data-testid="stMetricValue"] {
    font-size: 19px !important;
  }
  [data-testid="stMetricLabel"] p {
    font-size: 12px !important;
  }
  .sec-title {
    font-size: 17px !important;
  }
  .sec-note {
    font-size: 12px !important;
    line-height: 1.6;
  }
  /* 表格：允许触摸横向滚动，且不撑破页面 */
  [data-testid="stDataFrame"] {
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch;
    max-width: 100%;
  }
  [data-testid="stDataFrame"] > div {
    min-width: 0 !important;
  }
  /* Glide 数据网格滚动容器：确保横向触摸手势可被识别 */
  [data-testid="stDataFrame"] .dvn-scroller {
    touch-action: pan-x pan-y !important;
    -webkit-overflow-scrolling: touch;
    overscroll-behavior-x: contain;
    overflow-x: auto !important;
  }
  /* 手机上显示“表格可左右滑动”提示 */
  .mobile-swipe-hint {
    display: block;
    font-size: 11.5px;
    color: #7DD3FC;
    margin: 2px 0 6px;
    letter-spacing: .3px;
  }
  /* 单选/横向控件在小屏允许换行 */
  .stRadio [role="radiogroup"] {
    flex-wrap: wrap;
  }
  /* 图表容器宽度不超过视口 */
  [data-testid="stPlotlyChart"] {
    max-width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
  [data-testid="stPlotlyChart"] > div {
    width: 100% !important;
    min-width: 0 !important;
  }
  /* 侧栏开关按钮改造（手机“两页”切换）：
     左上角入口 = “场景参数”，侧栏内按钮 = “收起 · 查看结果” */
  [data-testid="stExpandSidebarButton"] {
    width: auto !important;
    min-width: 104px !important;
    height: 34px !important;
    margin: 0 !important;
    padding: 0 16px !important;
    border-radius: 999px !important;
    background: linear-gradient(92deg, #0EA5E9, #10B981) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 4px 14px rgba(34, 211, 238, .30);
  }
  [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {
    display: none !important;
  }
  [data-testid="stExpandSidebarButton"]::after {
    content: "场景参数";
    font-size: 14px;
    font-weight: 800;
    color: #06121F;
    letter-spacing: 2px;
    white-space: nowrap;
  }
  [data-testid="stSidebarCollapseButton"] {
    position: relative !important;
    width: auto !important;
    height: 30px !important;
    padding: 0 12px !important;
    border-radius: 999px !important;
    background: rgba(34, 211, 238, .14) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 4px !important;
  }
  [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {
    position: absolute !important;
    inset: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: transparent !important;
    font-size: 26px !important;
    z-index: 2 !important;
  }
  [data-testid="stSidebarCollapseButton"]::after {
    content: "收起 · 查看结果";
    font-size: 13px;
    font-weight: 700;
    color: #A5F3FC;
    letter-spacing: .5px;
    white-space: nowrap;
    position: relative;
    z-index: 1;
    pointer-events: none;
  }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 决策指标与权重映射（路径库/第一级规则/矩阵值统一来自 decision_core v2，
# 见 decision_v2_ui.py 与 11_决策内核v2_全温域/，此处不再维护第二套 RAW）
# ---------------------------------------------------------------
INDICATORS = ["能效%", "投资万元/MW", "回收期年", "CO2减排t/年", "政策分",
              "运行成本万元/MW·年"]
DIRECTIONS = ["max", "min", "min", "max", "max", "min"]
MAP_TO_TOPSIS = {
    "能效%": "系统能效", "投资万元/MW": "初始投资", "回收期年": "投资回收期",
    "CO2减排t/年": "CO2当量减排", "政策分": "政策补贴适配度",
}


def entropy_weights(X):
    """熵权法：基于当前候选矩阵动态计算客观权重（min-max 归一化 + 熵）。
    零方差列（无区分度）权重为 0，防止除零。
    注意：候选方案 ≤2 时，任意两方案在每列上只构成一个 0/1 区间，
    熵权会退化（信息量接近），此时 λ 敏感性主要体现主观权重的变化。"""
    n, m = X.shape
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    span = xmax - xmin
    # 方向处理：成本型指标取倒数方向（越小越好 → 越大越优）
    xr = np.zeros_like(X, dtype=float)
    for j in range(m):
        if span[j] < 1e-12:
            xr[:, j] = 1.0
        elif DIRECTIONS[j] == "max":
            xr[:, j] = (X[:, j] - xmin[j]) / span[j]
        else:
            xr[:, j] = (xmax[j] - X[:, j]) / span[j]
    p = (xr + 1e-10) / (xr.sum(axis=0) + 1e-10)
    e = -np.sum(p * np.log(p), axis=0) / np.log(n)
    w = (1.0 - e) / (1.0 - e).sum()
    return w


def combined_weights(lam, X=None):
    d = load_weights(_mtime("eval_weights.json"))
    names = d["指标"]
    # 主观部分：AHP 固定权重（演示假设），映射到 TOPSIS 6 指标并归一化
    lookup_sub = dict(zip(names, np.array(d["主观权重_AHP"], dtype=float)))
    w_sub6 = np.array([lookup_sub[MAP_TO_TOPSIS[c]] for c in INDICATORS[:5]]
                      + [0.25 * (lookup_sub["初始投资"] + lookup_sub["投资回收期"])])
    w_sub6 = w_sub6 / w_sub6.sum()
    if X is not None:
        w_obj6 = entropy_weights(X)          # 客观部分：熵权随当前候选矩阵动态计算
    else:
        lookup_obj = dict(zip(names, np.array(d["客观权重_熵权"], dtype=float)))
        w_obj6 = np.array([lookup_obj[MAP_TO_TOPSIS[c]] for c in INDICATORS[:5]]
                          + [0.25 * (lookup_obj["初始投资"] + lookup_obj["投资回收期"])])
        w_obj6 = w_obj6 / w_obj6.sum()
    w_all = lam * w_sub6 + (1 - lam) * w_obj6
    return w_all / w_all.sum()


def topsis(matrix, weights):
    # 单候选：贴近度按约定为 1（唯一可行路径即最优），避免页面显示 0.000
    if matrix.shape[0] == 1:
        return np.ones(1)
    # 防除零：全 0 / 零方差列（如减排列无区分度时）归一化不产生 nan
    norm = matrix / np.sqrt((matrix ** 2).sum(axis=0) + 1e-12)
    v = norm * weights
    ideal_pos = np.array([
        v[:, j].max() if DIRECTIONS[j] == "max" else v[:, j].min()
        for j in range(v.shape[1])])
    ideal_neg = np.array([
        v[:, j].min() if DIRECTIONS[j] == "max" else v[:, j].max()
        for j in range(v.shape[1])])
    d_pos = np.sqrt(((v - ideal_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((v - ideal_neg) ** 2).sum(axis=1))
    return d_neg / (d_pos + d_neg + 1e-12)


def run_v2_decision(t_src, m_dot, medium, demand, continuity, hours, dT,
                    driver):
    """调用统一内核（decision_v2_ui → decision_core）：主表与 λ 敏感性共用。"""
    scene = v2u.ui_scene(t_src, m_dot, medium, demand, continuity,
                         hours, dT, driver=driver)
    return v2u.run_decision(scene)


def _recovered_heat_kw(m_dot, medium, t_src, dT):
    """按热源流量/介质估算实际可回收热功率（与供热路径同一口径）。"""
    cp = {"热水/冷凝水": 4.2, "烟气": 1.1, "工艺液体": 2.5}[medium]
    dt = max(t_src - 40.0 - dT, 5.0)
    return m_dot * cp * dt


def orc_reduction(t_src, m_dot, medium, hours, dT):
    """ORC 发电减排：P50 统一取 decision_core 中位热效率（与决策矩阵同源）。"""
    sweep = load_sweep(_mtime("orc_sweep_coolprop.csv"))
    tmax_k = t_src + 273.15 - dT
    ok = sweep[sweep["heater_outlet_K"] <= tmax_k]
    if len(ok) == 0:
        return None
    effs = ok["thermal_eff"].values
    p10, p90 = np.percentile(effs, [10, 90])
    eff50_pct = dc.orc_eff_median_pct(t_src, dT)
    p50_kw_per_mw = eff50_pct * 10.0 if eff50_pct else np.percentile(effs * 1000.0, 50)
    q_kw = _recovered_heat_kw(m_dot, medium, t_src, dT)
    net_abs = p50_kw_per_mw * q_kw / 1000.0
    mwh = net_abs * hours / 1000.0
    co2 = mwh * GRID_EF
    money = mwh * 1000 * 0.65 / 10000.0
    return {"n_cond": len(ok), "p10": p10 * 1000.0, "p50": p50_kw_per_mw,
            "p90": p90 * 1000.0, "q_kw": q_kw, "net_abs_kW": net_abs,
            "mwh": mwh, "co2": co2, "money": money}


def steam_reduction(t_src, m_dot, medium, hours, dT):
    """高温蒸汽朗肯发电：P50 统一取 decision_core 插值效率（与决策矩阵同源）。"""
    sweep = load_steam_sweep(_mtime("steam_sweep_coolprop.csv"))
    # 热源温度 → 锅炉出口蒸汽温度：取 ~100℃ 端差，封顶 540℃（材料限制）
    t_boiler = min(max(t_src - 100.0, 180.0), 540.0) + 273.15
    ok = sweep[sweep["boiler_outlet_K"] <= t_boiler + 1e-6]
    if len(ok) == 0:
        return None
    effs = ok["thermal_eff"].values
    p10, p90 = np.percentile(effs, [10, 90])
    eff50_pct = dc.steam_eff_median_pct(t_src)
    p50_kw_per_mw = eff50_pct * 10.0 if eff50_pct else np.percentile(effs * 1000.0, 50)
    q_kw = _recovered_heat_kw(m_dot, medium, t_src, dT)
    net_abs = p50_kw_per_mw * q_kw / 1000.0
    mwh = net_abs * hours / 1000.0
    co2 = mwh * GRID_EF
    money = mwh * 1000 * 0.65 / 10000.0
    return {"n_cond": len(ok), "p10": p10 * 1000.0, "p50": p50_kw_per_mw,
            "p90": p90 * 1000.0, "q_kw": q_kw, "net_abs_kW": net_abs,
            "mwh": mwh, "co2": co2, "money": money}


def heat_reduction(t_src, m_dot, medium, hours, dT, cop=None):
    """供热路径减排：默认替代天然气口径；压缩式热泵（cop 给定）扣自身耗电排放。"""
    cp = {"热水/冷凝水": 4.2, "烟气": 1.1, "工艺液体": 2.5}[medium]
    t_out = 40.0
    dt = max(t_src - t_out - dT, 5.0)
    q_kw = m_dot * cp * dt
    heat_gj = q_kw * hours * 3.6 / 1000.0
    co2_replaced = heat_gj * 0.0561 / 0.90          # 替代天然气锅炉
    if cop:
        mwh_elec = heat_gj / 3.6 / cop              # 供热量折算耗电
        co2_elec = mwh_elec * GRID_EF               # 耗电排放
        co2 = max(co2_replaced - co2_elec, 0.0)     # 净减排
        money = (heat_gj * 98.0 - mwh_elec * 1000.0 * 0.65) / 10000.0
    else:
        mwh_elec, co2_elec = 0.0, 0.0
        co2 = co2_replaced
        money = heat_gj * 98.0 / 10000.0
    return {"q_kw": q_kw, "heat_gj": heat_gj, "co2": co2, "money": money,
            "co2_replaced": co2_replaced, "co2_elec": co2_elec,
            "mwh_elec": mwh_elec}


# ---------------------------------------------------------------
# 论文风图表（白底、细线、无网格）
# ---------------------------------------------------------------
CHART_FONT = dict(family="Microsoft YaHei, PingFang SC, sans-serif",
                  color="#E2E8F0", size=12)
GRID = "rgba(148,163,184,.18)"


def style_stage_table(df):
    """第一级筛选表：通过=绿、排除=红灰。"""
    def color_res(v):
        if v == "✓ 通过":
            return "background-color: rgba(52,211,153,.20); color: #A7F3D0; font-weight: 700;"
        return "background-color: rgba(239,68,68,.14); color: #FCA5A5;"
    return (df.style
              .map(color_res, subset=["结果"])
              .set_properties(**{"text-align": "left"})
              .set_table_styles([{
                  "selector": "th",
                  "props": [("background-color", "#16233B"),
                            ("color", "#7DD3FC"),
                            ("font-weight", "700"),
                            ("text-align", "left")]
              }]))


def style_topsis_table(df):
    """TOPSIS 表：第一名行绿色高亮。"""
    def row_style(r):
        if r.name == 0:
            return ["background-color: rgba(52,211,153,.14); font-weight: 700;"] * len(r)
        return [""] * len(r)
    return (df.style
              .apply(row_style, axis=1)
              .set_properties(**{"text-align": "left"})
              .set_table_styles([{
                  "selector": "th",
                  "props": [("background-color", "#16233B"),
                            ("color", "#7DD3FC"),
                            ("font-weight", "700"),
                            ("text-align", "left")]
              }]))


PATH_COLORS = {
    v2u.LABELS["direct"]: "#22D3EE",
    v2u.LABELS["whb_steam"]: "#38BDF8",
    v2u.LABELS["abs_self"]: "#34D399",
    v2u.LABELS["abs_ext"]: "#A78BFA",
    v2u.LABELS["comp"]: "#2DD4BF",
    v2u.LABELS["orc"]: "#4ADE80",
    v2u.LABELS["steam_pp"]: "#FB923C",
    v2u.LABELS["tc_storage"]: "#A3E635",
    v2u.LABELS["pcm_storage"]: "#FBBF24",
    v2u.LABELS["teg"]: "#F472B6",
    v2u.LABELS["abs_cool"]: "#38BDF8",
    v2u.LABELS["comp_cool"]: "#818CF8",
}


def style_lambda_table(df):
    """λ 敏感性表：第一名列按路径着色。"""
    def cell_color(v):
        c = PATH_COLORS.get(v, "#64748B")
        return f"background-color: {c}26; color: #E2E8F0; font-weight: 700;"
    return (df.style
              .map(cell_color, subset=["第一名"])
              .set_properties(**{"text-align": "left"})
              .set_table_styles([{
                  "selector": "th",
                  "props": [("background-color", "#16233B"),
                            ("color", "#7DD3FC"),
                            ("font-weight", "700"),
                            ("text-align", "left")]
              }]))


def lambda_figure(survivors, X_lam):
    """贴近度随 λ 的变化曲线：权重影响直观可见（即使第一名不变，曲线也在移动）。"""
    lam_grid = np.linspace(0.0, 1.0, 21)
    c_all = np.zeros((len(survivors), len(lam_grid)))
    for k, lam_t in enumerate(lam_grid):
        w = combined_weights(float(lam_t), X_lam)
        c_all[:, k] = topsis(X_lam, w)
    fig = go.Figure()
    for i, p in enumerate(survivors):
        fig.add_trace(go.Scatter(
            x=lam_grid, y=c_all[i], mode="lines+markers",
            name=p, line=dict(color=PATH_COLORS.get(p, "#64748B"), width=2.5),
            marker=dict(size=5),
            hovertemplate=f"{p}<br>λ=%{{x:.2f}} 贴近度=%{{y:.3f}}<extra></extra>"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,28,46,.55)", height=420, font=CHART_FONT,
        xaxis=dict(showgrid=False, zeroline=False,
                   title="λ（0=纯熵权，1=纯AHP）", tickmode="linear", dtick=0.1),
        yaxis=dict(showgrid=False, zeroline=False, title="TOPSIS 贴近度",
                   range=[0, 1.05]),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)),
        margin=dict(l=40, r=20, t=40, b=40))
    return fig


def _current_work_point(sweep, t_k, cost_fn, temp_col):
    """按当前热源温度折算蒸发/锅炉出口温度，取可行点热效率/成本的中位数（示意估算）。
    2026-08-13 审计 H4：原"最近单行"对数据稀疏区噪声大，改为可行集 P50。"""
    ok = sweep[sweep[temp_col] <= float(t_k)]
    if len(ok) == 0:
        return None
    eff_med = float(np.median(ok["thermal_eff"].values))
    net_mw = eff_med * 1000.0
    costs = ok.apply(cost_fn, axis=1).values
    cost = float(np.median(costs))
    return net_mw, cost, eff_med


def current_orc_point(t_src, dT):
    sweep = load_sweep(_mtime("orc_sweep_coolprop.csv"))

    def cost_fn(row):
        return (0.12 * 1000.0
                + 30.0 * (float(row["pump_outlet_Pa"]) / 1e6) ** 2
                + 120.0 * (1.0 - float(row["expander_eff"])))

    return _current_work_point(sweep, t_src + 273.15 - float(dT),
                               cost_fn, "heater_outlet_K")


def current_steam_point(t_src, dT):
    sweep = load_steam_sweep(_mtime("steam_sweep_coolprop.csv"))

    def cost_fn(row):
        return (0.12 * 1000.0
                + 80.0 * (float(row["boiler_Pa"]) / 1e6)
                + 120.0 * (1.0 - float(row["expander_eff"])))

    return _current_work_point(sweep, t_src + 273.15 - float(dT),
                               cost_fn, "boiler_outlet_K")


def pareto_mask(x, y, x_max=True, y_min=True):
    """返回非支配掩码：x 越大越好（默认），y 越小越好（默认）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x_better = (x[j] > x[i]) if x_max else (x[j] < x[i])
            y_better = (y[j] < y[i]) if y_min else (y[j] > y[i])
            if (x[j] >= x[i]) == x_max and (y[j] <= y[i]) == y_min and (x_better or y_better):
                mask[i] = False
                break
    return mask


def pareto_figure():
    """ORC 帕累托前沿：每 MW 回收热净功率 — 设备成本代理（与申报口径一致）。"""
    front = load_front_pymoo(_mtime("pymoo_pareto.csv"))
    real = load_front_real(_mtime("pareto_front_real.csv"))
    real = real.assign(cost_proxy=0.12 * real["Q_in_kW"]
                       + 30.0 * (real["pump_outlet_Pa"] / 1e6) ** 2
                       + 120.0 * (1.0 - real["expander_eff"]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=front["cost_proxy"], y=front["net_kW"], mode="markers",
        marker=dict(size=8, color=front["thermal_eff"],
                    colorscale=[[0, "#0EA5E9"], [0.5, "#22D3EE"], [1, "#34D399"]],
                    opacity=0.35, line=dict(width=0, color="#0B1220"),
                    colorbar=dict(title="热效率", thickness=14,
                                  tickfont=dict(color="#94A3B8", size=11))),
        name="pymoo 100 解（代理预测）",
        showlegend=True,
        hovertemplate="成本代理 %{x:.0f} 万元<br>净功率 %{y:.1f} kW/MW热<br>热效率 %{marker.color:.3f}<extra></extra>"))
    vf = load_front_orc_verified(_mtime("front_orc_verified.csv"))
    pf = vf.sort_values("cost_proxy")
    fig.add_trace(go.Scatter(
        x=pf["cost_proxy"], y=pf["net_kW"], mode="lines+markers",
        line=dict(color="#22D3EE", width=1.8),
        marker=dict(size=7, color="#22D3EE",
                    line=dict(color="#0B1220", width=1)),
        opacity=0.95,
        name=f"精确核验前沿（{len(pf)} 解）",
        hovertemplate="精确核验前沿<br>成本代理 %{x:.0f} 万元"
                      "<br>净功率 %{y:.1f} kW/MW热<extra></extra>"))
    real_nd = real.sort_values("cost_proxy")
    fig.add_trace(go.Scatter(
        x=real_nd["cost_proxy"], y=real_nd["net_kW"], mode="markers",
        marker=dict(size=10, color="#56B4E9",
                    line=dict(color="#0B1220", width=1)),
        name=f"CoolProp 复核抽样（{len(real_nd)} 解，已剔除被支配点）",
        hovertemplate="复核抽样<br>成本代理 %{x:.0f} 万元"
                      "<br>净功率 %{y:.1f} kW/MW热<extra></extra>"))
    best = real_nd.loc[real_nd["net_kW"].idxmax()]
    fig.add_trace(go.Scatter(
        x=[best["cost_proxy"]], y=[best["net_kW"]], mode="markers",
        marker=dict(symbol="star", size=15, color="#FBBF24",
                    line=dict(color="#0B1220", width=1)),
        name=f"最优点 {best['net_kW']:.1f} kW/MW热",
        hovertemplate="最优点<br>净功率 %{y:.1f} kW/MW热<extra></extra>"))
    current = None
    try:
        if demand == "发电" and t_src >= 110.0:
            current = current_orc_point(t_src, dT)
    except Exception:
        current = None
    if current is not None:
        fig.add_trace(go.Scatter(
            x=[current[1]], y=[current[0]], mode="markers+text",
            marker=dict(symbol="diamond", size=13, color="#F472B6",
                        line=dict(color="#0B1220", width=1)),
            text=[f"{current[0]:.0f}"], textposition="top center",
            textfont=dict(color="#F472B6", size=12),
            name="当前工况估算（示意）",
            hovertemplate=("当前工况估算<br>净功率 %{y:.1f} kW/MW热"
                           "<br>成本代理 %{x:.0f} 万元<extra></extra>")))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,28,46,.55)", height=470, font=CHART_FONT,
        xaxis=dict(showgrid=False, zeroline=False,
                   title="设备成本代理（万元，示意）"),
        yaxis=dict(showgrid=False, zeroline=False,
                   title="净功率（kW/MW 回收热）"),
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12)),
        margin=dict(l=40, r=20, t=30, b=40))
    return fig


def pareto_co2_figure():
    """碳减排—成本权衡图（精确核验前沿线 + 复核抽样点）。
    减碳 = 净功率×8000h×0.581（2021年度全国电网平均排放因子，推算）；成本为示意代理模型。"""
    real = load_front_real(_mtime("pareto_front_real.csv"))
    real = real.assign(co2=real["net_kW"] * 8000.0 / 1000.0 * GRID_EF)
    real = real.assign(cost=0.12 * real["Q_in_kW"]
                        + 30.0 * (real["pump_outlet_Pa"] / 1e6) ** 2
                        + 120.0 * (1.0 - real["expander_eff"]))
    fig = go.Figure()
    vf = load_front_orc_verified(_mtime("front_orc_verified.csv"))
    vf = vf.assign(co2=vf["net_kW"] * 8000.0 / 1000.0 * GRID_EF)
    pf = vf.sort_values("cost_proxy")
    fig.add_trace(go.Scatter(
        x=pf["cost_proxy"], y=pf["co2"], mode="lines+markers",
        line=dict(color="#56B4E9", width=1.8),
        marker=dict(size=6, color="#56B4E9",
                    line=dict(color="#0B1220", width=1)),
        opacity=0.95, name=f"精确核验前沿（{len(pf)} 解）",
        hovertemplate="成本代理 %{x:.0f} 万元<br>年碳减排 %{y:.0f} tCO2<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=real["cost"], y=real["co2"], mode="markers",
        marker=dict(size=9, color="#56B4E9",
                    line=dict(color="#0B1220", width=1)),
        name=f"CoolProp 复核抽样（{len(real)} 解，已剔除被支配点）",
        hovertemplate="成本代理 %{x:.0f} 万元<br>年碳减排 %{y:.0f} tCO2<br>净功率 %{customdata[0]:.1f} kW<extra></extra>",
        customdata=real[["net_kW"]].values))
    best = real.loc[real["net_kW"].idxmax()]
    fig.add_trace(go.Scatter(
        x=[best["cost"]], y=[best["co2"]], mode="markers",
        marker=dict(symbol="star", size=14, color="#FBBF24",
                    line=dict(color="#0B1220", width=1)),
        name=f"最优点 {best['net_kW']:.1f} kW",
        hovertemplate="最优点<br>净功率 %{customdata[0]:.1f} kW<extra></extra>",
        customdata=[[best["net_kW"]]]))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,28,46,.55)", height=470, font=CHART_FONT,
        xaxis=dict(showgrid=False, zeroline=False,
                   title="设备成本代理（万元，示意）"),
        yaxis=dict(showgrid=False, zeroline=False,
                   title="年碳减排（tCO2/年，推算口径）"),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)),
        margin=dict(l=40, r=20, t=30, b=40))
    return fig


def steam_pareto_figure():
    """高温蒸汽朗肯帕累托前沿（每 MW 回收热口径，成本代理示意）。"""
    sp = load_steam_pareto(_mtime("steam_pareto.csv"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sp["cost_proxy"], y=sp["net_kW"], mode="markers",
        marker=dict(size=8, color="#FBBF24", opacity=0.35,
                    line=dict(width=0, color="#0B1220")),
        name="蒸汽朗肯 100 解（CoolProp 复核搜索解）",
        hovertemplate="成本代理 %{x:.0f} 万元<br>净功率 %{y:.1f} kW/MW热<extra></extra>"))
    vf = load_front_steam_verified(_mtime("front_steam_verified.csv"))
    pf = vf.sort_values("cost_proxy")
    fig.add_trace(go.Scatter(
        x=pf["cost_proxy"], y=pf["net_kW"], mode="lines+markers",
        line=dict(color="#FBBF24", width=1.8),
        marker=dict(size=7, color="#FBBF24",
                    line=dict(color="#0B1220", width=1)),
        opacity=0.95,
        name=f"精确核验前沿（{len(pf)} 解）",
        hovertemplate="精确核验前沿<br>成本代理 %{x:.0f} 万元"
                      "<br>净功率 %{y:.1f} kW/MW热<extra></extra>"))
    best = sp.loc[sp["net_kW"].idxmax()]
    fig.add_trace(go.Scatter(
        x=[best["cost_proxy"]], y=[best["net_kW"]], mode="markers",
        marker=dict(symbol="star", size=14, color="#FBBF24",
                    line=dict(color="#0B1220", width=1)),
        name=f"最优点 {best['net_kW']:.0f} kW/MW热",
        hovertemplate="最优点<br>净功率 %{y:.0f} kW/MW热<extra></extra>"))
    current = None
    try:
        if demand == "发电" and t_src >= 200.0:
            current = current_steam_point(t_src, dT)
    except Exception:
        current = None
    if current is not None:
        fig.add_trace(go.Scatter(
            x=[current[1]], y=[current[0]], mode="markers+text",
            marker=dict(symbol="diamond", size=13, color="#F472B6",
                        line=dict(color="#0B1220", width=1)),
            text=[f"{current[0]:.0f}"], textposition="top center",
            textfont=dict(color="#F472B6", size=12),
            name="当前工况估算（示意）",
            hovertemplate=("当前工况估算<br>净功率 %{y:.1f} kW/MW热"
                           "<br>成本代理 %{x:.0f} 万元<extra></extra>")))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,28,46,.55)", height=400, font=CHART_FONT,
        xaxis=dict(showgrid=False, zeroline=False,
                   title="设备成本代理（万元，示意）"),
        yaxis=dict(showgrid=False, zeroline=False,
                   title="净功率（kW/MW 回收热）"),
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12)),
        margin=dict(l=40, r=20, t=30, b=40))
    return fig


def lca_figure():
    df = load_lca_monthly(_mtime("lca_monthly_real.csv"))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["月份"], y=df["月度降碳tCO2"], name="月度降碳 (tCO2)",
        marker=dict(color=df["月度降碳tCO2"],
                    colorscale=[[0, "#0EA5E9"], [1, "#34D399"]],
                    cornerradius=6, line=dict(width=0)),
        text=df["月度降碳tCO2"].round(1).astype(str),
        textposition="outside", textfont=dict(color="#94A3B8", size=9),
        hovertemplate="%{x}月 降碳 %{y:.1f} tCO2<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=df["月份"], y=df["累计降碳tCO2"], mode="lines+markers",
        name="累计 (tCO2)", yaxis="y2",
        line=dict(color="#FBBF24", width=2),
        marker=dict(size=6, color="#FBBF24", line=dict(color="#0B1220", width=1)),
        hovertemplate="%{x}月 累计 %{y:.1f} tCO2<extra></extra>"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,28,46,.55)", height=440, font=CHART_FONT,
        bargap=0.32,
        xaxis=dict(showgrid=False, zeroline=False, title="月份"),
        yaxis=dict(showgrid=False, zeroline=False, title="月度降碳 (tCO2)"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    zeroline=False, title="累计 (tCO2)",
                    tickfont=dict(color="#FBBF24")),
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12)),
        margin=dict(l=40, r=46, t=30, b=40))
    return fig


# ---------------------------------------------------------------
# 页面
# ---------------------------------------------------------------
def _advance_rt():
    """按波动曲线推进模拟实时工况（工厂 DCS 感，数据为示意）。"""
    rt = st.session_state.setdefault(
        "rt", {"t": 300.0, "m": 50.0, "hist": [], "n": 0, "base": 300.0})
    if st.session_state.get("rt_paused", False):
        return
    prof = st.session_state.get("rt_profile", "平稳波动")
    amp = float(st.session_state.get("rt_amp", 30.0))
    speed = float(st.session_state.get("rt_speed", 1.0))
    n = rt["n"]
    base = rt.get("base", rt["t"])
    if prof == "班次阶跃":
        t = base + amp * (1.0 if (n // 8) % 2 == 0 else -1.0)
    elif prof == "随机游走":
        t = (rt["t"] + (amp / 6.0) * math.sin(n * 0.4 * speed)
             + (amp / 12.0) * ((n * 7919) % 100 - 50) / 50.0)
    else:
        t = (base + amp * math.sin(n * 0.08 * speed)
             + (amp / 8.0) * math.sin(n * 0.017 * speed * 3.0))
    t = min(max(t, 40.0), 900.0)
    rt["t"] = t
    rt["m"] = min(max(rt.get("m", 50.0) * (1.0 + 0.03 * math.sin(n * 0.03 * speed)),
                      0.1), 200.0)
    rt["hist"] = rt.get("hist", [])[-59:] + [t]
    rt["n"] = n + 1


st.markdown(
    '<div class="hero">'
    '<span class="hero-badge">时代杯 · 智能控碳</span>'
    '<span class="hero-badge" style="background:linear-gradient(90deg,#34D399,#FBBF24)">零碳科技</span>'
    '<div class="hero-title">工业余热回收利用 · 智能决策演示平台</div>'
    '<div class="hero-sub">《工业废热或余热回收利用降碳技术路径与智能优化评价方法》｜路径筛选 → TOPSIS 排序 → 减碳估算 → 帕累托前沿 → 动态 LCA</div>'
    '</div>', unsafe_allow_html=True)

# 顶部导航（学校网站风格：三个独立页面，点击切换）
st.markdown('<div class="topnav-wrap">', unsafe_allow_html=True)
nav_cols = st.columns(3, gap="small")
for item in NAV_ITEMS:
    with nav_cols[NAV_ITEMS.index(item)]:
        if st.session_state.get("mode", "手动设定") == item:
            st.markdown(f'<div class="nav-item nav-active">{item}</div>',
                        unsafe_allow_html=True)
        else:
            if st.button(item, key=f"nav_{item}", width="stretch"):
                st.session_state["mode"] = item
                st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div style="font-size:17px;font-weight:800;color:#E2E8F0;margin-bottom:4px">场景参数</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="side-sec">热源与需求</div>', unsafe_allow_html=True)
    data_mode = st.session_state.get("mode", "手动设定")
    conditions = load_conditions()
    if data_mode == "手动设定":
        t_src = st.number_input("热源温度 (℃)", 40.0, 900.0, 120.0, 5.0)
        m_dot = st.number_input("热源流量 (kg/s)", 0.1, 200.0, 50.0, 0.5)
    elif data_mode == "典型工况":
        if conditions.empty:
            st.error("工况库 data/conditions_db.csv 缺失或格式错误，请检查 data/ 目录")
            t_src, m_dot = 120.0, 50.0
        else:
            preset = st.selectbox("典型工况", conditions["工况名称"].tolist())
            row = conditions[conditions["工况名称"] == preset].iloc[0]
            t_src = float(row["热源温度_℃"])
            m_dot = float(row["热源流量_kg_s"])
            # 与实时模拟分支一致：选中工况即同步需求/连续性/年运行小时，
            # 避免只改温度流量、仍按旧“用能需求”推荐造成误导
            demand_ui = str(row["用能需求"])
            if demand_ui in DEMAND_OPTIONS:
                st.session_state["demand_val"] = demand_ui
            cont_ui = str(row["热源连续性"])
            if cont_ui in ["连续", "间歇"]:
                st.session_state["continuity_val"] = cont_ui
            try:
                h_ui = int(float(row["年运行小时"]))
                if 4000 <= h_ui <= 8000:
                    st.session_state["hours_val"] = h_ui
            except (TypeError, ValueError):
                pass
            st.caption(
                f"{row['行业']} · {row['热源类型']} · 来源：{row['数据来源']}"
                + f" · 需求：{row['用能需求']} · 连续性：{row['热源连续性']}"
                + (f" · {row['说明']}" if row["说明"] else ""))
    else:
        st.session_state.setdefault(
            "rt", {"t": 300.0, "m": 50.0, "hist": [], "n": 0, "base": 300.0})
        if conditions.empty:
            st.error("工况库 data/conditions_db.csv 缺失或格式错误，请检查 data/ 目录")
        else:
            cond_names = conditions["工况名称"].tolist()
            cond_name = st.selectbox("工况库条目（实时模拟）", cond_names,
                                     key="rt_cond")
            row = conditions[conditions["工况名称"] == cond_name].iloc[0]
            if st.session_state.get("rt_cond_applied") != cond_name:
                base_t = float(row["热源温度_℃"])
                st.session_state["rt"] = {
                    "t": base_t, "m": float(row["热源流量_kg_s"]),
                    "hist": [], "n": 0, "base": base_t}
                st.session_state["rt_cond_applied"] = cond_name
                profile = str(row["波动模式"])
                st.session_state["rt_profile"] = (
                    profile if profile in FLOW_PROFILES else "平稳波动")
                st.session_state["rt_amp"] = float(row["波动幅度_℃"])
                st.session_state["rt_speed"] = float(row["波动速率"])
                demand = str(row["用能需求"])
                if demand in DEMAND_OPTIONS:
                    st.session_state["demand_val"] = demand
                continuity = str(row["热源连续性"])
                if continuity in ["连续", "间歇"]:
                    st.session_state["continuity_val"] = continuity
                try:
                    h = int(float(row["年运行小时"]))
                    if 4000 <= h <= 8000:
                        st.session_state["hours_val"] = h
                except (TypeError, ValueError):
                    pass
            st.caption(
                f"{row['行业']} · {row['热源类型']} · 来源：{row['数据来源']}"
                + (f" · {row['说明']}" if row["说明"] else ""))
            with st.expander("工况库（可编辑 data/conditions_db.csv）"):
                st.dataframe(
                    conditions[["工况名称", "行业", "热源类型", "热源温度_℃",
                                "热源流量_kg_s", "用能需求", "热源连续性",
                                "数据来源"]],
                    width="stretch", hide_index=True)
        _prof = st.session_state.get("rt_profile", "平稳波动")
        _prof = _prof if _prof in FLOW_PROFILES else "平稳波动"
        st.session_state["rt_profile"] = st.selectbox(
            "波动模式", FLOW_PROFILES, index=FLOW_PROFILES.index(_prof))
        st.session_state["rt_amp"] = st.slider(
            "波动幅度 (℃)", 5.0, 150.0,
            float(st.session_state.get("rt_amp", 30.0)), 5.0)
        st.session_state["rt_speed"] = st.slider(
            "波动速率", 0.5, 3.0,
            float(st.session_state.get("rt_speed", 1.0)), 0.5)
        st.session_state["rt_paused"] = st.toggle(
            "暂停实时数据", value=st.session_state.get("rt_paused", False))
        t_src = st.session_state["rt"]["t"]
        m_dot = st.session_state["rt"]["m"]
    demand = st.selectbox("用能需求", DEMAND_OPTIONS, key="demand_val")
    continuity = st.radio("热源连续性", ["连续", "间歇"], horizontal=True,
                          key="continuity_val")
    st.markdown('<div class="side-sec">运行与核算假设</div>', unsafe_allow_html=True)
    hours = st.slider("年运行小时 (h)", 4000, 8000, 8000, 500, key="hours_val")
    dT = st.slider("换热端差 (℃)", 5, 20, 10, 1)
    medium = st.selectbox("热介质（供热估算用）", ["热水/冷凝水", "烟气", "工艺液体"])
    driver = st.radio("驱动来源", v2u.DRIVER_OPTIONS, horizontal=True,
                      index=0,
                      help="外购蒸汽：供暖/热水场景启用蒸汽驱动吸收式热泵（abs_ext）；"
                           "外购电力：压缩式热泵/电压缩制冷按电驱动口径评价；"
                           "供冷场景严格区分驱动来源（余热自驱动→吸收式制冷、"
                           "外购电力→电压缩制冷、外购蒸汽→不设废热回收路径）",
                      key="driver_val")
    st.markdown('<div class="side-sec">评价权重</div>', unsafe_allow_html=True)
    lam = st.slider("组合权重 λ（主观占比）", 0.0, 1.0, 0.5, 0.05)
    st.caption("λ=主观(AHP)占比；1−λ=客观(熵权)占比")
    st.info("主观 AHP 权重为**演示假设**（正式应用需专家打分）；客观熵权随当前候选矩阵动态计算。")

    st.markdown('<div class="side-sec">工况库管理</div>', unsafe_allow_html=True)
    st.session_state.setdefault("_cond_default", conditions.copy())
    with st.expander("在线编辑工况库（实时/典型工况数据源）", expanded=False):
        st.caption(
            "增删改行后点“保存”。本地运行保存到 data/conditions_db.csv；"
            "部署版（Streamlit Cloud）服务器为临时环境，保存只对当前会话生效，"
            "长期修改请用下方“下载 CSV”后上传到 GitHub 仓库 data/ 目录。")
        editor_df = conditions.copy() if not conditions.empty else empty_conditions_template()
        edited = st.data_editor(
            editor_df, num_rows="dynamic", hide_index=True, width="stretch",
            key="cond_editor",
            column_config={
                "热源温度_℃": st.column_config.NumberColumn(
                    "热源温度_℃", min_value=40.0, max_value=900.0, step=5.0),
                "热源流量_kg_s": st.column_config.NumberColumn(
                    "热源流量_kg_s", min_value=0.1, max_value=200.0, step=1.0),
                "波动幅度_℃": st.column_config.NumberColumn(
                    "波动幅度_℃", min_value=5.0, max_value=150.0, step=5.0),
                "波动速率": st.column_config.NumberColumn(
                    "波动速率", min_value=0.5, max_value=3.0, step=0.1),
                "年运行小时": st.column_config.NumberColumn(
                    "年运行小时", min_value=4000, max_value=8000, step=500),
            })
        b1, b2 = st.columns(2)
        if b1.button("保存修改", type="primary"):
            errs = validate_conditions(edited)
            if errs:
                st.error("无法保存：\n" + "\n".join(errs))
            else:
                try:
                    path = save_conditions(edited)
                    st.session_state["_cond_default"] = edited.copy()
                    st.success(f"已保存 {len(edited)} 行 → {os.path.basename(path)}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"保存失败：{exc}")
        if b2.button("恢复初始库"):
            default = st.session_state.get("_cond_default")
            if default is not None and not default.empty:
                save_conditions(default)
                st.rerun()
        st.download_button(
            "下载当前 CSV",
            data=edited.to_csv(index=False).encode("utf-8-sig"),
            file_name="conditions_db.csv", mime="text/csv")
        up = st.file_uploader("上传 CSV 替换工况库（.csv）", type=["csv"])
        if up is not None:
            try:
                up_df = pd.read_csv(up, encoding="utf-8-sig").fillna("")
                errs = validate_conditions(up_df)
                if errs:
                    st.error("上传文件不合法：\n" + "\n".join(errs))
                else:
                    save_conditions(up_df)
                    st.success(f"已应用上传的工况库（{len(up_df)} 行）")
                    st.rerun()
            except Exception as exc:
                st.error(f"读取上传文件失败：{exc}")


def render_dashboard():
    if data_mode == "实时模拟":
        _advance_rt()
        globals()["t_src"] = st.session_state.get("rt", {}).get("t", 300.0)
        globals()["m_dot"] = st.session_state.get("rt", {}).get("m", 50.0)
        rt = st.session_state["rt"]
        cond_name = st.session_state.get("rt_cond_applied", "—")
        st.markdown(
            '<div class="sec-title"><span class="tag">LIVE</span>实时工况监测（模拟）</div>',
            unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("热源温度", f"{rt['t']:.0f} ℃")
        m2.metric("热源流量", f"{rt['m']:.1f} kg/s")
        m3.metric("当前工况", cond_name)
        st.caption("模拟实时数据（非实测）· 每 3 秒刷新 · 波动曲线全宽展示，无需手动拉伸")
        if rt["hist"]:
            st.line_chart(pd.DataFrame({"温度℃": rt["hist"]},
                                       index=range(len(rt["hist"]))),
                          height=220)
        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    elif data_mode == "手动设定":
        st.caption("手动设定：自由输入热源温度与流量，即时输出两级决策与降碳核算结果")
    elif data_mode == "典型工况":
        st.caption("典型工况：从工况库选择真实场景，一键查看推荐路径与核算结果")
    res = run_v2_decision(t_src, m_dot, medium, demand, continuity,
                          hours, dT, driver)
    keys = res["keys"]
    survivors = res["labels"]
    X = res["X"]

    st.markdown('<div class="sec-title"><span class="tag">01</span>第一级 · 热力学规则粗筛'
                '（统一内核 v2）</div>', unsafe_allow_html=True)
    if res["out_of_scope"]:
        st.warning("⚠ 能力圈外 / 无可行候选：" + res["message"])
        st.caption("红线（v2.1）：供冷/干燥为受支持需求；除湿未单列（并入供冷/干燥场景后评价）；"
                   "外购蒸汽驱动吸收式制冷不设废热回收路径；"
                   ">650℃ 发电/产汽需多压/再热锅炉专项设计，模型不做硬推荐（设计文档 §0/§9）。")
    else:
        st.markdown('<div class="mobile-swipe-hint">↔ 表格较宽：在表格上左右滑动可查看完整列</div>',
                    unsafe_allow_html=True)
        rows = [{"路径": v2u.LABELS[k],
                 "结果": "✓ 通过" if k in keys else "✗ 排除",
                 "原因": res["reasons"][k]} for k in dc.PATH_KEYS]
        st.dataframe(style_stage_table(pd.DataFrame(rows)), width="stretch",
                     hide_index=True)
        st.info(f"进入第二级候选：{'、'.join(survivors) if survivors else '无（请调整场景参数）'}")
        # C3.2（2026-09-05）：S2 型边界工况自动提示（预留端差裕量）
        for note in v2u.boundary_notices(res):
            st.warning(note)

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title"><span class="tag">02</span>第二级 · TOPSIS 精细排序'
                '（λ=AHP×熵权组合赋权）</div>', unsafe_allow_html=True)
    if keys:
        st.markdown('<div class="mobile-swipe-hint">↔ 表格较宽：在表格上左右滑动可查看完整列</div>',
                    unsafe_allow_html=True)
        w5 = combined_weights(lam, X)
        c = topsis(X, w5)
        df_r = pd.DataFrame({
            "路径": survivors,
            "能效%(仿真/参考)": X[:, 0],
            "投资(文献/估算)万元/MW": X[:, 1],
            "回收期(文献/估算)年": X[:, 2],
            "CO2减排t/年(推算)": X[:, 3],
            "政策分(政策规则)": X[:, 4],
            "运行成本(推算/估算)万元/MW·年": X[:, 5],
            "TOPSIS贴近度": np.round(c, 4),
        }).sort_values("TOPSIS贴近度", ascending=False).reset_index(drop=True)
        df_r["CO2减排t/年(推算)"] = df_r["CO2减排t/年(推算)"].apply(
            lambda v: f"{v:.1f}" if v > 0 else "—")
        df_r.insert(0, "排名", range(1, len(df_r) + 1))
        st.dataframe(style_topsis_table(df_r), width="stretch",
                     hide_index=True)
        st.markdown(
            f'<div class="rec-banner">推荐路径：<b>{df_r.iloc[0]["路径"]}</b>'
            f'<span class="muted">　贴近度 {df_r.iloc[0]["TOPSIS贴近度"]:.3f} · λ={lam:.2f}'
            f' · 驱动={driver} · 权重来自 AHP+熵权组合赋权</span></div>',
            unsafe_allow_html=True)
        if float(c.max()) >= 0.9995:
            st.caption("贴近度≈1.000 说明该路径在**全部指标上占优**（正理想解距离 d⁺≈0），"
                       "是数学结果而非写死；请结合第 2/3 名与贴近度差距判断稳健性。")
        st.caption("数据口径（统一内核 v2/P2）：路径规则/矩阵来自 decision_core；"
                   "ORC/蒸汽朗肯能效为 CoolProp 真值（ORC 累计中位数、蒸汽朗肯按锅炉出口温度插值）；"
                   "吸收式按需求拆分：供暖/储热=一类 COP1.7、工艺蒸汽=二类 COP0.45、"
                   "外购蒸汽驱动=COP_h1.7；**供冷/干燥（v2.1）**：吸收式制冷 COP_c 分段"
                   "（≥85℃ 0.7、80~84℃ 0.6，出处见 11/calibration 台账）、电压缩制冷 "
                   "COP_e=5.0（GB19577-2015 水冷式 3 级下限）、干燥按供热语义映射"
                   "（直接换热/产汽/热泵），除湿未单列并入供冷/干燥；"
                   "压缩式能效为一次能源效率（COP2.8×电网38%≈106%）；"
                   "减排列按当前参数推算（发电=替代购电；供热=替代天然气；压缩式另扣自身耗电；"
                   "吸收式制冷=替代电压缩电耗、电压缩制冷减排列记 0；"
                   "外购蒸汽吸收式扣驱动蒸汽燃料），可复算；"
                   "**投资/回收期**：ORC 引自《重庆大学学报》(23800元/kW、5.58年)；"
                   "吸收式外购蒸汽档按哈石化余热暖民（1.26 亿元/72.3 MW、回收期 4.2 年）；"
                   "吸收式/电压缩制冷的投资与回收期为工程估算 [待标定]（见 C2 台账）；"
                   "其余为工程估算/同一对比研究（李萌），详见《数据来源台账》与 11 目录 calibration；"
                   "主观权重 AHP 为演示假设（需专家打分），客观熵权随当前候选矩阵动态计算。")
        csv = df_r.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("下载排序结果 CSV", data=csv,
                           file_name="两级决策_演示结果.csv", mime="text/csv")

    st.divider()

    st.markdown('<div class="sec-title"><span class="tag">03</span>减碳与收益估算（推算口径，非实测）</div>',
                unsafe_allow_html=True)
    decision_ok = not res["out_of_scope"]
    orc = (orc_reduction(t_src, m_dot, medium, hours, dT) if demand in (
        "发电", "储热调峰") else None) if decision_ok else None
    heat = (heat_reduction(t_src, m_dot, medium, hours, dT) if demand in (
        "工艺蒸汽", "供暖/热水", "储热调峰", "干燥/烘干") else None) \
        if decision_ok else None
    hp = (heat_reduction(t_src, m_dot, medium, hours, dT, cop=HP_COP)
          if demand in ("工艺蒸汽", "供暖/热水", "储热调峰", "干燥/烘干")
          else None) if decision_ok else None
    top_path = df_r.iloc[0]["路径"] if survivors else None

    c1, c2, c3, c4 = st.columns(4)
    if orc is not None:
        c1.metric("ORC 可行工况数", f"{orc['n_cond']}",
                  f"{len(load_sweep(_mtime('orc_sweep_coolprop.csv')))} 工况中")
        c2.metric("实际净功率（按流量折算）", f"{orc['net_abs_kW']:.0f} kW",
                  f"P50 {orc['p50']:.0f} kW/MW热 × {orc['q_kw']:.0f} kW 回收热")
        c3.metric("年降碳（推算）", f"{orc['co2']:.1f} tCO2",
                  f"{orc['mwh']:.0f} MWh × 0.581")
        c4.metric("年节省电费（演示价 0.65 元/kWh）", f"{orc['money']:.1f} 万元",
                  "按替代购电口径")
    elif top_path == v2u.LABELS["abs_cool"]:
        s2 = res["scene"]
        q = dc.recovered_heat_kw(s2)
        cop = dc.abs_cool_cop(s2["热源温度_degC"]) or dc.COP_C_ABS
        q_cold = cop * q
        mwh = q_cold / dc.COP_E_COOL * s2["年运行小时"] / 1000.0
        red = mwh * dc.GRID_EF
        c1.metric("服务冷量（估算）", f"{q_cold:.0f} kW",
                  f"COP_c={cop:.2f} × 回收热 {q:.0f} kW")
        c2.metric("年制冷量（推算）", f"{q_cold * s2['年运行小时'] / 1000.0:.0f} MWh",
                  f"{s2['年运行小时']} h/年")
        c3.metric("替代电耗（推算）", f"{mwh / 10.0:.1f} 万kWh",
                  f"÷ COP_e={dc.COP_E_COOL:.1f}（GB19577 3级）")
        c4.metric("年降碳（推算）", f"{red:.1f} tCO2",
                  f"{mwh:.0f} MWh × 0.581")
        st.caption("吸收式制冷口径：废热为弃热不扣驱动排放；年节省电费（演示价 0.65 元/kWh）≈ "
                   f"{mwh * 0.065:.1f} 万元；辅机/泵运行成本约 3 万元/MW·年 [待标定]。"
                   "投资/回收期为工程估算，正式核算须以机组选型报价为准。")
    elif top_path == v2u.LABELS["comp_cool"]:
        s2 = res["scene"]
        c1.metric("电压缩制冷 COP_e", f"{dc.COP_E_COOL:.1f}",
                  "GB19577-2015 水冷式 3 级下限（保守取）")
        c2.metric("一次能源效率（演示口径）", f"{dc.COP_E_COOL * 0.38 * 100.0:.0f}%",
                  "COP_e × 电网发电效率 38%")
        c3.metric("耗电强度", f"{1000.0 / dc.COP_E_COOL:.0f} kW电/MW冷",
                  "按 1 MW 冷量服务计")
        c4.metric("年运行成本（电费演示价）",
                  f"{dc.comp_cool_opex_wan_mw(s2):.1f} 万元/MW·年",
                  f"减排 0：与电制冷现状同口径（{s2['年运行小时']} h/年）")
        st.caption("电压缩制冷是“无废热可用”时的基准方案：不假装比自身更低碳；"
                   "若要与吸收式制冷比较同一冷负荷，需按负荷口径核算（C2 待办），"
                   "本面板为 1 MW 冷量服务强度的演示口径。")
    elif top_path == v2u.LABELS["comp"] and hp is not None:
        c1.metric("回收热功率（估算）", f"{hp['q_kw']:.0f} kW", f"ṁ={m_dot} kg/s × cp×ΔT")
        c2.metric("年回收热量", f"{hp['heat_gj']:.0f} GJ", f"{hours} h/年")
        c3.metric("年降碳（净：替代天然气−耗电）", f"{hp['co2']:.1f} tCO2",
                  f"替代 {hp['co2_replaced']:.0f} − 耗电 {hp['co2_elec']:.0f} tCO2"
                  f"（COP {HP_COP}，{GRID_EF_LABEL}）")
        c4.metric("年净节省（燃气费−电费，演示价）", f"{hp['money']:.1f} 万元",
                  "按替代天然气、0.65 元/kWh 演示价")
    elif top_path == v2u.LABELS["abs_ext"]:
        s2 = res["scene"]
        q = dc.recovered_heat_kw(s2)
        c1.metric("回收热功率（估算）", f"{q:.0f} kW", f"ṁ={m_dot} kg/s × cp×ΔT")
        c2.metric("运行成本（外购蒸汽驱动）", f"{dc.steam_driven_abs_opex_wan_mw(s2):.1f} 万元/年",
                  "=(1/COP_h)×hours×3.6×100元/GJ÷1e4×1.03")
        c3.metric("年降碳（净：替代天然气−驱动蒸汽）",
                  f"{dc.steam_driven_abs_reduction(s2):.1f} tCO2",
                  f"COP_h={dc.COP_H_ABS_EXT} 口径")
        c4.metric("—", "—", "外购蒸汽驱动吸收式（哈石化余热暖民同型）")
    elif heat is not None:
        c1.metric("回收热功率（估算）", f"{heat['q_kw']:.0f} kW", f"ṁ={m_dot} kg/s × cp×ΔT")
        c2.metric("年回收热量", f"{heat['heat_gj']:.0f} GJ", f"{hours} h/年")
        c3.metric("年降碳（替代天然气，估算）", f"{heat['co2']:.1f} tCO2",
                  "0.0561 t/GJ ÷ 锅炉效率90%")
        c4.metric("年节省燃气费（演示价）", f"{heat['money']:.1f} 万元", "3.5 元/m³ 折算")
    else:
        for cc in (c1, c2, c3, c4):
            cc.metric("—", "—", "无可用估算")
    if demand == "供冷（制冷）" and decision_ok and res["scene"] is not None:
        with st.expander("制冷方式对比（同一冷负荷口径：每 1 MW 冷量服务 · 演示）"):
            s2 = res["scene"]
            h_run = s2["年运行小时"]
            cop_c = dc.abs_cool_cop(s2["热源温度_degC"])
            cop_c_txt = (f"{cop_c:.2f}" if cop_c else "—")
            heat_drive_txt = (
                f"{1.0 / cop_c:.2f} MW热（废热，弃热免费）" if cop_c
                else "不可行（热源 <80℃）")
            elec_mwh_per_mw = (1.0 / dc.COP_E_COOL) * h_run
            red_per_mw = elec_mwh_per_mw * dc.GRID_EF
            comp_cost = dc.comp_cool_opex_wan_mw(s2)
            # C2.1 口径强化（2026-09-05）：投资/回收期显式入表并按年运行小时动态计算，
            # 区分“新建/扩容增量回收期”与“存量改造全投资回收期”，避免数字被脱离口径引用。
            abs_inv = float(dc.BASE_INDICATORS["abs_cool"][1])    # 100 万元/MW冷 [待标定]
            comp_inv = float(dc.BASE_INDICATORS["comp_cool"][1])  # 60 万元/MW冷 [待标定]
            abs_aux = float(dc.BASE_INDICATORS["abs_cool"][5])    # 3 万元/MW·年（辅机）[待标定]
            inc_saving = comp_cost - abs_aux
            inc_extra = abs_inv - comp_inv
            pb_ok = cop_c is not None and inc_saving > 0
            inc_pb_txt = (f"≈{inc_extra / inc_saving:.1f} 年" if pb_ok
                          else "—（热源<80℃ 或节费≤辅机成本）")
            allin_pb_txt = (f"≈{abs_inv / inc_saving:.1f} 年" if pb_ok else "—")
            rows = [
                {"指标": "制冷 COP", "吸收式（余热驱动）": cop_c_txt,
                 "电压缩制冷（现状基准）": f"{dc.COP_E_COOL:.1f}"},
                {"指标": "驱动输入（每 MW 冷量）", "吸收式（余热驱动）": heat_drive_txt,
                 "电压缩制冷（现状基准）":
                     f"{1.0 / dc.COP_E_COOL:.3f} MW电"},
                {"指标": f"年耗电（{h_run} h，演示）",
                 "吸收式（余热驱动）": "≈0（废热驱动；辅机另计）",
                 "电压缩制冷（现状基准）":
                     f"{elec_mwh_per_mw / 10.0:.0f} 万kWh"},
                {"指标": "投资（万元/MW冷，[待标定]）",
                 "吸收式（余热驱动）": f"≈{abs_inv:.0f}（机组+换热/管网）",
                 "电压缩制冷（现状基准）": f"≈{comp_inv:.0f}（国产机组价上沿）"},
                {"指标": "年运行成本（演示价 0.65 元/kWh）",
                 "吸收式（余热驱动）":
                     f"≈{abs_aux:.0f} 万元/MW·年（辅机）[待标定]",
                 "电压缩制冷（现状基准）": f"≈{comp_cost:.0f} 万元/MW·年"},
                {"指标": f"增量回收期·新建/扩容（相对电压缩，{h_run} h/年）",
                 "吸收式（余热驱动）": inc_pb_txt,
                 "电压缩制冷（现状基准）": "—（基准方案）"},
                {"指标": f"全投资回收期·存量改造（电制冷为沉没成本，{h_run} h/年）",
                 "吸收式（余热驱动）": allin_pb_txt,
                 "电压缩制冷（现状基准）": "—（基准方案）"},
                {"指标": "相对“全用电压缩”的年减排",
                 "吸收式（余热驱动）": f"≈{red_per_mw:.0f} tCO2/MW冷·年",
                 "电压缩制冷（现状基准）": "0（同口径基准）"},
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption(
                "口径说明：① 本面板按 1 MW 冷量服务、台账基值（未按场景规模摊薄）演示；"
                "吸收式替代电压缩所避免的电耗只取决于冷量服务与 COP_e，与 COP_c 无关"
                "（COP_c 决定所需废热量）；减排按 "
                f"{dc.GRID_EF} tCO2/MWh（2021 全国电网平均因子）推算。"
                "② 投资与回收期均为工程估算 [待标定]（11/calibration 台账 §3.2）："
                "台账典型回收期 5.5/4.2 年是标定档位，非本面板推导值；本面板按"
                "“相对电压缩的增量投资（新建/扩容）”与“存量改造全投资（电制冷视为"
                f"沉没成本）”两种口径，绑定本场景年运行小时 {h_run} h 动态计算"
                f"（电价 0.65 元/kWh、辅机 {abs_aux:.0f} 万元/MW·年）。"
                "③ CAPEX 敏感性（2026-09-04，见 calibration/敏感性_C2.1_制冷CAPEX.md）："
                "吸收式投资若按国际文献系统安装口径上探至 210~426 万元/MW，"
                "8000 h 下增量回收期 0.7~3.6 年（电压缩取国产 60 或国际 81~141 "
                "万元/MW 的全部组合均 <5.5 年）；"
                "若年运行仅 4000 h 且吸收式取国际高位、电压缩按国产低位，最差约 7.5 年"
                "——回收期必须绑定年运行小时与造价档位，不得单独引用。"
                "④ 两列能效口径不同（COP_c×100 与 COP_e×电网效率 38%），不直接比较"
                "能效列；若废热本身有市场价值，应按其机会成本另行折算。")
    steam = (steam_reduction(t_src, m_dot, medium, hours, dT) if demand in (
        "发电", "储热调峰") else None) if decision_ok else None
    if steam is not None:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("蒸汽朗肯可行工况数", f"{steam['n_cond']}", "1600 工况中")
        s2.metric("实际净功率（按流量折算）", f"{steam['net_abs_kW']:.0f} kW",
                  f"P50 {steam['p50']:.0f} kW/MW热 × {steam['q_kw']:.0f} kW 回收热")
        s3.metric("年降碳（推算）", f"{steam['co2']:.1f} tCO2",
                  f"{steam['mwh']:.0f} MWh × 0.581")
        s4.metric("年节省电费（演示价 0.65 元/kWh）", f"{steam['money']:.1f} 万元",
                  "按替代购电口径")
    st.caption(f"说明：ORC/蒸汽发电数字基于 CoolProp 物性模型（净功率列按每 MW 回收热口径；"
               f"年发电量/降碳/电费按实际回收热功率 ṁ×cp×ΔT 折算，随流量实时变化；"
               f"电网因子采用{GRID_EF_LABEL}）；供热数字为替代天然气估算，其中压缩式热泵已扣耗电排放"
               "（COP 2.8）；制冷数字为替代电压缩电耗口径（吸收式 COP_c 分段 × 回收热 ÷ COP_e 5.0），"
               "电压缩制冷减排记 0（与现状同口径）；参数（电价/气价/锅炉效率/COP）均为演示/工程估算，"
               "正式核算须以项目实测为准。")

    st.divider()

    st.markdown('<div class="sec-title"><span class="tag">04</span>帕累托前沿 · 多目标优化结果</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sec-note">青色渐变点 = 代理预测 100 解（按热效率着色，含被支配点）｜青色线 = 精确核验前沿（10 解，CoolProp 扫描网格非支配子集）｜蓝点 = CoolProp 复核抽样（14 解，已剔除 2 个被支配点）｜金星 = 最优点；'
                '横轴为设备成本代理（示意），纵轴为每 MW 回收热净功率（kW/MW热）；'
                '品红菱形 = 当前输入估算工作点（按端差折算蒸发温度，示意；发电需求时显示）</div>',
                unsafe_allow_html=True)
    st.plotly_chart(pareto_figure(), width="stretch")
    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-note">碳减排—成本权衡：蓝线 = 精确核验前沿（10 解）；蓝点 = CoolProp 复核抽样（14 解，原 16 解中 2 个被支配点已剔除）；金星 = 最优点；净功率按每 MW 回收热口径，碳减排 = 净功率 × 8000h × 0.581（2021年度全国电网平均排放因子，推算口径），成本为示意代理模型</div>',
                unsafe_allow_html=True)
    st.plotly_chart(pareto_co2_figure(), width="stretch")
    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-note">高温蒸汽朗肯前沿（每 MW 回收热口径）：金点 = 100 解（CoolProp 复核搜索解），金色线 = 精确核验前沿（8 解）；'
                '金星 = 最优点；品红菱形 = 当前输入估算工作点（≥200℃ 发电需求时显示，示意）</div>',
                unsafe_allow_html=True)
    st.plotly_chart(steam_pareto_figure(), width="stretch")
    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title"><span class="tag">05</span>动态 LCA · 月度滚动核算</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sec-note">蓝绿柱 = 月度降碳｜金线 = 全年累计（推算口径，年 573.1 tCO2）；'
                '考虑设备制造排放 15.0 tCO2e（20 年摊销 0.75 t/年，工程估算），'
                '全生命周期口径年净降碳约 572.4 tCO2</div>',
                unsafe_allow_html=True)
    st.plotly_chart(lca_figure(), width="stretch")

    st.divider()

    st.markdown('<div class="sec-title"><span class="tag">06</span>敏感性 · λ 对排序的影响</div>',
                unsafe_allow_html=True)
    lam_grid = np.linspace(0.0, 1.0, 11)
    if survivors:
        X_lam = X  # 与上方 TOPSIS 表同一内核矩阵（decision_core v2）
        top1, top2, top3, c_top1 = [], [], [], []
        c_base = None
        for lam_t in lam_grid:
            w = combined_weights(float(lam_t), X_lam)
            c_t = topsis(X_lam, w)
            order = np.argsort(-c_t)
            top1.append(survivors[int(order[0])])
            top2.append(survivors[int(order[1])] if len(order) > 1 else "—")
            top3.append(survivors[int(order[2])] if len(order) > 2 else "—")
            c_top1.append(round(float(c_t[order[0]]), 3))
            if c_base is None:
                c_base = float(c_t[order[0]])
        delta = [f"{c - c_base:+.3f}" for c in c_top1]
        df_s = pd.DataFrame({"λ": [f"{v:.2f}" for v in lam_grid],
                             "第一名": top1, "第二名": top2, "第三名": top3,
                             "榜首贴近度 C(λ)": c_top1,
                             "ΔC vs λ=0": delta})
        st.dataframe(style_lambda_table(df_s), width="stretch", hide_index=True)
        # 贴近度-λ 曲线：权重对每个候选的影响一目了然
        st.plotly_chart(lambda_figure(survivors, X_lam), width="stretch")
        # 排序快照：λ=0 / 0.5 / 1 的完整排序（含贴近度）
        snap = []
        for lam_t in (0.0, 0.5, 1.0):
            w = combined_weights(float(lam_t), X_lam)
            c_t = topsis(X_lam, w)
            order = np.argsort(-c_t)
            snap.append(f"λ={lam_t:.1f}：" +
                        " ＞ ".join(f"{survivors[j]}({c_t[j]:.3f})" for j in order))
        st.markdown("**排序快照**　" + "　｜　".join(snap), unsafe_allow_html=True)
        if len(survivors) <= 2:
            st.info("当前场景候选路径较少（≤2 条）：熵权信息量低，λ 的影响主要体现在贴近度数值变化；"
                    "切换到「供暖/热水」或「储热调峰」等候选较多的场景，可看到第 2/3 名排序随 λ 变化。")
        stable = len(set(top1)) == 1
        if stable:
            lo, hi = min(c_top1), max(c_top1)
            st.success(f"λ∈[0,1] 全程第一名稳定为「{top1[0]}」（贴近度 {lo:.3f}~{hi:.3f}），"
                       f"首选路径对权重设定稳健；第 2/3 名仍随 λ 变化（见表格），说明权重确有影响。")
        else:
            st.success(f"第一名随 λ 变化：{' → '.join(dict.fromkeys(top1))}；"
                       f"结论对权重敏感，正式应用需收窄 λ 或补充专家打分。")
        st.caption("说明：本表与上方 TOPSIS 表使用同一决策矩阵（减排列/运行成本按当前参数推算）；"
                   "客观熵权随当前候选矩阵动态计算。第一名稳定不代表排序不变（请看第 2/3 名），"
                   "也不代表结论无风险——应结合贴近度差距判断。")
    else:
        st.warning("无候选路径")

    st.divider()
    with st.expander("数据口径与免责声明（答辩必讲）"):
        st.markdown(
            "1. **仿真数据**：CoolProp（IAPWS-97 / REFPROP）物性模型自建朗肯循环仿真，"
            "ORC 5925 工况（100~350℃，2026-08-13 扩展）+ 蒸汽朗肯 1600 工况"
            "（180~540℃，能量守恒偏差 <1e-9 kW）；"
            "旧版 DWSIM 9.0.5 400 工况数据保留存档（其工质曾混入水，已弃用并修正为纯异丁烷）；\n"
            "2. **代理模型**：scikit-learn MLP，ORC 净功率测试集 R²≈0.99、蒸汽朗肯 R²≈0.998；"
            "pymoo 各 100 解为代理预测搜索解；正式成果为 ORC 10 解精确核验前沿 + 14 个复核抽样点"
            "（原 16 解中 2 个被支配点已剔除）、蒸汽 8 解精确核验前沿；\n"
            "3. **减碳**：ORC/蒸汽发电为物性模型热效率 × 每 MW 回收热净功率 × 运行小时 × "
            "2021年度全国电网平均排放因子 0.5810（生态环境部 2022-03 发布）的"
            "**推算口径**，供热为替代天然气估算，均非实测；"
            "设备制造排放 15.0 tCO2e 为工程估算（20 年摊销 0.75 t/年），全生命周期口径年净降碳约 572.4 t；\n"
            "4. **成本/回收期**：示意性代理模型，非真实报价；CCER 收益为情景假设，未完成备案方法学前不计入基准财务指标；\n"
            "5. 本平台全部代码与数据随申报材料提交，可复算、可溯源；\n"
            "6. **统一决策内核 v2（P3 接线）**：本页第一/二级规则与 03 案例回溯共用同一 "
            "decision_core.py（全温域 25~650℃、12 条路径：热→热/电/冷/干燥四类转换、"
            "含高温蒸汽发电、外购蒸汽驱动吸收式热泵，以及 v2.1 新增吸收式/电压缩制冷）；"
            "**能力圈外不硬推荐**：>650℃ 的发电/产汽需多压/再热锅炉专项设计；"
            "供冷/干燥为受支持需求，除湿未单列（并入供冷/干燥场景后评价），"
            "外购蒸汽驱动吸收式制冷不设废热回收路径；\n"
            "7. **方法边界**：本页为**单路径比选**版本；工业最佳实践中的梯级利用"
            "（高温段先发电/产汽、低温段再供热）已列入后续扩展，答辩按此口径说明。")

    st.markdown(
        '<div class="footer">'
        '《工业废热或余热回收利用降碳技术路径与智能优化评价方法》· 时代杯零碳科技创新大赛 · '
        'CoolProp 物性仿真 + 机器学习 + 多目标优化 + 动态 LCA + 两级决策<br>'
        '数据口径：仿真/推算数据，非实测；成本为示意代理；CCER 为情景假设；'
        '投资/回收期/政策分来源见《数据来源台账》'
        '</div>', unsafe_allow_html=True)


if data_mode == "实时模拟":
    @st.fragment(run_every=3.0)
    def _live_body():
        render_dashboard()
    _live_body()
else:
    render_dashboard()
