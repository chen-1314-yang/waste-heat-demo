# -*- coding: utf-8 -*-
"""工业余热回收智能决策演示平台（正式版 · 深色科技风）。

数据链路：DWSIM 400 工况仿真 → sklearn 代理模型 → pymoo 多目标优化 →
动态 LCA 核算 → 两级智能决策（规则粗筛 + TOPSIS 精评）。
数据口径：仿真/推算/示意/公开文献四类，详见《数据来源台账》。
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="工业余热回收智能决策演示平台", layout="wide",
                   initial_sidebar_state="expanded")

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "data")


def _mtime(fname):
    """数据文件修改时间：作为缓存键，数据更新后页面自动刷新（无需重启服务）。"""
    return os.path.getmtime(os.path.join(DATA, fname))


@st.cache_data
def load_sweep(_t=None):
    return pd.read_csv(os.path.join(DATA, "dwsim_sweep_full.csv"),
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
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 路径库与指标（与正式版一致）
# ---------------------------------------------------------------
PATH_NAMES = [
    "直接换热供暖", "余热锅炉直接产汽", "吸收式热泵提温", "压缩式热泵提温",
    "ORC 余热发电", "热化学储热", "相变储热", "TEG 热电发电",
]
INDICATORS = ["能效%", "投资万元/MW", "回收期年", "CO2减排t/年", "政策分",
              "运行成本万元/MW·年"]
DIRECTIONS = ["max", "min", "min", "max", "max", "min"]
RAW = {
    # 列含义：[能效%(仿真/文献参考), 投资万元/MW(文献/工程估算), 回收期年(文献/工程估算),
    #          CO2减排t/年(推算,运行时覆盖), 政策分(按发改委支持目录规则打分),
    #          运行成本万元/MW·年(压缩式=公式推算,其余=工程估算)]
    # 能效：压缩式=COP2.8×电网效率38%≈106%(一次能源口径)；吸收式=余热自驱动 COP0.75；
    #       直接换热/余热锅炉=热回收率；ORC=仿真中位效率。
    # 投资/回收期：ORC=《重庆大学学报》2019(23800元/kW、5.58年)；压缩式/吸收式=李萌《基于余热
    #       回收用的热泵技术对比研究》(同一对比口径：压缩式4.23年、吸收式2.73年；投资为工程估算)；
    #       直接换热/余热锅炉=工程估算；储热/TEG=示意。政策分规则见《数据来源台账》。
    "直接换热供暖":   [90,  60, 2.5, 0, 3, 6],
    "余热锅炉直接产汽": [85, 100, 3.5, 0, 3, 12],
    "吸收式热泵提温": [75, 150, 2.7, 0, 4, 20],
    "压缩式热泵提温": [106, 120, 4.2, 0, 4, 186],
    "ORC 余热发电":  [12.3, 2380, 5.6, 0, 4, 15],
    "热化学储热":    [70, 900, 10.0, 0, 3, 28],
    "相变储热":      [75, 600, 8.0, 0, 3, 25],
    "TEG 热电发电":  [5, 1500, 12.0, 0, 2, 8],
}
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


def build_matrix(survivors, t_src, m_dot, medium, hours, dT):
    """构建 TOPSIS 决策矩阵：减排列/运行成本按当前参数动态推算（主表与 λ 敏感性共用）。"""
    X = np.array([RAW[p] for p in survivors], dtype=float)
    for i, p in enumerate(survivors):
        if p == "ORC 余热发电":
            r = orc_reduction(t_src, hours, dT)
            if r:
                X[i, 3] = round(r["co2"], 1)
        elif p == "压缩式热泵提温":
            # 运行成本随当前运行小时联动：0.357 MW电/MW热 × hours × 0.65 元/kWh
            X[i, 3] = round(heat_reduction(t_src, m_dot, medium, hours, dT)["co2"], 1)
            X[i, 5] = round(0.357 * hours * 0.65 * 1000 / 10000, 1)
        elif p in ("吸收式热泵提温", "直接换热供暖",
                   "余热锅炉直接产汽", "热化学储热", "相变储热"):
            r = heat_reduction(t_src, m_dot, medium, hours, dT)
            if r:
                X[i, 3] = round(r["co2"], 1)
    return X


def stage1(t_src, demand, continuity, t_steam=152):
    keep, reasons = {}, {}

    def excl(name, why):
        keep[name] = False
        reasons[name] = why

    if demand == "发电":
        keep["ORC 余热发电"] = t_src >= 110
        reasons["ORC 余热发电"] = (
            f"热源 {t_src}℃ 满足 ORC 驱动要求（≥110℃，含端差）" if t_src >= 110
            else f"热源仅 {t_src}℃，低于 ORC 驱动下限 110℃")
        keep["TEG 热电发电"] = t_src >= 40
        reasons["TEG 热电发电"] = "TEG 适合 40℃ 以上温差发电（兜底候选）" if t_src >= 40 \
            else "热源温度过低，TEG 温差不足"
        for p in ["直接换热供暖", "余热锅炉直接产汽", "吸收式热泵提温",
                  "压缩式热泵提温", "热化学储热", "相变储热"]:
            excl(p, "需求为发电：该路径不产出电力")
    elif demand == "工艺蒸汽":
        keep["余热锅炉直接产汽"] = t_src >= t_steam + 20
        reasons["余热锅炉直接产汽"] = (
            f"热源 {t_src}℃ ≥ 蒸汽饱和温度 {t_steam}℃ + 端差 20℃，可直接产汽"
            if t_src >= t_steam + 20 else
            f"热源 {t_src}℃ 不足以产生 {t_steam}℃ 蒸汽（需 ≥{t_steam + 20}℃）")
        keep["吸收式热泵提温"] = t_src >= 90
        reasons["吸收式热泵提温"] = (
            f"热源 {t_src}℃ 可驱动吸收式热泵（≥90℃）" if t_src >= 90
            else "吸收式热泵需 ≥90℃ 驱动热源，当前温度不足")
        keep["压缩式热泵提温"] = True
        reasons["压缩式热泵提温"] = "压缩式热泵以电驱动，不受热源温度下限限制"
        for p in ["直接换热供暖", "ORC 余热发电", "TEG 热电发电",
                  "热化学储热", "相变储热"]:
            excl(p, "需求为工艺蒸汽：该路径不产出蒸汽")
    elif demand == "供暖/热水":
        keep["直接换热供暖"] = t_src >= 60
        reasons["直接换热供暖"] = "直接换热适用 60℃ 以上热源" if t_src >= 60 \
            else "热源低于 60℃，直接换热效率过低"
        keep["吸收式热泵提温"] = t_src >= 90
        reasons["吸收式热泵提温"] = "热源 ≥90℃ 时可驱动吸收式热泵" if t_src >= 90 \
            else "吸收式热泵需 ≥90℃ 驱动热源"
        keep["压缩式热泵提温"] = True
        reasons["压缩式热泵提温"] = "压缩式热泵以电驱动，适用低温余热提温"
        for p in ["余热锅炉直接产汽", "ORC 余热发电", "TEG 热电发电",
                  "热化学储热", "相变储热"]:
            excl(p, "需求为供暖/热水：该路径不直接产热")
    else:
        keep["热化学储热"] = True
        reasons["热化学储热"] = "储热路径可跨时段调峰"
        keep["相变储热"] = True
        reasons["相变储热"] = "储热路径可跨时段调峰"
        keep["直接换热供暖"] = t_src >= 60
        reasons["直接换热供暖"] = "供暖需求场景候选" if t_src >= 60 else "热源温度过低"
        keep["吸收式热泵提温"] = t_src >= 90
        reasons["吸收式热泵提温"] = "提温需求候选" if t_src >= 90 else "驱动热源不足"
        keep["压缩式热泵提温"] = True
        reasons["压缩式热泵提温"] = "提温需求候选"
        keep["ORC 余热发电"] = t_src >= 110
        reasons["ORC 余热发电"] = "发电调峰候选" if t_src >= 110 else "驱动温度不足"
        keep["TEG 热电发电"] = t_src >= 40
        reasons["TEG 热电发电"] = "发电兜底候选" if t_src >= 40 else "温差不足"
        keep["余热锅炉直接产汽"] = t_src >= t_steam + 20
        reasons["余热锅炉直接产汽"] = "产汽候选" if t_src >= t_steam + 20 \
            else f"热源不足以直接产 {t_steam}℃ 蒸汽"

    if continuity == "连续":
        for p in ["热化学储热", "相变储热"]:
            if keep.get(p, True):
                reasons[p] = "热源连续：储热非必需（保留作调峰候选）"
    for p in PATH_NAMES:
        keep.setdefault(p, True)
        reasons.setdefault(p, "通过第一级筛选")
    return keep, reasons


def orc_reduction(t_src, hours, dT):
    sweep = load_sweep(_mtime("dwsim_sweep_full.csv"))
    tmax_k = t_src + 273.15 - dT
    ok = sweep[sweep["heater_outlet_K"] <= tmax_k]
    if len(ok) == 0:
        return None
    nets = ok["net_kW"].values
    p10, p50, p90 = np.percentile(nets, [10, 50, 90])
    mwh = p50 * hours / 1000.0
    co2 = mwh * 0.581
    money = mwh * 1000 * 0.65 / 10000.0
    return {"n_cond": len(ok), "p10": p10, "p50": p50, "p90": p90,
            "mwh": mwh, "co2": co2, "money": money}


def heat_reduction(t_src, m_dot, medium, hours, dT):
    cp = {"热水/冷凝水": 4.2, "烟气": 1.1, "工艺液体": 2.5}[medium]
    t_out = 40.0
    dt = max(t_src - t_out - dT, 5.0)
    q_kw = m_dot * cp * dt
    heat_gj = q_kw * hours * 3.6 / 1000.0
    co2 = heat_gj * 0.0561 / 0.90
    money = heat_gj * 98.0 / 10000.0
    return {"q_kw": q_kw, "heat_gj": heat_gj, "co2": co2, "money": money}


# ---------------------------------------------------------------
# 深色科技风图表
# ---------------------------------------------------------------
CHART_FONT = dict(family="Microsoft YaHei, PingFang SC, sans-serif",
                  color="#E2E8F0", size=13)
GRID = "rgba(148,163,184,.13)"


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
    "直接换热供暖": "#22D3EE", "余热锅炉直接产汽": "#38BDF8",
    "吸收式热泵提温": "#34D399", "压缩式热泵提温": "#2DD4BF",
    "ORC 余热发电": "#4ADE80", "热化学储热": "#A3E635",
    "相变储热": "#FBBF24", "TEG 热电发电": "#F472B6",
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
        xaxis=dict(gridcolor=GRID, zeroline=False,
                   title="λ（0=纯熵权，1=纯AHP）", tickmode="linear", dtick=0.1),
        yaxis=dict(gridcolor=GRID, zeroline=False, title="TOPSIS 贴近度",
                   range=[0, 1.05]),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)),
        margin=dict(l=40, r=20, t=40, b=40))
    return fig


def pareto_figure():
    front = load_front_pymoo(_mtime("pymoo_pareto.csv"))
    real = load_front_real(_mtime("pareto_front_real.csv")).sort_values("Q_in_kW")
    ver = load_verify(_mtime("dwsim_verify_pymoo.csv"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=front["Q_in_kW"], y=front["net_kW"], mode="markers",
        marker=dict(size=8, color=front["thermal_eff"],
                    colorscale=[[0, "#0EA5E9"], [0.5, "#22D3EE"], [1, "#34D399"]],
                    opacity=0.5, line=dict(width=0, color="#0B1220"),
                    colorbar=dict(title="热效率", thickness=14,
                                  tickfont=dict(color="#94A3B8", size=11))),
        name="pymoo 100 解（代理预测）",
        hovertemplate="吸热量 %{x:.1f} kW<br>净功率 %{y:.1f} kW<br>热效率 %{marker.color:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=real["Q_in_kW"], y=real["net_kW"], mode="lines+markers",
        line=dict(color="#34D399", width=2.5, shape="spline"),
        marker=dict(size=10, color="#34D399",
                    line=dict(color="#0B1220", width=1.5)),
        name="22 解（DWSIM 复核）",
        hovertemplate="吸热量 %{x:.1f} kW<br>净功率 %{y:.1f} kW<br><extra>22 解复核</extra>"))
    fig.add_trace(go.Scatter(
        x=ver["dwsim_qin"], y=ver["dwsim_net"], mode="markers+text",
        marker=dict(symbol="star", size=16, color="#FBBF24",
                    line=dict(color="#0B1220", width=1.5)),
        text=[f"{e:.1f}%" for e in ver["net_err_pct"]],
        textposition="top center", textfont=dict(color="#FBBF24", size=11),
        name="5 个 DWSIM 复核点",
        hovertemplate="吸热量 %{x:.1f} kW<br>净功率 %{y:.1f} kW<br>误差 %{text}<extra></extra>"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,28,46,.55)", height=470, font=CHART_FONT,
        xaxis=dict(gridcolor=GRID, zeroline=False, title="吸热量 Q_in (kW)"),
        yaxis=dict(gridcolor=GRID, zeroline=False, title="净功率 (kW)"),
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12)),
        margin=dict(l=40, r=20, t=30, b=40))
    return fig


def pareto_co2_figure():
    """碳减排—成本—能效 帕累托散点图（碳减排=净功率×8000h×0.581，推算口径）。"""
    front = load_front_pymoo(_mtime("pymoo_pareto.csv"))
    real = load_front_real(_mtime("pareto_front_real.csv"))
    front = front.assign(co2=front["net_kW"] * 8000.0 / 1000.0 * 0.581,
                         cost=front["cost_proxy"])
    real = real.assign(co2=real["net_kW"] * 8000.0 / 1000.0 * 0.581)
    real = real.assign(cost=0.12 * real["Q_in_kW"]
                       + 30.0 * (real["pump_outlet_Pa"] / 1e6) ** 2
                       + 120.0 * (1.0 - real["expander_eff"]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=front["cost"], y=front["co2"], mode="markers",
        marker=dict(size=7, color=front["thermal_eff"],
                    colorscale="Viridis", opacity=0.5,
                    colorbar=dict(title="热效率", thickness=14,
                                  tickfont=dict(color="#94A3B8", size=11))),
        name="pymoo 100 解（代理预测）", showlegend=False,
        hovertemplate="成本代理 %{x:.0f} 万元<br>年碳减排 %{y:.0f} tCO2<extra></extra>"))
    pf = real.sort_values("cost")
    fig.add_trace(go.Scatter(
        x=pf["cost"], y=pf["co2"], mode="lines",
        line=dict(color="#FBBF24", width=2, dash="dot"),
        opacity=0.7, name="DWSIM 复核前沿（22 解）",
        hovertemplate="成本代理 %{x:.0f} 万元<br>年碳减排 %{y:.0f} tCO2<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=real["cost"], y=real["co2"], mode="markers",
        marker=dict(symbol="star", size=14, color="#FBBF24",
                    line=dict(color="#0B1220", width=1)),
        name="22 解（DWSIM 真实复核）",
        hovertemplate="成本代理 %{x:.0f} 万元<br>年碳减排 %{y:.0f} tCO2<br>净功率 %{customdata[0]:.1f} kW<extra></extra>",
        customdata=real[["net_kW"]].values))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,28,46,.55)", height=470, font=CHART_FONT,
        xaxis=dict(gridcolor=GRID, zeroline=False,
                   title="设备成本代理（万元，示意）"),
        yaxis=dict(gridcolor=GRID, zeroline=False,
                   title="年碳减排（tCO2/年，推算口径）"),
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12)),
        annotations=[dict(
            x=1.0, y=1.0, xref="paper", yref="paper", xanchor="right",
            yanchor="top", showarrow=False,
            text="更优方向：成本 ↓ 且减碳 ↑",
            font=dict(color="#6EE7B7", size=12),
            bgcolor="rgba(14,23,41,.75)",
            bordercolor="rgba(52,211,153,.4)", borderwidth=1,
            borderpad=6)],
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
        textposition="outside", textfont=dict(color="#94A3B8", size=10),
        hovertemplate="%{x}月 降碳 %{y:.1f} tCO2<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=df["月份"], y=df["累计降碳tCO2"], mode="lines+markers",
        name="累计 (tCO2)", yaxis="y2",
        line=dict(color="#FBBF24", width=3),
        marker=dict(size=7, color="#FBBF24", line=dict(color="#0B1220", width=1)),
        hovertemplate="%{x}月 累计 %{y:.1f} tCO2<extra></extra>"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,28,46,.55)", height=440, font=CHART_FONT,
        bargap=0.32,
        xaxis=dict(gridcolor=GRID, zeroline=False, title="月份"),
        yaxis=dict(gridcolor=GRID, zeroline=False, title="月度降碳 (tCO2)"),
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
st.markdown(
    '<div class="hero">'
    '<span class="hero-badge">时代杯 · 智能控碳</span>'
    '<span class="hero-badge" style="background:linear-gradient(90deg,#34D399,#FBBF24)">零碳科技</span>'
    '<div class="hero-title">工业余热回收利用 · 智能决策演示平台</div>'
    '<div class="hero-sub">《工业废热或余热回收利用降碳技术路径与智能优化评价方法》｜路径筛选 → TOPSIS 排序 → 减碳估算 → 帕累托前沿 → 动态 LCA</div>'
    '</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div style="font-size:17px;font-weight:800;color:#E2E8F0;margin-bottom:4px">场景参数</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="side-sec">热源与需求</div>', unsafe_allow_html=True)
    t_src = st.slider("热源温度 (℃)", 60, 300, 120, 5)
    m_dot = st.slider("热源流量 (kg/s)", 0.5, 5.0, 1.0, 0.1)
    demand = st.selectbox("用能需求", ["发电", "工艺蒸汽", "供暖/热水", "储热调峰"])
    continuity = st.radio("热源连续性", ["连续", "间歇"], horizontal=True)
    st.markdown('<div class="side-sec">运行与核算假设</div>', unsafe_allow_html=True)
    hours = st.slider("年运行小时 (h)", 4000, 8000, 8000, 500)
    dT = st.slider("换热端差 (℃)", 5, 20, 10, 1)
    medium = st.selectbox("热介质（供热估算用）", ["热水/冷凝水", "烟气", "工艺液体"])
    st.markdown('<div class="side-sec">评价权重</div>', unsafe_allow_html=True)
    lam = st.slider("组合权重 λ（主观占比）", 0.0, 1.0, 0.5, 0.05)
    st.caption("λ=主观(AHP)占比；1−λ=客观(熵权)占比")

keep, reasons = stage1(t_src, demand, continuity)
survivors = [p for p in PATH_NAMES if keep[p]]

st.markdown('<div class="sec-title"><span class="tag">01</span>第一级 · 热力学规则粗筛</div>',
            unsafe_allow_html=True)
rows = [{"路径": p, "结果": "✓ 通过" if keep[p] else "✗ 排除", "原因": reasons[p]}
        for p in PATH_NAMES]
st.dataframe(style_stage_table(pd.DataFrame(rows)), use_container_width=True,
             hide_index=True)
st.info(f"进入第二级候选：{'、'.join(survivors) if survivors else '无（请调整场景参数）'}")

st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

st.markdown('<div class="sec-title"><span class="tag">02</span>第二级 · TOPSIS 精细排序</div>',
            unsafe_allow_html=True)
if survivors:
    X = build_matrix(survivors, t_src, m_dot, medium, hours, dT)
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
    st.dataframe(style_topsis_table(df_r), use_container_width=True,
                 hide_index=True)
    st.markdown(
        f'<div class="rec-banner">推荐路径：<b>{df_r.iloc[0]["路径"]}</b>'
        f'<span class="muted">　贴近度 {df_r.iloc[0]["TOPSIS贴近度"]:.3f} · λ={lam:.2f}'
        f' · 权重来自 AHP+熵权组合赋权</span></div>',
        unsafe_allow_html=True)
    st.caption("数据口径：能效列中 ORC 为 DWSIM 仿真推荐设计点热效率（12.3%）；压缩式热泵为一次能源效率"
               "（COP2.8×电网效率38%≈106%），吸收式热泵为余热自驱动 COP0.75，直接换热/余热锅炉为热回收率；"
               "减排列按当前参数推算（替代购电/替代天然气），可复算；**投资/回收期**：ORC 引自《重庆大学学报》"
               "(23800元/kW、5.58年)，压缩式与吸收式热泵的回收期引自同一对比研究"
               "（李萌：4.23年/2.73年），投资为工程估算（大庆改造案例 627万/14.4MW 为特例，见案例对标），"
               "直接换热/余热锅炉为工程估算，储热与 TEG 为示意；**运行成本**：压缩式按 COP2.8 反推耗电"
               "×当前运行小时×0.65元/kWh 动态推算，其余为工程估算；**政策分**按《节能降碳中央预算内投资专项管理办法》"
               "与《绿色低碳转型产业指导目录(2024年版)》收录情况打分，详见《数据来源台账》；"
               "主观权重 AHP 为演示假设（需专家打分），客观熵权随当前候选矩阵动态计算。")
    csv = df_r.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("下载排序结果 CSV", data=csv,
                       file_name="两级决策_演示结果.csv", mime="text/csv")
else:
    st.warning("无候选路径，请调整参数")

st.divider()

st.markdown('<div class="sec-title"><span class="tag">03</span>减碳与收益估算（推算口径，非实测）</div>',
            unsafe_allow_html=True)
orc = orc_reduction(t_src, hours, dT) if demand in ("发电", "储热调峰") else None
heat = heat_reduction(t_src, m_dot, medium, hours, dT) if demand in (
    "工艺蒸汽", "供暖/热水", "储热调峰") else None

c1, c2, c3, c4 = st.columns(4)
if orc is not None:
    c1.metric("ORC 可行工况数", f"{orc['n_cond']}", "400 工况中")
    c2.metric("净功率 P50 (仿真分布)", f"{orc['p50']:.1f} kW",
              f"P10~P90: {orc['p10']:.1f}~{orc['p90']:.1f}")
    c3.metric("年降碳（推算）", f"{orc['co2']:.1f} tCO2",
              f"{orc['mwh']:.0f} MWh × 0.581")
    c4.metric("年节省电费（演示价 0.65 元/kWh）", f"{orc['money']:.1f} 万元",
              "按替代购电口径")
elif heat is not None:
    c1.metric("回收热功率（估算）", f"{heat['q_kw']:.0f} kW", f"ṁ={m_dot} kg/s × cp×ΔT")
    c2.metric("年回收热量", f"{heat['heat_gj']:.0f} GJ", f"{hours} h/年")
    c3.metric("年降碳（替代天然气，估算）", f"{heat['co2']:.1f} tCO2",
              "0.0561 t/GJ ÷ 锅炉效率90%")
    c4.metric("年节省燃气费（演示价）", f"{heat['money']:.1f} 万元", "3.5 元/m³ 折算")
else:
    for cc in (c1, c2, c3, c4):
        cc.metric("—", "—", "无可用估算")
st.caption("说明：ORC 数字基于 DWSIM 400 工况仿真分布（推算口径）；供热数字为替代天然气估算，参数（电价/气价/锅炉效率）均为演示假设，正式核算须以项目实测为准。")

st.divider()

st.markdown('<div class="sec-title"><span class="tag">04</span>帕累托前沿 · 多目标优化结果</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sec-note">青色点 = 代理预测 100 解｜绿色线 = 22 解 DWSIM 复核｜金星 = 5 个复核点（标注误差）</div>',
            unsafe_allow_html=True)
st.plotly_chart(pareto_figure(), use_container_width=True)
st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
st.markdown('<div class="sec-note">碳减排—成本—能效权衡：碳减排 = 净功率 × 8000h × 0.581（推算口径）；金星 = 22 个 DWSIM 复核点</div>',
            unsafe_allow_html=True)
st.plotly_chart(pareto_co2_figure(), use_container_width=True)
st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title"><span class="tag">05</span>动态 LCA · 月度滚动核算</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sec-note">青绿柱 = 月度降碳｜金色线 = 全年累计（推算口径，年 173.2 tCO2）；'
            '考虑设备制造排放 7.2 tCO2e（20 年摊销 0.36 t/年，工程估算），'
            '全生命周期口径年净降碳约 172.8 tCO2</div>',
            unsafe_allow_html=True)
st.plotly_chart(lca_figure(), use_container_width=True)

st.divider()

st.markdown('<div class="sec-title"><span class="tag">06</span>敏感性 · λ 对排序的影响</div>',
            unsafe_allow_html=True)
lam_grid = np.linspace(0.0, 1.0, 11)
if survivors:
    X_lam = build_matrix(survivors, t_src, m_dot, medium, hours, dT)
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
    st.dataframe(style_lambda_table(df_s), use_container_width=True, hide_index=True)
    # 贴近度-λ 曲线：权重对每个候选的影响一目了然
    st.plotly_chart(lambda_figure(survivors, X_lam), use_container_width=True)
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
        "1. **仿真数据**：DWSIM 9.0.5 稳态仿真 400 工况（真实仿真，能量守恒最大偏差 0.013 kW）；\n"
        "2. **代理模型**：scikit-learn MLP，测试集 R² = 0.9968 / 0.9925 / 1.0000（热效率交叉验证 R² 较低，约 0.53，"
        "模型精度以净功率/吸热量为主）；pymoo 100 解为代理预测，其中 22 个前沿解 + 5 个代表点经 DWSIM 复核"
        "（净功率误差 <1%，热效率误差 ≤3.2%）；\n"
        "3. **减碳**：ORC 为仿真净功率 × 运行小时 × 华东电网因子 0.581 的**推算口径**，供热为替代天然气估算，均非实测；"
        "设备制造排放 7.2 tCO2e 为工程估算（20 年摊销 0.36 t/年），全生命周期口径年净降碳约 172.8 t；\n"
        "4. **成本/回收期**：示意性代理模型，非真实报价；CCER 收益为情景假设，未完成备案方法学前不计入基准财务指标；\n"
        "5. 本平台全部代码与数据随申报材料提交，可复算、可溯源。")

st.markdown(
    '<div class="footer">'
    '《工业废热或余热回收利用降碳技术路径与智能优化评价方法》· 时代杯零碳科技创新大赛 · '
    'DWSIM 仿真 + 机器学习 + 多目标优化 + 动态 LCA + 两级决策<br>'
    '数据口径：仿真/推算数据，非实测；成本为示意代理；CCER 为情景假设；'
    '投资/回收期/政策分来源见《数据来源台账》'
    '</div>', unsafe_allow_html=True)
