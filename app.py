# -*- coding: utf-8 -*-
"""
工业余热回收利用智能决策演示平台
=================================
用于"时代杯"答辩现场演示：输入热源温度、流量、用能需求等参数，
实时输出：路径筛选 → TOPSIS 排序 → 减碳/收益估算 → 帕累托前沿 → 动态 LCA。

数据口径说明（与申报书一致）：
  - DWSIM 400 工况：真实稳态仿真（dwsim_sweep_full.csv）
  - 帕累托前沿：22 解真实复核 + pymoo 100 解（代理预测） + 5 个 DWSIM 复核点
  - 减碳：ORC 用仿真净功率 × 运行小时 × 华东电网因子 0.581（推算口径）
  - 成本/回收期：示意性代理模型；TOPSIS 指标为典型演示值
  - 动态 LCA：lca_monthly_real.csv（推算口径，年 173.2 tCO2）

运行：streamlit run app.py
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="工业余热回收智能决策演示平台", layout="wide")

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "data")


@st.cache_data
def load_sweep():
    return pd.read_csv(os.path.join(DATA, "dwsim_sweep_full.csv"), encoding="utf-8-sig")


@st.cache_data
def load_front_real():
    return pd.read_csv(os.path.join(DATA, "pareto_front_real.csv"), encoding="utf-8-sig")


@st.cache_data
def load_front_pymoo():
    return pd.read_csv(os.path.join(DATA, "pymoo_pareto.csv"), encoding="utf-8-sig")


@st.cache_data
def load_verify():
    return pd.read_csv(os.path.join(DATA, "dwsim_verify_pymoo.csv"), encoding="utf-8-sig")


@st.cache_data
def load_lca_monthly():
    return pd.read_csv(os.path.join(DATA, "lca_monthly_real.csv"), encoding="utf-8-sig")


@st.cache_data
def load_weights():
    with open(os.path.join(DATA, "eval_weights.json"), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------
# 路径库（8 条主路径）
# ---------------------------------------------------------------
PATH_NAMES = [
    "直接换热供暖",
    "余热锅炉直接产汽",
    "吸收式热泵提温",
    "压缩式热泵提温",
    "ORC 余热发电",
    "热化学储热",
    "相变储热",
    "TEG 热电发电",
]

# TOPSIS 典型演示指标（行=路径，列=指标；正式应用须替换文献/实测数据）
INDICATORS = ["能效%", "投资万元/MW", "回收期年", "CO2减排t/年", "政策分"]
DIRECTIONS = ["max", "min", "min", "max", "max"]
RAW = {
    "直接换热供暖":   [90,  50, 2.0, 800, 4],
    "余热锅炉直接产汽": [85,  90, 3.0, 1000, 4],
    "吸收式热泵提温": [125, 160, 4.0, 1350, 5],
    "压缩式热泵提温": [130, 140, 3.5, 1250, 4],
    "ORC 余热发电":  [12, 200, 7.0, 600, 3],
    "热化学储热":    [85, 190, 5.5, 950, 3],
    "相变储热":      [80, 150, 4.8, 880, 3],
    "TEG 热电发电":  [5, 220, 9.0, 400, 2],
}

MAP_TO_TOPSIS = {
    "能效%": "系统能效",
    "投资万元/MW": "初始投资",
    "回收期年": "投资回收期",
    "CO2减排t/年": "CO2当量减排",
    "政策分": "政策补贴适配度",
}


def combined_weights(lam):
    """按 λ 重组主观/客观权重，并映射到 TOPSIS 5 指标。"""
    d = load_weights()
    names = d["指标"]
    w_sub = np.array(d["主观权重_AHP"], dtype=float)
    w_obj = np.array(d["客观权重_熵权"], dtype=float)
    w_all = lam * w_sub + (1 - lam) * w_obj
    w_all = w_all / w_all.sum()
    lookup = dict(zip(names, w_all))
    w5 = np.array([lookup[MAP_TO_TOPSIS[c]] for c in INDICATORS])
    return w5 / w5.sum()


def topsis(matrix, weights):
    norm = matrix / np.sqrt((matrix ** 2).sum(axis=0))
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


def stage1(t_src, demand, continuity, t_steam=152):
    """第一级：热力学规则粗筛（可解释、可审计）。"""
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
    else:  # 储热调峰
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
    """基于 400 工况仿真分布的 ORC 减碳/收益估算（推算口径）。"""
    sweep = load_sweep()
    tmax_k = t_src + 273.15 - dT
    ok = sweep[sweep["heater_outlet_K"] <= tmax_k]
    if len(ok) == 0:
        return None
    nets = ok["net_kW"].values
    p10, p50, p90 = np.percentile(nets, [10, 50, 90])
    mwh = p50 * hours / 1000.0
    co2 = mwh * 0.581
    money = mwh * 1000 * 0.65 / 10000.0  # 演示工业电价 0.65 元/kWh -> 万元
    return {
        "n_cond": len(ok), "p10": p10, "p50": p50, "p90": p90,
        "mwh": mwh, "co2": co2, "money": money,
    }


def heat_reduction(t_src, m_dot, medium, hours, dT):
    """供热/产汽路径的演示估算（替代天然气口径，明确为估算）。"""
    cp = {"热水/冷凝水": 4.2, "烟气": 1.1, "工艺液体": 2.5}[medium]
    t_out = 40.0  # 排放温度假设
    dt = max(t_src - t_out - dT, 5.0)
    q_kw = m_dot * cp * dt
    heat_gj = q_kw * hours * 3.6 / 1000.0
    co2 = heat_gj * 0.0561 / 0.90  # 替代天然气，锅炉效率 90%
    money = heat_gj * 98.0 / 10000.0  # 天然气 3.5 元/m³，热值 35.6 MJ/m³ -> 万元
    return {"q_kw": q_kw, "heat_gj": heat_gj, "co2": co2, "money": money}


def pareto_figure():
    front = load_front_pymoo()
    real = load_front_real()
    ver = load_verify()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=front["Q_in_kW"], y=front["net_kW"], mode="markers",
        marker=dict(color=front["thermal_eff"], colorscale="Viridis",
                    size=7, opacity=0.45, colorbar=dict(title="热效率")),
        name="pymoo 100 解（代理预测）"))
    fig.add_trace(go.Scatter(
        x=real["Q_in_kW"], y=real["net_kW"], mode="markers",
        marker=dict(color="red", size=9, symbol="circle"),
        name="22 解（DWSIM 复核）"))
    fig.add_trace(go.Scatter(
        x=ver["dwsim_qin"], y=ver["dwsim_net"], mode="markers",
        marker=dict(color="gold", size=14, symbol="star",
                    line=dict(color="black", width=1)),
        text=[f"误差 {e:.1f}%" for e in ver["net_err_pct"]],
        name="5 个 DWSIM 复核点"))
    fig.update_layout(
        height=460, title="帕累托前沿：净功率—吸热量（真实复核 + 代理扩展）",
        xaxis_title="吸热量 Q_in (kW)", yaxis_title="净功率 (kW)",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=40, r=20, t=60, b=40))
    return fig


def lca_figure():
    df = load_lca_monthly()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["月份"], y=df["月度降碳tCO2"], name="月度降碳 (tCO2)"))
    fig.add_trace(go.Scatter(
        x=df["月份"], y=df["累计降碳tCO2"], mode="lines+markers",
        name="累计 (tCO2)", line=dict(color="red", width=3)))
    fig.update_layout(
        height=360, title="动态 LCA：月度滚动核算（推算口径，年 173.2 tCO2）",
        xaxis_title="月份", yaxis_title="tCO2",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=40, r=20, t=60, b=40))
    return fig


# ---------------------------------------------------------------
# 页面
# ---------------------------------------------------------------
st.title("工业余热回收利用 · 智能决策演示平台")
st.caption("《工业废热或余热回收利用降碳技术路径与智能优化评价方法》 ｜ 时代杯零碳科技创新大赛")

with st.sidebar:
    st.header("场景参数")
    t_src = st.slider("热源温度 (℃)", 60, 300, 120, 5)
    m_dot = st.slider("热源流量 (kg/s)", 0.5, 5.0, 1.0, 0.1)
    demand = st.selectbox("用能需求", ["发电", "工艺蒸汽", "供暖/热水", "储热调峰"])
    continuity = st.radio("热源连续性", ["连续", "间歇"], horizontal=True)
    hours = st.slider("年运行小时 (h)", 4000, 8000, 8000, 500)
    dT = st.slider("换热端差 (℃)", 5, 20, 10, 1)
    medium = st.selectbox("热介质（供热估算用）", ["热水/冷凝水", "烟气", "工艺液体"])
    lam = st.slider("组合权重 λ（主观占比）", 0.0, 1.0, 0.5, 0.05)

st.sidebar.caption("λ=主观(AHP)占比；1−λ=客观(熵权)占比")

keep, reasons = stage1(t_src, demand, continuity)
survivors = [p for p in PATH_NAMES if keep[p]]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("① 第一级：热力学规则粗筛")
    rows = []
    for p in PATH_NAMES:
        rows.append({"路径": p, "结果": "✓ 通过" if keep[p] else "✗ 排除",
                     "原因": reasons[p]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.info(f"进入第二级候选：{'、'.join(survivors) if survivors else '无（请调整场景参数）'}")

with col2:
    st.subheader("② 第二级：TOPSIS 精细排序（组合权重）")
    if survivors:
        w5 = combined_weights(lam)
        X = np.array([RAW[p] for p in survivors], dtype=float)
        c = topsis(X, w5)
        df_r = pd.DataFrame({
            "路径": survivors,
            "能效%": X[:, 0],
            "投资万元/MW": X[:, 1],
            "回收期年": X[:, 2],
            "CO2减排t/年(演示)": X[:, 3],
            "政策分": X[:, 4],
            "TOPSIS贴近度": np.round(c, 4),
        }).sort_values("TOPSIS贴近度", ascending=False).reset_index(drop=True)
        df_r.insert(0, "排名", range(1, len(df_r) + 1))
        st.dataframe(df_r, use_container_width=True, hide_index=True)
        st.markdown(
            f"**推荐路径：{df_r.iloc[0]['路径']}**（贴近度 {df_r.iloc[0]['TOPSIS贴近度']:.3f}，λ={lam:.2f}）")
        st.caption("指标为典型演示值，正式应用须替换为文献/实测数据；权重来自 eval_index_system.py 的 AHP+熵权组合赋权。")
        csv = df_r.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("下载排序结果 CSV", data=csv,
                           file_name="两级决策_演示结果.csv", mime="text/csv")
    else:
        st.warning("无候选路径，请调整参数")

st.divider()

st.subheader("③ 减碳与收益估算（推算口径，非实测）")
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
    c1.metric("回收热功率（估算）", f"{heat['q_kw']:.0f} kW",
              f"ṁ={m_dot} kg/s × cp×ΔT")
    c2.metric("年回收热量", f"{heat['heat_gj']:.0f} GJ",
              f"{hours} h/年")
    c3.metric("年降碳（替代天然气，估算）", f"{heat['co2']:.1f} tCO2",
              "0.0561 t/GJ ÷ 锅炉效率90%")
    c4.metric("年节省燃气费（演示价）", f"{heat['money']:.1f} 万元",
              "3.5 元/m³ 折算")
else:
    for cc in (c1, c2, c3, c4):
        cc.metric("—", "—", "无可用估算")
st.caption("说明：ORC 数字基于 DWSIM 400 工况仿真分布（推算口径）；供热数字为替代天然气估算，参数（电价/气价/锅炉效率）均为演示假设，正式核算须以项目实测为准。")

st.divider()

left, right = st.columns([1.4, 1])
with left:
    st.subheader("④ 帕累托前沿（多目标优化结果）")
    st.plotly_chart(pareto_figure(), use_container_width=True)
with right:
    st.subheader("⑤ 动态 LCA 月度滚动核算")
    st.plotly_chart(lca_figure(), use_container_width=True)

st.divider()
st.subheader("⑥ 敏感性：λ 变化对排序的影响")
lam_grid = np.linspace(0.0, 1.0, 11)
if survivors:
    top1 = []
    for lam_t in lam_grid:
        w = combined_weights(float(lam_t))
        c_t = topsis(np.array([RAW[p] for p in survivors], dtype=float), w)
        top1.append(survivors[int(np.argmax(c_t))])
    df_s = pd.DataFrame({"λ": [f"{v:.2f}" for v in lam_grid], "第一名": top1})
    st.dataframe(df_s, use_container_width=True, hide_index=True)
    stable = len(set(top1)) == 1
    st.success(f"λ 从 0 到 1 全程第一名{'稳定为「' + top1[0] + '」' if stable else '发生变化'}；"
               f"{'排序对权重设定稳健。' if stable else '提示：排序对 λ 敏感，正式应用需收窄不确定区间。'}")
else:
    st.warning("无候选路径")

st.divider()
with st.expander("数据口径与免责声明（答辩必讲）"):
    st.markdown(
        "1. **仿真数据**：DWSIM 9.0.5 稳态仿真 400 工况（真实仿真，能量守恒最大偏差 0.013 kW）；\n"
        "2. **代理模型**：scikit-learn MLP，测试集 R² = 0.9968 / 0.9925 / 1.0000；pymoo 100 解为代理预测，其中 22 解 + 5 个代表点经 DWSIM 复核（误差 <1%）；\n"
        "3. **减碳**：ORC 为仿真净功率 × 运行小时 × 华东电网因子 0.581 的**推算口径**，供热为替代天然气估算，均非实测；\n"
        "4. **成本/回收期**：示意性代理模型，非真实报价；CCER 收益为情景假设，未完成备案方法学前不计入基准财务指标；\n"
        "5. 本平台全部代码与数据随申报材料提交，可复算、可溯源。")
