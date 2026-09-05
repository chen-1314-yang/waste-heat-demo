# -*- coding: utf-8 -*-
"""决策内核 v2：全温域工业废热回收两级决策（纯函数，无 UI）。

红线（v2.1 起）：
- 制冷（供冷）/干燥 = 受支持需求；除湿未单列为需求（并入供冷/干燥场景后评价）；
- 外购蒸汽驱动吸收式制冷是用能设备，不设废热回收路径；
- 能力圈外不硬给推荐；温度包络 25~650℃；ORC 110~350℃；蒸汽朗肯 280~650℃。
"""

import functools
import json
import os

import numpy as np
import pandas as pd

PATH_KEYS = ["direct", "whb_steam", "abs_self", "abs_ext", "comp",
             "orc", "steam_pp", "tc_storage", "pcm_storage", "teg",
             "abs_cool", "comp_cool"]
DISPLAY = {
    "direct": "直接换热",
    "whb_steam": "余热锅炉直接产汽",
    "abs_self": "吸收式热泵提温（余热自驱动）",
    "abs_ext": "吸收式热泵（外购蒸汽驱动）",
    "comp": "压缩式热泵提温（电驱动）",
    "orc": "ORC 余热发电",
    "steam_pp": "高温蒸汽发电（蒸汽朗肯）",
    "tc_storage": "热化学储热",
    "pcm_storage": "相变储热",
    "teg": "TEG 热电发电",
    "abs_cool": "吸收式制冷（余热驱动）",
    "comp_cool": "电压缩制冷（电驱动）",
}

DEMAND_SET = {"发电", "工艺蒸汽", "供暖·热水", "储热调峰", "供冷", "干燥"}
UNSUPPORTED_DEMAND = {"除湿", "除湿/干燥"}
CP = {"烟气": 1.1, "热水·冷凝水": 4.2, "工艺液体": 2.5}


class DecisionError(Exception):
    pass


class OutOfScopeError(DecisionError):
    pass


def validate_scene(scene):
    """校验并补默认值；越界抛出 OutOfScopeError。"""
    s = dict(scene)
    s.setdefault("需求温度_degC", None)
    s.setdefault("连续性", "连续")
    s.setdefault("驱动来源", "余热自驱动")
    s.setdefault("年运行小时", 8000)
    s.setdefault("换热端差_degC", 10)
    s.setdefault("规模_kW", None)
    s.setdefault("流量_kg_s", None)
    demand = s.get("需求")
    if demand in UNSUPPORTED_DEMAND:
        raise OutOfScopeError(
            f"需求「{demand}」未单列为 v2 需求类型：除湿并入供冷/干燥场景后评价，"
            "模型无法单独评价。")
    if demand not in DEMAND_SET:
        raise OutOfScopeError(f"未知需求类型：{demand}")
    t = s["热源温度_degC"]
    if not (25 <= t <= 650):
        raise OutOfScopeError(
            f"热源 {t}℃ 超出 v2 标定包络（25~650℃）"
            + ("；高于 650℃ 的发电/产汽需多压/再热锅炉专项设计"
               if t > 650 else "；低于 25℃ 需额外驱动能投入，不做评价"))
    return s


def recovered_heat_kw(scene):
    """可回收热功率：优先 流量×cp×ΔT；否则用规模_kW。"""
    m = scene.get("流量_kg_s")
    if m:
        dt = max(scene["热源温度_degC"] - 40.0 - scene["换热端差_degC"], 5.0)
        return m * CP[scene["载体"]] * dt
    q = scene.get("规模_kW")
    if q:
        return float(q)
    raise DecisionError("缺少流量或规模_kW，无法计算可回收热功率")


def scale_band(q_kw):
    return "小" if q_kw < 1000 else ("中" if q_kw <= 5000 else "大")


def stage1(scene):
    """第一级规则：返回 {path_key: (keep, reason)}，10 条全覆盖。"""
    t = scene["热源温度_degC"]
    demand = scene["需求"]
    t_dem = scene.get("需求温度_degC")
    driver = scene.get("驱动来源", "余热自驱动")
    cont = scene["连续性"]
    keep = {p: False for p in PATH_KEYS}
    why = {}

    def setk(p, ok, reason):
        keep[p] = ok
        why[p] = reason

    if demand == "发电":
        setk("orc", 110 <= t <= 350,
             (f"热源 {t}℃ 在 ORC 适用区间 110~350℃" if 110 <= t <= 350 else
              (f"热源 {t}℃ <110℃，ORC 温差不足" if t < 110 else
               f"热源 {t}℃ >350℃，超出 ORC 标定上限，应走蒸汽朗肯")))
        setk("steam_pp", 280 <= t <= 650,
             (f"热源 {t}℃ 满足蒸汽朗肯标定区间 280~650℃"
              "（锅炉出口按 min(max(t-100,180),540) 映射）"
              if 280 <= t <= 650 else
              (f"热源 {t}℃ <280℃，蒸汽朗肯经济性不足" if t < 280 else
               f"热源 {t}℃ >650℃，超出 v2 标定上限")))
        setk("teg", t >= 40,
             "TEG 适合 ≥40℃ 温差发电（兜底候选）" if t >= 40 else "温差不足")
        for p in ["direct", "whb_steam", "abs_self", "abs_ext", "comp",
                  "tc_storage", "pcm_storage", "abs_cool", "comp_cool"]:
            setk(p, False, "需求为发电：该路径不产出电力")
    elif demand == "工艺蒸汽":
        t_steam = t_dem if t_dem else 152.0
        setk("whb_steam", t >= t_steam + 20,
             (f"热源 {t}℃ ≥ 蒸汽 {t_steam}℃ + 端差 20℃，可直接产汽"
              if t >= t_steam + 20 else
              f"热源 {t}℃ 不足以直接产生 {t_steam}℃ 蒸汽（需 ≥{t_steam + 20}℃）"))
        setk("abs_self", t >= 90,
             f"热源 {t}℃ 可自驱动吸收式（≥90℃）" if t >= 90 else
             "吸收式自驱动需 ≥90℃ 驱动热源")
        setk("comp", True, "压缩式热泵以电驱动，不受热源温度下限限制")
        setk("abs_ext", False,
             "v2 中外购蒸汽驱动吸收式仅用于供暖·热水需求（扩展点）")
        for p in ["direct", "orc", "steam_pp", "teg",
                  "tc_storage", "pcm_storage", "abs_cool", "comp_cool"]:
            setk(p, False, "需求为工艺蒸汽：该路径不产出蒸汽")
    elif demand == "供暖·热水":
        lo = max(60.0, (t_dem if t_dem else 60.0) + scene["换热端差_degC"])
        setk("direct", t >= lo,
             (f"热源 {t}℃ ≥ 直接换热下限 {lo:.0f}℃（max(60, 需求温度+端差)）"
              if t >= lo else f"热源 {t}℃ < {lo:.0f}℃，直接换热不可行"))
        setk("abs_self", t >= 90,
             "热源 ≥90℃ 可自驱动吸收式" if t >= 90 else
             "吸收式自驱动需 ≥90℃ 驱动热源")
        setk("abs_ext", driver == "外购蒸汽" and t >= 25,
             ("外购蒸汽驱动吸收式：低温余热 + 蒸汽驱动（≥25℃ 可用）"
              if driver == "外购蒸汽" and t >= 25 else
              ("未提供外购蒸汽，外购蒸汽驱动吸收式不可用"
               if driver != "外购蒸汽" else "热源温度过低")))
        setk("comp", True, "压缩式热泵以电驱动，适用低温余热提温")
        for p in ["whb_steam", "orc", "steam_pp", "teg",
                  "abs_cool", "comp_cool"]:
            setk(p, False, "需求为供暖·热水：该路径不直接产热")
        for p in ["tc_storage", "pcm_storage"]:
            setk(p, True,
                 ("热源连续，储热非必需（保留作调峰候选）" if cont == "连续"
                  else "间歇热源，储热用于时空解耦"))
    elif demand == "供冷":
        if driver == "外购蒸汽":
            for p in PATH_KEYS:
                setk(p, False,
                     "外购蒸汽驱动吸收式制冷是用能设备（非余热回收）；"
                     "请将驱动来源改为余热自驱动（利用废热）或外购电力（电制冷）")
        else:
            setk("abs_cool",
                 driver == "余热自驱动" and ABS_COOL_T_MIN <= t <= 650,
                 (f"热源 {t}℃ ≥{ABS_COOL_T_MIN:.0f}℃ 且驱动来源为余热自驱动："
                  "可驱动吸收式制冷（≥85℃ COP≈0.7；80~84℃ COP≈0.6）"
                  if driver == "余热自驱动" and ABS_COOL_T_MIN <= t <= 650 else
                  ("吸收式制冷需 ≥" + f"{ABS_COOL_T_MIN:.0f}" + "℃ 驱动热源"
                   "（当前 " + str(t) + "℃）"
                   if driver == "余热自驱动"
                   else "吸收式制冷需“余热自驱动”口径（当前驱动来源="
                        + driver + "）")))
            setk("comp_cool", driver == "外购电力",
                 ("电压缩制冷以电驱动（外购电力口径），无废热时作为基准方案"
                  if driver == "外购电力"
                  else "电压缩制冷需“外购电力”口径（当前驱动来源="
                       + driver + "）"))
            for p in [p for p in PATH_KEYS
                      if p not in ("abs_cool", "comp_cool")]:
                setk(p, False, "需求为供冷：该路径不产出冷量")
    elif demand == "干燥":
        t_dry = t_dem if t_dem else 100.0
        lo = max(60.0, t_dry + scene["换热端差_degC"])
        setk("direct", t >= lo,
             (f"热源 {t}℃ ≥ 干燥热风下限 {lo:.0f}℃（默认需求温度 100℃+端差）"
              if t >= lo else
              f"热源 {t}℃ < {lo:.0f}℃，直接换热干燥不可行"))
        setk("whb_steam", t >= t_dry + 20,
             (f"热源 {t}℃ ≥ 干燥用蒸汽下限 {t_dry + 20:.0f}℃（默认 100℃+20℃）"
              if t >= t_dry + 20 else
              f"热源 {t}℃ 不足以直接产干燥用蒸汽（需 ≥{t_dry + 20:.0f}℃）"))
        setk("abs_self", t >= 90,
             "热源 ≥90℃ 可驱动吸收式热泵供热风（一类 COP1.7）" if t >= 90 else
             "吸收式热泵需 ≥90℃ 驱动热源")
        setk("comp", True, "热泵烘干以电驱动，适用低温干燥（60~90℃）提温")
        setk("abs_ext", False,
             "外购蒸汽驱动仅用于供暖·热水需求（干燥请用余热自驱动/热泵）")
        for p in ["orc", "steam_pp", "teg", "tc_storage", "pcm_storage",
                  "abs_cool", "comp_cool"]:
            setk(p, False, "需求为干燥：该路径不直接烘干")
    elif demand == "储热调峰":
        setk("tc_storage", True, "储热路径可跨时段调峰")
        setk("pcm_storage", True, "储热路径可跨时段调峰")
        setk("direct", t >= max(60.0, (t_dem if t_dem else 60.0)
                                + scene["换热端差_degC"]),
             "供暖调峰候选")
        setk("abs_self", t >= 90,
             "提温调峰候选" if t >= 90 else "驱动热源不足")
        setk("abs_ext", driver == "外购蒸汽" and t >= 25, "外购蒸汽驱动候选")
        setk("comp", True, "提温调峰候选")
        setk("orc", 110 <= t <= 350,
             "发电调峰候选" if 110 <= t <= 350 else "超出 ORC 区间（110~350℃）")
        setk("steam_pp", 280 <= t <= 650,
             "发电调峰候选" if 280 <= t <= 650 else "超出朗肯标定（280~650℃）")
        setk("teg", t >= 40, "发电兜底候选" if t >= 40 else "温差不足")
        setk("whb_steam", t >= (t_dem if t_dem else 152.0) + 20, "产汽调峰候选")
        setk("abs_cool", False, "储热调峰需求不产冷")
        setk("comp_cool", False, "储热调峰需求不产冷")
    for p in PATH_KEYS:
        if p not in why:
            setk(p, keep[p], "通过第一级筛选")
    return {p: (keep[p], why[p]) for p in PATH_KEYS}


# ---------------- 温度相关效率（真实仿真真值） ----------------

DATA_DIR = r"D:\Codex\2026-08-06\ni\时代杯项目材料\04_仿真数据与图表"


def set_data_dir(path):
    global DATA_DIR
    DATA_DIR = path


@functools.lru_cache(maxsize=1)
def _load_orc():
    return pd.read_csv(os.path.join(DATA_DIR, "orc_sweep_coolprop.csv"),
                       encoding="utf-8-sig")


@functools.lru_cache(maxsize=1)
def _load_steam():
    return pd.read_csv(os.path.join(DATA_DIR, "steam_sweep_coolprop.csv"),
                       encoding="utf-8-sig")


def orc_eff_median_pct(t_src, dT=10.0):
    df = _load_orc()
    ok = df[df["heater_outlet_K"] <= t_src + 273.15 - dT]
    if len(ok) == 0:
        return None
    return float(np.median(ok["thermal_eff"].values) * 100.0)


@functools.lru_cache(maxsize=1)
def _steam_eff_curve():
    """锅炉出口温度(℃) → thermal_eff 中位数的插值曲线（真实仿真真值）。"""
    df = _load_steam()
    g = df.groupby("boiler_outlet_K")["thermal_eff"].median()
    t_c = (g.index.values - 273.15).astype(float)
    eff = (g.values * 100.0).astype(float)
    order = np.argsort(t_c)
    return t_c[order], eff[order]


def steam_eff_median_pct(t_src):
    """蒸汽朗肯效率：t_boiler=clamp(t_src−100,180,540) 处的线性插值。

    P2 修正：旧实现按“≤t_boiler 的全部工况取中位数”，因仿真网格为 40℃ 步长，
    出现 250℃/300℃ 同值台阶；改为各锅炉出口温度点取中位数后线性插值。
    """
    df = _load_steam()
    t_boiler = min(max(t_src - 100.0, 180.0), 540.0)
    try:
        t_c, eff = _steam_eff_curve()
    except Exception:
        # 兜底：数据缺列/读取失败时退回旧“累计中位数”口径
        ok = df[df["boiler_outlet_K"] <= t_boiler + 273.15 + 1e-6]
        if len(ok) == 0:
            return None
        return float(np.median(ok["thermal_eff"].values) * 100.0)
    if t_boiler <= t_c[0]:
        return float(eff[0])
    if t_boiler >= t_c[-1]:
        return float(eff[-1])
    return float(np.interp(t_boiler, t_c, eff))


# ---------------- 第二级：动态矩阵 ----------------

INDICATORS = ["能效%", "投资万元/MW", "回收期年", "CO2减排t/年", "政策分",
              "运行成本万元/MW·年"]
DIRECTIONS = ["max", "min", "min", "max", "max", "min"]

# 与现有 08/案例引擎同一基值口径；P2 标定后更新如下：
# - abs_self 能效列占位=一类 COP1.7×100（供暖/储热调峰口径），工艺蒸汽场景由
#   build_matrix_v2 按二类 COP0.45 覆盖，不再混写 0.75；
# - abs_ext 投资/回收期按哈石化“余热暖民”公开工程数据定稿（宋大勇，2023）；
#   运行成本列置 0，由 build_matrix_v2 按蒸汽价动态计算。
BASE_INDICATORS = {
    "direct":     [90,  60, 2.5, 0, 3, 6],
    "whb_steam":  [85, 100, 3.5, 0, 3, 12],
    "abs_self":   [170, 150, 2.7, 0, 4, 20],
    "abs_ext":    [170, 174, 4.2, 0, 4, 0],
    "comp":       [106, 120, 4.2, 0, 4, 186],
    "orc":        [12.3, 2380, 5.6, 0, 4, 15],
    "steam_pp":   [25, 550, 4.5, 0, 4, 8],
    "tc_storage": [70, 900, 10.0, 0, 3, 28],
    "pcm_storage": [75, 600, 8.0, 0, 3, 25],
    "teg":        [5, 1500, 12.0, 0, 2, 8],
    # v2.1 新增：供冷（能效=COP×100 动态覆盖：≥85℃ 70%、80~84℃ 60%；
    #            投资/回收期/运行成本 [待标定]，工程估算口径）
    "abs_cool":   [70, 100, 5.5, 0, 4, 3],
    "comp_cool":  [190, 60, 4.2, 0, 3, 104],
}

# ---- 规模修正：生产能力指数法（0.6 次方/规模指数法，工程估算口径）----
# 单位投资随规模摊薄：I(rep) / I(3000kW) = (rep / 3000)^(n-1)。
# n 依据“生产能力指数法”规则：靠增大单机容量扩能 n≈0.6~0.7（蒸汽朗肯电站）；
# 靠并联增加机组扩能 n≈0.8~1.0（ORC 模块化取 0.85、热泵机组取 0.90 为默认）。
# 代表规模：小 500 / 中 3000 / 大 10000 kW。
SCALE_REP_KW = {"小": 500.0, "中": 3000.0, "大": 10000.0}
SCALE_EXP = {"steam_pp": 0.70, "orc": 0.85}
SCALE_EXP_DEFAULT = 0.90


def scale_multiplier(path, band):
    """规模单位投资修正系数（相对 3000 kW 中档基准，小规模 >1、大规模 <1）。"""
    rep = SCALE_REP_KW[band]
    n = SCALE_EXP.get(path, SCALE_EXP_DEFAULT)
    return (rep / 3000.0) ** (n - 1.0)


# 吸收式 COP 标定（P2 定稿，出处见 calibration/标定数据与出处_P2.md）
COP_I_ABS_SELF = 1.7    # 一类（增热型）自驱动 COP（供热/驱动热）：厂家口径 1.7~2.4，保守取 1.7
COP_II_ABS_SELF = 0.45  # 二类（升温型/热变换器）COP（制汽/驱动热）：公开口径 0.4~0.5，取 0.45
COP_H_ABS_EXT = 1.7     # 外购蒸汽驱动（一类）COP_h：哈石化口径 580 kW 蒸汽/1000 kW 供热≈1.72，保守取 1.7
COP_C_ABS = 0.7         # v2.1 吸收式制冷 COP_c 标称值（≥85℃ 驱动时）：单效溴化锂
                        #      85℃ 时最大 COP≈0.7（科普中国/机械工程名词口径）；
                        #      80~84℃ 低温段按 0.6 工程估算（见 abs_cool_cop）
ABS_COOL_T_MIN = 80.0   # v2.1 单效溴化锂热水驱动下限（最佳工作温度 80~100℃）
COP_E_COOL = 5.0        # v2.1 电压缩制冷 COP_e（冷量/电）：GB19577-2015 水冷式
                        #      冷水机组 3 级 COP≥5.0，保守取 5.0
STEAM_PRICE = 100.0     # 外购低压蒸汽价 元/GJ：2024 园区低压蒸汽 262 元/吨≈105 元/GJ、行业均价 39.7 元/GJ，保守取 100
STEAM_OPS_FACTOR = 1.03  # 水/泵/管网损耗附加 3%
GRID_EF = 0.581              # tCO2/MWh，2021 全国电网平均排放因子
GAS_EF = 0.0561              # tCO2/GJ 天然气
BOILER_EFF = 0.90
ELEC_PRICE = 0.65            # 元/kWh（沿用演示平台口径）


def abs_cool_cop(t_src):
    """吸收式制冷 COP_c（冷量/驱动热）分段：≥85℃ 取 0.7（单效标称），
    80~84℃ 取 0.6（低温端工程估算 [待标定]）；<80℃ 返回 None（不可行）。"""
    if t_src >= 85.0:
        return 0.7
    if t_src >= ABS_COOL_T_MIN:
        return 0.6
    return None


def heat_red_gas(scene, penalty=1.0):
    """供热减排列（替代天然气）：可回收热量 × 燃气因子 ÷ 锅炉效率 ÷ penalty。"""
    q = recovered_heat_kw(scene)
    heat_gj = q * scene["年运行小时"] * 3.6 / 1000.0
    return heat_gj * GAS_EF / BOILER_EFF / penalty


def steam_driven_abs_reduction(scene):
    """外购蒸汽驱动吸收式净减排 = 替代天然气 − 驱动蒸汽燃料排放。"""
    return heat_red_gas(scene, penalty=1.0) * (1.0 - 1.0 / COP_H_ABS_EXT)


def steam_driven_abs_opex_wan_mw(scene):
    """外购蒸汽驱动吸收式运行成本（万元/MW·年）。

    驱动蒸汽热耗 = 1/COP_H × 单位供热，单位供热 1 MW×hours = 3.6×hours GJ，
    成本 = 热耗 × 蒸汽价(元/GJ) ÷ 1e4（转万元）× 1.03（水/泵/管网损耗附加）。
    """
    hours = scene["年运行小时"]
    return ((1.0 / COP_H_ABS_EXT) * hours * 3.6
            * STEAM_PRICE / 10000.0 * STEAM_OPS_FACTOR)


def cooling_abs_reduction(scene):
    """吸收式制冷减排 = 替代电压缩制冷耗电（废热为弃热，不扣驱动排放）。

    服务冷量 Q_cold = COP_c(驱动温度分段) × Q_rec；替代电耗 = Q_cold ÷ COP_E_COOL。
    """
    cop = abs_cool_cop(scene["热源温度_degC"]) or COP_C_ABS
    q_cold = cop * recovered_heat_kw(scene)
    mwh_avoided = q_cold / COP_E_COOL * scene["年运行小时"] / 1000.0
    return mwh_avoided * GRID_EF


def comp_cool_opex_wan_mw(scene):
    """电压缩制冷运行成本（万元/MW·年）＝(1/COP_e)×hours×电价×1000÷1e4。"""
    return ((1.0 / COP_E_COOL) * scene["年运行小时"]
            * ELEC_PRICE * 1000.0 / 10000.0)


def comp_reduction(scene, cop=2.8):
    """压缩式热泵净减排 = 替代天然气 − 自身耗电对应电网排放。"""
    q = recovered_heat_kw(scene)
    heat_gj = q * scene["年运行小时"] * 3.6 / 1000.0
    elec_mwh = heat_gj / 3.6 / cop
    return max(heat_gj * GAS_EF / BOILER_EFF - elec_mwh * GRID_EF, 0.0)


def power_reduction(path, scene):
    """发电减排列：净功率(=热效率×可回收热功率) × 小时 × 电网因子。"""
    q = recovered_heat_kw(scene)
    hours = scene["年运行小时"]
    if path == "orc":
        e = orc_eff_median_pct(scene["热源温度_degC"], scene["换热端差_degC"])
    else:
        e = steam_eff_median_pct(scene["热源温度_degC"])
    if not e:
        return 0.0
    mwh = (e / 100.0) * q * hours / 1000.0
    return mwh * GRID_EF


def build_matrix_v2(survivors, scene):
    """构建 TOPSIS 决策矩阵；能效/投资/减排/运行成本按温段与规模动态化。"""
    X = np.array([BASE_INDICATORS[p].copy() for p in survivors], dtype=float)
    q = recovered_heat_kw(scene)
    band = scale_band(q)
    hours = scene["年运行小时"]
    demand = scene["需求"]
    for i, p in enumerate(survivors):
        sf = scale_multiplier(p, band)
        X[i, 1] = BASE_INDICATORS[p][1] * sf
        X[i, 2] = BASE_INDICATORS[p][2] * sf
        X[i, 5] = BASE_INDICATORS[p][5] * (1.0 if p == "comp" else sf)
        if p == "orc":
            e = orc_eff_median_pct(scene["热源温度_degC"], scene["换热端差_degC"])
            if e:
                X[i, 0] = round(e, 2)
        elif p == "steam_pp":
            e = steam_eff_median_pct(scene["热源温度_degC"])
            if e:
                X[i, 0] = round(e, 2)
        elif p == "abs_self":
            # 一类（增热型）用于供暖/储热调峰提温；二类（升温型/热变换器）用于工艺蒸汽
            cop = COP_II_ABS_SELF if demand == "工艺蒸汽" else COP_I_ABS_SELF
            X[i, 0] = round(cop * 100.0, 2)
            # 减排列口径（P3.5 审计修正）：替代天然气按“实际产出热量”计。
            # 一类供暖：产出热≈回收热（Q_rec 口径，保留原算法）；
            # 二类制蒸汽：产出热 = COP_II × Q_rec，按 0.45 折算，避免高估约 2.2 倍。
            red_factor = COP_II_ABS_SELF if demand == "工艺蒸汽" else 1.0
            X[i, 3] = round(heat_red_gas(scene) * red_factor, 1)
        elif p == "abs_ext":
            X[i, 0] = round(COP_H_ABS_EXT * 100.0, 2)
            X[i, 3] = round(steam_driven_abs_reduction(scene), 1)
            X[i, 5] = round(steam_driven_abs_opex_wan_mw(scene), 1)
        elif p == "abs_cool":
            cop = abs_cool_cop(scene["热源温度_degC"]) or COP_C_ABS
            X[i, 0] = round(cop * 100.0, 2)
            X[i, 3] = round(cooling_abs_reduction(scene), 1)
        elif p == "comp_cool":
            X[i, 0] = round(COP_E_COOL * 0.38 * 100.0, 2)
            X[i, 3] = 0.0  # 与“电制冷现状”同口径，不假装额外降碳
            X[i, 5] = round(comp_cool_opex_wan_mw(scene), 1)
        elif p == "comp":
            X[i, 3] = round(comp_reduction(scene), 1)
            X[i, 5] = round(0.357 * hours * ELEC_PRICE * 1000 / 10000, 1)
        elif p in ("direct", "whb_steam", "tc_storage", "pcm_storage"):
            X[i, 3] = round(heat_red_gas(scene), 1)
    return X


# ---------------- 权重 / TOPSIS / 总入口 ----------------

W_FALLBACK = np.array([0.20, 0.15, 0.16, 0.19, 0.18, 0.12])
_W_CACHE = None


def load_weights():
    global _W_CACHE
    if _W_CACHE is not None:
        return _W_CACHE
    p = os.path.join(DATA_DIR, "eval_weights.json")
    if not os.path.exists(p):
        _W_CACHE = W_FALLBACK
        return _W_CACHE
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    names = d["指标"]
    wsub = dict(zip(names, np.array(d["主观权重_AHP"], dtype=float)))
    eco = wsub["初始投资"] + wsub["投资回收期"]
    w = np.array([wsub["系统能效"], wsub["初始投资"], wsub["投资回收期"],
                  wsub["CO2当量减排"], wsub["政策补贴适配度"], 0.25 * eco])
    _W_CACHE = w / w.sum()
    return _W_CACHE


def topsis(matrix, weights):
    if matrix.shape[0] == 1:
        # 单候选：正负理想解重合，贴近度按约定为 1（唯一可行路径即最优）
        return np.ones(1)
    norm = matrix / np.sqrt((matrix ** 2).sum(axis=0) + 1e-12)
    v = norm * weights
    pos = np.array([v[:, j].max() if DIRECTIONS[j] == "max" else v[:, j].min()
                    for j in range(v.shape[1])])
    neg = np.array([v[:, j].min() if DIRECTIONS[j] == "max" else v[:, j].max()
                    for j in range(v.shape[1])])
    dpos = np.sqrt(((v - pos) ** 2).sum(axis=1))
    dneg = np.sqrt(((v - neg) ** 2).sum(axis=1))
    return dneg / (dpos + dneg + 1e-12)


def evaluate(scene):
    """总入口：场景画像 → 报告 dict。能力圈外 out_of_scope=True 且不硬推荐。"""
    report = {"out_of_scope": False, "top": None, "ranked": [],
              "reasons": {}, "matrix": None, "warnings": []}
    try:
        s = validate_scene(scene)
    except OutOfScopeError as e:
        report["out_of_scope"] = True
        report["warnings"].append(str(e))
        return report
    st = stage1(s)
    survivors = [p for p, (ok, _) in st.items() if ok]
    report["reasons"] = {p: r for p, (ok, r) in st.items()}
    if not survivors:
        report["out_of_scope"] = True
        report["warnings"].append(
            "无可行候选路径：" + "；".join(
                f"{DISPLAY[p]}：{r}" for p, (ok, r) in st.items() if not ok))
        return report
    X = build_matrix_v2(survivors, s)
    for i, p in enumerate(survivors):
        if p in ("orc", "steam_pp"):
            X[i, 3] = round(power_reduction(p, s), 1)
    c = topsis(X, load_weights())
    order = sorted(zip(survivors, c), key=lambda x: -x[1])
    q = recovered_heat_kw(s)
    if q < 500 and any(p == "steam_pp" for p, _ in order):
        report["warnings"].append(
            f"可回收热功率仅约 {q:.0f} kW（小规模），蒸汽朗肯单位造价×"
            f"{scale_multiplier('steam_pp', '小'):.2f}，经济性差，"
            "建议专项可行性评估，勿直接按推荐实施")
    report["ranked"] = [{"path": p, "closeness": round(float(cc), 4)}
                        for p, cc in order]
    report["matrix"] = X
    report["top"] = order[0][0]
    return report
