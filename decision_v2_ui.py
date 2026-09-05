# -*- coding: utf-8 -*-
"""演示平台 ↔ decision_core v2 适配层（纯逻辑，无 streamlit 依赖）。

职责：
- 把 UI 参数（温度/流量/介质/需求/连续性/驱动来源…）翻译成内核场景 schema；
- 调用统一内核（validate_scene / stage1 / build_matrix_v2）并回填发电减排列；
- 能力圈外（>650℃、制冷/干燥等）返回结构化提示，UI 只展示、不做硬推荐。

红线（v2.1）：供冷/干燥为受支持需求；除湿未单列（并入供冷/干燥场景后评价）；
外购蒸汽驱动吸收式制冷不设废热回收路径；能力圈外不硬给推荐（详见 11 目录设计文档）。

C3.2（2026-09-05）：新增 boundary_notices——把 S2 型边界工况（热源温度恰在
直接换热/驱动下限附近）的敏感性结论返回给 UI 自动提示，提醒预留端差裕量。
依据：outputs/decision_robustness_report.md（v2.1 内核，2026-09-04）——
S2 供暖 75℃（需求 65℃ + 端差 10℃）恰处“直接换热”可行边界，输入扰动下
名义推荐保持概率约 50%；吸收式制冷 80~84℃ 处于 COP_c=0.6 段且低于 80℃
即整体失效，同属需显式提示的边界区。
"""

import decision_core as dc

MEDIUM_MAP = {"热水/冷凝水": "热水·冷凝水", "烟气": "烟气", "工艺液体": "工艺液体"}
DEMAND_MAP = {"供暖/热水": "供暖·热水", "供冷（制冷）": "供冷",
              "干燥/烘干": "干燥"}  # UI 写法 → 内核写法
DRIVER_OPTIONS = ["余热自驱动", "外购蒸汽", "外购电力"]

# C3.2 边界提示阈值（℃）：裕量 < 高敏带 即视为“扰动可能翻转”，
# 裕量 ≥ 低敏带 但仍小于提示带 时按“较敏感、建议核实全年最低温度”提示。
BOUNDARY_HIGH_SENSITIVE = 5.0   # 裕量 <5℃：±3~5℃ 扰动即可越过边界
BOUNDARY_NOTICE_BAND = 10.0     # 裕量 <10℃：统一提示预留裕量
BOUNDARY_NEAR_BAND = 5.0        # 不可行但差 <5℃：提示“压缩端差即可恢复”

LABELS = {
    "direct": "直接换热供暖",
    "whb_steam": "余热锅炉直接产汽",
    "abs_self": "吸收式热泵提温（自驱动）",
    "abs_ext": "吸收式热泵提温（外购蒸汽）",
    "comp": "压缩式热泵提温（电驱动）",
    "orc": "ORC 余热发电",
    "steam_pp": "高温蒸汽发电",
    "tc_storage": "热化学储热",
    "pcm_storage": "相变储热",
    "teg": "TEG 热电发电",
    "abs_cool": "吸收式制冷（余热驱动）",
    "comp_cool": "电压缩制冷（电驱动）",
}
PATH_LABELS = [LABELS[p] for p in dc.PATH_KEYS]


def ui_scene(t_src, m_dot, medium, demand, continuity, hours, dT,
             driver="余热自驱动", t_dem=None):
    """把 UI 参数构造成内核场景（缺省项由内核补齐）。"""
    # 不做本地白名单校验：供冷驱动来源区分、干燥映射、除湿越界与未知需求
    # 统一交给内核，由 validate_scene/stage1 返回人话提示。
    return {
        "热源温度_degC": float(t_src),
        "载体": MEDIUM_MAP.get(medium, medium),
        "流量_kg_s": float(m_dot),
        "需求": DEMAND_MAP.get(demand, demand),
        "需求温度_degC": t_dem,
        "连续性": continuity,
        "驱动来源": driver,
        "年运行小时": int(hours),
        "换热端差_degC": float(dT),
    }


def run_decision(scene):
    """调用统一内核完成两级决策。

    返回 dict：
      out_of_scope: bool
      message: 能力圈外/无候选时的人话说明（正常时为空串）
      reasons: {path_key: str}（第一级原因，12 条全覆盖；越界时为空）
      keys / labels: 幸存路径（内核 key 与短显示名，顺序一致）
      X: 六列决策矩阵（发电 CO2 已按 power_reduction 回填）或 None
      scene: 校验补默认后的场景（供 UI 复算减排/运行成本）
    """
    try:
        s = dc.validate_scene(scene)
    except dc.OutOfScopeError as exc:
        return {"out_of_scope": True, "message": str(exc), "reasons": {},
                "keys": [], "labels": [], "X": None, "scene": None}
    st = dc.stage1(s)
    keys = [p for p in dc.PATH_KEYS if st[p][0]]
    reasons = {p: st[p][1] for p in dc.PATH_KEYS}
    if not keys:
        msg = "无可行候选路径：" + "；".join(
            f"{LABELS[p]}：{r}" for p, (ok, r) in st.items() if not ok)
        return {"out_of_scope": True, "message": msg, "reasons": reasons,
                "keys": [], "labels": [], "X": None, "scene": s}
    X = dc.build_matrix_v2(keys, s)
    for i, p in enumerate(keys):
        if p in ("orc", "steam_pp"):
            X[i, 3] = round(dc.power_reduction(p, s), 1)
    return {"out_of_scope": False, "message": "",
            "reasons": reasons, "keys": keys,
            "labels": [LABELS[p] for p in keys], "X": X, "scene": s}


def boundary_notices(res):
    """按两级决策结果返回 S2 型边界工况提示列表（空列表 = 无提示）。

    覆盖三类平台可达的边界：
      - 供暖·热水“直接换热”可行但裕量小 / 差一点不可行；
      - 干燥“直接热风”可行但裕量小（同直接换热机制）；
      - 供冷吸收式制冷处于 80~84℃ 驱动下限区（COP_c=0.6、低于 80℃ 失效）。
    纯逻辑无 streamlit 依赖，UI 层仅逐条渲染 st.warning。
    """
    if res["out_of_scope"] or not res["keys"] or not res["scene"]:
        return []
    s = res["scene"]
    t = s["热源温度_degC"]
    dT = s["换热端差_degC"]
    demand = s["需求"]
    driver = s.get("驱动来源", "余热自驱动")
    keys = set(res["keys"])
    notes = []

    if demand == "供暖·热水":
        t_dem = s.get("需求温度_degC") or 60.0
        lo = max(60.0, t_dem + dT)
        margin = t - lo
        if "direct" in keys and 0.0 <= margin < BOUNDARY_NOTICE_BAND:
            if margin < BOUNDARY_HIGH_SENSITIVE:
                notes.append(
                    f"⚠ 边界工况提示（S2 型）：热源 {t:.0f}℃ 仅比直接换热下限 "
                    f"{lo:.0f}℃（需求 {t_dem:.0f}℃ + 换热端差 {dT:.0f}℃）高 "
                    f"{margin:.1f}℃，温度/端差的小幅扰动即可让“直接换热”失效并转向"
                    "热泵（鲁棒性核验：恰处下限的场景保持概率约 50%）。"
                    "工程上建议预留 ≥5℃ 端差裕量，或按“直接换热优先 + 热泵兜底”"
                    "双方案设计。")
            else:
                notes.append(
                    f"⚠ 边界工况提示：热源 {t:.0f}℃ 距直接换热下限 {lo:.0f}℃"
                    f"（需求 {t_dem:.0f}℃ + 换热端差 {dT:.0f}℃）仅高 {margin:.1f}℃，"
                    "处于较敏感区。请用全年最低热源温度与换热器衰减后的真实端差复核，"
                    "预留 ≥5℃ 裕量，避免按理想工况定容。")
        elif "direct" not in keys and -BOUNDARY_NEAR_BAND <= margin < 0.0:
            notes.append(
                f"⚠ 边界工况提示：热源 {t:.0f}℃ 距直接换热下限 {lo:.0f}℃"
                f"（需求 {t_dem:.0f}℃ + 换热端差 {dT:.0f}℃）还差 {-margin:.1f}℃，"
                "当前直接换热不可行、推荐转向热泵；若现场实际端差可压缩"
                "（换热端差仍应 ≥5℃）或需求温度略低，直接换热路径可恢复，"
                "请以实测换热性能曲线为准后再定案。")
    elif demand == "干燥":
        t_dry = s.get("需求温度_degC") or 100.0
        lo = max(60.0, t_dry + dT)
        margin = t - lo
        if "direct" in keys and 0.0 <= margin < BOUNDARY_NOTICE_BAND:
            if margin < BOUNDARY_HIGH_SENSITIVE:
                notes.append(
                    f"⚠ 边界工况提示（S2 型）：热源 {t:.0f}℃ 仅比干燥直接热风下限 "
                    f"{lo:.0f}℃（需求 {t_dry:.0f}℃ + 换热端差 {dT:.0f}℃）高 "
                    f"{margin:.1f}℃，小幅扰动即可能让该路径失效并转向热泵烘干。"
                    "建议预留 ≥5℃ 端差裕量，或按“直接换热优先 + 热泵兜底”双方案设计。")
            else:
                notes.append(
                    f"⚠ 边界工况提示：热源 {t:.0f}℃ 距干燥直接热风下限 {lo:.0f}℃"
                    f"仅高 {margin:.1f}℃，处于较敏感区，请用全年最低温度复核后预留 ≥5℃ 裕量。")
    elif demand == "供冷" and driver == "余热自驱动" and "abs_cool" in keys:
        if t < 85.0:
            notes.append(
                f"⚠ 边界工况提示：热源 {t:.0f}℃ 处于吸收式制冷驱动下限区"
                f"（80~84℃ 按 COP_c=0.6 保守核算），运行温度一旦波动至 "
                f"{dc.ABS_COOL_T_MIN:.0f}℃ 以下该路径将整体失效。"
                "建议按 ≥85℃ 设计点预留裕量，或对该温度段做全年运行小时分布校核后再定容。")
    return notes
