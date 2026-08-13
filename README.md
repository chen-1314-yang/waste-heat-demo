# 工业余热回收利用 · 智能决策演示平台

《工业废热或余热回收利用降碳技术路径与智能优化评价方法》——时代杯零碳科技创新大赛（开放赛道·方向2 智能控碳）

## 这是什么

交互式智能决策工具：输入热源温度、流量、用能需求等参数，实时输出：

1. 第一级热力学规则粗筛（8 条技术路径，给出排除/通过原因）；
2. 第二级 TOPSIS 精细排序（AHP + 熵权组合权重，λ 可调）；
3. 减碳与收益估算（ORC 发电 / 供热替代天然气）；
4. 帕累托前沿（ORC 16 解精确模型复核 + pymoo 100 解 + 蒸汽朗肯 100 解）；
5. 动态 LCA 月度滚动核算（年降碳 573.1 tCO2，推算口径）；
6. λ 敏感性分析（排序稳定性）；
7. 工况库驱动的实时模拟（`data/conditions_db.csv` 可编辑，模拟 DCS 波动数据）。

## 工况库在线编辑

侧边栏底部“工况库管理 → 在线编辑工况库”面板可直接增删改工况行并保存
（本地运行写入 `data/conditions_db.csv`）。**Streamlit Cloud 为临时环境**：
网页内保存仅对当前会话有效，长期修改请点“下载当前 CSV”后上传覆盖
GitHub 仓库 `data/conditions_db.csv`，或把文件发给我们代传。

## 数据口径（务必阅读）

- **CoolProp 物性仿真**：ORC 1675 工况 + 蒸汽朗肯 1600 工况（能量守恒最大偏差 <1e-9 kW）；
- **帕累托前沿**：ORC 16 解经精确朗肯模型重算生成（109.6~150.5 kW/MW热）；pymoo 100 解为代理模型预测；
  5 个代表点复核（每 MW 回收热口径：ORC 净功率误差 0.17%~1.99%、热效率 0.03%~1.26%；
  蒸汽朗肯净功率误差 0.84%~1.41%、热效率 0.36%~0.64%；最优点 0.4%）；
- **工况库**：`data/conditions_db.csv` 为工程/文献示例（示意），可编辑；3 个对标案例有公开出处；
- **ORC 减碳**：每 MW 回收热净功率 × 实际回收热功率（ṁ×cp×ΔT）× 运行小时
  × 2021年度全国电网平均排放因子 0.5810，**推算口径，非实测**；
- **供热减碳**：替代天然气估算（0.0561 tCO2/GJ ÷ 锅炉效率 90%），演示估算；
- **电价 0.65 元/kWh、气价 3.5 元/m³**：演示单价假设；
- **TOPSIS 指标**：文献/工程估算/示意三类，页面与《数据来源台账》逐项标注；
- **成本/回收期**：示意性代理模型，非真实报价；**CCER 收益为情景假设**，未完成备案方法学前不计入基准财务指标。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开 http://localhost:8501 。

## 在线部署（Streamlit Community Cloud，免费）

1. 在 GitHub 新建公开仓库 `waste-heat-demo`（不要勾选 README/.gitignore 等初始化文件）；
2. 双击本目录下的 `push_to_github.bat`，按提示登录 GitHub 完成推送；
3. 打开 https://share.streamlit.io ，用 GitHub 登录；
4. New app → 选择仓库 `waste-heat-demo` → Main file 填 `app.py` → Deploy。

## 文件结构

```
waste-heat-demo/
├── app.py                  # 主程序
├── requirements.txt        # 依赖
├── push_to_github.bat      # 一键推送脚本（Windows）
├── README.md               # 本说明
├── .streamlit/config.toml  # 免邮箱提示 / headless
└── data/                   # 演示数据（可复算）
    ├── conditions_db.csv       # 工况库（可编辑：实时模拟/典型工况的数据源）
    ├── orc_sweep_coolprop.csv  # ORC 1675 工况（CoolProp）
    ├── steam_sweep_coolprop.csv # 蒸汽朗肯 1600 工况（CoolProp）
    ├── pareto_front_real.csv   # ORC 16 解复核前沿
    ├── steam_pareto.csv        # 蒸汽朗肯前沿
    ├── pymoo_pareto.csv        # pymoo 100 解
    ├── orc_surrogate.joblib / steam_surrogate.joblib  # 代理模型
    ├── lca_monthly_real.csv / lca_mc_annual_real.csv / lca_embodied.json  # 动态 LCA（推算）
    └── eval_weights.json       # AHP+熵权组合权重
```
