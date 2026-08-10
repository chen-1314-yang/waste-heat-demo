# 工业余热回收利用 · 智能决策演示平台

《工业废热或余热回收利用降碳技术路径与智能优化评价方法》——时代杯零碳科技创新大赛（开放赛道·方向2 智能控碳）

## 这是什么

交互式智能决策工具：输入热源温度、流量、用能需求等参数，实时输出：

1. 第一级热力学规则粗筛（8 条技术路径，给出排除/通过原因）；
2. 第二级 TOPSIS 精细排序（AHP + 熵权组合权重，λ 可调）；
3. 减碳与收益估算（ORC 发电 / 供热替代天然气）；
4. 帕累托前沿（22 解 DWSIM 复核 + pymoo 100 解 + 5 个复核点）；
5. 动态 LCA 月度滚动核算（年 173.2 tCO2，推算口径）；
6. λ 敏感性分析（排序稳定性）。

## 数据口径（务必阅读）

- **DWSIM 400 工况**：真实稳态仿真（能量守恒最大偏差 0.013 kW）；
- **帕累托前沿**：22 解经 DWSIM 复核；pymoo 100 解为代理模型预测；5 个代表点经 DWSIM 复核（误差 <1%）；
- **ORC 减碳**：仿真净功率 × 运行小时 × 华东电网因子 0.581，**推算口径，非实测**；
- **供热减碳**：替代天然气估算（0.0561 tCO2/GJ ÷ 锅炉效率 90%），演示估算；
- **电价 0.65 元/kWh、气价 3.5 元/m³**：演示单价假设；
- **TOPSIS 指标**：典型演示值，正式应用须替换文献/实测数据；
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
    ├── dwsim_sweep_full.csv   # DWSIM 400 工况
    ├── pareto_front_real.csv  # 22 解复核前沿
    ├── pymoo_pareto.csv       # pymoo 100 解
    ├── dwsim_verify_pymoo.csv # 5 个 DWSIM 复核点
    ├── lca_monthly_real.csv   # 动态 LCA 月度（推算）
    └── eval_weights.json      # AHP+熵权组合权重
```
