# CARC：年代感知稳健因果推断

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)

**CARC**（Chronology-Aware Robustness Causality，年代感知稳健因果推断）是一种从古气候
代用记录中检测因果先后关系的方法，同时解决两大挑战：

1. **年代模型不确定性** — 年代标尺误差可能产生虚假的领先/滞后信号
2. **共同强迫混淆** — 共享的外部驱动因子（如北大西洋气候）使独立记录表现出虚假因果关联

本方法引入**诚实性滤波器（honesty filter）**：仅当通过 ARC（自适应残差因果法）保守剥离
共同强迫信号后，在 ≥95% 的年代扰动系综测试中均通过 Granger 因果检验的因果边才被接受。

## 主要特性

- **ARC 预处理**：用偏线性回归以可调剥离强度 α 去除共同驱动代理（如 NGRIP δ¹⁸O）
  的共享分量
- **年代扰动系综**：按每条记录的年模型误差 (σ) 扰动其时间轴，每次扰动后重新检验
  Granger 因果性
- **稳健性评分**：在扰动中 p < 0.05 的方向性因果边所占比例
- **诚实性滤波器**：稳健性 < 95% 的边被拒绝
- **条件因果**：在不同气候状态（Heinrich 事件、间冰阶、冰盛期等）内分别检验因果方向

## 目录结构

```
CARC/
├── README.md                    # 英文说明
├── README_CN.md                 # 中文说明（本文件）
├── LICENSE                      # MIT 许可证
├── CITATION.cff                 # 引用元数据
├── requirements.txt             # Python 依赖
├── .env.example                 # 配置模板
├── .gitignore
├── data/                        # 处理后的输出和元数据
│   ├── data_tiers_final.csv     # Tier 分层结果
│   ├── record_quality_score.csv # 质量评分
│   ├── NGRIP_trend_800ka.csv    # 预计算的 ARC 代理
│   ├── orbital/                 # 轨道强迫数据
│   └── README.md                # 数据文档
├── proxy_records/               # ~80 条古气候代用记录 CSV
├── proxy_errors/                # 年代模型 1σ 误差文件
├── src/                         # 共享 Python 库
│   ├── __init__.py              # 公共 API
│   ├── config.py                # 中央配置
│   ├── io.py                    # 数据加载与预处理
│   ├── arc.py                   # ARC 去混淆
│   ├── granger.py               # Granger 因果 + 年代扰动
│   ├── stacking.py              # 区域堆叠
│   ├── synthetic.py             # 合成数据生成
│   ├── paleo_generator.py       # PaleoDataGenerator 类
│   └── validation.py            # 方法评估框架
├── scripts/                     # 编号分析驱动 01–45 + fignew/ 图件重建系列
│   ├── figure_tools/            # 制图/QA 工具（复现映射之外）
│   └── manuscript_tools/        # 文稿装配工具（需手稿 docx，不随仓库分发）
├── Tables/                      # 论文表格 xlsx（投稿审稿期 gitignore，投稿包内随附）
├── figures/                     # 生成的图表
└── outputs/                     # 论文运行的参考输出（CSV）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置数据路径

```bash
cp .env.example .env
# 编辑 .env 文件匹配你的系统路径
```

或直接设置环境变量:

```bash
export CARC_DATA_DIR=/path/to/proxy_records
export CARC_OUTPUT_DIR=./outputs
```

### 3. 运行分析

```bash
# 数据预处理：质量评分与分层
python scripts/01_preprocess_data.py

# 合成数据验证（无需外部数据）
python scripts/03_synthetic_validation.py

# Tier 1 记录的因果推断
python scripts/05_causal_inference_tier1.py

# 生成论文图表
python scripts/11_generate_figures.py
```

## 复现论文结果

论文全部数字表格均由 `scripts/` 中脚本产出，逐表对应关系见
[README_复现指南.md](README_复现指南.md)。摘要：

| 论文产物 | 脚本 | 输出（运行后） |
| :--- | :--- | :--- |
| Table 1（Scenario A，20×200） | `scripts/22_scenarioA_final.py` | `outputs/scenarioA_final_table1.csv` |
| Table 2（CCM / PCMCI） | `scripts/39_table2_ccm_pcmci.py` | `outputs/table2_ccm_pcmci.csv` |
| Table 4 / Table S4（六个气候窗口，180 有向格） | `scripts/06_conditional_causality.py` | `outputs/conditional_causality_allpairs.csv` |
| Table 5（区域相位滞后） | `scripts/08_regional_phase_lag.py` | — |
| Table S5（80 记录全对扫描 + 消融） | `scripts/19_full_pairwise_scan.py` | `outputs/pairwise_scan/` |
| Table S7（理论 FPR 剖面 + α×τ 门） | `scripts/42_theory_fpr_table.py` | `outputs/tableS7_fpr_profile.csv`、`outputs/tableS7_binomial_gate.csv` |
| β = 0.25 反解锚点（附录 S4.1） | `scripts/43_pairwise_beta_inversion.py` | `outputs/beta_inversion_pairs.csv` |
| 定理 2 数值例 MC（§2.2） | `scripts/44_theorem2_montecarlo.py` | `outputs/theorem2_mc_verification.csv` |
| Fig. 2(a) MC FPR 曲线 | `scripts/45_fig2a_fpr_sweep.py` | `outputs/fig2a_mc_sweep.csv` |
| §4.3 区域对 | `scripts/09_regional_robustness.py` → `scripts/17_regional_pairwise_matrix.py` | `outputs/regional_robustness_windows.csv` |
| Table S2 / S4 记录账目 | `scripts/01_preprocess_data.py` + `data/data_tiers_final.csv` | `data/` |

各脚本均有 `--quick` 冒烟模式（见各脚本 docstring）。全量运行固定随机种子、
确定性可复现，但耗时较长（如 Table 1 全量为 144 配置筛选网格 + 20 实现 × 200
扰动，分片 checkpoint 可续跑）。脚本输出落在 `outputs/`，论文表格 xlsx 在
`Tables/`（投稿审稿期间两者均被 `.gitignore` 保护，不上传）。

```bash
cd scripts
python 22_scenarioA_final.py --quick    # Table 1 冒烟（2×40）
python 39_table2_ccm_pcmci.py --quick         # Table 2 冒烟（1×20）
```

Table 2 另需 `tigramite>=5.0`（PCMCI/ParCorr）；CCM 列为 numpy 实现（已对
pyEDM 2.5 验证），pyEDM 可选（Windows 需 `legacy=True`）。

## 方法概述

### 问题

在古气候学中，检测区域 A 的气候变化是否超前于区域 B 是理解气候动力学的核心问题。
然而，标准 Granger 因果检验面临两个挑战：

1. **共享强迫**：两条记录同时响应同一外部驱动（如北大西洋温度），但响应灵敏度和
   时间常数不同，产生直接因果的假象。
2. **年代不确定性**：每条记录有独立的年代模型，误差为 0.1–3.0 kyr (1σ)。年代
   错位时，表观领先-滞后可能是伪影。

### CARC 解决方案

1. **ARC（自适应残差因果法）**：因果检验前，先回归去除共同强迫代理信号的 α 比例。
   默认 α = 0.5 为"保守"设置，去除一半共享方差而不假设代理完美捕获所有共同强迫。

2. **年代感知扰动**：对每个变量按 N(0, σ_age) 扰动时间轴，重新排序序列，重新运行
   Granger 检验。每个窗口重复 N=200 次。

3. **诚实性滤波器**：仅当在最优 VAR 滞后阶数下，≥95% 扰动中 Granger 检验
   (p < 0.05) 显著的方向边 A→B 才被接受。

### 关键理论

年代不确定性下假阳性率的理论下界：

$$FPR \geq 1 - \Phi\left(z_{\alpha} - \sqrt{\lambda_{eff}/2}\right)$$

其中 λ_eff 为有效信噪比，z_α 为所选显著性水平的临界值。

## 数据来源

代用记录覆盖 0–800 ka，代表全球各大洋盆和大陆：

| 档案类型 | 示例 | 年代不确定性 |
|---------|------|------------|
| 冰芯 | NGRIP, EDC, Dome Fuji, WAIS | 0.5–1.7 kyr (GICC05/AICC2012) |
| 海洋沉积 | ODP/U 系列站位 | 2–4 kyr (LR04 stack) |
| 石笋 | Cheng et al. 合成记录, Botuvera | 0.1–0.5 kyr (U-Th) |
| 黄土 | 中国黄土高原 | 2–3 kyr (轨道调谐) |
| 湖泊沉积 | Malawi, El'gygytgyn, Ohrid, Junín | 1–3 kyr |

详见 [data/README.md](data/README.md)。

## 引用

如在研究中使用 CARC，请引用配套论文（详见 [CITATION.cff](CITATION.cff)）：

> Ruan, X., Wang, H., & Zhang, Z. CARC: Chronology-Aware Regional Causal
> Inference on Paleoclimate Event Scales. 地学前缘（Earth Science Frontiers），投稿中。

## 许可证

MIT License — 详见 [LICENSE](LICENSE) 文件。
