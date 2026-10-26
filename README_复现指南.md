# CARC 复现指南 · Reproduction Guide

本仓库是 CARC 论文的**复现代码包**：`scripts/`（编号分析驱动 01–45，含 `fignew/`
图件重建系列）、`src/`（管线与检验器）、`proxy_records/` + `proxy_errors/` + `data/`
（运行所需数据）、`Tables/`（论文表格 xlsx 参考——公开仓库中因投稿保密被
`.gitignore` 忽略，投稿包内随附）。运行时输出写在本仓库的 `outputs/`、`figures/`
（自动创建，不跟踪）。

> 手稿 docx 与原始文献档案不随代码公开；`scripts/figure_tools/`（制图/QA 工具）与
> `scripts/manuscript_tools/`（文稿装配工具）为工作流辅助，不在下表复现映射内。

## 版本说明（先读）

论文早期版本的数字由 canonical 系列（`22`、`06` 等）直接产出；最终投稿版的数字由
重建系列 `34–39` 产出。2026-10-24 审计定案：**投稿版现值全部可复现**——例如
Table 4 的 54.0 由现存 `06_conditional_causality.py` 直接产出。下表两列并列两种
口径，投稿版数字以重建系列为准。

## 论文产物 → 脚本映射

| 论文产物 | 旧版数字（canonical 系列） | 投稿版数字（重建系列 34–39） | 盘上输出 |
| :--- | :--- | :--- | :--- |
| Table 1（Scenario A） | `scripts/22_scenarioA_final.py`（run22f：α=0.5/force/aic_f/test_joint/maxlag=50；β=0.25 行 = 6.0/6.5/4.5） | `scripts/34_scenarioA_submitted.py`（目标 70/60/96） | `outputs/scenarioA_final_table1.csv` |
| Table 2（CCM/PCMCI） | — | `scripts/39_table2_ccm_pcmci.py` | `outputs/table2_ccm_pcmci.csv` |
| Table 4（六窗口） | `scripts/06_conditional_causality.py`（**lagonly** 口径；54.0 即由此直出） | `scripts/35_table4_submitted.py`（`--all-pairs` → Table S4 180 格） | `outputs/conditional_causality_allpairs.csv` / `outputs/table4_submitted*.csv` |
| Table 5（相位） | `scripts/08_regional_phase_lag.py` | `scripts/37_phase_submitted.py` | — |
| §4.3 区域对 | `scripts/09_regional_robustness.py` → `scripts/17_regional_pairwise_matrix.py`（**min-p** 口径） | `scripts/36_regional_submitted.py` | `outputs/regional_robustness_windows.csv` |
| Table S2/S4（记录账目） | `scripts/01_preprocess_data.py` + `data/data_tiers_final.csv` | — | `data/` |
| Table S5（全对扫描 + 消融） | `scripts/19_full_pairwise_scan.py`（分片续跑） | — | `outputs/pairwise_scan/` |
| Table S6 / Fig.5（审查） | `scripts/41_scrutinization_figure.py` | — | `outputs/table_s6_scrutiny_profile.csv` |
| Table S7（理论 FPR 剖面 + α×τ 门） | `scripts/42_theory_fpr_table.py` | — | `outputs/tableS7_fpr_profile.csv` / `outputs/tableS7_binomial_gate.csv` |
| β=0.25 反解锚点（附录 S4.1） | `scripts/43_pairwise_beta_inversion.py` | — | `outputs/beta_inversion_pairs.csv` |
| Theorem 2 数值例 MC（§2.2） | `scripts/44_theorem2_montecarlo.py` | — | `outputs/theorem2_mc_verification.csv` |
| Fig. 2(a) MC FPR 曲线 | `scripts/45_fig2a_fpr_sweep.py` | — | `outputs/fig2a_mc_sweep.csv` |
| §5.3 误差压缩 | `scripts/24_age_error_scaling.py` | — | `outputs/age_error_scaling*.csv` |
| 滑窗 Fig.7 | `scripts/15_sliding_window_carc.py` | — | — |
| Scenario B | （论文已删，脚本保留备查） | `scripts/38_scenarioB.py` | `outputs/scenarioB_tableS6.csv` |
| 出版物图 | `scripts/13_publication_figures.py` | — | `figures/` |
| 图件重建系列 | `scripts/fignew/01–19` | — | `figures_new/`（gitignore，脚本保留） |

## 快速开始

```bash
# 环境：requirements.txt / environment.yml（仓库根）
pip install -r requirements.txt

python scripts/22_scenarioA_final.py --quick   # 冒烟；去 --quick 为全量 20×200（耗时）
```

各脚本 docstring 均写明 `--quick`/`--probe` 冒烟开关与输出 CSV 路径。

## 重跑必读的坑（既往结论）

1. **口径双语义**：Table 4 = lagonly（AIC 滞后单点 F）；§4.3 区域 = min-p（1..L 扫描）。混用差 ~4×。
2. **pyEDM 2.5（Windows）**：需 `legacy=True` + 列名净化；`EmbedDimension` 挂起 → CCM 的 E 固定 3。
3. **tigramite 5.x**：`ParCorr` 从 `tigramite.pcmci.parcorr` 导入（4.x 写法会挂）。
4. **21 与 34/22 的数字不可直接对比**：三者的年龄扰动注入顺序不同（生成期注入 / test_joint / --perdraw）。
5. **`13_publication_figures.py` 重生成前注意**：FigS10 图内数字已知陈旧（9/58/30 vs canonical 7/48/25），投稿前必修，勿随手覆盖定稿图。
6. `00_stacked_regions_800ka/*.csv` 旧孤儿堆叠（原始单位）勿用。
7. 数据路径全部由 `src/config.py` 解析（相对仓库根），亦可用 `CARC_DATA_DIR` / `CARC_ERROR_DIR` / `CARC_OUTPUT_DIR` / `CARC_FIGURE_DIR` / `CARC_CONFIG_OVERRIDE` 覆盖。
