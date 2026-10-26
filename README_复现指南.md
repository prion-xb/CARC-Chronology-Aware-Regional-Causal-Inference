# 完整复现代码 · 复现指南

本文件夹是 CARC 论文的**自包含复现代码包**：`scripts/`（全部分析脚本，含 `fignew/` 重建图系列）、`src/`（管线与检验器）、`proxy_records/` + `proxy_errors/` + `data/`（运行所需数据）、`Tables/`（论文表格 xlsx 参考）。运行时输出写在本文件夹的 `outputs/`、`figures/`（自动创建，不覆盖主仓库）。

> 主仓库中 `data 原始/`（原始文献档案，3.5k 文件）与 `paper/`（docx）未复制，需要时回主仓库取。

## 版本语义（先读）

论文有两个版本，数字不同：

- **现行工作文件 Research Pape5.docx** ← 对应 **run22f canonical 系列**（下表"现行论文"列）。
- **最终投稿版 CARC_v3_final-ZZF.docx** ← 其实现代码**已丢失**，仓库里是重建系列（`34–39`，目标数字见 [PLAN_submitted_reimplementation.md](PLAN_submitted_reimplementation.md)）。

## 论文产物 → 脚本映射

| 论文产物 | 现行论文（canonical） | 投稿版重建（目标版） | 盘上输出 |
| :--- | :--- | :--- | :--- |
| Table 1（Scenario A） | `scripts/22_scenarioA_final.py`（run22f：α=0.5/force/aic_f/test_joint/maxlag=50；β=0.25 行 = 6.0/6.5/4.5） | `scripts/34_scenarioA_submitted.py`（目标 70/60/96） | `outputs/scenarioA_final_table1.csv` |
| Table 2（CCM/PCMCI） | `scripts/39_table2_ccm_pcmci.py`（重建选择，见下） | 同左（原版丢失） | `outputs/table2_ccm_pcmci.csv` |
| Table 4（六窗口） | `scripts/06_conditional_causality.py`（**lagonly** 口径） | `scripts/35_table4_submitted.py`（`--all-pairs` → S5 180 格，目标 max 54.0） | `outputs/conditional_causality_allpairs.csv` / `outputs/table4_submitted*.csv` |
| Table 5（相位） | `scripts/08_regional_phase_lag.py` | `scripts/37_phase_submitted.py` | — |
| §4.3 区域对 | `scripts/09_regional_robustness.py` → `scripts/17_regional_pairwise_matrix.py`（**min-p** 口径） | `scripts/36_regional_submitted.py` | `outputs/regional_robustness_windows.csv` |
| Table S2/S4（记录账目） | `scripts/01_preprocess_data.py` + `data/data_tiers_final.csv` | — | `data/` |
| Table S5（全对扫描） | `scripts/19_full_pairwise_scan.py`（standard + nonoise/noage/noarc/alldiff 消融，分片续跑） | — | `outputs/pairwise_scan/` |
| Table S6 / Fig.5（审查） | `scripts/41_scrutinization_figure.py` | — | `outputs/table_s6_scrutiny_profile.csv` |
| Table S7（理论 FPR 剖面 + α×τ 门矩阵） | `scripts/42_theory_fpr_table.py` | — | `outputs/tableS7_fpr_profile.csv` / `outputs/tableS7_binomial_gate.csv` |
| β=0.25 反解锚点（§4.1 / 附录 S4.1） | `scripts/43_pairwise_beta_inversion.py` | — | `outputs/beta_inversion_pairs.csv` |
| Theorem 2 数值例 MC 验证（§2.2） | `scripts/44_theorem2_montecarlo.py` | — | `outputs/theorem2_mc_verification.csv` |
| Fig. 2(a) MC FPR 曲线（§2.2） | `scripts/45_fig2a_fpr_sweep.py` | — | `outputs/fig2a_mc_sweep.csv` |
| §5.3 误差压缩 | `scripts/24_age_error_scaling.py` | — | `outputs/age_error_scaling*.csv` |
| 滑窗 Fig.7 | `scripts/15_sliding_window_carc.py` | — | — |
| Scenario B | （论文已删，保持删除） | `scripts/38_scenarioB.py` | `outputs/scenarioB_tableS6.csv` |
| 出版物图 | `scripts/13_publication_figures.py` | — | `figures/` |
| figures_new 重建图 | `scripts/fignew/01–19`（规划见主仓库 `figures_new/README_规划.md`） | — | `figures_new/` |

## 快速开始

```bash
# 环境：requirements.txt / environment.yml（主仓库）
pip install -r ../requirements.txt

cd 完整复现代码
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
7. 数据路径全部由 `src/config.py` 解析（相对本文件夹根），亦可用 `CARC_DATA_DIR` / `CARC_ERROR_DIR` / `CARC_OUTPUT_DIR` / `CARC_FIGURE_DIR` / `CARC_CONFIG_OVERRIDE` 覆盖。
