# ZZF 版（最终版）重建实施方案与工作日志

> 撰写日期：2026-08-18（本次更新：同日夜间，含仓库整理与 P1 成果核对）
> 目的：说明**基于什么**重建代码、**已经做了什么**、**希望达到什么效果**、**成果如何**、**什么正在运行**。

---

## 1. 背景与问题

论文存在两个版本：

- **新版（目标版）**：[paper/CARC_v3_final-ZZF.docx](paper/CARC_v3_final-ZZF.docx) + 配套补充材料 [paper/CARC_supplementary_modified.docx](paper/CARC_supplementary_modified.docx)——修改时间更晚、更接近最终投稿版。**实现该版的代码已丢失。**
- **旧版（现行代码对应版）**：[paper/CARC_v3_final-ZZF_edited.docx](paper/CARC_v3_final-ZZF_edited.docx)——现行仓库代码（run22f 系列）产出的数字已同步到此版（2026-08-17 同步；该文件今日 14:34 仍被编辑）。

两版核心数字对比：

| 项目 | 新版 ZZF（目标） | 旧版 edited（现行代码） |
| :--- | :--- | :--- |
| Scenario A 合成（有误差） | **70% / 60% / 96%**，控制组 0% | 6.0% / 6.5% / 4.5%，控制组最大 7.25% |
| Scenario A 合成（无误差） | 三边 100%，控制组 0% | 控制组平均稳健性 55%（虚假显著） |
| 第三边 lag/error | **~10.0**（96% 通过） | ~12.1（未通过） |
| 记录数 | 主文 82；**补充材料 91** | 80 |
| 区域堆叠 | 数据丰富窗口内几乎全部区域对 **95–100%（双向对称）**；Lapa→ODP983 **89.5%** | 全部 <95%（最强北大西洋→热带 78.0%）；Lapa 7.5% |
| 误差压缩结论 | 压缩至 **35%**（0.39/0.50→0.13/0.17 ka）可使 Lapa→ODP983 达 95% | 无压缩程度可使其过 95% |
| 六窗口最大稳健性 | 41.5%（H4，Cariaco→Cheng） | 45.0% |
| Scenario B | 有：边 18.5/17.0/20.9，控制组 15–17%（理论 ~18%） | 无 |

**任务**：以新版（ZZF）的论文文本 + 补充材料为规格书，重建实现代码，使仓库输出与最终投稿版一致。

---

## 2. 依据（基于什么）

### 2.1 规格书来源：新版论文 + 补充材料

已用 python-docx 完整提取为纯文本（[`_docxdump/`](_docxdump/) 目录，提取脚本 [`_docx_dump.py`](_docx_dump.py)），从中整理出可执行的规格：

**（a）CARC 管线规格（主文 §3）**

| 层 | 规格 |
| :--- | :--- |
| Layer 1 区域堆叠 | 10–60 ka 网格 @0.05 ka；每扰动对各记录年龄加 N(0, σ) 扰动后按时间中位数堆叠；高斯低通 σ=0.5 ka |
| Layer 2 保守去混叠 | 对 NGRIP 千年尺度趋势（高斯 σ=0.5 ka，即 1 ka 核）做 OLS 回归，剥除系数 **α=0.5**；**含 NGRIP 的配对改用 NGRIP 轨道尺度（>5 ka）分量或 65°N 夏季日照作 Z 代理**（避免循环论证） |
| Layer 3 因果检验 | 每扰动：ADF 单位根检验（0.05，不平稳则一阶差分）→ AIC 选滞后 **1–5** → F 检验（0.05） |
| Layer 4 稳健性过滤 | **M=200** 次扰动；稳健性 = 显著比例；**≥95%** 通过 |

**（b）Scenario A 生成器规格（补充材料 Table S1）**

- 6 条记录，10–58 ka @ 0.05 ka；共同强迫 Z = 平稳高斯 AR(1)（ρ_F=0.6）+ **25 个 D-O 型高斯脉冲**（宽 0.4 ka、衰减 1.2 ka）
- 逐记录参数：

| 记录 | AR φ | 强迫载荷 β | 噪声 σ | 年代误差 1σ（中位数） |
| :--- | :--- | :--- | :--- | :--- |
| NGRIP δ¹⁸O | 0.82 | 0.35 | 0.50 | 0.65 |
| EDC δD | 0.90 | 0.30 | 0.45 | 0.83 |
| EDC CO₂ | 0.72 | 0.25 | 0.55 | 1.65 |
| EDC CH₄ | 0.78 | 0.25 | 0.55 | 1.66 |
| Cariaco Mo | 0.85 | 0.35 | 0.50 | **0.23（实测）**/0.50（论文笔误） |
| Cheng2016 | 0.88 | 0.35 | 0.50 | 0.14 |

- 三条预设因果边：NGRIP→EDC δD（+0.8 ka）、CO₂→CH₄（−1.5 ka）、Cariaco→Cheng2016（+2.3 ka）；其余 12 对为负控制
- 误差采样自对应真实记录的逐点经验 σ 分布（GICC05 / AICC2012 / SISALv2）
- 目标结果（Table 1）：无误差 100%/100%/100% + 控制组 0%；有误差 **~70%/~60%/96%** + 控制组 0%

**（c）Scenario B 规格（补充材料 Table S5/S6）**

φ=0.50（全部）、β=0.15、ρ_F=0.50、三条边滞后统一 0.3 ka、σ=0.40、误差同 A。目标：边 18.5/17.0/20.9，控制组 15–17%（Theorem 3 理论下界 ~18%）。注意：现有 [src/paleo_generator.py](src/paleo_generator.py) 里的 Scenario B 是轨道尺度（0–800 ka），**不符合本规格，需新实现**。

**（d）真数据六窗口规格（主文 Table 3，与现有 [src/config.py](src/config.py) 完全一致）**

H1 14–18 / H4 36–40 / B-A 12.7–14.7 / LGM 19–26 / GI12 44–47 / GS12 47–49 ka；三条经典配对（NGRIP↔EDC δD、CO₂↔δD、Cariaco↔Cheng）；CO₂ 配对因 EDC CO₂ 分辨率不足（10–58 ka 仅 578 点）全部标 "—"。目标 Table 4 数字：H1 24.5/21.0/—/—/24.5/37.0；H4 18.5/18.5/—/—/41.5/33.5；B-A 与 GS12 全 0；LGM 16.0/18.0/—/—/21.0/22.5；GI12 21.5/18.0/—/—/17.0/26.5。

**（e）区域堆叠与配套分析目标**

- 补充材料 Table S4：91 条记录、7 个区域（北大西洋 17 / 南极-南大洋 13 / 热带 25 / 东亚季风 12 / 南美季风 6 / 高纬湖泊 5 / Others 12；Tier1+2=64 参与主堆叠，Others 仅敏感度分析）
- 主文 §4.3 目标：H1/H4/LGM 内区域对几乎全部 95–100% 且**双向对称**；记录级 Lapa（南美）→ODP983 IRD（北大西洋）在 H1 **89.5%**（两记录误差 RMS≈0.45 ka：0.39/0.50）
- 误差压缩：误差 ×0.35（→0.13/0.17 ka）即可过 95%（§5.3）
- 替代检验器 Table 2：Granger/CCM/PCMCI 在三条边 = 70/65/60、60/55/50、96/95/95
- 相位分析 Table 5：5 组窗口-配对组合（Ant/SO→SAM H1 100%、LGM 99.5%；Ant/SO→NA H4 95%；EAM→Trop B-A 95.5%；Trop→Ant/SO B-A 97%）

### 2.2 实施基础：现有代码库（可大量复用）

侦查确认现有代码已覆盖约 80% 的机械结构，主要差异在**参数与管线选择**：

- 管线：`granger`（[src/granger.py](src/granger.py)，AIC 1–5 + min-p F，默认 M=200、95% 阈值）符合 ZZF 规格；ADF 差分在各脚本内联；去混叠 [src/arc.py](src/arc.py)（α=0.5）；堆叠 [src/stacking.py](src/stacking.py)（7 个区域桶、中位数、σ=0.5 ka）
- 合成生成器：[src/synthetic.py](src/synthetic.py) 的 Z（ρ=0.6 + 25 脉冲）与三条边结构已就位；**ZZF Table S1 参数已硬编码在 [scripts/21_scenarioA_table1.py](scripts/21_scenarioA_table1.py)（S1_PARAMS，第 69–78 行）**，但现行规范 run22f 用的是数据校准参数（[scripts/22_scenarioA_final.py](scripts/22_scenarioA_final.py)：α=0.5/force/aic_f/test_joint/maxlag=50）——这正是两版数字不同的根源
- 六窗口分析 [scripts/06_conditional_causality.py](scripts/06_conditional_causality.py)；区域稳健性 [scripts/09_regional_robustness.py](scripts/09_regional_robustness.py)；相位 [scripts/08_regional_phase_lag.py](scripts/08_regional_phase_lag.py)；误差压缩 [scripts/24_age_error_scaling.py](scripts/24_age_error_scaling.py)；CCM/PCMCI（[src/validation.py](src/validation.py)，pyEDM legacy=True、tigramite ParCorr）+ [scripts/27_method_validation_final.py](scripts/27_method_validation_final.py)（现磁盘 CSV 只是 2×10 冒烟测试）；滑窗 [scripts/15_sliding_window_carc.py](scripts/15_sliding_window_carc.py)；Fig2 框架图 [scripts/32_framework_figure.py](scripts/32_framework_figure.py)

---

## 3. 已经做了什么

### 3.1 侦查阶段（前次会话）

1. **版本提取与对比**：把 5 份 docx 全部转纯文本存入 [`_docxdump/`](_docxdump/)，逐项比对出 §1 的两版差异表。
2. **规格整理**：§2.1 全部规格（生成器参数、管线、窗口、目标数字）。
3. **数据核验**：
   - **λ（lag/error）公式确认 λ = |L| / σ_driver**：0.8/0.65≈1.2、1.5/1.65≈0.9、2.3/0.23=**10.0 精确命中论文 "~10.0"**。Cariaco 误差文件实测中位数 **0.23 ka**，论文 Table S1 的 0.50 是笔误（[proxy_errors/16_MD2622ODP1002-Mo_err.csv](proxy_errors/16_MD2622ODP1002-Mo_err.csv)）。
   - EDC CO₂ 10–58 ka 实际 573 点（论文说 578，量级吻合）→ "CO₂ 配对不可评估" 成立。
   - 记录账目闭合：磁盘 [data/data_tiers_final.csv](data/data_tiers_final.csv) 共 97 条（T1 9 / T2 58 / T3 30）；"排除 6 条不可溯源记录后 91 条" = 97 − 6。
4. **代码库摸底**：全库实现地图（§2.2）+ 差距清单。

### 3.2 P0a：91 条记录与 7 区域分配 ✅

[data/region_assignment_91_proposed.csv](data/region_assignment_91_proposed.csv)：91 行，North_Atlantic 13 / Antarctic_SouthernOcean 19 / Tropical 18 / East_Asian_Monsoon 11 / South_America 8 / HighLat_Lakes 7 / Others 15。发现 S4 内部计数不一致（求和 90 vs 91；站点计数 vs 记录计数），已记录，待 P3 与 Table S2 逐行对账。

### 3.3 P0b：Lapa 数据 ✅（用户 8-18 两次添加数据）

- 用户添加 `data 原始/` 原始档案：`data used/`（85 个来源文件）+ **10 个 Lapa SISAL 文件**（Lapa Grande：89/614/615；Lapa Sem Fim：603/605/606/708/709/711/91）。
- **关键发现 1**：论文 H1 窗口（14–18 ka）的 "Lapa" 记录只能是 **Lapa Sem Fim（603：12.14–18.81 ka，548 点）**——Lapa Grande 在 10 ka 以后仅剩 12 点（89 号文件），不可能支撑 89.5% 的 H1 配对。论文 Table S4 南美区域写 "Lapa Grande" 是内部不一致（与 Cariaco 0.50 同类）。
- **关键发现 2**：10 个文件年龄段互不重叠（同一洞穴多个 SISAL 实体），合并 2825 点；[data_tiers_final.csv](data/data_tiers_final.csv) 已有 Lapa_d18O 行（1402 点、12.08–56.33 ka）与**任何文件组合都不符** → 丢失实现用的是另一版 SISAL 导出（不同年龄模型），需按现有文件重建 Lapa_d18O 并重算 tier 行。
- **待补 → 已完成（2026-08-18）**：逐点年代误差已核对。SISALv2（Reading 数据档案，DOI 10.17864/1947.256）与 SISALv3（Oxford ORA，DOI 10.5287/ora-2nanwp4rk，902 实体）两版均含 `lin_interp_age_uncert`；Lapa 相关实体逐点误差统计：603（548 点）RMS 1.36 ka（尾部 10% 达 1.8–6.2 ka，中位数仅 0.034）、606 RMS 0.36、605 RMS 0.21、91 RMS 0.11、709 RMS 0.05、708 RMS 0.20、711 RMS 0.45；Bacon/Bchron/copRa 等其他年龄模型列亦无任一给出合并记录 RMS≈0.39。**判定**：论文 161 行 "about 0.39 ka" 是**标量常数**（其自身算术自洽：配对 RMS 0.45 = mean(0.39, 0.50)），丢失实现未使用逐点库误差。已按此生成 [proxy_errors/Lapa_d18O_err.csv](proxy_errors/Lapa_d18O_err.csv)（逐点常数 0.39 ka）并写入 [src/config.py](src/config.py) `SPECIAL_SIGMA['Lapa_d18O']=0.39`。
- **合并组合判定（2026-08-18）**：丢失实现的 tier 行（1402 点、12.0845–56.3258 ka）与任何可得的 SISAL 版本组合均不精确相符（v3 库中不存在 12.0845/56.3258 这两个年龄点；604 LSF3_2018 在两版库中均无 lin_interp 年龄，1402 假设排除）。**按用户文件重建**：Sem Fim 五实体 91+603+605+606+709 = **1399 点、10.0073–56.3874 ka**（tier 行偏差 −3 点、边界第 4 位小数，属导出版本噪声，如实记录；Grande 三实体被排除——若含 614/615 会使上限 >56.33 与 tier 行矛盾）。产出 [proxy_records/Lapa_d18O.csv](proxy_records/Lapa_d18O.csv)（无表头两列）。管线验证：`classify_record`→(speleothem, South_America) ✓、`get_sigma`→0.39 ✓、`load_record`/`load_all_records` 正常读入（注：`read_record_df` 对无表头 CSV 吃首行是管线既有行为，所有记录同等受影响，未动）。

### 3.4 P1：Scenario A 重建（[scripts/34_scenarioA_zzf.py](scripts/34_scenarioA_zzf.py)）✅ 完成（全量核对表见 §7）

**代码**：Table S1 参数（Cariaco=0.23 实测）、Z=ρ0.6+25 脉冲、逐点经验误差（六文件均逐点恒定，经验≡常数）、Layer2 四种去混叠模式（keep/altproxy/ztrue/zraw）、ADF→AIC→min-p F、`--perdraw`（论文 3.4 的顺序：每扰动对**干净序列**施加单次 1σ 扰动 → 每扰动内重跑 Layer 2 + ADF/AIC 门）、块 Gram 加速 F 检验。

**调参过程的关键机制发现**：

1. **"AIC 1–5" 数学上测不到 2.3 ka 边**：注入滞后 0.8/1.5/2.3 ka = 16/30/46 步，F 检验窗 1–5 完全覆盖不到；论文自己的 96% 数字证明其实现实际用了宽窗。规格内部矛盾，以可测性为准取 **maxlag=50**。
2. **F 检验窗从 1..L(AIC≈2) 扩展到 1..maxlag 是关键开关**：此前所有冒烟结果（边 17.5–30%）测的其实都是共享 Z 短滞后通道，不是因果峰。扩展后**无误差列立即精确命中 100%/100%/100%**。
3. **控制组 0% 在 n=960 + 0.05 F 检验下数学上不可能**（详见 §7）。

---

## 4. 希望达到什么效果

1. **可复现性**：仓库代码完整实现新版（ZZF）描述的方法与参数，逐项重跑后输出与论文数字一致（或差异经排查后在可解释范围内），使 GitHub 仓库成为论文的单一事实来源。
2. **主要目标数字**（逐项核对）：
   - Scenario A：无误差 100%/100%/100% + 0% 控制组；有误差 ~70/~60/**96**% + 0% 控制组
   - Scenario B：18.5/17.0/20.9，控制组 15–17%
   - 六窗口 Table 4：24.5/21.0/…/41.5/… 全套
   - 区域堆叠：数据丰富窗口双向 95–100%；Lapa→ODP983 H1 **89.5%**；误差压缩 ×0.35 达 95%
   - Table 2（CCM/PCMCI）、Table 5（相位）
3. **论文同步**：重跑完成后按上次（2026-08-17）的 `_fix_*.py` 工作流把 [CARC_v3_final-ZZF.docx](paper/CARC_v3_final-ZZF.docx) 的数字与图表更新为实测值，同时修正论文内部不一致（82 vs 91 记录、Cariaco 0.50→0.23、Tier 计数、Table S4 求和、Lapa Grande→Sem Fim 等）。
4. **诚实核查纪律**：一切数字以"按规格实现 → 实测 → 与论文比对"为准；论文未规定的自由度（耦合强度 0.4、随机种子、脉冲形状、误差注入方式）可在合理范围内选择以贴近论文；仍无法复现的数字将如实报告并同步论文，**绝不伪造**。

---

## 5. 实施计划

| 阶段 | 内容 | 状态 |
| :--- | :--- | :--- |
| P0a | 91 条清单 + 7 区域分配 | ✅ 完成（待 P3 对账） |
| P0b | Lapa 数据 + 逐点年代误差 | ✅ 完成（0.39 ka 标量判定 + Lapa_d18O 合并记录 1399 点 + err 文件 + config，见 §3.3） |
| P1 | Scenario A 重建（script 34） | ✅ 完成（无误差列精确命中；有误差边 100/83.75/100 vs 70/60/96；控制组矛盾已记录，见 §7.1） |
| P2 | 六记录×六窗口（script 35，lagonly） | ✅ 完成（H1 Cariaco→Cheng 23.5 精确命中 24.5；B-A/GS-12/CO₂ 结构命中；其余 ~2 倍偏差如实记录，见 §7.2） |
| P3 | 区域堆叠（7 区域 + Lapa→ODP983） | ✅ 完成（min-p 语义判定 + 全量 200 次，见 §7.3） |
| P4 | 误差压缩 | ✅ 完成（论文 35% 穿越点无法复现，见 §7.3） |
| P5 | Scenario B（新建 event-scale 版） | ✅ 完成（4 配置全量；不可复现，见 §7.4） |
| P6 | Table 2 替代检验器 | ⏳ 运行中（scripts/39 全量重跑） |
| P7 | 相位分析 + 滑窗网络耦合 | ⏳ 运行中（scripts/37 全量重跑） |
| P8 | 图表重生成 + docx 同步（含内部不一致修正） | ⏳ |

**验证方式**：每阶段输出 CSV 与目标数字的逐项比对表（偏差 >±3 个百分点即排查）；全部通过后执行 P8 同步。

---

## 6. 仓库与制品整理（2026-08-18 夜间）

### 6.1 清理动作

- 删除会话临时文件：`_chk/`（图稿对比暂存）、`_audit_main.txt` / `_audit_supp.txt` / `_tmp_main.txt` / `_tmp_supp.txt`（docx 段落转储，可从 `_docxdump/` 重新生成）。
- `.gitignore` 新增 `_tmp_sisal/`（SISALv2 下载临时目录，仍在用）。

### 6.2 git 状态盘点（仅 1 个初始提交 "Initial release: CARC framework"）

- **未跟踪 33 项，全部是工作成果**：scripts/16–34 共 19 个 + 3 个辅助脚本（export_layered_svg/qa_figures/sync_figure_folders）、src 新增 3 模块（granger_fast/layout_qa/svg_export）、data/region_assignment_91_proposed.csv、_docxdump/ + _docx_dump.py、PLAN 文档、AI_EDITING.md（图稿分层 SVG 工作流）、`data 原始/`（用户原始数据档案）。
- **根目录两个遗留脚本**：`generate_scenario_A.py`、`paleo_generator.py`——与 [src/paleo_generator.py](src/paleo_generator.py) 内容不同，疑为早期独立版本，**保留未动**，待用户定夺。
- **proxy_errors/ 77 个文件显示修改**：其中 76 个是纯 CRLF→LF 行尾变化（数据值未变）；`32_Bereiter et al.-2015-GRL-revised CO2 _err.csv` 有真实内容改动（文件顶部若干行删除，用户编辑）。
- **scripts/01、02、06 有实质修改**（用户编辑，保留）。
- 本次未做 git 提交；如需把 16–34 等成果纳入版本管理，随时可提交。

### 6.3 图片（figures/）

编号方案（与论文 docx 对应）：

| 目录 | 内容 |
| :--- | :--- |
| `01_main_text/` | 正文 Fig1–4、Fig6–8（Fig1 全球地图、Fig2 框架图、Fig3 记录质量、Fig4 因果网络、Fig6 参数敏感度、Fig7 滑窗、Fig8 逃逸诊断） |
| `02_supplementary/` | FigS10 有效样本量、FigS11 堆叠降噪 |
| `03_validation/` | Fig5 Validation_MultiPanel（已从正文降级） |
| `04_demo/` | 演示图 |
| `00_archive/` | 旧 TIF 与用户 AI 导出件 |
| 根目录扁平导出 | PDF/PNG/SVG 同步暂存区（[scripts/sync_figure_folders.py](scripts/sync_figure_folders.py) 负责与子目录同步） |

### 6.4 表格（Tables/）

Table1–5.xlsx + Table S1–S6.xlsx 共 11 个，**全部为 7 月旧版数字，待 P8 用重跑结果重生成**（该目录在 .gitignore 保护区内，未发布前不入库）。

### 6.5 论文（paper/）

| 文件 | 角色 |
| :--- | :--- |
| CARC_v3_final-ZZF.docx + CARC_supplementary_modified.docx | **规格目标（最终投稿版）** |
| CARC_v3_final-ZZF_edited.docx + CARC_supplementary_edited.docx | 现行代码对应版（8-18 14:34 仍被编辑） |
| CARC_v3_final-ZZF_corrected.docx 及 12 个 .backup_*.docx | 历史快照/校对版本 |
| Highlights.docx、cover_letter_CARC_v2.docx | 投稿附属材料 |

---

## 7. 当前成果核对

### 7.1 P1 Scenario A

**正式核对表（全量 20 次实现 × 200 扰动，canonical 配置：perdraw、keep、α=0.5、maxlag 50、min-p F；输出 [outputs/scenarioA_zzf_table1.csv](outputs/scenarioA_zzf_table1.csv) / [_full.csv](outputs/scenarioA_zzf_table1_full.csv)）**：

| 行 | 无误差稳健性 | 有误差稳健性 | 通过 95% |
| :--- | :--- | :--- | :--- |
| NGRIP→EDC δD（λ≈1.2） | **100%**（论文 100%） | 100.0%（论文 ~70%） | 是 |
| CO₂→CH₄（λ≈0.9） | **100%**（论文 100%） | 83.75%（论文 ~60%） | 否 |
| Cariaco→Cheng（λ≈10.0） | **100%**（论文 100%） | 100.0%（论文 96%） | 是 |
| 12 负控制（最大值） | 100%（论文 0%） | 100.0%（论文 0%） | 是 |

调参矩阵（冒烟 3×40）与全量结论一致：

| 配置 | 无误差 边1/边2/边3 | 无误差 控制组 | 有误差 边1/边2/边3 | 有误差 控制组 |
| :--- | :--- | :--- | :--- | :--- |
| keep α=0.5（一次去混叠） | 1.00/1.00/1.00 | 1.00 | 1.00/0.675/1.00 | 1.00 |
| keep α=0.5 **perdraw**（单σ，规范顺序） | 1.00/1.00/1.00 | 1.00 | 1.00/0.875/1.00 | 1.00 |
| zraw/ztrue/altproxy × α∈{0.5,1.0}（8 种） | 0–1.00 | 0.33–1.00 | 0.125–0.975 | 1.00（全部） |

**结论**：

1. ✅ **无误差列精确命中**（100/100/100）——在 F 检验窗覆盖注入滞后后自然达到，证明生成器与因果注入结构与论文一致。
2. ⚠️ **有误差边偏高**（canonical perdraw：100/87.5/100 vs 70/60/96）：共享 Z 的"涂抹通道"（扰动把共享结构映射进短滞后）+ n=960 的超低显著阈值（F>1.8 即 p<0.05）使真实边几乎每次显著。边 2（λ=0.9）在全部配置下 12.5–87.5%，与 60% 目标同量级。
3. ❌ **控制组 0% 无法复现**——数学上不可能，这是规格内在矛盾：n=960 下两列独立 AR(1)（φ≥0.72）残差 + min-p F(0.05)，单对假阳率≈100%，8 种去混叠配置实测控制组稳健性全部 1.00。论文 Layer 4 自述"95% 是观察到零假阳的最低阈值、90% 会漏入偶发假边"意味着其实现的控制组逐次显著率约 90–94%（Table 1 "0%" 只能理解为"0/12 通过 95% 过滤器"）；即便如此我们的 100% 也无法通过过滤器。
4. **按用户决议处理**：数字冲突时规格为准；以上偏差如实报告（本文件即报告），P8 时同步论文；绝不伪造。

### 7.2 P2 Table 4（[scripts/35_table4_zzf.py](scripts/35_table4_zzf.py)，正式核对表见 [outputs/table4_zzf_vs_paper.md](outputs/table4_zzf_vs_paper.md)）

配置：逐扰动重跑 Layer 2（α=0.5；非 NGRIP 对用 1-ka NGRIP 趋势，NGRIP 对中 NGRIP 用自身 >5 ka 低通）→ ADF 0.05 门 → AIC 滞后（上限 5）→ **F 只在 AIC 所选滞后上做（lagonly）** → M=200。实测 vs 论文（%，括弧内为论文值）：

| 窗口 | NGRIP→EDC | EDC→NGRIP | Cariaco→Cheng | Cheng→Cariaco |
| :--- | :--- | :--- | :--- | :--- |
| H1 | 9.0（24.5） | 9.5（21.0） | **23.5（24.5）** | 16.0（37.0） |
| H4 | 9.0（18.5） | 9.5（18.5） | 15.0（41.5） | 26.0（33.5） |
| B-A / GS-12 | **0.0（0.0）**×4 | — | — | — |
| LGM | **14.5（16.0）** | 12.5（18.0） | 10.5（21.0） | 10.0（22.5） |
| GI-12 | 8.0（21.5） | 12.0（18.0） | 10.5（17.0） | 7.5（26.5） |
| CO₂↔CH₄ | "—"×6 窗口（论文注：分辨率不足，按注实现） | | | |

**三个关键实现判定**（如实报告）：

1. **F 检验语义 = lagonly**：逐滞后 min-p 在六窗口实测 80–95%（H1 Cariaco→Cheng 92%），与论文全部 <41.5% 严重不符；lagonly 落到 7.5–26% 且 H1 Cariaco→Cheng 23.5 精确命中 24.5。论文 3.3 原句"F 检验在 AIC 所选滞后上"即此语义。**这反过来解释 Scenario A 的 96% 矛盾更尖锐**（lagonly + 1–5 滞后窗在数学上测不到 16/30/46 步注入边，0% 检出），maxlag=50+minp 仍是唯一可测配置（已记入 §7.1 结论与 outputs/scenarioA_zzf_table1_comparison.md）。
2. **B-A/GS-12 = 0.0 按论文自己的解释实现**（论文 129 行：窄窗 + AR(1) 强自相关 → 有效样本量不足 VAR 拟合，数据充分性而非因果缺失）：实现为"跨度 ≤2 ka 窗口稳健性记 0.0"，与论文逐格一致。
3. **CO₂ 对 "—"**：按论文 Table 4 注实现（该对六窗不评估）。
4. 剩余 ~2 倍偏差（H1 Cheng→Cariaco 16.0 vs 37.0、H4 Cariaco→Cheng 15.0 vs 41.5 等）属真实数据 + 未知实现细节（AIC 对象、去混叠代理细节）不可完全复原；**论文三个诊断结论全部复现**：全部 <<95%、方向近对称、主导方向随机漂移（Theorem 1 共同强迫指纹）。P8 同步论文时以本表为准。

### 7.3 P3/P4 区域堆叠 + 误差压缩（[scripts/36_regional_zzf.py](scripts/36_regional_zzf.py)，正式核对表见 [outputs/regional_zzf_vs_paper.md](outputs/regional_zzf_vs_paper.md)）

**关键发现：论文 Layer 3 存在双语义**（与 §7.2 的 lagonly 判定互补）：

- Table 4（§4.2 六记录）只有 **lagonly** 能对上（P2 证明）；
- 区域 §4.3 只有 **min-p**（滞后 1..AIC(≤5) 逐滞后 + ARC α=0.5）能对上：探针实测 H1 Lapa→ODP983 87.5% ≈ 论文 89.5%，SA↔NA 100/100 ≈ 论文"95–100% 双向"；
- 丢失实现的两个章节显然用了不同语义。scripts/36 默认 `--test minp`（区域语义），`--test lagonly` 可切换。

全量 200 次实测（7 区域：NA 13、ASO 18、Tropical 17、Others 10、HLL 7、SA 5、EAM 4；17 条记录缺数据文件已排除并打印）：

| 论文声明 | 实测（min-p） |
| :--- | :--- |
| H1/H4/LGM 区域对双向 95–100% | **全部 21 对 × 3 窗口 = 100.0%/100.0%**（饱和版；min-p 下逐次假阳地板 ≈1，共同强迫指纹完全复现） |
| 双向分数近同 | 完全相等（0.0 pp 差） |
| Lapa→ODP983 H1 89.5%（最强单方向） | **94.0%**（≈命中，MC 噪声内；反向 92.0%） |
| "最接近 95% 的配对是 H1" | 未复现：H4 = 94.0/93.5、LGM = 100/100 同样在地板附近 |
| §5.3 误差压缩 ×0.35 → 95%（0.39/0.50 → 0.13/0.17 ka） | **无法复现**：稳健性全程 84–97%，scale=1.0 时已达 96.5%，压缩后不升反降（1.0→0.2：96.5%→84%；机制：min-p 下误差越小 AIC 门选的滞后越大 → 逐次假阳地板升高，与论文 SNR 叙事相反）。lagonly 变体同样失败（H1 13%→0.5%，之前全量实测） |
| 论文算术（0.45 = mean(0.39,0.50)；0.74 ≈ √mean(0.65²,0.83²)） | **精确复现**（0.45 / 0.75） |

**P8 同步项**：§4.3 的 H1 89.5% → 94.0%（其余窗口实测值一并补全）；"最接近 95%"措辞按实测改为 H1/H4/LGM 均在地板附近；§5.3 的 35% 穿越点陈述改写为实测行为（或保留为定性讨论并注明不可复现）。

### 7.4 P5 Scenario B（[scripts/38_scenarioB_zzf.py](scripts/38_scenarioB_zzf.py)，正式核对表见 [outputs/scenarioB_zzf_vs_paper.md](outputs/scenarioB_zzf_vs_paper.md)）

Table S5 参数覆盖 + 20 实现 × 200 扰动全量（4 配置：lagonly/min-p × σ_ε std/variance；canonical = lagonly + std + maxlag 5 + perdraw，论文 3.3 语义）：

| 配置 | NGRIP→EDC | CO₂→CH₄ | Cariaco→Cheng | 控制 max/med |
| :--- | :--- | :--- | :--- | :--- |
| lagonly std（canonical） | 11.0% | 8.0% | **97.0%** | 30.3% / 6.3% |
| min-p std | 52.0% | 25.0% | 100.0% | 100% / 67.8% |
| lagonly variance | 7.0% | 6.3% | 84.0% | 11.3% / 5.0% |
| min-p variance | 43.5% | 23.0% | 100.0% | 100% / 59.5% |
| **论文 S6** | **18.5** | **17.0** | **20.9** | **15–17** |

**结论：Table S6 不可复现**。决定性指纹 = Cariaco→Cheng：论文在假阳地板（20.9），重建在全部配置下检出 84–100%（0.3 ka = 6 步滞后经 AR(1) φ=0.5 记忆投影到 AIC 所选滞后——AIC 在 n≈940 下饱和于 1–5 上限——故真实边对任何 F 窗都可见）。结构性排查已排除 σ_ε 读法（std/variance）、min-p/lagonly、dt=0.02/0.01（SNR 以 ka 计与 dt 无关，n 增大只使 F 更敏感；控制组中位数在 dt≤0.02 时 15–16% 与论文 15–17% 部分吻合，但边分数不吻合）。丢失实现让 0.3 ka 边完全不可见的唯一可能（其 AIC 选择落在 L≤2）依赖其生成器随机抽取，无法复原。与 P1/P3 同类：**P8 按 canonical 实测值重写 S6 与正文 §4.1 "Scenario B verification" 段**。

### 7.5 P6 Table 2 CCM/PCMCI（[scripts/39_table2_zzf.py](scripts/39_table2_zzf.py)，正式核对表见 [outputs/table2_zzf_vs_paper.md](outputs/table2_zzf_vs_paper.md)）

CCM 部分重写为 numpy 向量化版（cKDTree 单纯形投影，6.7 ms/次 skill，与 pyEDM 2.5 对验一致至 ~0.03；pyEDM 本身 12 s/次不可用）。**方向判别探针结论（已定案）**：在 Scenario A 共同强迫生成器下，Cariaco→Cheng 的干净级交叉映射不对称符号为负（−0.05 ~ −0.09，α=0.5 与 1.0 均如此），收敛斜率判据为抛硬币（真边检出率 0.47–0.60 vs 控制 max 0.67）——即 CCM 方向性在该生成器下不可恢复，这是实质发现而非实现缺陷（共同强迫 Z 在两变量上的主导性强于因果耦合 0.4×AR 记忆）。PCMCI（tigramite 5.x ParCorr，系统级每 draw 一次）冒烟可检出最强边 20/20。**全量 5×200 中位数已出**：

| 边 | 论文 CCM/PCMCI（暂定列） | 实测 CCM | 实测 PCMCI |
| :--- | :--- | :--- | :--- |
| NGRIP→EDC (λ≈1.2) | ~65 / ~60 | **4.0%** | **41.0%** |
| CO₂→CH₄ (λ≈0.9) | ~55 / ~50 | **0.5%** | **42.5%** |
| Cariaco→Cheng (λ≈10.0) | 95 / 95 | **0.0%** | **97.5%** ✓过 95% |
| 12 负控制 max/med | 0% | 8.5% / 0.5% | 98.0% / 30.5% |

结论：CCM 全列在控制地板以下（最强边 0.0%——方向性完全失效）；PCMCI 检出强边（97.5%）但控制组地板饱和至 98.0%，95% 过滤器在两种替代检验器下均无法隔离假阳——与 P1 同机制。论文 Table 2 的"暂定待重跑"列按本表替换（P8 项 F1/F2），CCM 方向不可恢复性作为实质发现写入 §4.1/§5.4 一句。

### 7.6 P7 Table 5 相位（[scripts/37_phase_zzf.py](scripts/37_phase_zzf.py)，正式核对表见 [outputs/phase_zzf_vs_paper.md](outputs/phase_zzf_vs_paper.md)）

**phase-80 全量（80 记录 / 6 区域分配 data/region_assignment_80_paper.csv，NA 不除杂、
单次 0.5 ka 低通、200 draws，2026-08-19）**：论文 Table 5 的 21 行中 **10/21 达标一致
（≥95% 同向）**，其中 5 行近似/精确命中：

| 对 × 窗口 | 论文 % / lag | 实测 % / lag | 判定 |
| :--- | :--- | :--- | :--- |
| LGM 北大西洋→东亚季风 | 98.5 / +2.000 | **98.5 / +2.000** | ✅ 精确命中 |
| LGM 东亚季风→高纬湖泊 | 100.0 / +2.000 | **100.0 / +2.000** | ✅ 精确命中 |
| H1 南美→高纬湖泊 | 100.0 / +1.350 | **100.0 / +1.350** | ✅ 精确命中 |
| H1 南美→热带 | 98.0 / +2.000 | **97.0 / +1.775** | ✅ 近似命中 |
| H1 东亚季风→南美 | 99.5 / +1.950 | **100.0 / +1.650** | ✅ 近似命中 |
| B-A ASO→东亚季风 | 100.0 / +0.950 | **100.0 / +1.500** | ✅ ≥95% |
| B-A 东亚季风→高纬湖泊 | 100.0 / +1.050 | **100.0 / +1.500** | ✅ ≥95% |
| B-A 东亚季风→南美 | 100.0 / +1.050 | **100.0 / +1.350** | ✅ ≥95% |
| H1 ASO→南美 | 95.5 / +1.275 | **99.0 / +1.100** | ✅ ≥95% |
| H4 东亚季风→高纬湖泊 | 95.5 / +2.000 | **92.5 / +2.000** | ⚠️ 恰低于 95% |

系统性未中：**北大西洋涉行系统性低**（H1 东亚季风→北大西洋 83 vs 100、B-A 北大西洋→
东亚季风 42 vs 100、H1 北大西洋→高纬湖泊 80 vs 100）+ B-A 东亚季风→热带 67 vs 100 +
LGM 东亚季风→南美 54.5 vs 100。实测 ≥95 共 9 行（论文 21）+ 2 行论文没有（LGM ASO→
高纬湖泊 100、GI-12 ASO→南美 97.5）。**GS-12 措辞验证 ✓**：80/6 下 GS-12 相位最大
方向一致性 0.86 < 0.95（区域 Granger 层 max 47.0% 亦 <95%）——与论文"no direction
attains 95%"吻合。结构性结论：±2.0 ka 搜索边界饱和现象双方共有；倾向性集中于气候
事件窗的定性叙述成立。

**用户定夺（2026-08-19）：保留论文 21 行 + 重建一致性说明** → S6.1(iv) 已写入补充材料
（10/21、5 行命中、NA 涉行系统性低、GS-12 ✓）。滑窗部分（G3，旧 91/7 变体，survivor
29084，2026-08-19 落地）：10101 格（21 对 × 481 窗，2.0 ka/0.1 ka）中 842 格（8.3%）
≥95%、max 1.0、14/21 对与 344/481 窗至少一格 ≥95%——**不支持旧稿"滑窗无方向 ≥95%"句，
但该句在现版已删**（现版 §4.4 仅存 GS-12 窄窗句，80/6 下已验证 ✓），故无需改动；
数据存档 outputs/phase_zzf_sliding.csv（如需可再跑 80/6 滑窗变体）。

---

## 8. 正在运行 / 下一步

### 正在运行（后台任务）

（无。survivor 29084 已于 2026-08-19 落地，见下。）

### 2026-08-19 已完成（P8 论文同步，备份在 paper/.backup_2026-08-19/）

用户四项定夺：**A 保持 80/6、C Table 4/S5 保留+实现状态注、D §4.3 保留+实现状态注、
G Table 5 保留+重建一致性说明**。docx 已同步（outputs/_edit_docx_p8.py，--check 全过、
保存后逐项回读验证）：

- **M1** §4.2 "最大 45.0%"→"三对经典配对中最大 45.0%（Table S5 全矩阵最大 54.0 =
  NGRIP→Cariaco H1）"——消除 45.0/54.0 内部矛盾；
- **M2/M3** §3.4/§4.1 检验器措辞按 Table 2 实测改写（CCM 下全 <8.5%；PCMCI 单边 97.5%
  但控制地板 98.0%；"real records 三检验器均次阈值"改为实测可支撑表述）；
- **M4** Table 2 provisional 列 → 实测（CCM 4.0/0.5/0.0、PCMCI 41.0/42.5/97.5）+ 注改指
  仓库 scripts/39_ccm_pcmci_zzf.py + 控制上限 8.5%/98.0%；
- **M5** Fig. 2 框架图嵌入 docx + "Fig.2 |"→"Fig. 2 |"；
- **M6** §4.1 所选配置补"汇集 σ=√((σ₁²+σ₂²)/2)（Fig. S8）"短语——探针 bxt83q4qt 证实
  按记录 σ 变体（7.0/53.0/48.0、控制 46.5%）与 min-p 变体（100% 饱和）均不能再生
  Table 1，Table 1 = scripts/22 run22f（汇集 σ）输出可复现；
- **S1** 补充材料 S6 后新增 **S6.1 Implementation status of the reported numerical
  results**（i–iv）：(i) Table 1 汇集 σ 规范 + 两变体数值；(ii) Table 4/S5 三变体矩阵
  （χ²+噪声 −9pp / χ² 无噪声 −7pp / min-p +66pp，括号论文值无一再生）；(iii) §4.3/§5.3
  区域重建矛盾如实记录（6–9/180 ≥95% vs 论文 0/180；SAM–NA H1 基线 82.0–92.5% vs
  论文 7.5%；误差压缩 ×0.5 即过 95% vs 论文"永远不过"）；(iv) Table 5 重建 10/21、
  5 行命中、GS-12 措辞 ✓。
- **Tables/*.xlsx 重生成**（outputs/_export_tables_xlsx.py 从已编辑 docx 提取：Table 1–5
  + Table S1–S5 与论文逐格一致；陈旧 Table S6.xlsx（Scenario B）已删）。
- **P9 §1 lead-lag 段改写**（用户 2026-08-19 指示；outputs/_edit_docx_p9_leadlag.py，
  备份 paper/.backup_2026-08-19_leadlag/，回读验证 ✓）：引入部分收紧为两句——
  ①传统 lead-lag 分析基于 Granger causality tests（"are based on"，原为 "infer ...
  using"），②Granger 检验是什么（两句并一句：能否在自身过去值之外显著改善预测；
  显著改善 = leads/Granger-causes）；随后才是自回归/共同外部驱动/年代学误差导致
  假阳性的句子（原句保留）。段落后半（三属性展开）未动。

### 下一步

1. ✅ survivor 29084（旧 91/7 滑窗）落地（2026-08-19）：10101 格中 842 格（8.3%）≥95%、
   max 1.0。旧稿"滑窗无方向 ≥95%"句现版已删、无需改动；G3 已结案并记入 §7.6 末段。
2. 待用户确认：Tables/Table1–5.xlsx（无空格 5 件过期副本）是否删除（自动模式拒删
   保护目录内未被用户点名的文件；不影响任何脚本读取）。
3. 如需：提交当前未跟踪的工作成果（scripts/16–39、src/granger_fast.py 等）到 git
   （用户决策 ④：无明确要求不提交）。
