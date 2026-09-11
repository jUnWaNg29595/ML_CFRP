# ⚗️ 环氧高分子机理与动力学特征增强工具 (Epoxy QSPR Enricher)

本工具专门针对 `ml_qspr_model_*.csv` 系列高分子配方与性能数据集，一键补齐三维交联网络、理论交联密度、化学计量偏离度以及多组分量子化学前线轨道动力学（$\Delta E$）特征，有效解决传统机器学习仅看孤立单体时“按配方划分跨域泛化能力暴跌”的痛点。

---

## 📁 1. 需要准备的数据文件

| 文件类型 | 推荐文件名 | 是否必须 | 说明 |
| :--- | :--- | :---: | :--- |
| **主训练数据文件** | `ml_qspr_model_tg_c.csv` | **必须** | 包含 `resin_1_structure`、`curing_agent_1_structure`、`formulation_r_value`、`tg_c` 等列的 QSPR 数据表。支持任意以 `ml_qspr_model_*.csv` 命名的目标性质表。 |
| **底层大宽表（补充）** | `ml_wide_samples.csv` | **可选（推荐）** | 与主文件放在同一目录下即可自动识别。用于为多组分样本补充精准的各组分 PHR 分数（如 `resin_2_amount_phr`）与精确当量。若不存在则自动采用自适应估算。 |

---

## 🚀 2. 运行方法

在 Anaconda 命令行（或激活 `CFRP_env` 环境的终端）中直接运行：

```bash
# 激活环境
conda activate CFRP_env

# 进入项目目录并运行（默认自动处理 ml_qspr_model_tg_c.csv）
python tools/epoxy_qspr_enricher/enrich_qspr_dataset.py

# 或者手动指定输入和输出路径：
python tools/epoxy_qspr_enricher/enrich_qspr_dataset.py \
    --input "C:\Users\wangj\Desktop\ml_dataset\ml_qspr_model_tg_c.csv" \
    --output "C:\Users\wangj\Desktop\ml_dataset\ml_qspr_model_tg_c_enhanced.csv"
```

处理完成后，将在目标路径直接生成增强版表格文件（如 `ml_qspr_model_tg_c_enhanced.csv`），无需等待耗时的 DFT/xTB 计算，数秒内即可完成全量 7400+ 样本的特征衍生！

---

## 📊 3. 新增的 18 个高分子物理与动力学特征解析

导出的表格会在原有列的基础上，自动追加以下 18 列特征（统一以前缀 `mech_` 标示）：

### A. 反应配比与交联缺陷特征
1. `mech_stoichiometry_r`：实际当量比 $r$（活性氢/环氧基当量比）。
2. `mech_stoich_deviation`：**对称偏离度 $|r - 1.0|$**。高分子交联网络缺陷度的核心表达，$r=1.0$ 时网络最致密，偏离 1.0 时产生大量未反应悬挂链缺陷。
3. `mech_stoich_log_ratio`：对数计量比 $\ln(r)$，区分环氧过量（负）还是固化剂过量（正）。
4. `mech_theoretical_alpha_max`：基于计量比的理论极限转化率 $\min(1.0, r, 1/r)$。
5. `mech_theoretical_alpha_gel`：**Flory-Stockmayer 凝胶点转化率 $\alpha_{\text{gel}}$**，反映体系形成宏观凝胶网络的难易程度。

### B. 高分子物理三维交联网络特征
6. `mech_weighted_epoxy_func`：树脂加权平均环氧官能度 $f_R$。
7. `mech_weighted_curer_func`：固化剂加权平均活性氢官能度 $f_H$。
8. `mech_average_functionality`：反应物体系平均分子官能度 $f_{\text{avg}}$。
9. `mech_theoretical_Mc`：**理论交联点间分子量 $M_c$**（$M_c = \frac{\overline{M}_{\text{formula}}}{f_{\text{avg}} - 2}$），高分子玻璃化转变 $T_g$ 的经典物理本质参数。
10. `mech_crosslink_density_proxy`：**理论交联密度代理值 $\rho_c = \frac{1000}{M_c}$**。与 $T_g$ 呈极强正相关。

### C. 多组分前线轨道能差动力学矩阵 ($\Delta E$)
11. `mech_delta_E_min`：**先发反应通道**（$\min_{i,j} |E_{\text{LUMO}}(R_i) - E_{\text{HOMO}}(H_j)|$）。体系中反应活性最高的那对组分，决定了起始反应温度和凝胶引发。
12. `mech_delta_E_max`：**最钝反应通道**（$\max_{i,j} |E_{\text{LUMO}}(R_i) - E_{\text{HOMO}}(H_j)|$）。体系中最难反应的组分，决定深层固化残余未反应缺陷风险。
13. `mech_delta_E_span`：**动力学色散度**（$\Delta E_{\text{max}} - \Delta E_{\text{min}}$）。反映体系分步固化与微观相分离倾向。
14. `mech_delta_E_weighted`：体系按配方加权的宏观平均反应活化能垒。

### D. 配方加权物理混合物特征
15. `mech_weighted_formula_mw`：配方平均单元分子量。
16. `mech_weighted_tpsa`：配方加权拓扑极性表面积（反映极性与氢键相互作用）。
17. `mech_weighted_aromatic_rings`：配方加权芳香环密度（反映链段刚性）。
18. `mech_weighted_rotatable_bonds`：配方加权可旋转键数（反映分子链柔顺度）。

---

## 💡 4. 后续建模建议
* **样本筛选**：根据用户偏好，可自行在训练前筛选 `tg_c > 0` 排除历史异常值。
* **模型推荐**：XGBoost / CatBoost / LightGBM 在引入上述带有明确物理极值点（如 `|r - 1.0|`、`rho_c`、`delta_E_min`）的特征后，在按配方分组（GroupKFold / Formulation Split）的外推测试集上，$R^2$ 通常会有显著改善。
