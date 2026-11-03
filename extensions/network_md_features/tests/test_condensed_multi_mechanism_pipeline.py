"""
test_condensed_multi_mechanism_pipeline.py
验证阶段 1 ~ 5 新增的凝聚态多机理自动交联簇建模、统一 AM1-BCC 参数化、孤立气相单分子参考态 CED、
多随机种子复现性 ANOVA 报告以及 15 个真实文献配方凝聚态试点与 Group-CV 消融产物。
"""

import json
import pytest
import pandas as pd
from pathlib import Path

EXT_ROOT = Path("/home/wangj/project/CFRP/extensions/network_md_features")


def test_stage1_condensed_dgeba_dds_1524_benchmark_verified():
    """验证阶段 1：1524 原子 DGEBA/DDS 凝聚态基准体系三态全部达到真实凝聚态密度且通过门禁"""
    diag_csv = (
        EXT_ROOT
        / "artifacts"
        / "condensed_benchmark_stage1"
        / "postprocess_condensed_dgeba_dds_1524"
        / "md_features_diagnostic.csv"
    )
    assert diag_csv.exists()
    df = pd.read_csv(diag_csv)
    assert len(df) == 3
    assert (df["atom_count"] == 1524).all()
    assert (df["validation_status"] == "CONDENSED_PHASE_VERIFIED").all()
    assert (df["admitted_to_formal_ml_training"] == 1).all()
    assert (df["md_density_g_cm3"] >= 1.05).all() and (df["md_density_g_cm3"] <= 1.25).all()
    assert (df["md_ffv_probe_0A"] >= 0.30).all() and (df["md_ffv_probe_0A"] <= 0.45).all()
    assert (df["md_cohesive_energy_density_CED_MPa"] >= 300.0).all()
    assert (df["md_hbond_intermolecular_average"] >= 20.0).all()
    assert (df["trajectory_max_bond_strain_dev_A"] <= 0.25).all()


def test_stage2_and_3_five_mechanisms_and_seed_reproducibility():
    """验证阶段 2 与阶段 3：5 大固化机理（含酸酐与环氧均聚自固化）× 3 个随机种子 = 15 次凝聚态模拟全部通过门禁"""
    s23_dir = EXT_ROOT / "artifacts" / "stage2_3_multi_mechanism_and_seeds"
    df = pd.read_csv(s23_dir / "stage2_3_multi_mechanism_seed_reproducibility.csv")
    rep = json.loads((s23_dir / "stage2_3_anova_and_reproducibility_report.json").read_text(encoding="utf-8"))

    assert len(df) == 15
    assert set(df["mechanism_type"].unique()) == {"amine", "anhydride", "phenol", "thiol", "homopolymerization"}
    assert (df["validation_status"] == "CONDENSED_PHASE_VERIFIED").all()
    assert (df["trajectory_max_bond_strain_dev_A"] <= 0.25).all()
    assert rep["all_runs_passed_condensed_gates"] is True
    assert rep["anova_signal_to_noise_summary"]["md_coulomb_energy_density_MJ_m3"]["statistically_significant_p_lt_0_01"] is True
    assert rep["anova_signal_to_noise_summary"]["md_hbond_count_average"]["statistically_significant_p_lt_0_01"] is True


def test_stage4_and_5_literature_pilot_and_group_cv_ablation():
    """验证阶段 4 与阶段 5：15 个分层文献配方凝聚态试点全部通过门禁并生成准入特征表与 Group-CV 消融报告"""
    s4_dir = EXT_ROOT / "artifacts" / "stage4_pilot_condensed_md"
    df_adm = pd.read_parquet(s4_dir / "md_features_admitted_pilot.parquet")
    assert len(df_adm) == 15
    assert (df_adm["validation_status"] == "CONDENSED_PHASE_VERIFIED").all()
    assert (df_adm["admitted_to_formal_ml_training"] == 1).all()
    assert (df_adm["md_density_g_cm3"] >= 0.98).all() and (df_adm["md_density_g_cm3"] <= 1.25).all()
    assert (df_adm["trajectory_max_bond_strain_dev_A"] <= 0.25).all()
    assert "md_hbond_inter_density_nm3" in df_adm.columns


def test_stoich_cure_coupled_md_and_ml_optimization_v3():
    """验证化学计量比 r 与固化温度 T_cure 动态耦合的凝聚态 MD 及全数据集模型优化结果"""
    opt_json = (
        EXT_ROOT
        / "artifacts"
        / "coupled_md_ml_optimization_v3"
        / "coupled_md_ml_optimization_results.json"
    )
    assert opt_json.exists()
    res = json.loads(opt_json.read_text(encoding="utf-8"))
    assert res["real_gromacs_pme_condensed_runs_count"] >= 44
    assert len(res["coupled_md_features_generated"]) == 16

    # 验证四个数据集在 Upgraded_Ensemble_Plus_Coupled_MD 下训练集 R² 均 >= 0.93，且测试集 Random/Group 5F R² 均优于旧基线
    for task_key in [
        "Tg_644_Clean_Subset",
        "Tg_2229_Full_Dataset",
        "Storage_Modulus_546_Full_Structure_Dataset",
        "Storage_Modulus_566_Benchmark_Dataset",
    ]:
        t_res = res[task_key]
        leg = t_res["Legacy_Baseline_RF_BaseOnly"]
        base_ens = t_res["Upgraded_Ensemble_BaseOnly"]
        md_ens = t_res["Upgraded_Ensemble_Plus_Coupled_MD"]
        assert md_ens["Train_R2_mean"] >= 0.93
        assert md_ens["Random_5Fold_OOF_R2"] > leg["Random_5Fold_OOF_R2"]
        assert md_ens["Random_5Fold_OOF_R2"] > base_ens["Random_5Fold_OOF_R2"]
        assert md_ens["Group_5Fold_OOF_R2"] > leg["Group_5Fold_OOF_R2"]
        assert md_ens["Group_5Fold_OOF_R2"] > base_ens["Group_5Fold_OOF_R2"]


def test_condensed_md_pinn_and_hybrid_pinm_v4():
    """验证凝聚态 MD 物理场深度融合的 GPU PINN 与 Hybrid PINN-PINM 在 5 大数据集上的训练与交叉验证表现"""
    pinn_json = (
        EXT_ROOT
        / "artifacts"
        / "condensed_md_pinn_optimization_v4"
        / "condensed_md_pinn_optimization_results.json"
    )
    assert pinn_json.exists()
    res = json.loads(pinn_json.read_text(encoding="utf-8"))

    for task_key in [
        "Tg_644_Clean_Subset",
        "Tg_2229_Full_Dataset",
        "Storage_Modulus_546_Full_Structure_Dataset",
        "Storage_Modulus_566_Benchmark_Dataset",
        "Storage_Modulus_566_Clean_Deduplicated",
        "Storage_Modulus_706_PINN_Benchmark_Dataset",
        "Storage_Modulus_706_Physically_Consistent",
    ]:
        assert task_key in res
        t_res = res[task_key]
        leg = t_res["Legacy_Baseline_RF_BaseOnly"]
        orig_pinn = t_res["Original_EpoxyPINN_BaseOnly"]
        orig_pinm = t_res["Original_EpoxyPINM_BaseOnly"]
        cond_pinn = t_res["Condensed_MD_Kernel_PINN_GPU"]
        hyb = t_res["Condensed_MD_Hybrid_PINN_PINM"]

        # 1. 纯神经网络 GPU Condensed-MD PINN 在随机 5 折与分组 5 折上全面超越原版 EpoxyPINNRegressor
        assert cond_pinn["Random_5Fold_OOF_R2"] > orig_pinn["Random_5Fold_OOF_R2"]
        assert cond_pinn["Group_5Fold_OOF_R2"] > orig_pinn["Group_5Fold_OOF_R2"]

        # 2. Hybrid PINN-PINM 训练集 R² 全部 >= 0.968，且在随机 5 折与分组 5 折上全面超越 Legacy RF 与 Original PINN
        assert hyb["Train_R2_mean"] >= 0.968
        assert hyb["Random_5Fold_OOF_R2"] > leg["Random_5Fold_OOF_R2"]
        assert hyb["Random_5Fold_OOF_R2"] > orig_pinn["Random_5Fold_OOF_R2"]
        assert hyb["Random_5Fold_OOF_R2"] > orig_pinm["Random_5Fold_OOF_R2"]
        assert hyb["Group_5Fold_OOF_R2"] > leg["Group_5Fold_OOF_R2"]
        assert hyb["Group_5Fold_OOF_R2"] > orig_pinn["Group_5Fold_OOF_R2"]


def test_pure_formulation_process_md_566_and_706_breakthrough():
    """验证仅输入【配方 + 工艺 + 凝聚态 MD 物理特征】、剥离杂乱向量后在 566 去重模量集、706 模量集和 2,229 Tg 集上的突破"""
    import pandas as pd

    clean_566_csv = EXT_ROOT / "artifacts" / "clean_566_formulation_process_physics.csv"
    clean_tg2229_csv = EXT_ROOT / "artifacts" / "clean_tg2229_formulation_process_physics.csv"
    assert clean_566_csv.exists()
    assert clean_tg2229_csv.exists()

    df_566 = pd.read_csv(clean_566_csv, nrows=5)
    df_tg = pd.read_csv(clean_tg2229_csv, nrows=5)
    for df_chk in (df_566, df_tg):
        bad_cols = [
            c for c in df_chk.columns
            if "maccs" in c.lower() or "morgan" in c.lower() or "_polymer_" in c.lower() or "_bigsmiles_" in c.lower()
        ]
        assert len(bad_cols) == 0

    pinn_json = (
        EXT_ROOT
        / "artifacts"
        / "condensed_md_pinn_optimization_v4"
        / "condensed_md_pinn_optimization_results.json"
    )
    res = json.loads(pinn_json.read_text(encoding="utf-8"))

    m566_clean = res["Storage_Modulus_566_Clean_Deduplicated"]["Condensed_MD_Hybrid_PINN_PINM"]
    assert m566_clean["feature_count"] <= 105
    assert m566_clean["Random_5Fold_OOF_R2"] >= 0.705
    assert m566_clean["Random_5Fold_OOF_MAE"] <= 0.350

    m706_clean = res["Storage_Modulus_706_Physically_Consistent"]["Condensed_MD_Hybrid_PINN_PINM"]
    assert m706_clean["feature_count"] <= 130
    assert m706_clean["Random_5Fold_OOF_R2"] >= 0.705
    assert m706_clean["Group_5Fold_OOF_R2"] >= 0.685

    tg2229_pure = res["Tg_2229_Full_Dataset"]["Condensed_MD_Hybrid_PINN_PINM"]
    assert tg2229_pure["feature_count"] <= 115
    assert tg2229_pure["Random_5Fold_OOF_R2"] >= 0.815
    assert tg2229_pure["Group_5Fold_OOF_R2"] >= 0.705
