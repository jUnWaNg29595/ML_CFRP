"""
run_stage4_stage5_pilot_and_group_cv_ablation.py
执行总体计划的阶段四（代表性文献配方凝聚态 MD 试点 Pilot Run）与阶段五（严格按配方分组 Group CV 消融实验）：
1. 从 data_fixed/latest_data_clean_594.csv 中分层抽取覆盖 6 类固化机理（amine, anhydride, phenol, carboxyl, thiol, homopolymerization）
   且跨越 25 ℃ ~ 300 ℃ 宽广 Tg 梯度的真实文献配方；
2. 严格零标签泄漏（Zero Data Leakage）：仅使用部署时可获得的树脂与固化剂 SMILES 构建凝聚态交联团簇，
   运行 AM1-BCC 参数化、孤立气相单分子参考态标定、GROMACS NPT 压密 + PME 采样与 postprocess_trajectory.py 提取；
3. 将全部通过 CONDENSED_PHASE_VERIFIED 门禁的样本导出为 md_features_admitted_pilot.parquet 与 .csv；
4. 在严格 Formulation Group 5-Fold 与 Leave-One-Formulation-Out (LOFO) 下对比：
   - Baseline_2D_Mech (仅原系统 2D RDKit + 配方机理特征)
   - Baseline_Plus_Static3D (原系统特征 + RDKit 静态 3D 构象描述符)
   - Condensed_MD_Augmented (原系统特征 + 真实凝聚态 MD 特征)
   并输出完整消融统计报告。
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.*")

CFRP_ROOT = Path("/home/wangj/project/CFRP")
EXT_ROOT = CFRP_ROOT / "extensions" / "network_md_features"
for p in [str(CFRP_ROOT), str(EXT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core.epoxy_mechanism_features import EpoxyMechanismEngine
from core.reaction_core import inspect_curer_functionality
from src.condensed_phase_multi_mechanism_engine import (
    generate_multi_mechanism_reaction_adducts,
    parameterize_species_am1bcc_with_gas_ref,
    build_and_run_condensed_state,
)
from src.postprocess_trajectory import run_postprocessing


def compute_static_3d_descriptors(smiles: str) -> Dict[str, float]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    cid = AllChem.EmbedMolecule(mol, params)
    if cid == -1:
        params.useRandomCoords = True
        cid = AllChem.EmbedMolecule(mol, params)
    if cid == -1:
        return {"static3d_npr1": 0.5, "static3d_npr2": 0.5, "static3d_spherocity": 0.5, "static3d_rg": 5.0}
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    return {
        "static3d_npr1": float(rdMolDescriptors.CalcNPR1(mol)),
        "static3d_npr2": float(rdMolDescriptors.CalcNPR2(mol)),
        "static3d_spherocity": float(rdMolDescriptors.CalcSpherocityIndex(mol)),
        "static3d_rg": float(rdMolDescriptors.CalcRadiusOfGyration(mol)),
    }


def select_stratified_pilot_formulations(df_tg: pd.DataFrame) -> pd.DataFrame:
    eng = EpoxyMechanismEngine()
    allowed_elems = {"H", "C", "N", "O", "S", "F", "Cl"}
    candidates = []

    for idx, r in df_tg.iterrows():
        tg_val = float(r["tg_c"])
        if tg_val < 20.0 or tg_val > 310.0:
            continue
        mr = eng.parse_molecule_safe(r.get("resin_1_structure", ""))
        if mr is None or mr.GetNumHeavyAtoms() > 28 or mr.GetNumHeavyAtoms() < 8:
            continue
        if not set(a.GetSymbol() for a in mr.GetAtoms()).issubset(allowed_elems):
            continue
        rs_can = Chem.MolToSmiles(mr, canonical=True)

        raw_h = r.get("curing_agent_1_structure", "")
        if pd.isna(raw_h) or not str(raw_h).strip():
            hs_can = ""
            mech = "homopolymerization"
        else:
            mh = eng.parse_molecule_safe(raw_h)
            if mh is None or mh.GetNumHeavyAtoms() > 22 or mh.GetNumHeavyAtoms() < 4:
                continue
            if not set(a.GetSymbol() for a in mh.GetAtoms()).issubset(allowed_elems):
                continue
            hs_can = Chem.MolToSmiles(mh, canonical=True)
            c_info = inspect_curer_functionality(hs_can)
            mech = c_info.mechanism if c_info else "unknown"

        if mech not in ("amine", "anhydride", "phenol", "thiol", "homopolymerization", "carboxyl"):
            continue
        try:
            rxn = generate_multi_mechanism_reaction_adducts(rs_can, hs_can if hs_can else None, mech)
            m_ad2 = Chem.AddHs(Chem.MolFromSmiles(rxn["adduct_2_smiles"]))
            if 55 <= m_ad2.GetNumAtoms() <= 145:
                candidates.append({
                    "row_idx": int(idx),
                    "r_smi": rs_can,
                    "h_smi": hs_can,
                    "mech": mech,
                    "tg_c": tg_val,
                    "ad2_atoms": m_ad2.GetNumAtoms(),
                })
        except Exception:
            continue

    df_c = pd.DataFrame(candidates).drop_duplicates(subset=["r_smi", "h_smi"])

    selected = []
    for m in ["thiol", "carboxyl", "homopolymerization", "phenol"]:
        sub = df_c[df_c["mech"] == m].sort_values("tg_c")
        selected.append(sub.head(2))

    sub_anh = df_c[df_c["mech"] == "anhydride"].sort_values("tg_c")
    if len(sub_anh) >= 3:
        selected.append(sub_anh.iloc[[0, len(sub_anh) // 2, -1]])

    sub_am = df_c[df_c["mech"] == "amine"].sort_values("tg_c")
    if len(sub_am) >= 6:
        idxs = [int(round(i * (len(sub_am) - 1) / 5)) for i in range(6)]
        selected.append(sub_am.iloc[idxs])

    df_sel = pd.concat(selected).drop_duplicates(subset=["row_idx"]).sort_values("tg_c").reset_index(drop=True)
    return df_sel


def main():
    stage4_root = EXT_ROOT / "artifacts" / "stage4_pilot_condensed_md"
    tpl_root = stage4_root / "species_library"
    attempt_dir = stage4_root / "attempt_pilot_literature"
    post_dir = stage4_root / "postprocess_pilot_literature"
    stage4_root.mkdir(parents=True, exist_ok=True)

    df_tg = pd.read_csv(CFRP_ROOT / "data_fixed" / "latest_data_clean_594.csv", low_memory=False)
    df_pilot_spec = select_stratified_pilot_formulations(df_tg)
    print(f"Selected {len(df_pilot_spec)} stratified literature formulations for Stage 4 Pilot Run:")
    print(df_pilot_spec[["row_idx", "mech", "tg_c", "ad2_atoms", "r_smi", "h_smi"]].to_string(), flush=True)

    pilot_records: List[Dict[str, Any]] = []
    static_3d_rows: List[Dict[str, Any]] = []

    for p_idx, row in df_pilot_spec.iterrows():
        r_idx = int(row["row_idx"])
        r_smi = str(row["r_smi"])
        h_smi = str(row["h_smi"]) if str(row["h_smi"]) else None
        mech = str(row["mech"])
        fid = f"PILOT_{p_idx:02d}_ROW{r_idx}_{mech.upper()}"

        t0 = time.time()
        rxn = generate_multi_mechanism_reaction_adducts(r_smi, h_smi, mech)
        code_prefix = f"{chr(65 + int(p_idx))}0"
        sp_r = parameterize_species_am1bcc_with_gas_ref(rxn["resin_smiles"], f"{code_prefix}R", str(tpl_root))
        sp_h = parameterize_species_am1bcc_with_gas_ref(rxn["hardener_smiles"], f"{code_prefix}H", str(tpl_root))
        sp_a1 = parameterize_species_am1bcc_with_gas_ref(rxn["adduct_1_smiles"], f"{code_prefix}A", str(tpl_root))
        sp_a2 = parameterize_species_am1bcc_with_gas_ref(rxn["adduct_2_smiles"], f"{code_prefix}B", str(tpl_root))

        # 动态配平分子个数使总原子数在 850 ~ 1250 之间（满足 >= 500 原子代表性体元门禁且高效）
        comp = [(sp_a2, 4), (sp_a1, 3), (sp_r, 2), (sp_h, 2)]
        tot_atoms = sum(sp["atom_count"] * c for sp, c in comp)
        if tot_atoms < 1050:
            scale_m = int(np.ceil(1050.0 / max(1, tot_atoms)))
            comp = [(sp, c * scale_m) for sp, c in comp]
            tot_atoms = sum(sp["atom_count"] * c for sp, c in comp)

        init_epoxy_sites = (comp[0][1] * 2 + comp[1][1] * 1 + comp[2][1]) * 2

        single_attempt = stage4_root / f"attempt_{fid}"
        single_post = stage4_root / f"postprocess_{fid}"

        build_and_run_condensed_state(
            state_dir=str(single_attempt / "state_0_addition"),
            state_index=0,
            state_name=fid,
            species_Composition=comp,
            initial_epoxy_sites=init_epoxy_sites,
            seed=20261019 + p_idx,
            nt_threads=8,
            compress_steps=10000,
            anneal_steps=8000,
            prod_steps=8000,
            mechanism_type=mech,
            formulation_id=fid,
        )

        res_post = run_postprocessing(
            input_attempt_dir=str(single_attempt),
            output_dir=str(single_post),
            stride=1,
            grid_spacing_nm=0.10,
        )
        st_row = res_post["df_summary"].iloc[0].to_dict()
        st_row["literature_row_idx"] = r_idx
        st_row["formulation_id"] = fid
        st_row["mechanism_type"] = mech
        st_row["resin_smiles_canonical"] = rxn["resin_smiles"]
        st_row["hardener_smiles_canonical"] = rxn["hardener_smiles"]
        st_row["adduct_2_smiles"] = rxn["adduct_2_smiles"]
        st_row["literature_tg_c"] = float(row["tg_c"])
        pilot_records.append(st_row)

        s3d = compute_static_3d_descriptors(rxn["adduct_2_smiles"])
        s3d["literature_row_idx"] = r_idx
        static_3d_rows.append(s3d)

        print(
            f"[{p_idx+1:2d}/{len(df_pilot_spec)}] {fid:28s} | Tg={row['tg_c']:6.1f} C | "
            f"status={st_row['validation_status']} | dens={st_row['md_density_g_cm3']:.4f} | "
            f"FFV0={st_row['md_ffv_probe_0A']:.4f} | CED={st_row['md_cohesive_energy_density_CED_MPa']} MPa | "
            f"HB={st_row['md_hbond_count_average']:.2f} | time={time.time()-t0:.1f}s",
            flush=True,
        )

    df_pilot_all = pd.DataFrame(pilot_records)
    df_pilot_all.to_csv(stage4_root / "md_features_pilot_all.csv", index=False)

    # 仅准入通过 CONDENSED_PHASE_VERIFIED 的样本进入正式试点特征库
    df_admitted = df_pilot_all[df_pilot_all["admitted_to_formal_ml_training"] == 1].copy().reset_index(drop=True)
    admitted_parquet = stage4_root / "md_features_admitted_pilot.parquet"
    admitted_csv = stage4_root / "md_features_admitted_pilot.csv"
    df_admitted.to_parquet(admitted_parquet, index=False)
    df_admitted.to_csv(admitted_csv, index=False)
    print(f"\nAdmitted {len(df_admitted)}/{len(df_pilot_all)} pilot formulations -> {admitted_parquet}")

    # ==========================================================================
    # 阶段五：在准入文献样本上执行严格 Group 5-Fold 与 LOFO 消融实验
    # ==========================================================================
    df_s3d = pd.DataFrame(static_3d_rows)
    admitted_indices = df_admitted["literature_row_idx"].tolist()
    df_lit_sub = df_tg.iloc[admitted_indices].copy().reset_index(drop=True)
    df_lit_sub["literature_row_idx"] = admitted_indices

    # 选择文献基线表中的核心 2D 与机理特征（排除目标、ID 与全空列）
    base_candidate_cols = [
        "formulation_resin_hardener_equivalent_ratio",
        "mech_stoich_log_ratio",
        "stoich_defect_ratio",
        "rigid_carbon_ratio",
        "tg_c_heating_rate_c_min",
        "resin_mw",
        "hardener_mw",
        "resin_aromatic_rings",
        "hardener_aromatic_rings",
        "resin_tpsa",
        "hardener_tpsa",
        "resin_h_donors",
        "hardener_h_donors",
        "resin_rotatable_bonds",
        "hardener_rotatable_bonds",
    ]
    numeric_all = df_lit_sub.select_dtypes(include=[np.number]).columns.tolist()
    exclude_set = {"tg_c", "literature_row_idx", "row_index", "id", "sample_id"}
    avail_base_cols = [c for c in base_candidate_cols if c in numeric_all]
    if len(avail_base_cols) < 8:
        # 补充方差最大的前 12 个合法数值特征
        extra = [
            c for c in numeric_all
            if c not in exclude_set and c not in avail_base_cols and df_lit_sub[c].notna().sum() >= len(df_lit_sub) - 2
        ]
        avail_base_cols.extend(extra[: (12 - len(avail_base_cols))])

    md_feature_cols = [
        "md_density_g_cm3",
        "md_ffv_probe_0A",
        "md_ffv_probe_1_2A",
        "md_vdw_energy_density_MJ_m3",
        "md_coulomb_energy_density_MJ_m3",
        "md_cohesive_energy_density_CED_MPa",
        "md_hbond_count_average",
        "md_hbond_intermolecular_average",
    ]
    s3d_cols = ["static3d_npr1", "static3d_npr2", "static3d_spherocity", "static3d_rg"]

    df_merged = (
        df_lit_sub[["literature_row_idx", "tg_c"] + avail_base_cols]
        .merge(df_s3d, on="literature_row_idx", how="inner")
        .merge(df_admitted[["literature_row_idx", "formulation_id", "mechanism_type"] + md_feature_cols], on="literature_row_idx", how="inner")
    )

    # 1. 计算每个凝聚态 MD 特征与文献实测 Tg 的物理相关性
    correlations = {}
    y_tg = df_merged["tg_c"].to_numpy(dtype=float)
    for col in md_feature_cols:
        x_val = df_merged[col].to_numpy(dtype=float)
        r_p, p_p = stats.pearsonr(x_val, y_tg)
        r_s, p_s = stats.spearmanr(x_val, y_tg)
        correlations[col] = {
            "pearson_r_with_tg": round(float(r_p), 4),
            "pearson_p_value": round(float(p_p), 4),
            "spearman_rho_with_tg": round(float(r_s), 4),
            "spearman_p_value": round(float(p_s), 4),
        }

    # 2. Group 5-Fold & Leave-One-Formulation-Out 消融对比
    feature_sets = {
        "Baseline_2D_Mech": avail_base_cols,
        "Baseline_Plus_Static3D": avail_base_cols + s3d_cols,
        "Condensed_MD_Only": md_feature_cols,
        "Baseline_Plus_Condensed_MD": avail_base_cols + md_feature_cols,
    }

    models_dict = {
        "Ridge": lambda: Ridge(alpha=5.0, random_state=42),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=60, max_depth=4, min_samples_leaf=2, random_state=42),
        "GradientBoosting": lambda: GradientBoostingRegressor(n_estimators=40, max_depth=2, learning_rate=0.08, random_state=42),
    }

    groups = df_merged["formulation_id"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    lofo = LeaveOneGroupOut()

    ablation_results: Dict[str, Any] = {"Group_5Fold": {}, "Leave_One_Formulation_Out": {}}

    for cv_name, splitter in [("Group_5Fold", gkf), ("Leave_One_Formulation_Out", lofo)]:
        ablation_results[cv_name] = {}
        for fset_name, f_cols in feature_sets.items():
            ablation_results[cv_name][fset_name] = {}
            X_raw = df_merged[f_cols].copy()
            for m_name, m_factory in models_dict.items():
                y_true_list = []
                y_pred_list = []
                for tr_idx, te_idx in splitter.split(X_raw, y_tg, groups):
                    X_tr = X_raw.iloc[tr_idx].copy()
                    X_te = X_raw.iloc[te_idx].copy()
                    y_tr = y_tg[tr_idx]
                    y_te = y_tg[te_idx]

                    med = X_tr.median().fillna(0.0)
                    X_tr = X_tr.fillna(med)
                    X_te = X_te.fillna(med)

                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_te_s = scaler.transform(X_te)

                    model = m_factory()
                    model.fit(X_tr_s, y_tr)
                    pred = model.predict(X_te_s)

                    y_true_list.extend(y_te.tolist())
                    y_pred_list.extend(pred.tolist())

                yt = np.array(y_true_list, dtype=float)
                yp = np.array(y_pred_list, dtype=float)
                mae = float(mean_absolute_error(yt, yp))
                rmse = float(np.sqrt(mean_squared_error(yt, yp)))
                r2 = float(r2_score(yt, yp))
                ablation_results[cv_name][fset_name][m_name] = {
                    "feature_count": len(f_cols),
                    "OOF_MAE_C": round(mae, 3),
                    "OOF_RMSE_C": round(rmse, 3),
                    "OOF_R2": round(r2, 4),
                }

    report_payload = {
        "stage_4_pilot_admitted_count": len(df_admitted),
        "stage_4_pilot_total_attempted": len(df_pilot_all),
        "mechanisms_covered": sorted(df_admitted["mechanism_type"].unique().tolist()),
        "tg_range_C": [float(df_admitted["literature_tg_c"].min()), float(df_admitted["literature_tg_c"].max())],
        "physical_correlations_with_literature_tg": correlations,
        "stage_5_group_cv_ablation_metrics": ablation_results,
    }
    report_json = stage4_root / "stage4_5_pilot_and_ablation_report.json"
    report_json.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ PHYSICAL CORRELATIONS WITH LITERATURE Tg ================")
    print(pd.DataFrame(correlations).T.to_string())
    print("\n================ STAGE 5 GROUP 5-FOLD ABLATION RESULTS ================")
    print(json.dumps(ablation_results["Group_5Fold"], indent=2))
    print("\n================ STAGE 5 LOFO ABLATION RESULTS ================")
    print(json.dumps(ablation_results["Leave_One_Formulation_Out"], indent=2))


if __name__ == "__main__":
    main()
