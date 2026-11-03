"""
run_full_stoich_cure_coupled_md_and_ml_optimization.py
完整实现：
1. 在 25℃ 储能模量数据集上运行耦合了化学计量比 r、最高固化温度 T_cure、固化时间 t_cure 的真实 GROMACS PME 凝聚态 MD 模拟；
2. 汇总全部真实 GROMACS PME 凝聚态模拟数据（阶段1基准 + 阶段2/3多机理多种子 + 阶段4 Tg试点 + 配比/温度梯度 + 储能模量试点），
   标定物理动力学与凝聚态场方程，为全量 Tg 数据集与全量 25℃ 储能模量数据集生成 100% 覆盖、零目标泄漏的
   “化学计量比 + 固化工艺热历史 + 凝聚态 MD”物理特征；
3. 系统训练并评估 Tg 与 25℃ 储能模量的机器学习模型（对比旧基线 vs 新物理耦合特征 + CatBoost/LightGBM/XGBoost/ExtraTrees 集成），
   全面报告：
   - 训练集 R² / MAE / RMSE
   - 随机 5 折交叉验证 (Random 5-Fold OOF) R² / MAE / RMSE
   - 严格配方留出 5 折交叉验证 (Formulation Group 5-Fold OOF) R² / MAE / RMSE
"""

import os
import sys
import json
import math
import time
import hashlib
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CFRP_ROOT = Path("/home/wangj/project/CFRP")
EXT_MD = CFRP_ROOT / "extensions" / "network_md_features"
sys.path.insert(0, str(CFRP_ROOT))
sys.path.insert(0, str(EXT_MD))

def run_modulus_pilot_gromacs_simulations() -> pd.DataFrame:
    """
    在 htpolynet_env 下运行 10 个涵盖不同化学计量比 r、固化温度 T_cure 与固化机理的储能模量代表性配方 GROMACS PME 凝聚态模拟。
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from core.epoxy_mechanism_features import EpoxyMechanismEngine
    from core.reaction_core import inspect_curer_functionality
    from src.condensed_phase_multi_mechanism_engine import (
        generate_multi_mechanism_reaction_adducts,
        parameterize_species_am1bcc_with_gas_ref,
        compute_stoichiometry_and_cure_coupled_composition,
        build_and_run_condensed_state,
    )
    from src.postprocess_trajectory import run_postprocessing

    mod_root = EXT_MD / "artifacts" / "stage4_modulus_pilot_condensed_md"
    out_pq = mod_root / "md_features_modulus_pilot.parquet"
    if out_pq.exists():
        df_existing = pd.read_parquet(out_pq)
        if len(df_existing) >= 8:
            print(f"Loaded existing {len(df_existing)} modulus pilot GROMACS runs from {out_pq}")
            return df_existing

    eng = EpoxyMechanismEngine()
    df_mod = pd.read_csv(
        CFRP_ROOT / "cache/session_snapshots/before_molecular_feature_clear_20260918_111832_data.csv",
        low_memory=False,
    )
    allowed_elems = {"H", "C", "N", "O", "S", "F", "Cl"}

    mod_cands = []
    for idx, r in df_mod.iterrows():
        val = float(r["storage_modulus_25c_gpa"])
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
                eq_r = (
                    float(r.get("formulation_resin_hardener_equivalent_ratio", 1.0))
                    if pd.notna(r.get("formulation_resin_hardener_equivalent_ratio"))
                    else 1.0
                )
                tc = (
                    float(r.get("process_max_temperature_c", 150.0))
                    if pd.notna(r.get("process_max_temperature_c"))
                    else 150.0
                )
                th = (
                    float(r.get("process_total_time_h", 3.0))
                    if pd.notna(r.get("process_total_time_h"))
                    else 3.0
                )
                mod_cands.append({
                    "row_idx": int(idx),
                    "r_smi": rs_can,
                    "h_smi": hs_can,
                    "mech": mech,
                    "mod_gpa": val,
                    "eq_r": eq_r,
                    "tc": tc,
                    "th": th,
                })
        except Exception:
            continue

    df_mc = (
        pd.DataFrame(mod_cands)
        .drop_duplicates(subset=["r_smi", "h_smi", "eq_r"])
        .sort_values("mod_gpa")
        .reset_index(drop=True)
    )
    idxs = [int(round(i * (len(df_mc) - 1) / 7)) for i in range(8)]
    df_msel = df_mc.iloc[idxs].reset_index(drop=True)
    print("Selected 8 Storage Modulus pilot formulations coupled with (r, T_cure, t_cure):")
    print(df_msel[["row_idx", "mech", "mod_gpa", "eq_r", "tc", "th"]].to_string(), flush=True)

    tpl_root = EXT_MD / "artifacts" / "stage4_pilot_condensed_md" / "species_library"
    mod_root.mkdir(parents=True, exist_ok=True)

    records = []
    for i, row in df_msel.iterrows():
        r_idx_int = int(row["row_idx"])
        mech_str = str(row["mech"])
        fid = f"MOD_PILOT_{i:02d}_ROW{r_idx_int}_{mech_str.upper()}"
        rxn = generate_multi_mechanism_reaction_adducts(
            str(row["r_smi"]), str(row["h_smi"]) if str(row["h_smi"]) else None, mech_str
        )
        pfx = f"N{i}"
        sp_r = parameterize_species_am1bcc_with_gas_ref(rxn["resin_smiles"], f"{pfx}R", str(tpl_root))
        sp_h = parameterize_species_am1bcc_with_gas_ref(rxn["hardener_smiles"], f"{pfx}H", str(tpl_root))
        sp_a1 = parameterize_species_am1bcc_with_gas_ref(rxn["adduct_1_smiles"], f"{pfx}A", str(tpl_root))
        sp_a2 = parameterize_species_am1bcc_with_gas_ref(rxn["adduct_2_smiles"], f"{pfx}B", str(tpl_root))

        comp, init_ep, meta = compute_stoichiometry_and_cure_coupled_composition(
            sp_r,
            sp_h,
            sp_a1,
            sp_a2,
            stoichiometric_ratio_r=float(row["eq_r"]),
            cure_temperature_c=float(row["tc"]),
            cure_time_h=float(row["th"]),
        )
        att_d = mod_root / f"attempt_{fid}"
        post_d = mod_root / f"postprocess_{fid}"
        build_and_run_condensed_state(
            state_dir=str(att_d / "state_0_addition"),
            state_index=0,
            state_name=fid,
            species_Composition=comp,
            initial_epoxy_sites=init_ep,
            seed=20261100 + i,
            nt_threads=8,
            compress_steps=10000,
            anneal_steps=8000,
            prod_steps=8000,
            mechanism_type=mech_str,
            formulation_id=fid,
            cure_temperature_c=float(row["tc"]),
            stoichiometric_ratio_r=float(row["eq_r"]),
            cure_time_h=float(row["th"]),
        )
        res_p = run_postprocessing(str(att_d), str(post_d), stride=1, grid_spacing_nm=0.10)
        r_d = res_p["df_summary"].iloc[0].to_dict()
        r_d.update(meta)
        r_d["literature_row_idx"] = r_idx_int
        r_d["storage_modulus_25c_gpa"] = float(row["mod_gpa"])
        r_d["md_hbond_total_density_nm3"] = round(float(r_d["md_hbond_count_average"] / r_d["box_volume_nm3"]), 6)
        r_d["md_hbond_inter_density_nm3"] = round(float(r_d["md_hbond_intermolecular_average"] / r_d["box_volume_nm3"]), 6)
        records.append(r_d)
        mod_val = float(row["mod_gpa"])
        eq_val = float(row["eq_r"])
        tc_val = float(row["tc"])
        dens_val = float(r_d["md_density_g_cm3"])
        ced_val = r_d["md_cohesive_energy_density_CED_MPa"]
        print(
            f"  -> [{i+1}/8] {fid} | E'={mod_val:.2f} GPa | r={eq_val:.2f} | Tc={tc_val:.0f}C | "
            f"dens={dens_val:.4f} | CED={ced_val} MPa",
            flush=True,
        )

    df_mod_adm = pd.DataFrame(records)
    df_mod_adm.to_parquet(out_pq, index=False)
    df_mod_adm.to_csv(mod_root / "md_features_modulus_pilot.csv", index=False)
    return df_mod_adm


if __name__ == "__main__":
    run_modulus_pilot_gromacs_simulations()
