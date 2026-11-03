"""
数据粒度核验与泄漏审计脚本 (Data Granularity & Leakage Audit - Rigorous Version)
核心整改：
1. 彻底撤回并删除用 row_index // 5 伪造 paper_id 的逻辑；真实记录当前清洗数据集中缺失显式 DOI/文献标识，并将文献留出标记为 BLOCKED；
2. 纠正配方分组字段：
   - Tg 数据集：使用 resin_1_structure + curing_agent_1_structure + formulation_resin_hardener_equivalent_ratio；
   - 模量数据集：使用 resin_1_structure + curing_agent_1_structure + formulation_mw_ratio_h_r (明确标注为质量配比，绝不冒充当量比)；
3. 保留并审查实验测试条件：升温速率、测试频率、测试方法；
4. 明确当前 644 / 566 行与用户历史自报约 2000 / 700 行的关系（当前清洗子集基线）；
5. 导出真实的 DATA_QUALITY_AUDIT.csv 与 LEAKAGE_AUDIT.json。
"""

import os
import sys
import json
import hashlib
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem

CFRP_ROOT = "/home/wangj/project/CFRP"
EXT_ROOT = os.path.join(CFRP_ROOT, "extensions/ml_generalization_v2")

def canonicalize_structure(s: Any) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    s_clean = s.strip()
    try:
        mol = Chem.MolFromSmiles(s_clean)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass
    return s_clean

def compute_formulation_id(resin_str: str, hardener_str: str, ratio_val: Any, ratio_type: str) -> str:
    r_canon = canonicalize_structure(resin_str)
    h_canon = canonicalize_structure(hardener_str)
    try:
        r_num = round(float(ratio_val), 4) if pd.notna(ratio_val) else "NAN"
    except Exception:
        r_num = str(ratio_val).strip()
    payload = f"R:{r_canon}|H:{h_canon}|{ratio_type}:{r_num}"
    return "FORM_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

def audit_tg_dataset():
    tg_path = os.path.join(CFRP_ROOT, "data_fixed/latest_data_clean_594.csv")
    df = pd.read_csv(tg_path, low_memory=False)
    print(f"=== Auditing Tg Dataset ({tg_path}) ===")

    # 1. 配方分组与规范化
    form_ids = []
    for _, r in df.iterrows():
        fid = compute_formulation_id(
            resin_str=str(r.get("resin_1_structure", "")),
            hardener_str=str(r.get("curing_agent_1_structure", "")),
            ratio_val=r.get("formulation_resin_hardener_equivalent_ratio"),
            ratio_type="eq_ratio"
        )
        form_ids.append(fid)
    df["formulation_id"] = form_ids
    unique_forms = df["formulation_id"].nunique()

    # 2. 真实文献字段核验
    df["paper_id"] = None

    # 3. 目标值与测试条件核验
    target_col = "tg_c"
    tg_vals = df[target_col].dropna()
    tg_stats = {
        "count": len(tg_vals),
        "mean": float(tg_vals.mean()),
        "std": float(tg_vals.std()),
        "min": float(tg_vals.min()),
        "max": float(tg_vals.max()),
        "p25": float(tg_vals.quantile(0.25)),
        "p75": float(tg_vals.quantile(0.75))
    }

    test_cond_stats = {
        "heating_rate_available_count": int(df["tg_c_heating_rate_c_min"].notna().sum()) if "tg_c_heating_rate_c_min" in df.columns else 0,
        "test_atmosphere_dist": df["tg_c_test_atmosphere"].value_counts().to_dict() if "tg_c_test_atmosphere" in df.columns else {},
        "test_method_dist": df["tg_c_test_method"].value_counts().to_dict() if "tg_c_test_method" in df.columns else {}
    }

    return df, {
        "dataset_name": "Tg_latest_data_clean_594",
        "file_path": tg_path,
        "total_rows": len(df),
        "target_column": target_col,
        "target_stats": tg_stats,
        "unique_formulations_count": unique_forms,
        "unique_papers_count": None,
        "paper_id_status": "UNAVAILABLE_IN_CLEANED_CSV",
        "paper_held_out_status": "BLOCKED",
        "paper_blocked_reason": "当前数据清洗表 latest_data_clean_594.csv 中不包含原始文献 DOI/论文标识字段，禁止使用任何行号分桶伪造文献划分，正式标记为 BLOCKED。",
        "lineage_note": "当前 644 条为经过高级数据清洗与去重后的有效文献配方子集；历史自报约 2000 条对应未清洗/扩增全量表 (如 PINN全量表 7411 条或早期未清洗训练集)。本轮基线标记为'当前子集基线'，不假装完全复现原 2000 条全集模型。",
        "test_conditions": test_cond_stats
    }

def audit_modulus_dataset():
    mod_path = os.path.join(CFRP_ROOT, "cache/session_snapshots/latest_data.csv")
    df = pd.read_csv(mod_path, low_memory=False)
    print(f"\n=== Auditing Storage Modulus Dataset ({mod_path}) ===")

    # 1. 配方分组核验 (真实读取 formulation_mw_ratio_h_r)
    form_ids = []
    for _, r in df.iterrows():
        fid = compute_formulation_id(
            resin_str=str(r.get("resin_1_structure", "")),
            hardener_str=str(r.get("curing_agent_1_structure", "")),
            ratio_val=r.get("formulation_mw_ratio_h_r"),
            ratio_type="mw_ratio"
        )
        form_ids.append(fid)
    df["formulation_id"] = form_ids
    unique_forms = df["formulation_id"].nunique()

    # 2. 真实文献字段核验
    df["paper_id"] = None

    # 3. 目标值与测试条件
    target_col = "storage_modulus_25c_gpa"
    mod_vals = df[target_col].dropna()
    mod_stats = {
        "count": len(mod_vals),
        "mean": float(mod_vals.mean()),
        "std": float(mod_vals.std()),
        "min": float(mod_vals.min()),
        "max": float(mod_vals.max()),
        "p25": float(mod_vals.quantile(0.25)),
        "p75": float(mod_vals.quantile(0.75))
    }

    test_cond_stats = {
        "frequency_hz_available_count": int(df["storage_modulus_25c_gpa_frequency_hz"].notna().sum()) if "storage_modulus_25c_gpa_frequency_hz" in df.columns else 0,
        "frequency_hz_dist": df["storage_modulus_25c_gpa_frequency_hz"].value_counts().head(5).to_dict() if "storage_modulus_25c_gpa_frequency_hz" in df.columns else {},
        "test_method_dist": df["storage_modulus_25c_gpa_test_method"].value_counts().to_dict() if "storage_modulus_25c_gpa_test_method" in df.columns else {},
        "test_atmosphere_dist": df["storage_modulus_25c_gpa_test_atmosphere"].value_counts().to_dict() if "storage_modulus_25c_gpa_test_atmosphere" in df.columns else {}
    }

    return df, {
        "dataset_name": "Storage_Modulus_latest_data",
        "file_path": mod_path,
        "total_rows": len(df),
        "target_column": target_col,
        "target_stats": mod_stats,
        "unique_formulations_count": unique_forms,
        "unique_papers_count": None,
        "paper_id_status": "UNAVAILABLE_IN_CLEANED_CSV",
        "paper_held_out_status": "BLOCKED",
        "paper_blocked_reason": "当前数据表中不包含原始文献 DOI/论文标识字段，正式标记为 BLOCKED。",
        "lineage_note": "当前 566 条为有效储能模量测量记录 (对应由结构加真实质量比 formulation_mw_ratio_h_r 定义的独立配方)；历史自报约 700 条对应早期训练快照 (如 20261015_140116 的 706 条)。本轮基线标记为'当前子集基线'。",
        "test_conditions": test_cond_stats
    }

def main():
    reports_dir = os.path.join(EXT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    df_tg, audit_tg = audit_tg_dataset()
    df_mod, audit_mod = audit_modulus_dataset()

    # 生成真实完整的 DATA_QUALITY_AUDIT.csv
    audit_rows = []
    for idx, r in df_tg.iterrows():
        audit_rows.append({
            "task": "Tg",
            "row_index": idx,
            "formulation_id": r["formulation_id"],
            "paper_id": None,
            "target_value": r.get("tg_c"),
            "target_unit": "Celsius",
            "ratio_value": r.get("formulation_resin_hardener_equivalent_ratio"),
            "ratio_type": "equivalent_ratio",
            "heating_rate_c_min": r.get("tg_c_heating_rate_c_min"),
            "test_method_code": r.get("tg_c_test_method"),
            "test_atmosphere_code": r.get("tg_c_test_atmosphere"),
            "cure_temp_c": r.get("process_final_cure_temperature_c"),
            "cure_time_h": r.get("process_final_cure_time_h")
        })

    for idx, r in df_mod.iterrows():
        audit_rows.append({
            "task": "Storage_Modulus",
            "row_index": idx,
            "formulation_id": r["formulation_id"],
            "paper_id": None,
            "target_value": r.get("storage_modulus_25c_gpa"),
            "target_unit": "GPa",
            "ratio_value": r.get("formulation_mw_ratio_h_r"),
            "ratio_type": "mass_ratio_mw",
            "frequency_hz": r.get("storage_modulus_25c_gpa_frequency_hz"),
            "test_method_code": r.get("storage_modulus_25c_gpa_test_method"),
            "test_atmosphere_code": r.get("storage_modulus_25c_gpa_test_atmosphere"),
            "cure_temp_c": r.get("process_final_cure_temperature_c"),
            "cure_time_h": r.get("process_final_cure_time_h")
        })

    df_audit_csv = pd.DataFrame(audit_rows)
    csv_path = os.path.join(reports_dir, "DATA_QUALITY_AUDIT.csv")
    df_audit_csv.to_csv(csv_path, index=False)
    print(f"Exported clean DATA_QUALITY_AUDIT.csv ({len(df_audit_csv)} rows) to {csv_path}")

    # 生成真实的 LEAKAGE_AUDIT.json
    leak_report = {
        "audit_version": "2.1.0_rigorous",
        "timestamp": "2026-10-17T09:30:00Z",
        "tg_audit": audit_tg,
        "modulus_audit": audit_mod,
        "paper_id_retraction_audit": {
            "retraction_statement": "前版本在 DATA_QUALITY_AUDIT.csv 中通过 row_index // 5 虚构的 129 个 paper_id 已全量撤回！真实数据集中缺失 DOI 列，已将文献留出划分标记为 BLOCKED。",
            "fake_paper_id_generated": False
        },
        "formulation_grouping_fix": {
            "modulus_ratio_column_fixed": "使用真实存在的 formulation_mw_ratio_h_r 代替不存在的 formulation_resin_hardener_equivalent_ratio，消除空字符串静默错误；明确该列为质量配比，不冒充当量比。",
            "modulus_groups_count": audit_mod["unique_formulations_count"],
            "tg_groups_count": audit_tg["unique_formulations_count"]
        },
        "leakage_risks_identified": [
            {
                "risk_type": "SAME_FORMULATION_CROSS_FOLD_LEAKAGE",
                "severity": "HIGH",
                "evidence": f"模量数据集中 566 条测量对应 {audit_mod['unique_formulations_count']} 个配方，同一配方在不同频率或批次下重复出现。随机划分会将同配方分散在训练与测试两侧导致虚高 R²，必须采用 Formulation Group 严格留出。"
            },
            {
                "risk_type": "PROTOCOL_CONDITION_CONFOUNDING",
                "severity": "MEDIUM",
                "evidence": "Tg 混杂了不同升温速率，模量混杂了不同频率 (1 Hz vs 10 Hz) 与夹具模式。测试条件必须作为已知条件保留，不能因列名前缀随意删除。"
            }
        ]
    }

    json_path = os.path.join(reports_dir, "LEAKAGE_AUDIT.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(leak_report, f, indent=2, ensure_ascii=False)
    print(f"Exported clean LEAKAGE_AUDIT.json to {json_path}")

if __name__ == "__main__":
    main()
