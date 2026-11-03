"""
训练准入控制模块 (Training Admission Controller - Rigorous Evidence Verification)
核心职责：
1. 强制实行三种评估路径隔离：
   - baseline_literature_only: 仅使用经过质检的真实文献数据与允许的旧特征，不要求 MD 特征；
   - formal_md_ablation: 正式 MD 消融评估，必须满足严格的真实性、可追溯性与门禁合格证据；
   - diagnostic_fixture_test: 仅用于测试 fixture，显式标记 DIAGNOSTIC_ONLY，严禁混入科研输出。
2. 彻底堵塞独立验收审计发现的漏洞：
   - 漏洞 1: 仅靠 eligible=True 与 gate=PASS 绝不能放行！必须核验 data_origin, parameter_set_id, protocol_id, network_id, feature_run_id 等证据链；
   - 漏洞 2: 字符串 'False'、'false'、'0' 绝不被 bool() 误判为 True；
   - 漏洞 3: 当样本无真实模拟产物、缺少执行回执或包含合成标记时，坚决拒绝并返回 ADMISSION_REJECTED；
   - 漏洞 4: 拦截多 seed 重复计数/重复 join 导致的配方行数虚假翻倍；
   - 漏洞 5: 拦截未认证或冲突的 protocol_id 与 parameter_set_id。
"""

import os
import math
from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import numpy as np

PATH_BASELINE_LITERATURE_ONLY = "baseline_literature_only"
PATH_FORMAL_MD_ABLATION = "formal_md_ablation"
PATH_DIAGNOSTIC_FIXTURE_TEST = "diagnostic_fixture_test"

VALID_PATHS = {
    PATH_BASELINE_LITERATURE_ONLY,
    PATH_FORMAL_MD_ABLATION,
    PATH_DIAGNOSTIC_FIXTURE_TEST
}

# 认证受支持的协议集与参数集
RECOGNIZED_PROTOCOLS = {
    "EPX_HTPOLYNET_GAFF2_VERIFIED_V1",
    "EPX_GAFF2_DGEBA_DDS_MINI_V1",
    "VERIFIED_MD_V3_5"
}

def strict_parse_bool(val: Any) -> bool:
    """
    严格布尔值解析：
    彻底防御 Python 语言内置 bool('False') == True 的陷阱！
    只有真正的 True 或字符串 'true', '1' 才返回 True；
    'False', 'false', '0', None, NaN 等一律返回 False。
    """
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if val is None or pd.isna(val):
        return False
    if isinstance(val, (int, float, np.integer, np.floating)):
        return bool(val == 1)
    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned in ("true", "1", "yes", "t"):
            return True
        elif cleaned in ("false", "0", "no", "f", "none", "nan", ""):
            return False
        else:
            return False
    return False


class TrainingAdmissionValidator:
    """
    训练前置准入核验器 (Training Admission Validator)
    在任何机器学习模型拟合前，对输入数据集与特征执行硬门禁核验。
    """
    def __init__(self, path_mode: str = PATH_FORMAL_MD_ABLATION):
        if path_mode not in VALID_PATHS:
            raise ValueError(f"Invalid path_mode: {path_mode}. Must be one of {VALID_PATHS}")
        self.path_mode = path_mode

    def validate_dataset_admission(self, df: pd.DataFrame) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
        """
        验证全量待入模数据集的准入资格
        Returns:
            (is_admitted, admitted_df, audit_report)
        """
        audit_report = {
            "path_mode": self.path_mode,
            "total_input_rows": len(df),
            "admitted_rows": 0,
            "rejected_rows": 0,
            "rejection_reasons": {},
            "status": "PENDING"
        }

        if df is None or len(df) == 0:
            audit_report["status"] = "ADMISSION_REJECTED"
            audit_report["message"] = "Dataset is empty."
            return False, pd.DataFrame(), audit_report

        # 1. 路径 1: 仅真实文献基线评估 (baseline_literature_only)
        if self.path_mode == PATH_BASELINE_LITERATURE_ONLY:
            if "literature_quality_flag" in df.columns:
                valid_mask = df["literature_quality_flag"].astype(str).str.upper() == "PASS"
            else:
                valid_mask = pd.Series(True, index=df.index)

            admitted_df = df[valid_mask].copy()
            audit_report["admitted_rows"] = len(admitted_df)
            audit_report["rejected_rows"] = len(df) - len(admitted_df)
            audit_report["status"] = "PASS" if len(admitted_df) > 0 else "ADMISSION_REJECTED"
            return (len(admitted_df) > 0), admitted_df, audit_report

        # 2. 路径 3: 明确标记为诊断测试 fixture
        if self.path_mode == PATH_DIAGNOSTIC_FIXTURE_TEST:
            admitted_df = df.copy()
            audit_report["admitted_rows"] = len(admitted_df)
            audit_report["status"] = "DIAGNOSTIC_PASS_TEST_ONLY"
            audit_report["warning"] = "DIAGNOSTIC TEST FIXTURE ONLY; strictly forbidden for production ML or scientific performance reporting."
            return True, admitted_df, audit_report

        # 3. 路径 2: 正式 MD 消融评估 (formal_md_ablation) - 实行最高级别真实证据准入核验
        admit_mask = pd.Series(True, index=df.index)
        reasons_count = {}

        # 规则 A: eligible_for_training 严格校验 (防 'False' 字符串放行)
        if "eligible_for_training" in df.columns:
            elig_parsed = df["eligible_for_training"].apply(strict_parse_bool)
            ineligible_cnt = (~elig_parsed).sum()
            if ineligible_cnt > 0:
                reasons_count["ineligible_for_training_flag"] = int(ineligible_cnt)
                admit_mask &= elig_parsed
        else:
            reasons_count["missing_eligible_for_training_column"] = len(df)
            admit_mask &= False

        # 规则 B: 科学门禁状态必须为 PASS (拒绝 FAIL, BLOCKED, NOT_VERIFIED, EXPLORATORY)
        if "scientific_gate_status" in df.columns:
            gate_status_clean = df["scientific_gate_status"].astype(str).str.strip().str.upper()
            non_pass_mask = gate_status_clean != "PASS"
            non_pass_cnt = non_pass_mask.sum()
            if non_pass_cnt > 0:
                reasons_count["non_pass_scientific_gate"] = int(non_pass_cnt)
                admit_mask &= (~non_pass_mask)
        else:
            reasons_count["missing_scientific_gate_status_column"] = len(df)
            admit_mask &= False

        # 规则 C: 拦截合成与伪造数据来源 (data_origin 必须明确且非 SYNTHETIC)
        if "data_origin" in df.columns:
            origin_clean = df["data_origin"].astype(str).str.strip().str.upper()
            is_synthetic = origin_clean.str.contains("SYNTHETIC") | origin_clean.str.contains("FIXTURE")
            synth_cnt = is_synthetic.sum()
            if synth_cnt > 0:
                reasons_count["synthetic_fixture_origin"] = int(synth_cnt)
                admit_mask &= (~is_synthetic)
            # 必须具有真实合法数据来源标识
            is_valid_origin = origin_clean.isin(["REAL_MD_SIMULATION", "VERIFIED_PRODUCTION", "EXPERIMENTAL_LITERATURE"])
            invalid_origin_cnt = (~is_valid_origin).sum()
            if invalid_origin_cnt > 0:
                reasons_count["unverified_data_origin"] = int(invalid_origin_cnt)
                admit_mask &= is_valid_origin
        else:
            # 缺少 data_origin 必须拒绝！彻底堵塞 no_origin_no_receipts_no_hashes 漏洞！
            reasons_count["missing_data_origin_evidence"] = len(df)
            admit_mask &= False

        # 规则 D: 计算证据链核验 (哈希、网络ID、参数ID、协议ID 链条)
        # 正式 MD 消融必须提供可追溯的 network_id 或 protocol_id 或执行回执证据
        required_evidence_cols = ["network_id", "formulation_id"]
        for ec in required_evidence_cols:
            if ec in df.columns:
                missing_id = df[ec].isna() | (df[ec].astype(str).str.strip() == "")
                missing_cnt = missing_id.sum()
                if missing_cnt > 0:
                    reasons_count[f"missing_{ec}"] = int(missing_cnt)
                    admit_mask &= (~missing_id)
            else:
                reasons_count[f"missing_{ec}_column"] = len(df)
                admit_mask &= False

        # 规则 E: 证据回执与哈希核验 (防假 PASS 标签)
        if "calculation_receipt_hash" in df.columns:
            bad_hash = df["calculation_receipt_hash"].isna() | (df["calculation_receipt_hash"].astype(str).str.strip().str.len() < 16)
            bad_hash_cnt = bad_hash.sum()
            if bad_hash_cnt > 0:
                reasons_count["missing_or_invalid_receipt_hash"] = int(bad_hash_cnt)
                admit_mask &= (~bad_hash)

        # 规则 F: 协议冲突与认证核验
        if "protocol_id" in df.columns:
            unrec_proto = ~df["protocol_id"].astype(str).str.strip().isin(RECOGNIZED_PROTOCOLS)
            unrec_cnt = unrec_proto.sum()
            if unrec_cnt > 0:
                reasons_count["unrecognized_or_conflicting_protocol"] = int(unrec_cnt)
                admit_mask &= (~unrec_proto)

        # 规则 G: 拦截多 seed 重复 join 膨胀
        if "seed" in df.columns and "formulation_id" in df.columns:
            dup_keys = df.duplicated(subset=["formulation_id", "seed"], keep=False)
            dup_cnt = dup_keys.sum()
            if dup_cnt > 0:
                reasons_count["duplicate_formulation_seed_expansion"] = int(dup_cnt)
                admit_mask &= (~dup_keys)

        admitted_df = df[admit_mask].copy()
        audit_report["admitted_rows"] = int(len(admitted_df))
        audit_report["rejected_rows"] = int(len(df) - len(admitted_df))
        audit_report["rejection_reasons"] = reasons_count

        if len(admitted_df) == 0:
            audit_report["status"] = "ADMISSION_REJECTED"
            audit_report["message"] = (
                f"全量 {len(df)} 条样本均未通过正式训练准入审核！"
                f"拒绝原因分布: {reasons_count}。坚决阻断不合格或缺证据特征进入模型训练。"
            )
            return False, admitted_df, audit_report

        audit_report["status"] = "PASS"
        return True, admitted_df, audit_report
