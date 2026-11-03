"""
安全特征对比与评估引擎 (Safe Feature Comparison & Evaluation Engine - Rigorous Anti-Imputation)
核心安全与科学修复：
1. 修复漏洞 1: 当样本缺少 data_origin、network_id、计算回执或标记为不合格时，在 formal_md_ablation 下坚决拒绝准入并阻断，返回 status='ADMISSION_REJECTED'；
2. 修复漏洞 2: 当请求的特征组增量列缺失时，该组坚决标记为 BLOCKED，绝不静默退化；
3. 修复漏洞 3: 严格物理量非插补铁律！对 B/C/D 组的正式物理特征 (md_* 或增量特征列)，严格禁止用中位数或0将缺失值包装成“已计算值”！若存在物理特征缺失或 Inf/NaN 异常，该组必须坚决标记为 BLOCKED；
4. 普通文献协变量可在训练折内执行中位数插补与标准化，测试折仅 transform。
"""

import hashlib
import json
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .admission import (
    TrainingAdmissionValidator,
    PATH_BASELINE_LITERATURE_ONLY,
    PATH_FORMAL_MD_ABLATION,
    PATH_DIAGNOSTIC_FIXTURE_TEST
)
from .feature_contract import FeatureContractValidator, CONTRACT_GROUPS_DEF

class SafeFeatureComparisonEngine:
    def __init__(self, random_state: int = 42, path_mode: str = PATH_FORMAL_MD_ABLATION):
        self.random_state = random_state
        self.path_mode = path_mode
        self.admission_validator = TrainingAdmissionValidator(path_mode=path_mode)
        self.contract_validator = FeatureContractValidator()

    def evaluate_feature_groups(
        self,
        df: pd.DataFrame,
        target_col: str,
        group_col: str,
        baseline_feature_cols: List[str],
        topology_cols: List[str],
        free_volume_cols: List[str],
        interaction_cols: List[str],
        n_splits: int = 3,
        allow_diagnostic: bool = False
    ) -> Dict[str, Any]:
        """
        在严格准入、契约防御与物理防伪插补下评估阶梯特征组表现
        """
        # 1. 强制准入审查 (防御漏洞 1: 缺少来源证据或不合格样本直接阻断)
        effective_mode = PATH_DIAGNOSTIC_FIXTURE_TEST if allow_diagnostic else self.path_mode
        validator = TrainingAdmissionValidator(path_mode=effective_mode)
        is_admitted, admitted_df, admission_audit = validator.validate_dataset_admission(df)

        if not is_admitted:
            return {
                "status": "ADMISSION_REJECTED",
                "sample_count": 0,
                "rejection_audit": admission_audit,
                "message": admission_audit.get("message", "Dataset failed training admission review."),
                "results": {}
            }

        # 2. 检查目标列与分组列有效性
        if target_col not in admitted_df.columns:
            return {
                "status": "TARGET_COLUMN_MISSING",
                "sample_count": 0,
                "message": f"Target column '{target_col}' not found in dataset.",
                "results": {}
            }
        if group_col not in admitted_df.columns:
            return {
                "status": "GROUP_COLUMN_MISSING",
                "sample_count": 0,
                "message": f"Group column '{group_col}' not found in dataset.",
                "results": {}
            }

        sub_df = admitted_df.dropna(subset=[target_col, group_col]).copy()
        n_samples = len(sub_df)
        unique_groups = sub_df[group_col].nunique()

        if unique_groups < 2 or n_samples < 4:
            return {
                "status": "INSUFFICIENT_SAMPLE_SIZE_FOR_CV",
                "sample_count": n_samples,
                "group_count": unique_groups,
                "message": "有效样本数或独立分组数过少，不足以执行严密的交叉验证。",
                "results": {}
            }

        # 3. 特征契约严格核查 (防御漏洞 2: 防静默退化)
        groups_declaration = {
            "A_Baseline": (baseline_feature_cols, []),
            "B_Plus_Topology": (baseline_feature_cols, topology_cols),
            "C_Plus_FreeVolume": (baseline_feature_cols + topology_cols, free_volume_cols),
            "D_Plus_Interactions": (baseline_feature_cols + topology_cols + free_volume_cols, interaction_cols)
        }

        group_contracts = {}
        for grp_k, (base_c, incr_c) in groups_declaration.items():
            contract_res = self.contract_validator.validate_group_contract(
                df_columns=sub_df.columns.tolist(),
                grp_name=grp_k,
                declared_base_cols=base_c,
                declared_incremental_cols=incr_c
            )
            group_contracts[grp_k] = contract_res

        # 4. 执行交叉验证训练
        actual_splits = min(n_splits, unique_groups)
        gkf = GroupKFold(n_splits=actual_splits)

        results = {}
        all_groups_blocked = True

        for grp_k, c_info in group_contracts.items():
            # 契约不合格或增量列缺失，坚决标为 BLOCKED！
            if c_info["contract_status"] != "PASS":
                results[grp_k] = {
                    "status": "BLOCKED",
                    "reason": c_info["reason"],
                    "missing_columns": c_info.get("missing_columns", []),
                    "features_used_count": 0,
                    "features_list": []
                }
                continue

            active_cols = c_info["active_features"]
            if not active_cols:
                results[grp_k] = {
                    "status": "NO_FEATURES_AVAILABLE",
                    "features_used_count": 0,
                    "features_list": []
                }
                continue

            # 关键防御：物理特征缺失值与 Inf 异常严禁中位数插补！(防御 missing_physical_values 漏洞)
            # 对 B/C/D 组中的增量物理特征，若存在缺失值或 Inf，不能用中位数插补伪造成有效模拟特征
            if grp_k != "A_Baseline":
                declared_incr = groups_declaration[grp_k][1]
                missing_or_inf_cols = []
                for c in declared_incr:
                    if c in sub_df.columns:
                        col_vals = sub_df[c]
                        has_na = col_vals.isna().any()
                        has_inf = np.isinf(col_vals).any() if np.issubdtype(col_vals.dtype, np.number) else False
                        if has_na or has_inf:
                            missing_or_inf_cols.append(c)

                if missing_or_inf_cols:
                    results[grp_k] = {
                        "status": "BLOCKED",
                        "reason": f"特征组 {grp_k} 中物理特征存在缺失值或 Inf 异常 ({missing_or_inf_cols})。严格禁止将未模拟或异常物理量用插补包装为已计算特征！",
                        "missing_columns": missing_or_inf_cols,
                        "features_used_count": 0,
                        "features_list": []
                    }
                    continue

            all_groups_blocked = False
            y_true_all = []
            y_pred_all = []

            for train_idx, test_idx in gkf.split(sub_df, sub_df[target_col], sub_df[group_col]):
                train_data = sub_df.iloc[train_idx]
                test_data = sub_df.iloc[test_idx]

                X_train = train_data[active_cols].copy()
                X_test = test_data[active_cols].copy()
                y_train = train_data[target_col].copy()
                y_test = test_data[target_col].copy()

                # 普通文献协变量在折内拟合插补与缩放
                imputer_values = X_train.median().fillna(0.0)
                X_train_imp = X_train.fillna(imputer_values)
                X_test_imp = X_test.fillna(imputer_values)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_imp)
                X_test_scaled = scaler.transform(X_test_imp)

                model = RandomForestRegressor(n_estimators=50, random_state=self.random_state, max_depth=5)
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)

                y_true_all.extend(y_test.values)
                y_pred_all.extend(preds)

            y_true_arr = np.array(y_true_all)
            y_pred_arr = np.array(y_pred_all)

            mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
            rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
            r2 = float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 1 else 0.0

            data_hash = self.contract_validator.compute_features_sha256(sub_df, active_cols)

            results[grp_k] = {
                "status": "PASS",
                "features_used_count": len(active_cols),
                "features_list": active_cols,
                "features_data_sha256": data_hash,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4)
            }

        overall_status = "BLOCKED" if all_groups_blocked else "PASS"

        return {
            "status": overall_status,
            "sample_count": n_samples,
            "unique_groups": unique_groups,
            "cross_validation_splits": actual_splits,
            "admission_audit": admission_audit,
            "contract_audit": group_contracts,
            "results": results
        }

FeatureComparisonEngine = SafeFeatureComparisonEngine
