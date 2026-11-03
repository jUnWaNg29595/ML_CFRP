# -*- coding: utf-8 -*-
"""
core/feature_deduplication.py

CFRP 机器学习特征矩阵清洗、别名去重与训练就绪化模块。
提供 `prepare_ml_feature_matrix` 函数与 `MLFeaturePipeline` 数据契约流水线，
保证进入模型的特征矩阵无同名重复列、无同义物理别名、无单位换算冗余，并严格阻断目标/ID泄漏。
"""

from typing import Tuple, Optional, List, Dict, Any, Set, Union
import numpy as np
import pandas as pd
from .feature_registry import (
    get_feature_registry,
    get_canonical_feature_names,
    get_features_for_set,
    FeatureMeta,
    FEATURE_SETS,
)

# 默认排除的预测目标字段黑名单 (防止数据泄漏 P0-4)
DEFAULT_TARGET_COLS: Set[str] = {
    "tg",
    "tg_c",
    "target_tg_c",
    "tg_c_test_method",
    "tg_c_test_atmosphere",
    "tg_c_heating_rate_c_min",
    "tensile_strength_mpa",
    "tensile_strength",
    "flexural_strength_mpa",
    "flexural_strength",
    "flexural_modulus_gpa",
    "flexural_modulus",
    "tensile_modulus_gpa",
    "tensile_modulus",
    "impact_strength_kj_m2",
    "impact_strength",
    "elongation_at_break_percent",
    "elongation_at_break",
    "fracture_toughness_k1c",
}

# 默认排除的样本标识符黑名单
DEFAULT_ID_COLS: Set[str] = {
    "sample_id",
    "row_id",
    "doi",
    "formula_id",
    "id",
    "index",
    "dataset_id",
    "source_file",
    "paper_id",
}

# 默认排除的分组/批次元数据黑名单
DEFAULT_GROUP_COLS: Set[str] = {
    "group_id",
    "batch_id",
    "cluster_id",
    "fold_id",
    "split",
    "split_group",
}

# 明确允许 One-Hot 编码的分类特征白名单 (P1-1)
ALLOWED_CATEGORICAL_COLS: Set[str] = {
    "cp_r_source",
    "xl_r_source",
    "cp_mechanism",
    "xl_mechanism",
    "curing_mechanism",
    "process_has_post_cure",
}

# 各分层特征集合专属的分类变量映射 (防止消融交叉污染 P1)
CATEGORICAL_FEATURE_SETS: Dict[str, Set[str]] = {
    "reaction_core": {"cp_r_source", "xl_r_source", "cp_mechanism", "xl_mechanism", "curing_mechanism"},
    "molecular": set(),  # 分子结构集合是纯数值化学描述符，不引入反应来源或工艺分类
    "formulation_process": {"process_has_post_cure", "curing_mechanism"},
    "full_safe": {"cp_r_source", "xl_r_source", "cp_mechanism", "xl_mechanism", "curing_mechanism", "process_has_post_cure"},
}


def validate_and_resolve_density(
    raw_density: Any,
    default_density: float = 1.2,
    allow_fallback: bool = True,
    index: Optional[pd.Index] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """严格验证并解析密度序列 (P0-2)。

    返回三元组:
        (density_used_s, density_source_s, density_fallback_used_s)

    规则:
    1. 密度必须是有限且严格正数 (np.isfinite(val) and val > 0.0)。
    2. NaN, None, 0, 负数, inf, -inf, 非法字符串等无效输入绝不能直接参与换算。
    3. 若为有效有限正数: density_used = val, density_source = "sample_provided", fallback_used = False。
    4. 若为无效值:
       - allow_fallback=True (默认): density_used = default_density (1.2),
         density_source = "reference_fallback", fallback_used = True。
       - allow_fallback=False: density_used = np.nan,
         density_source = "invalid_unresolved", fallback_used = False。
    5. 返回的 Series 索引与输入保持完全一致。
    """
    if isinstance(raw_density, pd.Series):
        s = raw_density.copy()
    elif isinstance(raw_density, (list, tuple, np.ndarray)):
        s = pd.Series(raw_density, index=index)
    else:
        s = pd.Series([raw_density], index=index if index is not None else [0])

    s_num = pd.to_numeric(s, errors="coerce")
    is_valid = s_num.notna() & np.isfinite(s_num) & (s_num > 0.0)

    used = s_num.copy()
    src = pd.Series("invalid_unresolved", index=s.index, dtype=object)
    fallback = pd.Series(False, index=s.index, dtype=bool)

    if allow_fallback:
        used = used.where(is_valid, float(default_density))
        src = pd.Series(np.where(is_valid, "sample_provided", "reference_fallback"), index=s.index)
        fallback = ~is_valid
    else:
        used = used.where(is_valid, np.nan)
        src = pd.Series(np.where(is_valid, "sample_provided", "invalid_unresolved"), index=s.index)
        fallback = pd.Series(False, index=s.index, dtype=bool)

    return used, src, fallback


def prepare_ml_feature_matrix(
    df: pd.DataFrame,
    registry: Optional[Dict[str, FeatureMeta]] = None,
    *,
    feature_set: Union[str, List[str]] = "reaction_core",
    target_columns: Optional[List[str]] = None,
    id_columns: Optional[List[str]] = None,
    group_columns: Optional[List[str]] = None,
    encode_categoricals: bool = False,
    categorical_vocabularies: Optional[Dict[str, List[str]]] = None,
    allow_unregistered_numeric: bool = False,
    conflict_policy: str = "raise",  # "raise" | "keep_first" | "ignore"
    conflict_tolerance: float = 1e-4,
    reference_density_g_cm3: float = 1.2,
    allow_fallback_density: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """为下游模型训练清洗并提取规范特征矩阵。

    步骤：
    1. 输入重复列名检测与累积冲突判定（P0-1: 3列及以上同名列逐行与累积合并结果比较，冲突默认 raise）。
    2. 目标泄漏与元数据阻断（P0-4: 严格排除 Tg、力学目标、样本 ID 及 Group 字段）。
    3. 注册表物理别名一致性校验与补齐（P0-1B, P0-3: 共同非空冲突 raise，缺失安全补齐）。
    4. 规范量纲单位换算（P0-2, P0-3: 严格校验密度有限正数，统一上游生成与下游转换基准）。
    5. 分类特征 One-Hot 编码与白名单保留（P1-1: 仅对授权分类列编码，编码列受白名单保护）。
    6. 分层特征集合筛选与白名单过滤（Feature Sets A/B/C/D 灵活消融支持）。
    """
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    reg = registry or get_feature_registry()
    n_rows = len(df)
    orig_index = df.index
    out = df.copy()
    out.index = pd.RangeIndex(n_rows)
    dropped_audit: List[Dict[str, Any]] = []
    canonical_conflict_masks: Dict[str, pd.Series] = {}

    # -----------------------------------------------------------------------
    # 步骤 1: 同名重复列名检测与累积冲突处理 (P0-1)
    # -----------------------------------------------------------------------
    if not out.columns.is_unique:
        unique_col_names = list(dict.fromkeys(out.columns))
        merged_cols: Dict[str, pd.Series] = {}

        for col_name in unique_col_names:
            sub = out[col_name]
            if isinstance(sub, pd.DataFrame):
                k = sub.shape[1]
                merged_s = sub.iloc[:, 0].copy()
                dup_conflict_mask = pd.Series(False, index=out.index)

                for j in range(1, k):
                    cand_col = sub.iloc[:, j]
                    # 与当前累计合并结果 merged_s 逐行比较冲突 (P0-1)
                    both_valid = merged_s.notna() & cand_col.notna()
                    if both_valid.any():
                        cand_num = pd.to_numeric(cand_col, errors="coerce")
                        merged_num = pd.to_numeric(merged_s, errors="coerce")
                        both_num = both_valid & cand_num.notna() & merged_num.notna()

                        has_conflict = False
                        first_idx = None
                        val0 = None
                        valj = None
                        conflict_indices = pd.Index([])

                        if both_num.any():
                            diff = (merged_num.loc[both_num] - cand_num.loc[both_num]).abs()
                            conflicts = diff > conflict_tolerance
                            if conflicts.any():
                                has_conflict = True
                                first_idx = conflicts[conflicts].index[0]
                                val0 = merged_s.loc[first_idx]
                                valj = cand_col.loc[first_idx]
                                conflict_indices = diff[conflicts].index

                        non_num = both_valid & (~both_num)
                        if not has_conflict and non_num.any():
                            str_m = merged_s.loc[non_num].astype(str)
                            str_c = cand_col.loc[non_num].astype(str)
                            str_diff = str_m != str_c
                            if str_diff.any():
                                has_conflict = True
                                first_idx = str_diff[str_diff].index[0]
                                val0 = merged_s.loc[first_idx]
                                valj = cand_col.loc[first_idx]
                                conflict_indices = str_diff[str_diff].index

                        if has_conflict:
                            if conflict_policy == "raise":
                                orig_first_idx = orig_index[first_idx]
                                raise ValueError(
                                    f"Duplicate column '{col_name}' has conflicting values at index '{orig_first_idx}': "
                                    f"{val0} vs {valj} (policy='raise')"
                                )
                            elif conflict_policy == "mask_conflicts":
                                dup_conflict_mask.loc[conflict_indices] = True
                                canon_key = reg[col_name].canonical_name if (col_name in reg and reg[col_name].canonical_name) else (reg[col_name].alias_of if (col_name in reg and reg[col_name].alias_of) else col_name)
                                canonical_conflict_masks.setdefault(canon_key, pd.Series(False, index=out.index)).loc[conflict_indices] = True
                                for ci in conflict_indices:
                                    pos = int(ci)
                                    dropped_audit.append({
                                        "feature_name": col_name,
                                        "action": "masked_conflict_to_nan",
                                        "reason": "duplicate_column_conflict_masked",
                                        "row_position": pos,
                                        "row_index": orig_index[pos],
                                    })
                            else:
                                dropped_audit.append({
                                    "feature_name": col_name,
                                    "action": "resolved_conflict_kept_first",
                                    "reason": "duplicate_column_name_value_mismatch",
                                })
                        else:
                            dropped_audit.append({
                                "feature_name": col_name,
                                "action": "deduplicated_identical_columns",
                                "reason": "duplicate_column_name_exact_match",
                            })
                    else:
                        dropped_audit.append({
                            "feature_name": col_name,
                            "action": "complementary_merged",
                            "reason": "duplicate_column_name_nan_complementary",
                        })

                    # 互补合并累计值
                    merged_s = merged_s.combine_first(cand_col)

                if conflict_policy == "mask_conflicts" and dup_conflict_mask.any():
                    merged_s.loc[dup_conflict_mask] = np.nan
                merged_cols[col_name] = merged_s
            else:
                merged_cols[col_name] = sub

        out = pd.DataFrame(merged_cols, index=out.index)

    # -----------------------------------------------------------------------
    # 步骤 2: 目标泄漏、ID 与分组元数据阻断 (P0-4)
    # -----------------------------------------------------------------------
    target_set = {str(c).strip().lower() for c in (target_columns or [])} | DEFAULT_TARGET_COLS
    id_set = {str(c).strip().lower() for c in (id_columns or [])} | DEFAULT_ID_COLS
    group_set = {str(c).strip().lower() for c in (group_columns or [])} | DEFAULT_GROUP_COLS

    cols_to_drop_step2 = []
    for c in list(out.columns):
        c_low = str(c).strip().lower()
        if c_low in target_set or (target_columns and c in target_columns):
            cols_to_drop_step2.append(c)
            dropped_audit.append({
                "feature_name": c,
                "action": "excluded_target_column",
                "reason": "target_column_excluded",
            })
        elif c_low in id_set or (id_columns and c in id_columns):
            cols_to_drop_step2.append(c)
            dropped_audit.append({
                "feature_name": c,
                "action": "excluded_id_column",
                "reason": "id_column_excluded",
            })
        elif c_low in group_set or (group_columns and c in group_columns):
            cols_to_drop_step2.append(c)
            dropped_audit.append({
                "feature_name": c,
                "action": "excluded_group_column",
                "reason": "group_column_excluded",
            })

    if cols_to_drop_step2:
        out = out.drop(columns=cols_to_drop_step2)

    # -----------------------------------------------------------------------
    # 步骤 3: 严格密度解析、别名协调与量纲换算 (P0-1B, P0-2, P0-3)
    # -----------------------------------------------------------------------
    # 样本级输入密度严格解析 (P0-2)
    rho_col_name = "rho_g_cm3" if "rho_g_cm3" in out.columns else ("density_g_cm3" if "density_g_cm3" in out.columns else None)
    if rho_col_name is not None:
        raw_rho_s = out[rho_col_name]
    elif "rho_g_cm3" in df.columns:
        raw_rho_s = df["rho_g_cm3"]
    elif "density_g_cm3" in df.columns:
        raw_rho_s = df["density_g_cm3"]
    else:
        raw_rho_s = pd.Series(np.nan, index=out.index)

    sample_density_s, density_src_s, density_fallback_s = validate_and_resolve_density(
        raw_rho_s,
        default_density=reference_density_g_cm3,
        allow_fallback=allow_fallback_density,
        index=out.index,
    )

    # 提取上游机理生成时实际采用的密度 (P0-3 & P0-4: 全路径严格尊重 allow_fallback_density)
    mech_rho_col = "rho_used_for_mech_nu_g_cm3"
    if mech_rho_col in out.columns:
        mech_gen_density_s, _, _ = validate_and_resolve_density(
            out[mech_rho_col],
            default_density=reference_density_g_cm3,
            allow_fallback=allow_fallback_density,
            index=out.index,
        )
    elif mech_rho_col in df.columns:
        mech_gen_density_s, _, _ = validate_and_resolve_density(
            df[mech_rho_col],
            default_density=reference_density_g_cm3,
            allow_fallback=allow_fallback_density,
            index=out.index,
        )
    else:
        mech_gen_density_s = None

    # P0-4: 禁止密度回退时，清除违规保留的历史参考计算值
    if not allow_fallback_density:
        bad_rho_mask = (density_src_s != "sample_provided")
        if bad_rho_mask.any():
            for d_col in ("mech_crosslink_density_mol_m3", "xl_nu_theory_mol_m3", "xl_nu_junction_mol_m3"):
                if d_col in out.columns:
                    out.loc[bad_rho_mask, d_col] = np.nan

    cols_to_drop_step3 = []

    for col in list(out.columns):
        if col in reg:
            meta = reg[col]
            if not meta.include_in_ml and meta.alias_of:
                canon = meta.alias_of

                # 判断别名是同一量纲还是需要单位换算 (P0-2, P0-3, P0-4)
                if meta.relationship == "unit_conversion":
                    # 确定单位换算所采用的密度基准 (P0-3 & P0-4):
                    # 1. 若样本给出了真实有效密度，优先采用真实密度覆盖旧参考缓存 (P0-4)；
                    # 2. 若样本给出了无效密度或缺失密度：
                    #    - allow_fallback_density=False: 严禁回退，密度置为 NaN (P0-4)；
                    #    - allow_fallback_density=True: 允许使用上游生成密度或参考基准 (1.2 g/cm³)。
                    is_sample_prov = (density_src_s == "sample_provided")
                    if not allow_fallback_density:
                        conv_density_s = sample_density_s.where(is_sample_prov, np.nan)
                    else:
                        if mech_gen_density_s is not None:
                            conv_density_s = mech_gen_density_s.copy()
                            conv_density_s = conv_density_s.where(~is_sample_prov, sample_density_s)
                        elif canon in out.columns:
                            conv_density_s = sample_density_s
                        else:
                            conv_density_s = sample_density_s

                    mult = conv_density_s * 1000.0
                    col_num = pd.to_numeric(out[col], errors="coerce")
                    converted_val = col_num * mult

                    # 若密度为 NaN (例如禁用回退时的非法或缺失密度)，转换值严格为 NaN (P0-4)
                    converted_val = converted_val.where(conv_density_s.notna() & (conv_density_s > 0), np.nan)

                    if canon in out.columns:
                        # 共同非空校验
                        both_valid = out[canon].notna() & converted_val.notna()
                        if both_valid.any():
                            diff = (pd.to_numeric(out.loc[both_valid, canon], errors="coerce") -
                                    converted_val.loc[both_valid]).abs()
                            rel_diff = diff / pd.to_numeric(out.loc[both_valid, canon], errors="coerce").abs().clip(lower=1e-4)
                            # 单位换算由于浮点舍入允许 1% 相对容差 (如 3.33 mmol/g 对应 4000 mol/m³)
                            unit_tol = 0.01 if "crosslink_density" in col else conflict_tolerance
                            if (rel_diff > unit_tol).any():
                                bad_indices = rel_diff[rel_diff > unit_tol].index
                                bad_idx = bad_indices[0]
                                if conflict_policy == "raise":
                                    orig_bad_idx = orig_index[bad_idx]
                                    raise ValueError(
                                        f"Unit conversion conflict between '{col}' (converted: {converted_val.loc[bad_idx]}) "
                                        f"and canonical '{canon}' ({out[canon].loc[bad_idx]}) at index '{orig_bad_idx}' (policy='raise')"
                                    )
                                elif conflict_policy == "mask_conflicts":
                                    canonical_conflict_masks.setdefault(canon, pd.Series(False, index=out.index)).loc[bad_indices] = True
                                    for bi in bad_indices:
                                        pos = int(bi)
                                        dropped_audit.append({
                                            "feature_name": col,
                                            "action": f"masked_conflict_with_{canon}_to_nan",
                                            "reason": f"unit_conversion_conflict_with_{canon}",
                                            "row_position": pos,
                                            "row_index": orig_index[pos],
                                        })
                                else:
                                    dropped_audit.append({
                                        "feature_name": col,
                                        "action": f"conflict_with_{canon}_kept_canonical",
                                        "reason": f"unit_conversion_conflict_with_{canon}",
                                    })
                        # 安全利用别名补齐规范主列 (P0-3)
                        out[canon] = out[canon].combine_first(converted_val)
                        cols_to_drop_step3.append(col)
                        dropped_audit.append({
                            "feature_name": col,
                            "action": f"converted_and_merged_into_{canon}",
                            "reason": f"unit_conversion_into_{canon}",
                        })
                    else:
                        # 规范主列缺失，将别名换算后晋升为规范主列 (P0-2, P0-3)
                        out[canon] = converted_val
                        cols_to_drop_step3.append(col)
                        dropped_audit.append({
                            "feature_name": col,
                            "action": f"promoted_and_converted_to_{canon}",
                            "reason": f"unit_conversion_promoted_to_{canon}",
                        })
                else:
                    # 同一量纲物理别名 (exact_alias P0-1B)
                    alias_val = pd.to_numeric(out[col], errors="coerce") if pd.api.types.is_numeric_dtype(out[col]) else out[col]

                    if canon in out.columns:
                        canon_val = pd.to_numeric(out[canon], errors="coerce") if pd.api.types.is_numeric_dtype(out[canon]) else out[canon]
                        both_valid = canon_val.notna() & alias_val.notna()
                        if both_valid.any():
                            # 官能度由于历史舍入允许 0.05 容差，其余允许 conflict_tolerance
                            tol = 0.05 if ("average_functionality" in col or "f_avg" in col) else conflict_tolerance
                            diff = (canon_val.loc[both_valid] - alias_val.loc[both_valid]).abs()
                            if (diff > tol).any():
                                bad_indices = diff[diff > tol].index
                                bad_idx = bad_indices[0]
                                if conflict_policy == "raise":
                                    orig_bad_idx = orig_index[bad_idx]
                                    raise ValueError(
                                        f"Physical alias conflict between '{col}' ({alias_val.loc[bad_idx]}) "
                                        f"and canonical '{canon}' ({canon_val.loc[bad_idx]}) at index '{orig_bad_idx}' (policy='raise')"
                                    )
                                elif conflict_policy == "mask_conflicts":
                                    canonical_conflict_masks.setdefault(canon, pd.Series(False, index=out.index)).loc[bad_indices] = True
                                    for bi in bad_indices:
                                        pos = int(bi)
                                        dropped_audit.append({
                                            "feature_name": col,
                                            "action": f"masked_conflict_with_{canon}_to_nan",
                                            "reason": f"alias_conflict_with_{canon}",
                                            "row_position": pos,
                                            "row_index": orig_index[pos],
                                        })
                                else:
                                    dropped_audit.append({
                                        "feature_name": col,
                                        "action": f"conflict_with_{canon}_kept_canonical",
                                        "reason": f"alias_conflict_with_{canon}",
                                    })
                        # 安全利用别名补齐规范主列 (P0-3)
                        out[canon] = out[canon].combine_first(alias_val)
                        cols_to_drop_step3.append(col)
                        dropped_audit.append({
                            "feature_name": col,
                            "action": f"merged_into_{canon}",
                            "reason": f"exact_alias_of_{canon}",
                        })
                    else:
                        # 规范主列缺失，将别名晋升为规范主列 (P0-3)
                        out[canon] = alias_val
                        cols_to_drop_step3.append(col)
                        dropped_audit.append({
                            "feature_name": col,
                            "action": f"promoted_to_{canon}",
                            "reason": f"promoted_alias_to_{canon}",
                        })
            elif not meta.include_in_ml and meta.dedup_reason:
                cols_to_drop_step3.append(col)
                dropped_audit.append({
                    "feature_name": col,
                    "action": f"dropped_{meta.dedup_reason}",
                    "reason": meta.dedup_reason,
                })

    if cols_to_drop_step3:
        out = out.drop(columns=[c for c in cols_to_drop_step3 if c in out.columns])

    # 统一应用所有规范特征的累积冲突掩码，确保 mask_conflicts 下冲突行严格为 NaN，不被后续别名重填 (P0-1)
    if conflict_policy == "mask_conflicts":
        for canon_name, conf_mask in canonical_conflict_masks.items():
            if canon_name in out.columns and conf_mask.any():
                out.loc[conf_mask, canon_name] = np.nan

    # -----------------------------------------------------------------------
    # 步骤 4: 分类变量 One-Hot 编码与白名单保留 (P1-1 & P1 特征集合隔离)
    # -----------------------------------------------------------------------
    allowed_dummy_cols: Set[str] = set()

    if encode_categoricals:
        fs_key = str(feature_set).strip().lower() if isinstance(feature_set, str) else "full_safe"
        if fs_key in ("reaction_core", "a"):
            target_cat_cols = CATEGORICAL_FEATURE_SETS["reaction_core"]
        elif fs_key in ("molecular", "b"):
            target_cat_cols = CATEGORICAL_FEATURE_SETS["molecular"]
        elif fs_key in ("formulation_process", "c"):
            target_cat_cols = CATEGORICAL_FEATURE_SETS["formulation_process"]
        elif fs_key in ("a+b", "b+a"):
            target_cat_cols = CATEGORICAL_FEATURE_SETS["reaction_core"]
        elif fs_key in ("b+c", "c+b"):
            target_cat_cols = CATEGORICAL_FEATURE_SETS["formulation_process"]
        elif fs_key in ("a+c", "c+a"):
            target_cat_cols = CATEGORICAL_FEATURE_SETS["reaction_core"] | CATEGORICAL_FEATURE_SETS["formulation_process"]
        else:
            target_cat_cols = CATEGORICAL_FEATURE_SETS["full_safe"]

        for sc in list(out.columns):
            if sc in target_cat_cols:
                s_series = out[sc].astype(str)

                # 若提供了预定词表（如测试集/推理阶段），严格使用预定词表防止维度漂移
                if categorical_vocabularies and sc in categorical_vocabularies:
                    vocab = categorical_vocabularies[sc]
                else:
                    # 训练阶段：基于数据提取唯一有效类别
                    vocab = sorted([c for c in s_series.unique() if c not in ("nan", "None", "<NA>", "")])

                for cat in vocab:
                    d_col = f"{sc}_{cat}"
                    out[d_col] = (s_series == cat).astype(float)
                    allowed_dummy_cols.add(d_col)
                    dropped_audit.append({
                        "feature_name": d_col,
                        "action": "generated_one_hot_column",
                        "reason": f"encoded_from_{sc}",
                    })

                out = out.drop(columns=[sc])
                dropped_audit.append({
                    "feature_name": sc,
                    "action": "encoded_and_dropped_raw_category",
                    "reason": "categorical_feature_encoded",
                })

    # 排除所有非数值、非布尔字符串列 (SMILES, 文件路径等)
    str_cols = [c for c in out.columns if not pd.api.types.is_numeric_dtype(out[c]) and not pd.api.types.is_bool_dtype(out[c])]
    if str_cols:
        for sc in str_cols:
            out = out.drop(columns=[sc])
            dropped_audit.append({
                "feature_name": sc,
                "action": "dropped_string_column",
                "reason": "non_numeric_ml_exclusion",
            })

    # -----------------------------------------------------------------------
    # 步骤 5: 分层特征集合筛选与白名单检查 (Section IX)
    # -----------------------------------------------------------------------
    if isinstance(feature_set, (list, tuple, set)):
        target_features = set(feature_set)
    else:
        target_features = set(get_features_for_set(feature_set))

    cols_to_keep: List[str] = []
    cols_to_drop_step5: List[str] = []

    for c in list(out.columns):
        if c in allowed_dummy_cols:
            # 合法 One-Hot 编码列受到明确保护 (P1-1)
            cols_to_keep.append(c)
        elif c in target_features:
            cols_to_keep.append(c)
        elif allow_unregistered_numeric and pd.api.types.is_numeric_dtype(out[c]):
            cols_to_keep.append(c)
        else:
            cols_to_drop_step5.append(c)
            dropped_audit.append({
                "feature_name": c,
                "action": "excluded_from_feature_set",
                "reason": f"not_in_feature_set_{feature_set}" if (c in reg and reg[c].include_in_ml) else "unregistered_numeric_excluded",
            })

    if cols_to_drop_step5:
        out = out.drop(columns=cols_to_drop_step5)

    out.index = orig_index
    summary_df = pd.DataFrame(dropped_audit)
    return out, summary_df


class MLFeaturePipeline:
    """训练与验证/测试阶段严格一致的 ML 特征提取流水线 (Section X 数据契约)。

    保证:
    1. 训练集确定特征名列表与列顺序，测试集严格对齐，零维度漂移。
    2. 分类编码词表在训练集固定，测试集遇到未知类别统一赋值为 0.0。
    3. 严格数据隔离，不向测试集泄漏分布信息。
    """

    def __init__(
        self,
        feature_set: Union[str, List[str]] = "reaction_core",
        encode_categoricals: bool = False,
        conflict_policy: str = "raise",
        conflict_tolerance: float = 1e-4,
        reference_density_g_cm3: float = 1.2,
        allow_fallback_density: bool = True,
        target_columns: Optional[List[str]] = None,
        id_columns: Optional[List[str]] = None,
        group_columns: Optional[List[str]] = None,
    ):
        self.feature_set = feature_set
        self.encode_categoricals = encode_categoricals
        self.conflict_policy = conflict_policy
        self.conflict_tolerance = conflict_tolerance
        self.reference_density_g_cm3 = reference_density_g_cm3
        self.allow_fallback_density = allow_fallback_density
        self.target_columns = target_columns
        self.id_columns = id_columns
        self.group_columns = group_columns

        self.selected_features_: List[str] = []
        self.categorical_vocabularies_: Dict[str, List[str]] = {}
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "MLFeaturePipeline":
        """在训练集上拟合特征契约与分类词表。"""
        # 记录分类变量词表 (仅对目标特征集合允许的分类变量 P1)
        if self.encode_categoricals:
            fs_key = str(self.feature_set).strip().lower() if isinstance(self.feature_set, str) else "full_safe"
            target_cats = CATEGORICAL_FEATURE_SETS.get(fs_key, CATEGORICAL_FEATURE_SETS["full_safe"])
            for sc in target_cats:
                if sc in df.columns:
                    s_vals = df[sc].dropna().astype(str).unique()
                    cats = sorted([c for c in s_vals if c not in ("nan", "None", "<NA>", "")])
                    self.categorical_vocabularies_[sc] = cats

        clean_df, _ = prepare_ml_feature_matrix(
            df,
            feature_set=self.feature_set,
            encode_categoricals=self.encode_categoricals,
            categorical_vocabularies=self.categorical_vocabularies_,
            conflict_policy=self.conflict_policy,
            conflict_tolerance=self.conflict_tolerance,
            reference_density_g_cm3=self.reference_density_g_cm3,
            allow_fallback_density=self.allow_fallback_density,
            target_columns=self.target_columns,
            id_columns=self.id_columns,
            group_columns=self.group_columns,
        )
        self.selected_features_ = list(clean_df.columns)
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """在验证集/测试集上应用拟合好的特征契约。"""
        if not self.is_fitted_:
            raise ValueError("MLFeaturePipeline 必须先调用 fit() 拟合，才能调用 transform()。")

        clean_df, _ = prepare_ml_feature_matrix(
            df,
            feature_set=self.feature_set,
            encode_categoricals=self.encode_categoricals,
            categorical_vocabularies=self.categorical_vocabularies_,
            conflict_policy=self.conflict_policy,
            conflict_tolerance=self.conflict_tolerance,
            reference_density_g_cm3=self.reference_density_g_cm3,
            allow_fallback_density=self.allow_fallback_density,
            target_columns=self.target_columns,
            id_columns=self.id_columns,
            group_columns=self.group_columns,
        )

        # 严格按照训练集的列和顺序对齐，缺失列安全填补 NaN 或 0.0，杜绝维度漂移
        out = pd.DataFrame(index=clean_df.index)
        for col in self.selected_features_:
            if col in clean_df.columns:
                out[col] = clean_df[col]
            else:
                is_dummy = any(col.startswith(f"{c}_") for c in ALLOWED_CATEGORICAL_COLS)
                out[col] = 0.0 if is_dummy else np.nan

        return out

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """拟合并返回训练矩阵及审计数据。"""
        self.fit(df)
        clean_df, audit_df = prepare_ml_feature_matrix(
            df,
            feature_set=self.feature_set,
            encode_categoricals=self.encode_categoricals,
            categorical_vocabularies=self.categorical_vocabularies_,
            conflict_policy=self.conflict_policy,
            conflict_tolerance=self.conflict_tolerance,
            reference_density_g_cm3=self.reference_density_g_cm3,
            allow_fallback_density=self.allow_fallback_density,
            target_columns=self.target_columns,
            id_columns=self.id_columns,
            group_columns=self.group_columns,
        )
        return clean_df, audit_df
