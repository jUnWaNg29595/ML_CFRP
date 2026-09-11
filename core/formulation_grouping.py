# -*- coding: utf-8 -*-
"""
多组分环氧树脂配方组智能识别与分组划分模块
用于在多组分、高频树脂（如 E51 占 50%+）场景下，智能识别化学配方组，并提供平衡、无泄漏的分组划分支持。
"""

import re
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np


def detect_formulation_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    智能检测数据表中属于环氧配方的各类列：
    - resin_cols: 树脂/环氧单体结构列
    - hardener_cols: 固化剂结构列
    - additive_cols: 助剂/促进剂/增韧剂/填料结构列
    - meta_group_cols: 显式的配方ID/批次/文献ID列
    """
    if df is None or df.empty:
        return {"resin": [], "hardener": [], "additive": [], "meta": []}

    cols = list(df.columns)
    resin_cols = []
    hardener_cols = []
    additive_cols = []
    meta_group_cols = []

    for c in cols:
        cl = str(c).lower().strip()
        # 排除纯数值统计、温度、时间、比例等连续工艺指标列
        if any(
            skip in cl
            for skip in [
                "count",
                "total",
                "ratio",
                "phr",
                "weight",
                "mass",
                "temperature",
                "time",
                "integral",
                "rate",
                "present",
                "basis",
                "tg",
                "modulus",
                "strength",
            ]
        ):
            continue

        # 1. 树脂/环氧单体列检测
        if re.search(r"(resin|epoxy).*?(structure|smiles|formula)", cl) or cl in [
            "resin_smiles",
            "epoxy_smiles",
            "resin",
            "epoxy",
        ]:
            resin_cols.append(c)
        # 2. 固化剂列检测
        elif re.search(
            r"(curing_agent|hardener|curer|curing).*?(structure|smiles|formula)", cl
        ) or cl in [
            "curing_agent_smiles",
            "hardener_smiles",
            "curing_agent",
            "hardener",
            "curer",
        ]:
            hardener_cols.append(c)
        # 3. 助剂/催化/增韧剂列检测
        elif re.search(
            r"(additive|modifier|filler|initiator|catalyst|accelerator|toughener).*?(structure|smiles|formula)",
            cl,
        ):
            additive_cols.append(c)
        # 4. 元数据显式分组键检测
        elif any(
            k in cl
            for k in [
                "formulation_id",
                "formula_id",
                "group_id",
                "batch_id",
                "doi",
                "paper_id",
                "sample_id",
                "system_id",
            ]
        ):
            meta_group_cols.append(c)

    return {
        "resin": resin_cols,
        "hardener": hardener_cols,
        "additive": additive_cols,
        "meta": meta_group_cols,
    }


def get_formulation_group_options(
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    根据检测到的配方列构建推荐的分组选项列表
    """
    detected = detect_formulation_columns(df)
    resin_cols = detected["resin"]
    hardener_cols = detected["hardener"]
    additive_cols = detected["additive"]
    meta_cols = detected["meta"]

    options = []

    # 选项 1: 主树脂 + 主固化剂配对 (首选推荐)
    if resin_cols and hardener_cols:
        cols = [resin_cols[0], hardener_cols[0]]
        options.append(
            {
                "key": "primary_pair",
                "label": f"🎯 主树脂 + 主固化剂配对 ({resin_cols[0]} + {hardener_cols[0]}) [推荐]",
                "columns": cols,
                "description": "以核心骨架组合为组。完美化解 E51 单体占比过高问题，同时彻底杜绝相同化学配对跨集泄漏。",
            }
        )

    # 选项 2: 全多组分配方签名
    all_chem_cols = resin_cols + hardener_cols + additive_cols
    if len(all_chem_cols) >= 2:
        options.append(
            {
                "key": "full_signature",
                "label": f"🧪 全组分配方签名 ({len(all_chem_cols)} 个化学结构列联合签名)",
                "columns": all_chem_cols,
                "description": "综合考虑所有树脂、固化剂与助剂结构，各组分完全一致时归为同组，最细粒度的化学配方隔离。",
            }
        )

    # 选项 3: 仅按主固化剂分组 (适合测试固化剂外推能力)
    if hardener_cols:
        options.append(
            {
                "key": "hardener_only",
                "label": f"🧪 仅按主固化剂分组 ({hardener_cols[0]})",
                "columns": [hardener_cols[0]],
                "description": "测试集将包含训练集从未出现过的固化剂（测试对全新未见固化剂的外推预测能力）。",
            }
        )

    # 选项 4: 仅按主树脂分组
    if resin_cols:
        options.append(
            {
                "key": "resin_only",
                "label": f"🧪 仅按主树脂分组 ({resin_cols[0]})",
                "columns": [resin_cols[0]],
                "description": "仅按树脂骨架分组。⚠️ 注意：若单一树脂（如 E51）占比过高，会导致划分严重不均。",
            }
        )

    # 选项 5: 显式元数据分组列 (如有)
    for mc in meta_cols:
        options.append(
            {
                "key": f"meta_{mc}",
                "label": f"📋 按元数据列分组 ({mc})",
                "columns": [mc],
                "description": f"直接依据数据表中的现成分组标识列 [{mc}] 进行独立划分。",
            }
        )

    # 选项 6: 自定义多选
    options.append(
        {
            "key": "custom",
            "label": "⚙️ 自定义选择分组列...",
            "columns": [],
            "description": "手动从所有数据列中挑选 1~N 个列自由组合作为分组键。",
        }
    )

    return options


def build_group_series(
    df: pd.DataFrame, columns: List[str]
) -> pd.Series:
    """
    根据选定的列构建分组标签序列
    """
    if df is None or df.empty or not columns:
        return pd.Series(["all_samples"] * len(df) if df is not None else [], index=df.index if df is not None else None)

    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        return pd.Series(["all_samples"] * len(df), index=df.index)

    if len(valid_cols) == 1:
        return df[valid_cols[0]].fillna("").astype(str)

    # 多列按 || 拼接
    return df[valid_cols].fillna("").astype(str).agg(" || ".join, axis=1)


def analyze_group_distribution(
    groups: pd.Series, test_size: float = 0.2
) -> Dict[str, Any]:
    """
    分析分组的统计特性与平衡性，检测是否存在极端倾斜（如单一组 > 30% 导致 E51 无法划分的问题）
    """
    if groups is None or len(groups) == 0:
        return {
            "n_groups": 0,
            "max_group_size": 0,
            "max_group_pct": 0.0,
            "is_skewed": False,
            "warning": "未检测到有效分组数据。",
        }

    total_samples = len(groups)
    vc = groups.value_counts()
    n_groups = len(vc)
    max_group_size = int(vc.iloc[0]) if n_groups > 0 else 0
    max_group_pct = float(max_group_size / total_samples) * 100.0
    top_group_name = str(vc.index[0]) if n_groups > 0 else ""

    # 如果单一最大组占总样本比例超过测试集比例 (例如 test_size=0.2 时，单一组 > 25%~30%)
    is_skewed = max_group_pct > max(25.0, test_size * 100.0)

    warning = ""
    recommendation = ""
    if is_skewed:
        warning = (
            f"⚠️ 分组严重倾斜预警：最大单一组占全数据集的 {max_group_pct:.1f}% ({max_group_size}/{total_samples} 样本)。"
            f"在划分测试集 (设定比例 {test_size*100:.0f}%) 时，由于同组样本不可分割，该大组无法被切分，"
            f"将导致测试集样本量严重偏离设定值或无法放入测试集！"
        )
        recommendation = "强烈建议改用【主树脂 + 主固化剂配对】或【全组分配方签名】，通过组合键将高频单体进一步细分为多个独立的反应配方组。"
    elif n_groups < 5:
        warning = f"⚠️ 分组数量偏少：总共仅有 {n_groups} 个独立组，分组交叉验证 (GroupKFold) 可能会因 Fold 数量不足而报错。"
        recommendation = "建议增加分组维度的列，以细化分组粒度。"

    return {
        "n_groups": n_groups,
        "max_group_size": max_group_size,
        "max_group_pct": max_group_pct,
        "top_group_name": top_group_name,
        "is_skewed": is_skewed,
        "warning": warning,
        "recommendation": recommendation,
    }
