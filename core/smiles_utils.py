# -*- coding: utf-8 -*-
"""
SMILES 工具函数：
- 多组分/多片段 SMILES 分列（自动拆分成 *_1, *_2, ...）
- RDKit canonical 化（统一写法，便于统计/类别平衡）
- 配方键（composition key）生成：对组分排序/去重后拼接，避免因顺序不同被误判为不同配方

说明：
- 分隔符规则与平台保持一致：先按 ';'、'；'、'|' 分割；再按“带空格的 +”分割；最后按 '.' 分割。
- 注意：不对 'C[N+](C)(C)C' 这种 SMILES 中的 '+' 误切分。
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False
    Chem = None


_SPLIT_SEMI = re.compile(r"\s*[;；|]\s*")
_SPLIT_PLUS = re.compile(r"\s+\+\s+")


def split_smiles_cell(cell) -> List[str]:
    """把单元格内容拆成多个 SMILES 片段（字符串列表）。"""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []

    # 先按 ;/；/| 分割
    parts = _SPLIT_SEMI.split(s)

    # 再按“带空格的 +”分割
    parts2: List[str] = []
    for p in parts:
        parts2.extend(_SPLIT_PLUS.split(p))

    # 再按 '.' 分割（SMILES 规范的多片段分隔）
    frags: List[str] = []
    for p in parts2:
        frags.extend([x.strip() for x in str(p).split('.') if x and str(x).strip()])

    return [f for f in frags if f]


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    RDKit canonical SMILES。
    - 失败返回 None
    """
    if not RDKIT_AVAILABLE:
        return None
    if smiles is None:
        return None
    s = str(smiles).strip()
    if not s:
        return None
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        # isomericSmiles=True 有助于保留立体信息（若有）
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def make_composition_key(components: List[str], canonicalize: bool = True, unique: bool = True, sort: bool = True) -> Optional[str]:
    """
    把多个组分生成一个稳定的“配方键”（composition key）。
    - canonicalize: 是否对每个组分做 RDKit canonical 化
    - unique: 是否去重
    - sort: 是否排序（避免组分顺序差异导致 key 不同）
    """
    if not components:
        return None

    comps = []
    for c in components:
        c = str(c).strip()
        if not c:
            continue
        if canonicalize:
            cc = canonicalize_smiles(c)
            comps.append(cc if cc else c)
        else:
            comps.append(c)

    if not comps:
        return None

    if unique:
        comps = list(dict.fromkeys(comps))  # 保序去重

    if sort:
        comps = sorted(comps)

    return ".".join(comps) if comps else None


def split_smiles_column(
    df: pd.DataFrame,
    column: str,
    max_components: int = 6,
    canonicalize: bool = True,
    add_key: bool = True,
    add_n_components: bool = True,
    keep_original: bool = True,
    prefix: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    将 df[column] 自动分列成 column_1...column_k。

    Returns:
        (new_df, created_columns)
    """
    if column not in df.columns:
        return df, []

    if max_components < 1:
        max_components = 1

    pref = prefix or column
    new_df = df.copy()

    # 1) 逐行拆分
    all_components: List[List[str]] = []
    max_len = 0
    for v in new_df[column].tolist():
        comps = split_smiles_cell(v)
        if canonicalize:
            comps2 = []
            for c in comps:
                cc = canonicalize_smiles(c)
                comps2.append(cc if cc else c)
            comps = comps2
        all_components.append(comps)
        max_len = max(max_len, len(comps))

    # 2) 决定列数
    k = min(max_len, max_components)
    created_cols: List[str] = []

    for i in range(k):
        col_i = f"{pref}_{i+1}"
        created_cols.append(col_i)
        new_df[col_i] = [comps[i] if len(comps) > i else np.nan for comps in all_components]

    if add_n_components:
        ncol = f"{pref}_n_components"
        created_cols.append(ncol)
        new_df[ncol] = [len([c for c in comps if c]) for comps in all_components]

    if add_key:
        kcol = f"{pref}_key"
        created_cols.append(kcol)
        new_df[kcol] = [make_composition_key(comps, canonicalize=False, unique=True, sort=True) for comps in all_components]

    if not keep_original:
        new_df = new_df.drop(columns=[column])

    return new_df, created_cols


def build_formulation_key(
    df: pd.DataFrame,
    resin_key_col: str,
    hardener_key_col: str,
    new_col: str = "formulation_key",
) -> pd.DataFrame:
    """
    基于 resin_key_col + hardener_key_col 构建体系配方键，用于类别平衡/分组划分。
    """
    if resin_key_col not in df.columns or hardener_key_col not in df.columns:
        return df
    new_df = df.copy()
    new_df[new_col] = (
        new_df[resin_key_col].astype(str).fillna("") + "||" + new_df[hardener_key_col].astype(str).fillna("")
    ).replace({"||": np.nan})
    return new_df


def top_value_counts(series: pd.Series, top_n: int = 10) -> pd.Series:
    """安全的 value_counts(top_n)。"""
    try:
        vc = series.value_counts(dropna=False)
        return vc.head(top_n)
    except Exception:
        return pd.Series(dtype=int)
