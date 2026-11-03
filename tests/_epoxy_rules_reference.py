"""自动生成（勿手工改）：git HEAD 版 filter_candidates_by_epoxy_rules 原实现。

用于与向量化版本做逐位等价对比。手工重写判定分支正是历史上出错的原因，
所以这里直接抽取原码。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.virtual_screening import (
    DEFAULT_EPOXY_RULES,
    RDKIT_AVAILABLE,
    _calc_rule_features,
)

def filter_candidates_by_epoxy_rules(
    df: pd.DataFrame,
    resin_col: Optional[str] = "resin_smiles",
    hardener_col: Optional[str] = "hardener_smiles",
    rules: Optional[Dict] = None,
    keep_invalid: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty or not RDKIT_AVAILABLE:
        return df
    rules = rules or {}
    base = DEFAULT_EPOXY_RULES
    rules = {
        "global": {**base.get("global", {}), **(rules.get("global", {}) if isinstance(rules, dict) else {})},
        "resin": {**base.get("resin", {}), **(rules.get("resin", {}) if isinstance(rules, dict) else {})},
        "hardener": {**base.get("hardener", {}), **(rules.get("hardener", {}) if isinstance(rules, dict) else {})},
        "pair": {**base.get("pair", {}), **(rules.get("pair", {}) if isinstance(rules, dict) else {})},
    }

    allowed_elements = rules.get("global", {}).get("allowed_elements")
    if allowed_elements:
        try:
            allowed_elements = set(allowed_elements)
        except Exception:
            allowed_elements = None
    reject_charged = bool(rules.get("global", {}).get("reject_charged", True))

    resin_rules = rules.get("resin", {})
    hardener_rules = rules.get("hardener", {})
    pair_rules = rules.get("pair", {})

    cache: Dict[str, Dict[str, float]] = {}

    def _get_feat(smiles: str) -> Dict[str, float]:
        key = str(smiles or "")
        if key in cache:
            return cache[key]
        feats = _calc_rule_features(smiles, allowed_elements)
        cache[key] = feats
        return feats

    def _in_range(val: float, lo: Optional[float], hi: Optional[float]) -> bool:
        if val is None or not np.isfinite(val):
            return False
        if lo is not None and val < float(lo):
            return False
        if hi is not None and val > float(hi):
            return False
        return True

    def _passes_resin(feat: Dict[str, float]) -> bool:
        if not feat.get("valid"):
            return bool(keep_invalid)
        if reject_charged and feat.get("has_charge"):
            return False
        if not _in_range(feat.get("mol_wt"), resin_rules.get("min_mw"), resin_rules.get("max_mw")):
            return False
        if not _in_range(feat.get("heavy_atoms"), resin_rules.get("min_heavy_atoms"), resin_rules.get("max_heavy_atoms")):
            return False
        if feat.get("epoxide", 0) < int(resin_rules.get("min_epoxide", 0)):
            return False
        max_epoxide = resin_rules.get("max_epoxide")
        if max_epoxide is not None and feat.get("epoxide", 0) > int(max_epoxide):
            return False
        min_ar = resin_rules.get("min_aromatic_rings")
        if min_ar is not None:
            ar_val = feat.get("aromatic_rings")
            if ar_val is None or not np.isfinite(ar_val) or ar_val < float(min_ar):
                return False
        if resin_rules.get("ban_strong_acids", True):
            if feat.get("carboxylic_acid", 0) > 0:
                return False
            if feat.get("sulfonic_acid", 0) > 0:
                return False
            if feat.get("phosphoric_acid", 0) > 0:
                return False
        if resin_rules.get("ban_amines", True):
            if feat.get("primary_amine", 0) > 0 or feat.get("secondary_amine", 0) > 0:
                return False
        if resin_rules.get("ban_phenols", True) or resin_rules.get("ban_phenolic_hydroxyl", True):
            if feat.get("phenol_oh", 0) > 0:
                return False
        return True

    def _passes_hardener(feat: Dict[str, float]) -> Tuple[bool, str]:
        if not feat.get("valid"):
            return bool(keep_invalid), ""
        if reject_charged and feat.get("has_charge"):
            return False, ""
        if not _in_range(
            feat.get("mol_wt"), hardener_rules.get("min_mw"), hardener_rules.get("max_mw")
        ):
            return False, ""
        if not _in_range(
            feat.get("heavy_atoms"),
            hardener_rules.get("min_heavy_atoms"),
            hardener_rules.get("max_heavy_atoms"),
        ):
            return False, ""
        if hardener_rules.get("ban_epoxide", True) and feat.get("epoxide", 0) > 0:
            return False, ""
        if hardener_rules.get("ban_strong_acids", True):
            if feat.get("carboxylic_acid", 0) > 0:
                return False, ""
            if feat.get("sulfonic_acid", 0) > 0:
                return False, ""
            if feat.get("phosphoric_acid", 0) > 0:
                return False, ""
        # 固化剂最小官能团限制（按固化剂类型分别判断）
        min_func = int(hardener_rules.get("min_active_hydrogen", 0) or 0)
        if min_func > 0:
            amine_h = 2 * int(feat.get("primary_amine", 0)) + int(feat.get("secondary_amine", 0))
            anhydride_n = int(feat.get("anhydride", 0))
            phenol_n = int(feat.get("phenol_oh", 0))
            thiol_n = int(feat.get("thiol", 0))
            imidazole_n = int(feat.get("imidazole", 0))
            tertiary_n = int(feat.get("tertiary_amine", 0))
            # 胺类：按活泼氢数（伯胺~2 + 仲胺~1）
            if amine_h > 0:
                if amine_h < min_func:
                    return False, ""
            # 酸酐：按酸酐基团数（每个酸酐开环消耗1个环氧基）
            elif anhydride_n > 0:
                if anhydride_n < min_func:
                    return False, ""
            # 酚类：按酚羟基数
            elif phenol_n > 0:
                if phenol_n < min_func:
                    return False, ""
            # 硫醇：按巯基数
            elif thiol_n > 0:
                if thiol_n < min_func:
                    return False, ""
            # 咪唑：按咪唑环数（催化型，每个咪唑催化多个环氧开环）
            elif imidazole_n > 0:
                if imidazole_n < min_func:
                    return False, ""
            # 叔胺：按叔胺基数（催化型）
            elif tertiary_n > 0:
                if tertiary_n < min_func:
                    return False, ""
            else:
                # 未能识别类型的固化剂，保守过滤
                if min_func > 1:
                    return False, ""

        has_amine = (feat.get("primary_amine", 0) + feat.get("secondary_amine", 0)) > 0
        has_anhydride = feat.get("anhydride", 0) > 0
        has_phenol = feat.get("phenol_oh", 0) > 0
        has_thiol = feat.get("thiol", 0) > 0
        has_imidazole = feat.get("imidazole", 0) > 0
        has_tertiary = feat.get("tertiary_amine", 0) > 0

        if pair_rules.get("reject_mixed_class", True) and has_amine and has_anhydride:
            return False, ""

        allowed = hardener_rules.get("allowed_classes")
        allowed_set = None
        if allowed:
            try:
                allowed_set = {str(x).strip().lower() for x in allowed if str(x).strip()}
            except Exception:
                allowed_set = None

        class_flags = {
            "amine": has_amine,
            "anhydride": has_anhydride,
            "phenol": has_phenol,
            "thiol": has_thiol,
            "imidazole": has_imidazole,
            "tertiary_amine": has_tertiary,
        }
        if allowed_set is not None and not any(class_flags.get(k, False) for k in allowed_set):
            return False, ""

        for name in ("amine", "anhydride", "phenol", "thiol", "imidazole", "tertiary_amine"):
            if class_flags.get(name) and (allowed_set is None or name in allowed_set):
                return True, name
        # The permissive default accepts other valid curing components. A
        # caller that needs class-specific chemistry can pass allowed_classes.
        return (True, "other") if allowed_set is None else (False, "")

    amine_ratio = pair_rules.get("amine_ratio", (0.0, float("inf")))
    anhydride_ratio = pair_rules.get("anhydride_ratio", (0.0, float("inf")))
    phenol_ratio = pair_rules.get("phenol_ratio", amine_ratio)
    thiol_ratio = pair_rules.get("thiol_ratio", amine_ratio)

    mask = []
    has_resin = bool(resin_col and resin_col in df.columns)
    has_hardener = bool(hardener_col and hardener_col in df.columns)
    for _, row in df.iterrows():
        resin_feat = {}
        if has_resin:
            resin_smiles = row.get(resin_col)
            resin_feat = _get_feat(resin_smiles)
            if not _passes_resin(resin_feat):
                mask.append(False)
                continue

        if not has_hardener:
            mask.append(True)
            continue

        hardener_smiles = row.get(hardener_col)
        hardener_feat = _get_feat(hardener_smiles)
        ok, cls = _passes_hardener(hardener_feat)
        if not ok:
            mask.append(False)
            continue

        if not has_resin:
            mask.append(True)
            continue

        if cls == "amine":
            H = 2 * int(hardener_feat.get("primary_amine", 0)) + int(
                hardener_feat.get("secondary_amine", 0)
            )
            if H <= 0:
                mask.append(False)
                continue
            ratio = float(resin_feat.get("epoxide", 0)) / float(H)
            if not _in_range(ratio, amine_ratio[0], amine_ratio[1]):
                mask.append(False)
                continue
        elif cls == "anhydride":
            A = int(hardener_feat.get("anhydride", 0))
            if A <= 0:
                mask.append(False)
                continue
            ratio = float(resin_feat.get("epoxide", 0)) / float(A)
            if not _in_range(ratio, anhydride_ratio[0], anhydride_ratio[1]):
                mask.append(False)
                continue
        elif cls == "phenol":
            H = int(hardener_feat.get("phenol_oh", 0))
            if H <= 0:
                mask.append(False)
                continue
            ratio = float(resin_feat.get("epoxide", 0)) / float(H)
            if not _in_range(ratio, phenol_ratio[0], phenol_ratio[1]):
                mask.append(False)
                continue
        elif cls == "thiol":
            H = int(hardener_feat.get("thiol", 0))
            if H <= 0:
                mask.append(False)
                continue
            ratio = float(resin_feat.get("epoxide", 0)) / float(H)
            if not _in_range(ratio, thiol_ratio[0], thiol_ratio[1]):
                mask.append(False)
                continue
        mask.append(True)

    return df.loc[mask].reset_index(drop=True)


