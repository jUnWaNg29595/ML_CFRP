"""
严格无泄漏分组划分管理器 (Strict Leakage-Free Split Manager)
核心职责：
1. 生成并冻结多维度泛化评估划分方案：
   - random_5fold: 随机 5 折 (对照复现历史自报随机划分)；
   - formulation_group_5fold: 配方严格留出 5 折 (基于结构与真实配比构成的 formulation_id，同一配方绝不跨折)；
   - structural_family_group: 骨架化学家族留出 (测试跨未见单体家族的泛化外推)；
   - paper_group_5fold: 文献留出，因清洗数据表中缺失真实 DOI 列，明确标记为 BLOCKED (绝不使用行号分桶伪造)。
2. 固化划分清单至 SPLIT_MANIFEST.json，包含样本索引、配方哈希、折叠归属与内容 SHA256。
"""

import os
import sys
import json
import hashlib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold

class SplitManager:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def generate_splits(
        self,
        df: pd.DataFrame,
        target_name: str,
        group_col: str = "formulation_id",
        family_col: Optional[str] = None,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        生成严格的评估划分方案
        """
        n_samples = len(df)
        indices = np.arange(n_samples)

        # 1. Random 5-Fold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        random_splits = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df)):
            random_splits.append({
                "fold": int(fold_idx),
                "train_indices": [int(i) for i in train_idx],
                "test_indices": [int(i) for i in test_idx],
                "train_count": len(train_idx),
                "test_count": len(test_idx)
            })

        # 2. Formulation Group 5-Fold (配方严格留出)
        unique_groups = df[group_col].nunique()
        actual_group_splits = min(n_splits, unique_groups)
        gkf = GroupKFold(n_splits=actual_group_splits)
        
        group_splits = []
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df[group_col])):
            train_groups = set(df.iloc[train_idx][group_col])
            test_groups = set(df.iloc[test_idx][group_col])
            overlap = train_groups.intersection(test_groups)
            assert len(overlap) == 0, f"Critical Leakage: Groups overlap across folds: {overlap}"

            group_splits.append({
                "fold": int(fold_idx),
                "train_indices": [int(i) for i in train_idx],
                "test_indices": [int(i) for i in test_idx],
                "train_groups_count": len(train_groups),
                "test_groups_count": len(test_groups),
                "train_count": len(train_idx),
                "test_count": len(test_idx),
                "leakage_overlap_count": len(overlap)
            })

        # 3. 结构家族外推留出 (如果提供了 family_col 且组数足够)
        family_splits = []
        if family_col and family_col in df.columns:
            u_fams = df[family_col].nunique()
            if u_fams >= 3:
                f_splits = min(4, u_fams)
                fgkf = GroupKFold(n_splits=f_splits)
                for fold_idx, (train_idx, test_idx) in enumerate(fgkf.split(df, groups=df[family_col])):
                    train_fams = set(df.iloc[train_idx][family_col])
                    test_fams = set(df.iloc[test_idx][family_col])
                    family_splits.append({
                        "fold": int(fold_idx),
                        "train_indices": [int(i) for i in train_idx],
                        "test_indices": [int(i) for i in test_idx],
                        "test_families": list(test_fams),
                        "train_count": len(train_idx),
                        "test_count": len(test_idx)
                    })

        # 4. 文献留出 (因缺失 DOI 列，严格标记为 BLOCKED)
        paper_held_out_status = {
            "status": "BLOCKED",
            "reason": "当前数据清洗表中缺失原始文献 DOI/论文标识字段，禁止使用任何行号分桶或伪造分组进行文献留出评估，正式标记为 BLOCKED。"
        }

        # 计算数据哈希
        data_hash = hashlib.sha256(df[group_col].astype(str).str.cat().encode("utf-8")).hexdigest()

        manifest = {
            "target_name": target_name,
            "total_samples": n_samples,
            "unique_formulations_count": unique_groups,
            "data_group_hash": data_hash,
            "random_5fold": random_splits,
            "formulation_group_5fold": group_splits,
            "structural_family_group_splits": family_splits,
            "paper_group_held_out": paper_held_out_status
        }
        return manifest
