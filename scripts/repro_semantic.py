# -*- coding: utf-8 -*-
"""复现 semantic(聚合物语义特征)环节: 计时 + 重复列名 + 契约校验"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

from core.molecular_features import extract_configured_semantic_features
from core.molecular_feature_workflow import validate_feature_frame_contract
from core.molecular_feature_workflow import find_duplicate_feature_names

DATA = "cache/session_snapshots/latest_data.csv"
df = pd.read_csv(DATA)

col = "resin_1_structure"
smiles_list_input = df[col].tolist()
valid_indices = [i for i, s in enumerate(smiles_list_input)
                 if s is not None and not (isinstance(s, float) and np.isnan(s)) and str(s).strip()]

semantic_params_batch = {
    "append_polymer_string_features": True,
    "append_polymer_semantic_features": True,
    "append_ionic_semantic_features": False,
    "bigsmiles_semantic_num_samples": 8,
    "bigsmiles_semantic_min_repeat_units": 1,
    "bigsmiles_semantic_max_repeat_units": 4,
    "bigsmiles_semantic_random_state": 17,
    "preserve_duplicate_columns": True,
}

print(f"{col}: {len(valid_indices)} 有效样本, 开始 semantic 提取...")
t0 = time.time()
semantic_full_df = extract_configured_semantic_features(
    smiles_list_input, semantic_params_batch, prefix="resin_"
)
dt = time.time() - t0
print(f"semantic 完成: {semantic_full_df.shape}, 耗时 {dt:.1f}s")

dup = find_duplicate_feature_names(semantic_full_df.columns.tolist())
print(f"semantic 内部重复列名: {dup[:10] if dup else '无'}")

# 模拟 3D 分支后的 semantic 子集合并
sub = semantic_full_df.iloc[valid_indices].reset_index(drop=True)
print(f"semantic 子集: {sub.shape} (valid_indices {len(valid_indices)})")
errs = validate_feature_frame_contract(sub, valid_indices, len(df))
print("契约:", "✅ 通过" if not errs else f"❌ {errs[:3]}")

# 3D 特征 + semantic 合并后查重
from core.molecular_features import RDKit3DDescriptorExtractor
ex = RDKit3DDescriptorExtractor(coulomb_top_k=20)
valid_smiles = [smiles_list_input[i] for i in valid_indices]
feat3d, svi = ex.smiles_to_3d_descriptors(valid_smiles, n_jobs=1)
ext_idx = [valid_indices[i] for i in svi]
f3 = feat3d.reset_index(drop=True)
sem_sub = semantic_full_df.iloc[ext_idx].reset_index(drop=True)
merged = pd.concat([f3, sem_sub], axis=1)
print(f"3D+semantic 合并: {merged.shape}")
dup2 = find_duplicate_feature_names(list(f3.columns) + list(sem_sub.columns))
print(f"合并后重复列名: {dup2[:10] if dup2 else '无'}")
