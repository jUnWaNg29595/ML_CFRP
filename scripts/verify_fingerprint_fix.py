# -*- coding: utf-8 -*-
"""回归验证: extract_fingerprints 返回 (df, valid_indices) 后，
批量循环的 sub_valid_idx 映射能通过 feature contract 校验。
模拟用户场景: 954 个有效 SMILES 中有 1 条解析失败的 BigSMILES (953 != 954)。"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from core.molecular_features import extract_fingerprints
from core.molecular_feature_workflow import validate_feature_frame_contract, align_extracted_features_to_rows

# 1) 构造 954 行: 953 条有效 + 1 条化学无效 BigSMILES（七元芳环, RDKit 无法 Kekulize）
good = "C1CC2CC1C(C2)O"  # 简单脂肪族, 稳定可解析
smiles_list = []
for i in range(954):
    if i == 18:
        smiles_list.append("c1cccccc1")  # 七元芳环, 不满足 Hückel 规则, repair 链也修不好
    else:
        smiles_list.append(good)

# 模拟批量循环的 source_valid_indices: 全部非空 => 954 个
valid_indices = [i for i, s in enumerate(smiles_list) if str(s).strip()]
assert len(valid_indices) == 954, f"valid_indices={len(valid_indices)}"
valid_smiles = [smiles_list[i] for i in valid_indices]

# 2) 修复后: 便捷函数返回 (df, sub_valid_idx)
features_df, sub_valid_idx = extract_fingerprints(valid_smiles, fp_type="MACCS", n_bits=167, radius=2)
print(f"features_df rows = {len(features_df)}, sub_valid_idx = {len(sub_valid_idx)}")
assert len(features_df) == len(sub_valid_idx), "便捷函数返回的 df 与 valid_indices 行数不一致!"

# 3) 批量循环的映射逻辑
extracted_valid_indices = valid_indices  # 默认
if len(features_df) > 0 and sub_valid_idx:
    extracted_valid_indices = [valid_indices[i] for i in sub_valid_idx]
print(f"extracted_valid_indices = {len(extracted_valid_indices)}")

# 4) contract 校验: 行数必须一致 (修复前这里报 953 != 954)
errors = validate_feature_frame_contract(features_df.reset_index(drop=True), extracted_valid_indices, 954)
if errors:
    print("CONTRACT ERRORS:", errors)
    sys.exit(1)
print("✓ feature contract 校验通过 (953 行特征 == 953 个有效行索引)")

# 5) 回填后: 无效行 (原行18) 应为全 NaN, 其余 953 行有值
full = align_extracted_features_to_rows(features_df.reset_index(drop=True), extracted_valid_indices, 954)
assert len(full) == 954
row18 = full.iloc[18]
row19 = full.iloc[19]
assert row18.isna().all(), "行18 应全为 NaN (解析失败被跳过)"
assert row19.notna().any(), "行19 应有指纹值"
print(f"✓ 回填: 行18 全 NaN ✓, 行19 有值 (非零列数={int(row19.notna().sum())})")

# 6) 无效行占比极小, 不再中断批处理
print(f"✓ 总行数 954, 成功提取 {len(features_df)} 行, 跳过 {954 - len(features_df)} 行 — 批处理不再中断")
print("ALL CHECKS PASSED")
