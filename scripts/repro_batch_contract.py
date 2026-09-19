# -*- coding: utf-8 -*-
"""复现批量提取流程(3D分支) + 特征契约校验, 抓出真实异常"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

DATA = "data_fixed/latest_data_fixed.csv"
df = pd.read_csv(DATA)
print(f"数据 {df.shape[0]} 行")

from core.molecular_features import RDKit3DDescriptorExtractor
from core.molecular_feature_workflow import validate_feature_frame_contract

# 模拟 UI: 用户可能一次勾选多个 SMILES 列批量提取
batch_cols = ["resin_1_structure", "resin_2_structure", "resin_3_structure",
              "curing_agent_1_structure", "curing_agent_2_structure"]

for col in batch_cols:
    smiles_list_input = df[col].tolist()
    valid_indices = [i for i, s in enumerate(smiles_list_input)
                     if s is not None and not (isinstance(s, float) and np.isnan(s)) and str(s).strip()]
    valid_smiles = [smiles_list_input[i] for i in valid_indices]
    print(f"\n=== {col}: {len(valid_indices)} 有效样本 ===")

    try:
        extractor = RDKit3DDescriptorExtractor(coulomb_top_k=20)
        feat_df_3d, sub_valid_idx = extractor.smiles_to_3d_descriptors(valid_smiles, n_jobs=1)
        print(f"3D 提取: feat_df {feat_df_3d.shape}, sub_valid_idx {len(sub_valid_idx)}")
        if len(feat_df_3d) > 0:
            extracted_valid_indices = [valid_indices[i] for i in sub_valid_idx]
            features_df = feat_df_3d.reset_index(drop=True)
            errs = validate_feature_frame_contract(features_df, extracted_valid_indices, len(df))
            if errs:
                print("❌ 契约违规:")
                for e in errs:
                    print("   -", e)
            else:
                print("✅ 契约通过")
        else:
            print("❌ feature contract violation: no features extracted (3D 返回空)")
    except Exception as e:
        import traceback
        print(f"❌ 异常: {type(e).__name__}: {e}")
        traceback.print_exc()
