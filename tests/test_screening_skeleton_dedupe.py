"""
测试配方筛选中的候选化学骨架去重与多样性筛选特征对齐。
"""
import pandas as pd
import numpy as np
import pytest


def test_skeleton_dedupe_keeps_highest_scoring_condition():
    """验证化学骨架去重：对同一种树脂+固化剂组合，即使在多种温度/时间/配比下展开，仅保留评分最高的一行。"""
    # 模拟 2 个化学分子对，每对展开了 3 种工艺条件
    df = pd.DataFrame({
        "formulation_id": [1, 2, 3, 4, 5, 6],
        "resin_smiles": ["CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"] * 3 + ["c1cc(OCC2CO2)ccc1Cc3ccc(OCC4CO4)cc3"] * 3,
        "hardener_smiles": ["Nc1ccc(Cc2ccc(N)cc2)cc1"] * 3 + ["Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1"] * 3,
        "process_temp": [150.0, 180.0, 200.0, 160.0, 180.0, 210.0],
        "prediction": [160.0, 185.0, 182.0, 210.0, 240.0, 235.0],
        "total_score": [70.0, 95.0, 92.0, 80.0, 98.0, 93.0],
    })

    # 按总分降序排列
    sorted_df = df.sort_values(["total_score", "prediction"], ascending=[False, False])

    # 执行化学骨架去重
    dedupe_cols = ["resin_smiles", "hardener_smiles"]
    deduped_df = sorted_df.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)

    assert len(deduped_df) == 2
    # 第一名应该是 (BPF + DDS) 在 180°C 的最优条件 (score=98.0, pred=240.0)
    assert deduped_df.iloc[0]["process_temp"] == 180.0
    assert deduped_df.iloc[0]["total_score"] == 98.0
    # 第二名应该是 (DGEBA + DDM) 在 180°C 的最优条件 (score=95.0, pred=185.0)
    assert deduped_df.iloc[1]["process_temp"] == 180.0
    assert deduped_df.iloc[1]["total_score"] == 95.0
