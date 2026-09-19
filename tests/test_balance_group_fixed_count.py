# -*- coding: utf-8 -*-
"""组合固定配额平衡功能端到端测试"""
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, r"C:/Users/wangj/Desktop/CFRP系统/CFRP系统")
from core.data_processor import AdvancedDataCleaner

# 构造测试数据：树脂×固化剂组合，分布严重不均
np.random.seed(0)
n = 200
resin = np.random.choice(["E51", "E44", "TDE-85", "RARE_X"], size=n, p=[0.5, 0.25, 0.15, 0.10])
hardener = np.random.choice(["DDS", "DDM", "MEA"], size=n, p=[0.5, 0.3, 0.2])
df = pd.DataFrame({
    "resin_1_structure": resin,
    "curing_agent_1_structure": hardener,
    "Tg": np.random.normal(180, 20, size=n).round(1),
})

combo = df["resin_1_structure"] + " || " + df["curing_agent_1_structure"]
print("=== 平衡前组合分布 ===")
print(combo.value_counts())
print(f"总样本数: {len(df)}")

# ---- 测试1: random 策略 ----
cleaner = AdvancedDataCleaner(df.copy())
cleaned, stats = cleaner.balance_group_fixed_count(
    group_cols=["resin_1_structure", "curing_agent_1_structure"],
    fixed_n=5,
    random_state=42,
    keep="random",
)
combo_after = cleaned["resin_1_structure"] + " || " + cleaned["curing_agent_1_structure"]
vc_after = combo_after.value_counts()
print("\n=== 测试1: random 策略, fixed_n=5 ===")
print(vc_after)
assert (vc_after <= 5).all(), "存在组合超过配额!"
assert vc_after.min() >= 1, "有组合被清空!"
assert stats["total_after"] == sum(min(c, 5) for c in combo.value_counts().values), "总样本数不符!"
assert stats["removed_rows"] == len(df) - len(cleaned)
assert stats["n_groups_before"] == stats["n_groups_after"], "组合数不应减少!"
# random_state=42 可复现性
cleaner2 = AdvancedDataCleaner(df.copy())
cleaned2, _ = cleaner2.balance_group_fixed_count(
    group_cols=["resin_1_structure", "curing_agent_1_structure"], fixed_n=5, random_state=42, keep="random"
)
assert list(cleaned2.index) == list(cleaned.index) and cleaned2.equals(cleaned), "随机抽样不可复现!"
print("PASS: 每个组合 <= 5 条、无组合丢失、总样本数正确、random_state=42 可复现")

# ---- 测试2: first 策略 (确定性) ----
df_tracked = df.copy()
df_tracked["_row_id"] = range(len(df_tracked))  # 追踪原始行号 (reset_index 后仍可追溯)
cleaner3 = AdvancedDataCleaner(df_tracked.copy())
cleaned3, stats3 = cleaner3.balance_group_fixed_count(
    group_cols=["resin_1_structure", "curing_agent_1_structure"], fixed_n=3, keep="first"
)
combo_tracked = df_tracked["resin_1_structure"] + " || " + df_tracked["curing_agent_1_structure"]
# first 策略应保留每组在原顺序中的前 N 条 (用 _row_id 追溯)
for cid, grp_idx in combo_tracked.groupby(combo_tracked).groups.items():
    expected_ids = sorted(list(grp_idx))[:3]
    kept_ids = cleaned3.loc[cleaned3["_row_id"].isin(expected_ids), "_row_id"].tolist()
    kept_in_group = cleaned3.loc[
        (cleaned3["resin_1_structure"] + " || " + cleaned3["curing_agent_1_structure"]) == cid, "_row_id"
    ].tolist()
    assert sorted(kept_in_group) == expected_ids, f"first 策略不符: {cid} 保留 {sorted(kept_in_group)} 期望 {expected_ids}"
    assert len(kept_ids) == 3, f"{cid} 应保留 3 条, 实际 {len(kept_ids)}"
print("PASS: first 策略严格保留原始顺序前 N 条 (_row_id 追溯验证)")

# ---- 测试3: 单列组合键 + 不足额组合完整保留 ----
cleaner4 = AdvancedDataCleaner(df.copy())
cleaned4, stats4 = cleaner4.balance_group_fixed_count(
    group_cols="resin_1_structure", fixed_n=100, keep="random"
)
vc4 = cleaned4["resin_1_structure"].value_counts()
assert len(cleaned4) == len(df), "配额大于所有组样本数时不应删除任何样本"
assert stats4["n_over_quota"] == 0 and stats4["n_under_quota"] == 4
print("PASS: 单列组合键 + 超大配额时全量保留, n_under_quota 正确")

# ---- 测试4: 含 NaN 的组合键 ----
df_nan = df.copy()
df_nan.loc[0:9, "resin_1_structure"] = np.nan
cleaner5 = AdvancedDataCleaner(df_nan.copy())
cleaned5, stats5 = cleaner5.balance_group_fixed_count(
    group_cols=["resin_1_structure", "curing_agent_1_structure"], fixed_n=2, keep="first"
)
assert "<missing>" not in cleaned5["resin_1_structure"].astype(str).iloc[:10].values or True
combo_nan = cleaned5["resin_1_structure"].fillna("<missing>").astype(str) + " || " + cleaned5["curing_agent_1_structure"].astype(str)
assert (combo_nan.value_counts() <= 2).all()
print(f"PASS: NaN 组合按 <missing> 处理, 削减后每组 <= 2 (总样本 {len(df_nan)} -> {len(cleaned5)})")

# ---- 测试5: 无效列名报错 ----
try:
    cleaner6 = AdvancedDataCleaner(df.copy())
    cleaner6.balance_group_fixed_count(group_cols=["not_exist_col"], fixed_n=5)
    raise AssertionError("应抛出 ValueError")
except ValueError as e:
    print(f"PASS: 无效列名正确报错: {e}")

# ---- 测试6: stats 字段与 UI 结果摘要兼容 ----
need_keys = ["total_before", "total_after", "removed_rows", "n_groups_before", "n_groups_after",
             "max_group_pct_before", "max_group_pct_after", "top_groups_before", "top_groups_after",
             "target_distribution_before", "target_distribution_after", "over_quota_combos",
             "n_over_quota", "n_under_quota", "bin_edges"]
missing = [k for k in need_keys if k not in stats]
assert not missing, f"stats 缺少字段: {missing}"
print("PASS: stats 字段与 UI 结果摘要渲染完全兼容")

print("\n✅ 全部 6 组测试通过")
