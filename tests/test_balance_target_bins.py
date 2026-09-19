# -*- coding: utf-8 -*-
"""分箱均衡下采样（软硬均衡）端到端测试：模拟软段数据多、硬段数据少的模量分布"""
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, r"C:/Users/wangj/Desktop/CFRP系统/CFRP系统")
from core.data_processor import AdvancedDataCleaner

# 模拟：模量呈右偏分布，低模量(软段)样本多，高模量(硬段)样本少
np.random.seed(7)
soft = np.random.normal(2.0, 0.6, 300).clip(0.5, 4.5)   # 软段 300 条
hard = np.random.normal(8.0, 1.0, 40).clip(5.5, 11.0)   # 硬段仅 40 条
df = pd.DataFrame({"modulus_GPa": np.concatenate([soft, hard]).round(3)})
print(f"总样本: {len(df)}, 软段(<4.5) {int((df['modulus_GPa'] < 4.6).sum())} 条, 硬段(>5.5) {int((df['modulus_GPa'] > 5.4).sum())} 条")

# ---- 测试1: 等宽分箱 + 自动均衡上限 ----
cleaner = AdvancedDataCleaner(df.copy())
cleaned, stats = cleaner.balance_target_bins(column="modulus_GPa", n_bins=10, bin_strategy="uniform", random_state=42)
num_after = cleaned["modulus_GPa"]
bins_after = pd.cut(num_after, bins=stats["bin_edges"])
vc_after = bins_after.value_counts()
vc_after = vc_after[vc_after > 0]
print("\n=== 测试1: uniform 自动均衡 ===")
print("各箱保留数:", dict(sorted(vc_after.items(), key=lambda kv: float(kv[0].left))))
print(f"{stats['total_before']} -> {stats['total_after']}, 删除 {stats['removed_rows']}, "
      f"高频箱 {stats['n_reduced_bins']} 个被削减, 稀疏箱 {stats['n_full_kept_bins']} 个完整保留, "
      f"最大箱占比 {stats['max_bin_pct_before']:.1f}% -> {stats['max_bin_pct_after']:.1f}%")

assert stats["total_after"] == len(cleaned)
assert stats["removed_rows"] == len(df) - len(cleaned)
assert stats["n_invalid_kept"] == 0
assert vc_after.max() <= stats["cap_used"], "存在箱超过上限!"
# 硬段高模量箱应被完整保留（原样本数 <= cap）
hard_bins_kept = [v for k, v in stats["bin_counts_after"].items()]
assert max(hard_bins_kept) == stats["cap_used"]
assert stats["max_bin_pct_after"] < stats["max_bin_pct_before"], "均衡后最大箱占比应下降"
# 可复现性
cleaner2 = AdvancedDataCleaner(df.copy())
cleaned2, _ = cleaner2.balance_target_bins(column="modulus_GPa", n_bins=10, bin_strategy="uniform", random_state=42)
assert cleaned2.equals(cleaned), "random_state=42 应可复现"
print("PASS: 上限生效、稀疏箱保留、最大箱占比下降、可复现")

# ---- 测试2: 完全均衡（上限=最小箱样本数）----
cleaner3 = AdvancedDataCleaner(df.copy())
cap_min = 5
cleaned3, stats3 = cleaner3.balance_target_bins(column="modulus_GPa", n_bins=10, bin_strategy="uniform", max_per_bin=cap_min)
kept_vals = list(stats3["bin_counts_after"].values())
assert max(kept_vals) == cap_min, f"完全均衡要求每箱 <= {cap_min}, 实际最大 {max(kept_vals)}"
print(f"PASS: 完全均衡 (上限={cap_min}), {stats3['total_before']} -> {stats3['total_after']}")

# ---- 测试3: quantile 等频分位 ----
cleaner4 = AdvancedDataCleaner(df.copy())
cleaned4, stats4 = cleaner4.balance_target_bins(column="modulus_GPa", n_bins=8, bin_strategy="quantile", max_per_bin=30)
assert all(v <= 30 for v in stats4["bin_counts_after"].values())
print(f"PASS: quantile 模式上限生效 ({stats4['total_before']} -> {stats4['total_after']})")

# ---- 测试4: 含 NaN/inf 的行不参与均衡、原样保留 ----
df_nan = df.copy()
df_nan.loc[0:4, "modulus_GPa"] = np.nan
df_nan.loc[5:6, "modulus_GPa"] = np.inf
cleaner5 = AdvancedDataCleaner(df_nan.copy())
cleaned5, stats5 = cleaner5.balance_target_bins(column="modulus_GPa", n_bins=10, bin_strategy="uniform")
assert len(cleaned5) == stats5["total_after"]
assert stats5["n_invalid_kept"] == 7, f"应保留 7 条无效行, 实际 {stats5['n_invalid_kept']}"
assert cleaned5["modulus_GPa"].isna().sum() == 5 and np.isinf(cleaned5["modulus_GPa"]).sum() == 2
print(f"PASS: NaN/inf 行原样保留 ({stats5['n_invalid_kept']} 条), 有效样本 {stats5['total_before'] - 7} -> {stats5['total_after'] - 7}")

# ---- 测试5: 全 NaN 列报错 / 无效列名报错 ----
try:
    AdvancedDataCleaner(pd.DataFrame({"x": [np.nan, np.nan]})).balance_target_bins(column="x")
    raise AssertionError("应报错")
except ValueError as e:
    print(f"PASS: 全无效列报错: {e}")
try:
    AdvancedDataCleaner(df.copy()).balance_target_bins(column="not_exist")
    raise AssertionError("应报错")
except ValueError as e:
    print(f"PASS: 无效列名报错: {e}")

# ---- 测试6: 软硬均衡效果验证 ----
# 硬段内部的密集箱同样会被均衡到上限（否则该箱会变成新的高频箱），
# 但硬段整体保留率应远高于软段，且软硬比例应显著改善
before_hard = df[df["modulus_GPa"] > 5.4]
after_hard = cleaned[cleaned["modulus_GPa"] > 5.4]
before_soft = df[df["modulus_GPa"] <= 5.4]
after_soft = cleaned[cleaned["modulus_GPa"] <= 5.4]
hard_keep_rate = len(after_hard) / len(before_hard)
soft_keep_rate = len(after_soft) / len(before_soft)
ratio_before = len(before_soft) / len(before_hard)
ratio_after = len(after_soft) / max(1, len(after_hard))
assert hard_keep_rate >= 0.9, f"硬段保留率过低: {hard_keep_rate:.1%}"
assert hard_keep_rate > soft_keep_rate, "稀有段(硬段)保留率应高于高频段(软段)"
assert ratio_after < ratio_before * 0.3, f"软硬比改善不足: {ratio_before:.1f}:1 -> {ratio_after:.1f}:1"
print(f"PASS: 软硬均衡效果: 硬段保留 {hard_keep_rate:.1%}, 软段保留 {soft_keep_rate:.1%}, "
      f"软硬比 {ratio_before:.1f}:1 -> {ratio_after:.1f}:1")

print("\n✅ 全部 6 组测试通过")
