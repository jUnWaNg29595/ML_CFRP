# -*- coding: utf-8 -*-
"""正态分布整形与特定区间处理端到端测试"""
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, r"C:/Users/wangj/Desktop/CFRP系统/CFRP系统")
from core.data_processor import AdvancedDataCleaner

# 模拟右偏分布：软段(低模量)数据多，硬段(高模量)少
np.random.seed(7)
soft = np.random.normal(2.0, 0.6, 300).clip(0.5, 4.5)
hard = np.random.normal(8.0, 1.0, 40).clip(5.5, 11.0)
df = pd.DataFrame({"modulus_GPa": np.concatenate([soft, hard]).round(3)})

# ================= shape_to_normal =================
print("=== shape_to_normal 测试 ===")
cleaner = AdvancedDataCleaner(df.copy())
cleaned, stats = cleaner.shape_to_normal(column="modulus_GPa", n_bins=15, random_state=42)
print(f"{stats['total_before']} -> {stats['total_after']} (删 {stats['removed_rows']}), "
      f"mu={stats['mu']:.3f}, sigma={stats['sigma']:.3f}, peak={stats['peak_samples']}")

# 钟形验证：用 plan 表的箱中心与保留数，按距 μ 的距离分组比较
mu = stats["mu"]
plan_df = pd.DataFrame(stats["plan"])
plan_df["dist"] = (plan_df["箱中心"] - mu).abs()
near = plan_df[plan_df["dist"] <= stats["sigma"]]["计划保留"]
far = plan_df[plan_df["dist"] > 2 * stats["sigma"]]["计划保留"]
avg_near = float(near.mean()) if len(near) else 0.0
avg_far = float(far.mean()) if len(far) else 0.0
print(f"中心区(≤1σ)平均保留 {avg_near:.1f} 条/箱, 远端(>2σ)平均保留 {avg_far:.1f} 条/箱")
assert avg_near > avg_far * 2, f"钟形不明显: 近 {avg_near} vs 远 {avg_far}"
assert stats["total_after"] < stats["total_before"], "应有削减"
# 只减不增 + 目标公式逐箱验证：计划保留 == min(原样本数, ceil(peak × 权重))
assert (plan_df["计划保留"] <= plan_df["原样本数"]).all(), "出现超出原样本数的箱!"
_expected = np.minimum(
    plan_df["原样本数"].to_numpy(),
    np.maximum(1, np.ceil(stats["peak_samples"] * plan_df["正态权重"].to_numpy()).astype(int)),
)
assert (plan_df["计划保留"].to_numpy() == _expected).all(), "存在箱的保留数不符合正态目标公式"
# 可复现
cleaner2 = AdvancedDataCleaner(df.copy())
cleaned2, _ = cleaner2.shape_to_normal(column="modulus_GPa", n_bins=15, random_state=42)
assert cleaned2.equals(cleaned), "random_state=42 应可复现"
# plan 表完整性
assert all({"分箱区间", "正态权重", "原样本数", "计划保留"} <= set(r) for r in stats["plan"])
print("PASS: 钟形形态成立、只减不增、中心峰值生效、可复现、plan 完整")

# 自定义 mu/sigma/peak
cleaner3 = AdvancedDataCleaner(df.copy())
cleaned3, stats3 = cleaner3.shape_to_normal(column="modulus_GPa", n_bins=15, mu=2.0, sigma=1.0, peak_samples=30)
assert stats3["mu"] == 2.0 and stats3["sigma"] == 1.0 and stats3["peak_samples"] == 30
assert max(stats3["bin_counts_after"].values()) <= 30
print(f"PASS: 自定义 μ=2.0, σ=1.0, peak=30 生效 ({stats3['total_before']} -> {stats3['total_after']})")

# ================= treat_range =================
print("\n=== treat_range 测试 ===")
lo, hi = 1.0, 4.0
n_in = int(((df["modulus_GPa"] >= lo) & (df["modulus_GPa"] <= hi)).sum())
n_out = len(df) - n_in

# 测试1: remove 模式
c4 = AdvancedDataCleaner(df.copy())
cl4, st4 = c4.treat_range(column="modulus_GPa", lower=lo, upper=hi, mode="remove")
assert len(cl4) == n_out and st4["removed_in_range"] == n_in
assert not ((cl4["modulus_GPa"] >= lo) & (cl4["modulus_GPa"] <= hi)).any(), "区间内未删净"
assert (cl4["modulus_GPa"] > hi).sum() == int((df["modulus_GPa"] > hi).sum()), "区间外硬段被误删!"
print(f"PASS: remove 模式: 区间内 {n_in} 条全删, 区间外 {n_out} 条原样保留")

# 测试2: downsample_ratio 模式
c5 = AdvancedDataCleaner(df.copy())
cl5, st5 = c5.treat_range(column="modulus_GPa", lower=lo, upper=hi, mode="downsample_ratio", ratio=0.3)
kept_in = int(((cl5["modulus_GPa"] >= lo) & (cl5["modulus_GPa"] <= hi)).sum())
expect = max(1, round(n_in * 0.3))
assert kept_in == expect, f"应保留 {expect} 条, 实际 {kept_in}"
assert int((cl5["modulus_GPa"] > hi).sum()) == int((df["modulus_GPa"] > hi).sum()), "区间外被误删!"
c5b = AdvancedDataCleaner(df.copy())
cl5b, _ = c5b.treat_range(column="modulus_GPa", lower=lo, upper=hi, mode="downsample_ratio", ratio=0.3, random_state=42)
assert cl5b.equals(cl5), "ratio 模式应可复现"
print(f"PASS: ratio=30% 模式: 区间内 {n_in} -> {kept_in}, 区间外不受影响, 可复现")

# 测试3: downsample_bins 模式（区间内均衡，区间外不动）
c6 = AdvancedDataCleaner(df.copy())
cl6, st6 = c6.treat_range(column="modulus_GPa", lower=lo, upper=hi, mode="downsample_bins", n_bins=6, max_per_bin=10)
assert all(v <= 10 for v in st6["bin_counts_after"].values()), "区间内存在箱超上限"
assert int((cl6["modulus_GPa"] > hi).sum()) == int((df["modulus_GPa"] > hi).sum()), "区间外被误删!"
print(f"PASS: bins 模式: 区间内 {n_in} -> {st6['n_in_range'] - st6['removed_in_range']}, 每箱<=10, 区间外不动")

# 测试4: 下限>上限自动交换
c7 = AdvancedDataCleaner(df.copy())
cl7, st7 = c7.treat_range(column="modulus_GPa", lower=hi, upper=lo, mode="remove")
assert st7["range_lower"] == lo and st7["range_upper"] == hi and len(cl7) == n_out
print("PASS: 下限>上限自动交换")

# 测试5: 区间外全保留 + NaN 行保留
df_nan = df.copy()
df_nan.loc[0, "modulus_GPa"] = np.nan
c8 = AdvancedDataCleaner(df_nan.copy())
cl8, st8 = c8.treat_range(column="modulus_GPa", lower=lo, upper=hi, mode="remove")
assert cl8["modulus_GPa"].isna().sum() == 1, "NaN 行应保留"
print("PASS: NaN 行不受区间处理影响")

# 测试6: 无效 mode 报错 / 非法比例报错
try:
    AdvancedDataCleaner(df.copy()).treat_range(column="modulus_GPa", lower=lo, upper=hi, mode="bad_mode")
    raise AssertionError("应报错")
except ValueError as e:
    print(f"PASS: 非法 mode 报错: {e}")
try:
    AdvancedDataCleaner(df.copy()).treat_range(column="modulus_GPa", lower=lo, upper=hi, mode="downsample_ratio", ratio=1.5)
    raise AssertionError("应报错")
except ValueError as e:
    print(f"PASS: 非法 ratio 报错: {e}")

print("\n✅ 全部测试通过")
