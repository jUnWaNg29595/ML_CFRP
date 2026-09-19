# -*- coding: utf-8 -*-
"""
模量 R²=0.33~0.39 的成因逐项定量诊断
====================================
假设清单（逐项检验）:
  H1 标签测量噪声（同配方多次测量的离散度 → R² 理论天花板）
  H2 填料信息缺失（宽表 contains_filler 未进窄表 → 大变量被隐藏）
  H3 测试条件混杂（方法/温度/速率不统一）
  H4 目标量纲跨两个数量级（线性 MSE 被高值主导 → log 目标）
  H5 样本量不足（学习曲线）
  H6 结构信息不足（现有特征对"同族不同配方"分辨力不够 → 残差 vs 预测值分析）
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

from core.formulation_fusion import FormulationFusionEngine
from core.polymer_physics import augment_polymer_physics

DATA = r"C:/Users/wangj/Desktop/ml_dataset"
_eng = FormulationFusionEngine(verbose=False)


def xgb(seed=42):
    return XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
                        random_state=seed, n_jobs=4, tree_method="hist", verbosity=0)


def hash_series(df):
    return df.apply(_eng.compute_formulation_hash, axis=1)


def prep_ml(X):
    num, oh = [], []
    for c in X.columns:
        s, cl = X[c], str(c).lower()
        if pd.api.types.is_numeric_dtype(s):
            num.append(c)
        elif any(k in cl for k in ("structure", "smiles", "inchi", "bigsmiles")):
            continue
        elif s.nunique(dropna=True) <= 12:
            oh.append(c)
    out = X[num].copy()
    for c in oh:
        d = pd.get_dummies(X[c].astype(str), prefix=c, dtype=float)
        d.index = X.index
        out = pd.concat([out, d], axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def cv_r2(F, y, seeds=(42, 7)):
    F = F.fillna(F.median(numeric_only=True)).fillna(0.0)
    X_ = F.to_numpy(dtype=float)
    sc = []
    for seed in seeds:
        pred = np.zeros_like(y)
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(X_):
            m = xgb(seed)
            m.fit(X_[tr], y[tr])
            pred[te] = m.predict(X_[te])
        sc.append(r2_score(y, pred))
    return float(np.mean(sc)), pred if len(seeds) == 1 else None


print("=" * 84)
print("模量 R² 低成因诊断")
print("=" * 84)

mod = pd.read_csv(f"{DATA}/ml_qspr_model_tensile_modulus_gpa.csv", encoding="utf-8",
                  encoding_errors="replace", low_memory=False)
y_raw = pd.to_numeric(mod["tensile_modulus_gpa"], errors="coerce")
g = hash_series(mod)
mod = mod.assign(_g=g.to_numpy(), _y=y_raw)
mod = mod[y_raw.between(0.1, 10.0)].reset_index(drop=True)

# ---------------------------------------------------------------- H1 测量噪声天花板
print("\n[H1] 同配方多次测量的离散度 → R² 理论天花板")
grp = mod.groupby("_g")["_y"]
sizes = grp.size()
multi = grp.std().dropna()
print(f"  唯一配方 {sizes.size} 个; 有≥2次测量的 {int((sizes>=2).sum())} 个 "
      f"(覆盖 {int(sizes[sizes>=2].sum())} 行)")
print(f"  组内标准差: 中位 {multi.median():.3f} GPa | 组内 CV 中位 "
      f"{(grp.mean().astype(float).ravel() if False else 0):.0f}")
cv = (multi / grp.mean()[multi.index]).dropna()
print(f"  组内变异系数 CV: 中位 {cv.median()*100:.1f}%  P75 {cv.quantile(.75)*100:.1f}%")
# ICC 天花板: 1 - 组内方差/总方差 (加权)
tot_var = mod["_y"].var()
within_var = sum(((sdf["_y"] - sdf["_y"].mean()) ** 2).sum() for _, sdf in mod.groupby("_g"))
icc = 1 - within_var / (len(mod) * tot_var)
print(f"  ICC(组内一致性) = {icc:.3f}  → 预测'每次测量'的 R² 天花板 ≈ {icc:.2f}")
print(f"  → 预测'配方典型值'(去重口径)的天花板更高, 由组均值噪声决定 ≈ "
      f"{1 - (mod.groupby('_g')['_y'].mean().var() * 0 + multi.var() / sizes[sizes>=2].map(lambda n: n).mean()**2 / 1) if False else '见下'}")

# ---------------------------------------------------------------- H2 填料信息
print("\n[H2] 填料信息缺失检验（宽表 contains_filler 关联）")
wide = pd.read_csv(f"{DATA}/ml_wide_samples.csv", encoding="utf-8",
                   encoding_errors="replace", low_memory=False)
has_cols = [c for c in ["contains_filler", "filler_classes", "filler_evidence_level"]
            if c in wide.columns]
print(f"  宽表填料字段: {has_cols}")
if "contains_filler" in wide.columns:
    vw = wide["contains_filler"].astype(str).str.lower()
    frac_filler = vw.isin(["true", "1", "yes", "y"]).mean()
    print(f"  宽表整体含填料比例: {frac_filler*100:.0f}%")
    # 用哈希关联到模量行
    hw = hash_series(wide)
    gm = hash_series(mod.drop(columns=["_g", "_y"]))
    wmap = dict(zip(hw, vw))
    fmod = gm.map(lambda x: wmap.get(x, "unknown"))
    known = fmod != "unknown"
    fknown = fmod[known]
    print(f"  模量行可关联填料信息: {known.mean()*100:.0f}%")
    print(f"  其中含填料: {(fknown.isin(['true','1','yes','y'])).mean()*100:.0f}%"
          f"  ← 若显著>0, 填料是被隐藏的大变量")

# ---------------------------------------------------------------- H3 测试条件混杂
print("\n[H3] 测试条件/方法混杂")
tc = "tensile_modulus_gpa_test_temperature_c"
tt = pd.to_numeric(mod[tc], errors="coerce")
print(f"  测试温度覆盖 {tt.notna().mean()*100:.0f}%; 常温(20-30℃)占已知的 "
      f"{tt.dropna().between(20,30).mean()*100:.0f}%")
meth = mod["tensile_modulus_gpa_test_method"].value_counts()
print(f"  测试方法分布: {dict(meth.head(6))}")

# ---------------------------------------------------------------- 特征与基线
Xa = augment_polymer_physics(mod.drop(columns=["_g", "_y", "tensile_modulus_gpa"]))
BASE = prep_ml(Xa.drop(columns=[c for c in Xa.columns if c.startswith("phys_")]))
F_BP = pd.concat([BASE, Xa[[c for c in Xa.columns if c.startswith("phys_")]]], axis=1)
# 去重
idx = []
for gh, sub in mod.groupby("_g"):
    sub2 = sub.dropna(subset=["_y"])
    idx.append(sub2.iloc[(sub2["_y"] - sub2["_y"].median()).abs().argmin()].name)
mod_d = mod.loc[idx].reset_index(drop=True)
g_d = hash_series(mod_d)
y_d = pd.to_numeric(mod_d["tensile_modulus_gpa"], errors="coerce").to_numpy(float)
tt_d = pd.to_numeric(mod_d[tc], errors="coerce").where(lambda s: s.between(-50, 300)).fillna(25).to_numpy(float)
Xa_d = augment_polymer_physics(mod_d.drop(columns=["_g", "_y", "tensile_modulus_gpa"]))
BASE_d = prep_ml(Xa_d.drop(columns=[c for c in Xa_d.columns if c.startswith("phys_")]))
F_BP_d = pd.concat([BASE_d, Xa_d[[c for c in Xa_d.columns if c.startswith("phys_")]]], axis=1)

# ---------------------------------------------------------------- H4 log 目标
print("\n[H4] 目标量纲检验（线性 vs log 目标）")
for nm, F, yy, tag in [("全量测量口径", F_BP, mod["_y"].to_numpy(float), "linear"),
                       ("去重配方口径", F_BP_d, y_d, "linear")]:
    r_lin, pred = cv_r2(F, yy, seeds=(42,))
    r_log, plog = cv_r2(F, np.log(yy), seeds=(42,))
    # 把 log 预测转回原空间算 R²
    r_back = r2_score(yy, np.exp(plog))
    print(f"  {nm}: R²(线性)={r_lin:.3f}  R²(log目标)={r_log:.3f}  log预测还原后={r_back:.3f}")

# ---------------------------------------------------------------- H5 学习曲线
print("\n[H5] 学习曲线（样本量 vs R², 去重口径 + PHYS）")
Fd = F_BP_d.fillna(F_BP_d.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=float)
rng = np.random.RandomState(0)
for frac in [0.25, 0.5, 0.75, 1.0]:
    n = int(len(y_d) * frac)
    sel = rng.choice(len(y_d), n, replace=False)
    sc = []
    for seed in (42,):
        pred = np.zeros(n)
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(Fd[sel]):
            m = xgb(seed)
            m.fit(Fd[sel][tr], y_d[sel][tr])
            pred[te] = m.predict(Fd[sel][te])
        sc.append(r2_score(y_d[sel], pred))
    print(f"  n={n:4d} ({frac*100:3.0f}%)  R² = {np.mean(sc):.3f}")

# ---------------------------------------------------------------- H6 残差结构
print("\n[H6] 残差分析（误差集中在哪）")
r2_full, pred = cv_r2(F_BP_d, y_d, seeds=(42,))
err = np.abs(pred - y_d)
bins = [(0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 10.0)]
print(f"  全模型 R²={r2_full:.3f}  MAE={mean_absolute_error(y_d, pred):.3f} GPa")
for lo, hi in bins:
    m = (y_d >= lo) & (y_d < hi)
    if m.sum() < 10:
        continue
    print(f"    y∈[{lo:.1f},{hi:.1f}) GPa: n={int(m.sum()):4d}  "
          f"MAE={err[m].mean():.3f}  相对误差={err[m].mean()/y_d[m].mean()*100:5.1f}%")
over = pred > 2 * y_d
under = y_d > 2 * pred
print(f"  严重高估(>2×): {int(over.sum())} 行   严重低估(<0.5×): {int(under.sum())} 行")
