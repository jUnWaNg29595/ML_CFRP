# -*- coding: utf-8 -*-
"""
按配方去重（每配方抽 1 个代表行）后的模量特征测试
==================================================
用户方案：每个配方只保留 1 行 → 从根源上消除"同配方重复"泄漏，然后随机 CV。

对照：
  - 去重 + 随机 5 折（3 种子）  ← 本方案
  - 全量 + GroupKFold(5)       ← 上一轮方案（应与去重结果一致，互为验证）

特征集消融：BASE → +PHYS → +Tg(实测关联) → +Tg_pred(OOF, 折内配方组对 Tg 模型不可见)
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
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


def load(t):
    return pd.read_csv(f"{DATA}/ml_qspr_model_{t}.csv", encoding="utf-8",
                       encoding_errors="replace", low_memory=False)


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


def dedup_one_per_group(df, g, y):
    """每配方取 1 个代表行：取 y 最接近组中位数的行（确定性、代表性最好）。"""
    df = df.assign(_g=g.to_numpy(), _y=y)
    pick = []
    for gh, sub in df.groupby("_g"):
        sub = sub.dropna(subset=["_y"])
        if len(sub) == 0:
            continue
        med = sub["_y"].median()
        pick.append(sub.iloc[(sub["_y"] - med).abs().argmin()].name)
    out = df.loc[pick].drop(columns=["_g", "_y"])
    return out


print("=" * 84)
print("去重（每配方 1 行）后的模量特征测试")
print("=" * 84)

# ---------------------------------------------------------------- 数据与去重
mod = load("tensile_modulus_gpa")
y_raw = pd.to_numeric(mod["tensile_modulus_gpa"], errors="coerce")
g_mod = hash_series(mod)
mod = mod.assign(_y=y_raw)
mod = mod[y_raw.between(0.1, 10.0)].reset_index(drop=True)
g_mod = hash_series(mod)
mod_d = dedup_one_per_group(mod, g_mod, mod["_y"]).reset_index(drop=True)
print(f"模量表: {len(mod)} 行(清洗后) → 去重 {len(mod_d)} 个唯一配方")

tg = load("tg_c")
y_tg = pd.to_numeric(tg["tg_c"], errors="coerce")
g_tg = hash_series(tg)
tg = tg.assign(_y=y_tg)
tg = tg[y_tg.notna()].reset_index(drop=True)
g_tg = hash_series(tg)
tg_d = dedup_one_per_group(tg, g_tg, tg["_y"]).reset_index(drop=True)
print(f"Tg  表: {len(tg)} 行 → 去重 {len(tg_d)} 个唯一配方")

# ---------------------------------------------------------------- 特征
y = pd.to_numeric(mod_d["tensile_modulus_gpa"], errors="coerce").to_numpy(float)
g_mod_d = hash_series(mod_d)
tt = pd.to_numeric(mod_d["tensile_modulus_gpa_test_temperature_c"], errors="coerce")
tt = tt.where(tt.between(-50, 300), np.nan).fillna(25.0).to_numpy(float)

Xa = augment_polymer_physics(mod_d.drop(columns=["tensile_modulus_gpa"]))
BASE = prep_ml(Xa.drop(columns=[c for c in Xa.columns if c.startswith("phys_")]))
PH = Xa[[c for c in Xa.columns if c.startswith("phys_")]]
F_BP = pd.concat([BASE, PH], axis=1)

# Tg 实测关联（去重表 → 中位数）
tg_meas = dict(zip(g_tg, pd.to_numeric(tg_d["tg_c"], errors="coerce")))
Tgv = g_mod_d.map(lambda x: tg_meas.get(x, np.nan)).to_numpy(dtype=float)
F_TG = pd.concat([F_BP, pd.DataFrame({"derived_Tg": Tgv, "derived_dT": Tgv - tt},
                                     index=F_BP.index)], axis=1)

# ---------------------------------------------------------------- OOF 预测 Tg（去重表训练，折内配方不可见）
tg_pred = np.full(len(mod_d), np.nan)
Xt_all = prep_ml(augment_polymer_physics(tg_d.drop(columns=["tg_c"])))
y_tg_d = pd.to_numeric(tg_d["tg_c"], errors="coerce").to_numpy(float)
g_tg_d = hash_series(tg_d).to_numpy()
for tr, te in KFold(5, shuffle=True, random_state=42).split(mod_d):
    held = set(g_mod_d[te])
    mask_tr = ~np.isin(g_tg_d, list(held))
    m = xgb()
    m.fit(Xt_all[mask_tr], y_tg_d[mask_tr])
    tg_pred[te] = m.predict(Xt_all.iloc[te])
print(f"OOF 预测 Tg: 覆盖 {np.isfinite(tg_pred).mean()*100:.0f}%（Tg 模型用去重表训练，折内配方组不可见）")

F_TGP = pd.concat([F_BP, pd.DataFrame({"derived_TgPred": tg_pred,
                                       "derived_dT_pred": tg_pred - tt}, index=F_BP.index)], axis=1)

# ---------------------------------------------------------------- 评估（去重 + 随机 5 折，3 种子）
sets = {"BASE": BASE, "BASE+PHYS": F_BP, "BASE+PHYS+TG(实测)": F_TG,
        "BASE+PHYS+TG_PRED": F_TGP}
print()
print("─" * 84)
print(f"{'特征集':26s} {'R²(3种子均值)':>14s} {'±std':>8s} {'vs BASE':>9s}")
base_r2 = None
for name, F in sets.items():
    F = F.loc[:, ~F.columns.duplicated()].fillna(F.median(numeric_only=True)).fillna(0.0)
    X_ = F.to_numpy(dtype=float)
    scores = []
    for seed in (42, 7, 2024):
        pred = np.zeros_like(y)
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(X_):
            m = xgb(seed)
            m.fit(X_[tr], y[tr])
            pred[te] = m.predict(X_[te])
        scores.append(r2_score(y, pred))
    r2 = float(np.mean(scores))
    if base_r2 is None:
        base_r2 = r2
    print(f"{name:26s} {r2:14.4f} {np.std(scores):8.4f} {r2-base_r2:+9.4f}")

print()
print("互验: 全量+GroupKFold(上一轮) = 0.173 (BASE) / 0.274 (+PHYS+TG_PRED)")
print("     若去重结果与 GroupKFold 接近 → 两种无泄漏协议互证成立")
