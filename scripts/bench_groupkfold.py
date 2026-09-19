# -*- coding: utf-8 -*-
"""
GroupKFold（按配方哈希分组）对比：剔除"同配方不同测试条件"的跨折泄漏
随机切分下同一配方的多次测量会同时出现在训练/测试集 → 树模型靠记忆重复样本获益
分组切分更接近真实部署（预测全新配方）
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score
from core.pinn_model import EpoxyPINNRegressor
from core.polymer_physics import augment_polymer_physics
from core.formulation_fusion import FormulationFusionEngine

DATA = r"C:/Users/wangj/Desktop/ml_dataset"
_eng = FormulationFusionEngine(verbose=False)


def xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                        reg_lambda=1.0, random_state=42, n_jobs=4,
                        tree_method="hist", verbosity=0)


def prep_xgb(X):
    num, oh = [], []
    for c in X.columns:
        cl = str(c).lower()
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            num.append(c)
        elif any(p in cl for p in ("structure", "smiles", "inchi", "bigsmiles")):
            continue
        elif s.nunique(dropna=True) <= 12:
            oh.append(c)
    out = X[num].copy() if num else pd.DataFrame(index=X.index)
    for c in oh:
        d = pd.get_dummies(pd.Categorical(X[c].astype(str),
                                          categories=sorted(X[c].dropna().astype(str).unique()) + ["__other__"]),
                           prefix=c, dtype=float)
        d.index = X.index
        out = pd.concat([out, d], axis=1)
    return out


print("=" * 92)
print("按配方分组 GroupKFold(3折) vs 随机 KFold(3折) — 同特征(XGB+物理 / PINN+物理)")
print("=" * 92)
print(f"{'目标':20s} {'切分':10s} {'XGB(+物理)':>14s} {'PINN(+物理)':>14s} {'差值':>10s}")

for target in ["tg_c", "td10_c", "tensile_modulus_gpa"]:
    df = pd.read_csv(f"{DATA}/ml_qspr_model_{target}.csv", encoding="utf-8",
                     encoding_errors="replace", low_memory=False).dropna(subset=[target]).reset_index(drop=True)
    y = df[target].values.astype(float)
    X = df.drop(columns=[target])
    Xp = augment_polymer_physics(X)
    Xn = prep_xgb(Xp)
    groups = df.apply(_eng.compute_formulation_hash, axis=1).to_numpy()

    for scheme in ["random", "grouped"]:
        if scheme == "random":
            splits = list(KFold(3, shuffle=True, random_state=42).split(X))
        else:
            splits = list(GroupKFold(3).split(X, y, groups))
        r_x, r_p = [], []
        for tr, te in splits:
            m = xgb(); m.fit(Xn.iloc[tr], y[tr]); r_x.append(r2_score(y[te], m.predict(Xn.iloc[te])))
            pin = EpoxyPINNRegressor(mode="auto", target_name=target, seed=42,
                                     physics_weight=0.25, use_polymer_physics=True)
            pin.fit(X.iloc[tr], y[tr])
            r_p.append(r2_score(y[te], pin.predict(X.iloc[te])))
        mx, mp = float(np.mean(r_x)), float(np.mean(r_p))
        print(f"{target:20s} {scheme:10s} {mx:14.4f} {mp:14.4f} {mp-mx:+10.4f}")
    print()
