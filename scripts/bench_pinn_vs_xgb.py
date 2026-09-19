# -*- coding: utf-8 -*-
"""
PINN vs XGBoost 严格同切分对比
  - 同 80/20 切分、同 seed、同特征（XGB 分别测"原特征"与"+物理指数"）
  - 评估: tg_c / td10_c / td5_c / 模量族
  - 并测试两种优化杠杆: 多种子平均、PINN+XGB 融合
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from core.pinn_model import EpoxyPINNRegressor
from core.polymer_physics import augment_polymer_physics, DEFAULT_PHYSICS_FEATURES

DATA = r"C:/Users/wangj/Desktop/ml_dataset"
TARGETS = ["tg_c", "td10_c", "td5_c", "tensile_modulus_gpa", "storage_modulus_25c_gpa"]
SEEDS = [42, 7, 2024]


def xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                        reg_lambda=1.0, random_state=42, n_jobs=4,
                        tree_method="hist", verbosity=0)



def prep_xgb(X):
    """与 PINN 等价的文本处理: 结构列剔除, 低基数分类 one-hot, 高基数剔除。"""
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
        # 用一个"is-na"列保留缺失信息
        d.index = X.index
        out = pd.concat([out, d], axis=1)
    return out

def load(target):
    df = pd.read_csv(f"{DATA}/ml_qspr_model_{target}.csv", encoding="utf-8",
                     encoding_errors="replace", low_memory=False).dropna(subset=[target])
    return df.reset_index(drop=True)


print("=" * 96)
print("PINN vs XGBoost 严格同切分（80/20；3 seeds；均值±标准差）")
print("=" * 96)
print(f"{'目标':22s} {'XGB(原)':>15s} {'XGB(+物理)':>15s} {'PINN(+物理)':>15s} {'融合0.5':>13s} {'PINN/多种子':>14s}")

rows = {}
for target in TARGETS:
    df = load(target)
    y = df[target].values.astype(float)
    X = df.drop(columns=[target])
    Xp = augment_polymer_physics(X)
    Xn, Xnp = prep_xgb(X), prep_xgb(Xp)
    acc = {k: [] for k in ["xgb", "xgbp", "pinn", "blend", "pinn_ens"]}
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        idx_tr, idx_te = Xtr.index, Xte.index
        # XGB 原特征
        m = xgb(); m.fit(Xn.loc[idx_tr], ytr); p_xgb = m.predict(Xn.loc[idx_te])
        # XGB + 物理指数（用同索引取增强后的列）
        m2 = xgb(); m2.fit(Xnp.loc[idx_tr], ytr); p_xgbp = m2.predict(Xnp.loc[idx_te])
        # PINN + 物理指数
        pin = EpoxyPINNRegressor(mode="auto", target_name=target, seed=seed,
                                 physics_weight=0.25, use_polymer_physics=True)
        t0 = time.time(); pin.fit(Xtr, ytr); p_pinn = pin.predict(Xte)
        # PINN 多种子平均（3 个内部种子）
        ps = [p_pinn]
        for s2 in (101, 202):
            p2 = EpoxyPINNRegressor(mode="auto", target_name=target, seed=s2,
                                    physics_weight=0.25, use_polymer_physics=True)
            p2.fit(Xtr, ytr); ps.append(p2.predict(Xte))
        p_ens = np.mean(ps, axis=0)

        acc["xgb"].append(r2_score(yte, p_xgb))
        acc["xgbp"].append(r2_score(yte, p_xgbp))
        acc["pinn"].append(r2_score(yte, p_pinn))
        acc["blend"].append(r2_score(yte, 0.5 * p_pinn + 0.5 * p_xgbp))
        acc["pinn_ens"].append(r2_score(yte, p_ens))
        print(f"   [{target} seed={seed}] xgb={acc['xgb'][-1]:.4f} xgb+phys={acc['xgbp'][-1]:.4f} "
              f"pinn={acc['pinn'][-1]:.4f} blend={acc['blend'][-1]:.4f} pinn_ens={acc['pinn_ens'][-1]:.4f}"
              f"  ({time.time()-t0:.0f}s)")
    rows[target] = {k: (np.mean(v), np.std(v)) for k, v in acc.items()}

print()
print("─" * 96)
print(f"{'目标':22s} {'XGB(原)':>15s} {'XGB(+物理)':>15s} {'PINN(+物理)':>15s} {'融合0.5':>13s} {'PINN多种子':>14s}")
for t, r in rows.items():
    print(f"{t:22s} " + " ".join(f"{r[k][0]:7.4f}±{r[k][1]:.4f}" for k in
                                 ["xgb", "xgbp", "pinn", "blend", "pinn_ens"]))
print()
print("判决（PINN 是否优于 XGB）:")
for t, r in rows.items():
    d = r["pinn"][0] - r["xgbp"][0]
    de = r["pinn_ens"][0] - r["xgbp"][0]
    print(f"  {t:22s} PINN-XGB(+物理) = {d:+.4f}    PINN多种子-XGB = {de:+.4f}   "
          f"{'✅胜' if d > 0 else '❌负'}")
