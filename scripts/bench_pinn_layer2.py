# -*- coding: utf-8 -*-
"""
PINN 第 2 层（ν 交联密度中间层）基准脚本

对比项（同切分 seed=42，随机 80/20）：
1. EpoxyPINN 无 ν 监督（理论锚 only）
2. EpoxyPINN + 实测 ν 监督（配方哈希关联 crosslink_density_mol_m3）
3. XGBoost（净化特征）
4. XGBoost + 实测 ν 特征（缺失=-1）
并报告 ν 头预测 vs 实测 ν 的 Spearman 秩相关（物理层理论基线 ≈0.235）。

运行： C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe scripts/bench_pinn_layer2.py
"""
import sys, time, warnings
sys.path.insert(0, r"C:/Users/wangj/Desktop/CFRP系统/CFRP系统")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

DATA = r"C:/Users/wangj/Desktop/ml_dataset"
TARGETS = ["tg_c", "td5_c", "tensile_strength_mpa", "tensile_modulus_gpa"]
NU_RAW_MAX = 1e4  # 实测 ν 合理上限（脏数据清洗）


def build_hash2nu(eng):
    xl_tab = pd.read_csv(f"{DATA}/ml_qspr_model_crosslink_density_mol_m3.csv",
                         encoding="utf-8", encoding_errors="replace")
    h = xl_tab.apply(eng.compute_formulation_hash, axis=1)
    ok = (xl_tab["crosslink_density_mol_m3"] >= 100) & (xl_tab["crosslink_density_mol_m3"] <= NU_RAW_MAX)
    h2nu = {}
    for hh, v, k in zip(h, xl_tab["crosslink_density_mol_m3"], ok):
        if k and pd.notna(v):
            h2nu.setdefault(hh, float(v))
    return h2nu


def main():
    from core.pinn_model import EpoxyPINNRegressor
    from core.formulation_fusion import FormulationFusionEngine
    from xgboost import XGBRegressor

    eng = FormulationFusionEngine(verbose=False)
    h2nu = build_hash2nu(eng)

    for target in TARGETS:
        try:
            df = pd.read_csv(f"{DATA}/ml_qspr_model_{target}.csv", encoding="utf-8", encoding_errors="replace")
        except FileNotFoundError:
            print(f"[skip] {target}: 窄表不存在")
            continue
        df = df.dropna(subset=[target]).reset_index(drop=True)
        dh = df.apply(eng.compute_formulation_hash, axis=1)
        df["crosslink_density_mol_m3"] = dh.map(lambda hh: h2nu.get(hh, np.nan))

        X = df.drop(columns=[target])
        y = df[target].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        cov = Xtr["crosslink_density_mol_m3"].notna().mean()
        print(f"\n===== {target} (n={len(df)}, 实测ν覆盖={cov*100:.0f}%) =====")

        # PINN + 实测 ν 监督
        t0 = time.time()
        m = EpoxyPINNRegressor(mode="auto", target_name=target, seed=42)
        m.fit(Xtr, ytr)
        t1 = time.time()
        p = m.predict(Xte)
        line = f"  PINN L2(+实测ν监督)   R2={r2_score(yte, p):.4f}  {t1-t0:5.1f}s"
        if getattr(m, "nu_pred_", None) is not None:
            mk = Xte["crosslink_density_mol_m3"].notna().values
            if mk.sum() > 50:
                rho = spearmanr(m.nu_pred_[mk], Xte.loc[mk, "crosslink_density_mol_m3"].values)[0]
                line += f"  | ν头Spearman={rho:.3f}"
        print(line)

        # PINN 无 ν 监督（理论锚 only）
        t0 = time.time()
        m0 = EpoxyPINNRegressor(mode="auto", target_name=target, seed=42)
        m0.fit(Xtr.drop(columns=["crosslink_density_mol_m3"]), ytr)
        t1 = time.time()
        p0 = m0.predict(Xte.drop(columns=["crosslink_density_mol_m3"]))
        print(f"  PINN L1(理论锚only)   R2={r2_score(yte, p0):.4f}  {t1-t0:5.1f}s")

        # XGBoost ± 实测 ν 特征
        Xmtr, _ = m._transform(Xtr)
        Xmte, _ = m._transform(Xte)
        def xgb_run(Xa, Xb, tag):
            xgb = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                               colsample_bytree=0.8, random_state=42, n_jobs=-1,
                               tree_method="hist", verbosity=0)
            xgb.fit(Xa, ytr)
            print(f"  XGB{tag}: R2={r2_score(yte, xgb.predict(Xb)):.4f}")
        xgb_run(Xmtr, Xmte, "(净化特征)      ")
        nu_tr = Xtr["crosslink_density_mol_m3"].fillna(-1).values.reshape(-1, 1)
        nu_te = Xte["crosslink_density_mol_m3"].fillna(-1).values.reshape(-1, 1)
        xgb_run(np.hstack([Xmtr, nu_tr]), np.hstack([Xmte, nu_te]), "+实测ν特征    ")


if __name__ == "__main__":
    main()
