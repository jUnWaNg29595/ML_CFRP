# -*- coding: utf-8 -*-
"""
阶段 1：训练 配方→交联密度(ν) 嵌入模型（两阶段 PINN 的第一阶）

- 数据: ml_qspr_model_crosslink_density_mol_m3.csv，清理到物理区间 [100, 1e4] mol/m³
- 评估: 80/20 留出，报告 R²(log ν) 与 Spearman(ν)
- 产物: models/crosslink_nu_encoder.joblib（冻结后由 EpoxyPINNRegressor 自动发现并嵌入）

运行: C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe scripts/train_nu_encoder.py
"""
import sys, time, warnings
sys.path.insert(0, r"C:/Users/wangj/Desktop/CFRP系统/CFRP系统")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

DATA = r"C:/Users/wangj/Desktop/ml_dataset/ml_qspr_model_crosslink_density_mol_m3.csv"
OUT = r"C:/Users/wangj/Desktop/CFRP系统/CFRP系统/models/crosslink_nu_encoder.joblib"


def main():
    from core.crosslink_nu_model import CrosslinkNuEncoder
    from core import crosslink_physics as xphy

    df = pd.read_csv(DATA, encoding="utf-8", encoding_errors="replace")
    nu = df["crosslink_density_mol_m3"].to_numpy(dtype=float)
    print(f"原始样本: {len(df)} | 区间内: {((nu >= 100) & (nu <= 1e4)).sum()} "
          f"| 脏数据(区间外): {((nu < 100) | (nu > 1e4)).sum()}")

    X_df = df.drop(columns=["crosslink_density_mol_m3",
                            "crosslink_density_mol_m3_test_method",
                            "crosslink_density_mol_m3_test_atmosphere"])
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(df))
    n_te = int(0.2 * len(df))
    te_idx, tr_idx = idx[:n_te], idx[n_te:]

    def run(tag, use_physics):
        enc = CrosslinkNuEncoder(use_physics_features=use_physics)
        enc.fit(X_df.iloc[tr_idx].reset_index(drop=True), nu[tr_idx])
        lp = enc.predict_log_nu(X_df.iloc[te_idx].reset_index(drop=True))
        r2 = r2_score(np.log(nu[te_idx]), lp)
        rho = spearmanr(np.exp(lp), nu[te_idx])[0]
        print(f"  [{tag:22s}] R2(logν)={r2:.4f}  Spearman(ν)={rho:.3f}")
        return enc, r2, rho

    print("阶段1 嵌入模型（留出 20%）:")
    enc0, r2_0, rho_0 = run("XGB 纯配方特征", False)
    enc1, r2_1, rho_1 = run("XGB + 物理特征(xl_*)", True)

    # 理论基线参照
    xl = xphy.compute_crosslink_features(X_df.iloc[te_idx].reset_index(drop=True))
    m = xl["xl_nu_theory_mol_m3"].notna().values
    rho_t = spearmanr(xl.loc[m, "xl_nu_theory_mol_m3"], nu[te_idx][m])[0] if m.sum() > 30 else float("nan")
    print(f"  [理论基线 ν_theory      ] Spearman={rho_t:.3f} (n={int(m.sum())})")

    # 用全量数据重训最终产物
    print("用全量数据训练最终编码器 ...")
    t0 = time.time()
    final = CrosslinkNuEncoder(use_physics_features=(r2_1 >= r2_0))
    final.fit(X_df, nu)
    final.save(OUT)
    print(f"已保存: {OUT}  ({time.time()-t0:.0f}s)")
    print(f"  provenance: {final.provenance_}")


if __name__ == "__main__":
    main()
