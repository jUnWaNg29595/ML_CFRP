# -*- coding: utf-8 -*-
"""PINN 第 1 层优化后的快速基准对比：EpoxyPINN(新默认) vs XGBoost，同切分同特征矩阵。"""
import sys, time, warnings
sys.path.insert(0, r"C:/Users/wangj/Desktop/CFRP系统/CFRP系统")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

DATA = r"C:/Users/wangj/Desktop/ml_dataset"
TARGETS = ["tg_c", "td5_c", "tensile_strength_mpa", "tensile_modulus_gpa"]

def eval_reg(name, y_true, y_pred, t0, t1):
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f"  {name:22s} R2={r2:7.4f}  RMSE={rmse:9.3f}  时间={t1-t0:6.1f}s")
    return r2, rmse

def main():
    from core.pinn_model import EpoxyPINNRegressor

    for target in TARGETS:
        try:
            df = pd.read_csv(f"{DATA}/ml_qspr_model_{target}.csv", encoding="utf-8", encoding_errors="replace")
        except FileNotFoundError:
            print(f"[skip] {target}: 窄表不存在")
            continue
        df = df.dropna(subset=[target]).reset_index(drop=True)
        if len(df) < 100:
            print(f"[skip] {target}: 样本不足 ({len(df)})")
            continue
        # 防泄漏：剔除目标相关泄露列（*_extrapolated 等不在窄表中；测试条件列保留在特征里）
        X_df = df.drop(columns=[target])
        y = df[target].values
        X_tr, X_te, y_tr, y_te = train_test_split(X_df, y, test_size=0.2, random_state=42)
        print(f"\n===== {target} (n={len(df)}) =====")

        # ---- 用 PINN 的净化特征矩阵给两个模型同喂（对比模型本身而非特征） ----
        prep = EpoxyPINNRegressor(mode="auto", target_name=target)
        prep._identify_special_columns(X_tr)
        num_tr = prep._build_numeric_features(X_tr, fit_categoricals=True)
        pack = prep._fit_preprocess(num_tr)
        prep._prep_ = pack  # 启用修好的 _transform 路径（自动展开到插补器全列集）

        Xm_tr, _ = prep._transform(X_tr)
        Xm_te, _ = prep._transform(X_te)

        # ---- XGBoost 基线 ----
        try:
            from xgboost import XGBRegressor
            t0 = time.time()
            xgb = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                               subsample=0.8, colsample_bytree=0.8, random_state=42,
                               n_jobs=-1, tree_method="hist", verbosity=0)
            xgb.fit(Xm_tr, y_tr)
            t1 = time.time()
            eval_reg("XGBoost", y_te, xgb.predict(Xm_te), t0, t1)
        except ImportError:
            print("  [skip] xgboost 未安装")

        # ---- EpoxyPINN 新默认 ----
        t0 = time.time()
        pinn = EpoxyPINNRegressor(mode="auto", target_name=target, seed=42)
        pinn.fit(X_tr, y_tr)
        t1 = time.time()
        eval_reg("EpoxyPINN(新默认)", y_te, pinn.predict(X_te), t0, t1)

        # ---- EpoxyPINN 物理加权 (physics_weight=0.3) ----
        t0 = time.time()
        pinn2 = EpoxyPINNRegressor(mode="auto", target_name=target, seed=42, physics_weight=0.3)
        pinn2.fit(X_tr, y_tr)
        t1 = time.time()
        eval_reg("EpoxyPINN(pw=0.3)", y_te, pinn2.predict(X_te), t0, t1)

if __name__ == "__main__":
    main()
