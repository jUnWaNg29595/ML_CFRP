# -*- coding: utf-8 -*-
"""
模量提升路径 ① + ② 实测
=========================
① 换度量: log 目标训练 + MAPE 评估（对跨两个数量级的目标，R² 是不公平度量）
② 分段专用模型: 软端(<1 GPa, 柔性体系) 与 主群(玻璃态) 各建专用模型 + 路由器
   - 路由器两种: 规则(f_ar 阈值, 阈值在训练折内标定) / 分类器(XGB, 训练折内学习)
   - 全部 5 折 CV 内完成, 无测试信息泄漏

口径: 去重(每配方1行, n=748), 随机5折 × 2种子, 特征 BASE+PHYS
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor, XGBClassifier

from core.formulation_fusion import FormulationFusionEngine
from core.polymer_physics import augment_polymer_physics

DATA = r"C:/Users/wangj/Desktop/ml_dataset"
_eng = FormulationFusionEngine(verbose=False)


def xgb(seed=42):
    return XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
                        random_state=seed, n_jobs=4, tree_method="hist", verbosity=0)


def prep(X):
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


# ---------------------------------------------------------------- 数据（去重）
mod = pd.read_csv(f"{DATA}/ml_qspr_model_tensile_modulus_gpa.csv", encoding="utf-8",
                  encoding_errors="replace", low_memory=False)
y_all = pd.to_numeric(mod["tensile_modulus_gpa"], errors="coerce")
g_all = mod.apply(_eng.compute_formulation_hash, axis=1)
mod = mod[y_all.between(0.1, 10.0)].copy()
g_all = mod.apply(_eng.compute_formulation_hash, axis=1)
idx = []
for gh, sub in mod.assign(_g=g_all).groupby("_g"):
    s2 = sub.dropna(subset=["tensile_modulus_gpa"])
    idx.append(s2.iloc[(s2["tensile_modulus_gpa"] - s2["tensile_modulus_gpa"].median()).abs().argmin()].name)
d = mod.loc[idx].reset_index(drop=True)
y = pd.to_numeric(d["tensile_modulus_gpa"], errors="coerce").to_numpy(float)
print(f"去重 n={len(y)}  软端(<1GPa) {int((y<1).sum())} 行  主群(>=1) {int((y>=1).sum())} 行")

Xa = augment_polymer_physics(d.drop(columns=["tensile_modulus_gpa"]))
F = prep(Xa)
F = F.fillna(F.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=float)
f_ar = pd.to_numeric(Xa["phys_f_ar"], errors="coerce").to_numpy(float)
f_ar = np.where(np.isfinite(f_ar), f_ar, np.nanmedian(f_ar))
SOFT = y < 1.0

SEEDS = (42, 7)


def cv_eval(predict_fn, tag):
    """predict_fn(Xtr, ytr, Xte) -> yhat（原空间）"""
    preds = np.zeros(len(y))
    for seed in SEEDS:
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(F):
            preds[te] += predict_fn(F[tr], y[tr], F[te]) / len(SEEDS)
    r2 = r2_score(y, preds)
    mape = mean_absolute_percentage_error(y, preds) * 100
    soft_m = mean_absolute_percentage_error(y[SOFT], preds[SOFT]) * 100 if SOFT.sum() else np.nan
    main_m = mean_absolute_percentage_error(y[~SOFT], preds[~SOFT]) * 100
    r2s, _ = (lambda p: (r2_score(np.log(y), np.log(np.clip(p, 0.05, None))), 0))(preds)
    print(f"  {tag:34s} R²={r2:.3f}  R²log={r2s:.3f}  MAPE={mape:5.1f}%  "
          f"软端MAPE={soft_m:6.1f}%  主群MAPE={main_m:5.1f}%")
    return dict(tag=tag, r2=r2, mape=mape, soft=soft_m, main=main_m, pred=preds)


results = {}

# ---------------------------------------------------------------- 0 基线: 单模型线性
results["baseline"] = cv_eval(lambda Xt, yt, Xe: xgb().fit(Xt, yt).predict(Xe),
                              "基线: 单模型·线性目标")

# ---------------------------------------------------------------- ① log 目标
def pred_log(Xt, yt, Xe):
    m = xgb()
    m.fit(Xt, np.log(yt))
    return np.exp(m.predict(Xe))

results["log"] = cv_eval(pred_log, "① 单模型·log目标")

# ---------------------------------------------------------------- ② 规则路由分段
def make_rule_router_calib(Xt, yt, f_ar_tr):
    """训练折内标定 f_ar 阈值: 使 软端标签(y<1) 与 f_ar<阈值的错分最少"""
    best_t, best_err = 0.35, 1e9
    for t in np.unique(np.quantile(f_ar_tr, np.linspace(0.1, 0.9, 17))):
        pred_soft = f_ar_tr < t
        err = np.mean(pred_soft != (yt < 1.0))
        if err < best_err:
            best_err, best_t = err, float(t)
    return best_t


def pred_seg_rule(Xt, yt, Xe, f_ar_tr=None, f_ar_te=None, mode="rule"):
    yt = np.asarray(yt, float)
    far_tr = f_ar_tr if f_ar_tr is not None else np.zeros(len(yt))
    far_te = f_ar_te if f_ar_te is not None else np.zeros(len(Xe))
    if mode == "rule":
        t = make_rule_router_calib(Xt, yt, far_tr)
        s_tr, s_te = far_tr < t, far_te < t
    else:  # classifier
        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                            random_state=42, n_jobs=4, verbosity=0)
        clf.fit(Xt, (yt < 1.0).astype(int))
        s_te = clf.predict(Xe).astype(bool)
        # 训练折内自路由（软模型用软行训练）
        s_tr = clf.predict(Xt).astype(bool)
    yhat = np.full(len(Xe), np.nan)
    # 软端模型（训练折内被路由为软的行）
    if s_tr.sum() >= 12:
        ms = xgb()
        ms.fit(Xt[s_tr], np.log(yt[s_tr]))
        if s_te.any():
            yhat[s_te] = np.exp(ms.predict(Xe[s_te]))
    # 主群模型
    if (~s_tr).sum() >= 12:
        mh = xgb()
        mh.fit(Xt[~s_tr], np.log(yt[~s_tr]))
        todo = np.isnan(yhat)
        if todo.any():
            yhat[todo] = np.exp(mh.predict(Xe[todo]))
    yhat[np.isnan(yhat)] = np.median(yt)
    return yhat, s_te


def wrap_rule(Xt, yt, Xe, f_tr, f_te):
    return pred_seg_rule(Xt, yt, Xe, f_tr, f_te, mode="rule")[0]


def wrap_clf(Xt, yt, Xe, f_tr, f_te):
    return pred_seg_rule(Xt, yt, Xe, f_tr, f_te, mode="clf")[0]


# 路由准确率（折外）
route_acc = []
for seed in SEEDS:
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(F):
        _, s_te = pred_seg_rule(F[tr], y[tr], F[te], f_ar[tr], f_ar[te], mode="clf")
        route_acc.append(np.mean(s_te == SOFT[te]))
print(f"\n分类器路由折外准确率: {np.mean(route_acc)*100:.1f}%  (基线: 多数类 {max(SOFT.mean(),1-SOFT.mean())*100:.0f}%)")
print()

f_tr_cache = {}
results["seg_rule"] = cv_eval(
    lambda Xt, yt, Xe: wrap_rule(Xt, yt, Xe, f_ar[np.isin(np.arange(len(y)), [])] if False else None,
                                 None) if False else None,
    "占位") if False else None

# 手动跑两个分段方案（需要把 f_ar 一并传入）
def run_segmented(mode, tag):
    preds = np.zeros(len(y))
    for seed in SEEDS:
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(F):
            p, _ = pred_seg_rule(F[tr], y[tr], F[te], f_ar[tr], f_ar[te], mode=mode)
            preds[te] += p / len(SEEDS)
    r2 = r2_score(y, preds)
    mape = mean_absolute_percentage_error(y, preds) * 100
    soft_m = mean_absolute_percentage_error(y[SOFT], preds[SOFT]) * 100
    main_m = mean_absolute_percentage_error(y[~SOFT], preds[~SOFT]) * 100
    r2s = r2_score(np.log(y), np.log(np.clip(preds, 0.05, None)))
    print(f"  {tag:34s} R²={r2:.3f}  R²log={r2s:.3f}  MAPE={mape:5.1f}%  "
          f"软端MAPE={soft_m:6.1f}%  主群MAPE={main_m:5.1f}%")
    return dict(r2=r2, mape=mape, soft=soft_m, main=main_m)


print("② 分段专用模型:")
r_rule = run_segmented("rule", "②a 规则路由(f_ar阈值, 折内标定)")
r_clf = run_segmented("clf", "②b 分类器路由(XGB)")

# ---------------------------------------------------------------- 汇总
print()
print("=" * 84)
b = results["baseline"]; l = results["log"]
print(f"{'方案':36s} {'R²':>7s} {'MAPE':>8s} {'软端MAPE':>9s} {'主群MAPE':>9s}")
for nm, r in [("基线 单模型·线性", b), ("① 单模型·log", l), ("②a 规则路由分段", r_rule),
              ("②b 分类器路由分段", r_clf)]:
    print(f"{nm:36s} {r['r2']:7.3f} {r['mape']:7.1f}% {r['soft']:8.1f}% {r['main']:8.1f}%")
