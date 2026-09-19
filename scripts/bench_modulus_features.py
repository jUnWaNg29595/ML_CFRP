# -*- coding: utf-8 -*-
"""
模量特征构造与严格测试（GroupKFold 按配方分组）
=============================================
基于实测准入的候选特征，构造并做特征集消融：

  BASE                    原始配方/工艺/测试条件列（含 EEW、测试温度、加载速率、固化温度）
  +PHYS                   加 5 个物理指数（芳香度/sp3/柔性/键能/内聚能，单体加权）
  +TG                     加实测 Tg（按配方哈希关联两表）+ ΔT = Tg − T_test
  +TG_PRED                OOF 预测 Tg（严格无泄漏：折内该配方组的 Tg 模型未见过）+ ΔT_pred

同时做：
  - 标签清洗（剔除 <0.1 GPa 的 278 行异常 + 1 行 47 GPa）
  - 测试温度清洗（剔除 −273/462 等错误值，缺失按 25℃ 填充）
  - 分组切分（同配方不同测试条件不跨折）—— 避免随机切分的泄漏
"""
import os
import re
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from core.formulation_fusion import FormulationFusionEngine
from core.polymer_physics import augment_polymer_physics

DATA = r"C:/Users/wangj/Desktop/ml_dataset"
_eng = FormulationFusionEngine(verbose=False)


def xgb(**kw):
    p = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
             colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
             random_state=42, n_jobs=4, tree_method="hist", verbosity=0)
    p.update(kw)
    return XGBRegressor(**p)


def hash_series(df):
    return df.apply(_eng.compute_formulation_hash, axis=1)


def load(target):
    return pd.read_csv(f"{DATA}/ml_qspr_model_{target}.csv", encoding="utf-8",
                       encoding_errors="replace", low_memory=False)


def prep_ml(X: pd.DataFrame) -> pd.DataFrame:
    """传统 ML 预处理：数值直通 + 低基数分类 one-hot + 结构列剔除。"""
    num, oh = [], []
    for c in X.columns:
        s = X[c]
        cl = str(c).lower()
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
    out = out.loc[:, ~out.columns.duplicated()]
    return out


# ================================================================ 1. 数据准备
print("=" * 88)
print("模量特征构造与测试")
print("=" * 88)

mod = load("tensile_modulus_gpa")
y_raw = pd.to_numeric(mod["tensile_modulus_gpa"], errors="coerce")
g_mod = hash_series(mod)

# 目标清洗
keep = y_raw.between(0.1, 10.0)
print(f"模量标签: 原始 {len(mod)} 行 → 清洗后 {int(keep.sum())} 行"
      f"（剔除 <0.1 GPa: {int((y_raw < 0.1).sum())}, >10 GPa: {int((y_raw > 10).sum())}）")

# 测试温度清洗 + 填充
tcol = "tensile_modulus_gpa_test_temperature_c"
tt = pd.to_numeric(mod[tcol], errors="coerce")
bad_t = (~tt.between(-50, 300)) & tt.notna()
tt = tt.where(~bad_t, np.nan).fillna(25.0)
print(f"测试温度: 剔除错误值 {int(bad_t.sum())} 个，缺失(38%)按 25℃ 填充")

# Tg 表关联
tgdf = load("tg_c")
g_tg = hash_series(tgdf)
y_tg = pd.to_numeric(tgdf["tg_c"], errors="coerce")
_tgm = {}
for x, v in zip(g_tg, y_tg):
    if pd.notna(v):
        _tgm.setdefault(x, []).append(float(v))
tg_meas = {k: float(np.median(v)) for k, v in _tgm.items()}
tg_join = g_mod.map(lambda x: tg_meas.get(x, np.nan))
print(f"Tg 两表关联: {int(tg_join.notna().sum())}/{len(mod)} = {tg_join.notna().mean()*100:.0f}%")

# ================================================================ 2. 特征构造
Xb_raw = mod.drop(columns=["tensile_modulus_gpa"])
Xb_raw = augment_polymer_physics(Xb_raw)          # 加 phys_*
BASE = prep_ml(Xb_raw.drop(columns=[c for c in Xb_raw.columns
                                    if c.startswith("phys_")]))

PHYS = Xb_raw[[c for c in Xb_raw.columns if c.startswith("phys_")]
              if False else [c for c in Xb_raw.columns if c.startswith("phys_")]]
PHYS.index = BASE.index
FRAME_BP = pd.concat([BASE, PHYS], axis=1)

Tgv = tg_join.to_numpy(dtype=float)
dt = Tgv - tt.to_numpy()
FRAME_BPT = pd.concat([FRAME_BP,
                       pd.DataFrame({"derived_Tg": Tgv, "derived_dT": dt,
                                     "derived_Ttest": tt.to_numpy()}, index=BASE.index)],
                      axis=1)

print(f"特征维度: BASE={BASE.shape[1]}  BASE+PHYS={FRAME_BP.shape[1]}  "
      f"BASE+PHYS+TG={FRAME_BPT.shape[1]}")

# ================================================================ 3. OOF 预测 Tg（严格无泄漏）
# 联合分组：折内该配方组的 Tg 行不参与训练
groups_mod = g_mod.to_numpy()
gkf = GroupKFold(n_splits=5)

tg_pred_oof = np.full(len(mod), np.nan)
gi = 0
for tr_idx, te_idx in gkf.split(mod, groups=groups_mod):
    held = set(groups_mod[te_idx])
    tg_mask = ~g_tg.isin(held).to_numpy()
    Xt = prep_ml(augment_polymer_physics(tgdf.drop(columns=["tg_c"])))
    m = xgb()
    m.fit(Xt[tg_mask], y_tg[tg_mask].to_numpy())
    tg_pred_oof[te_idx] = m.predict(Xt.iloc[te_idx])
    gi += 1
print(f"OOF 预测 Tg 完成（{gi} 折，折内该配方组对 Tg 模型不可见），"
      f"覆盖 {np.isfinite(tg_pred_oof).mean()*100:.0f}%")

FRAME_BPTP = FRAME_BPT.copy()
FRAME_BPTP["derived_TgPred"] = tg_pred_oof
FRAME_BPTP["derived_dT_pred"] = tg_pred_oof - tt.to_numpy()

# ================================================================ 4. 分组消融
sets = {
    "BASE（原始配方/工艺/测试条件）": BASE,
    "BASE + PHYS（物理指数）": FRAME_BP,
    "BASE + PHYS + TG（实测 Tg + ΔT）": FRAME_BPT,
    "BASE + PHYS + TG_PRED（预测 Tg + ΔT）": FRAME_BPTP,
    "BASE + PHYS + TG_PRED（仅 ΔT_pred）": pd.concat(
        [FRAME_BP, pd.DataFrame({"derived_dT_pred": tg_pred_oof - tt.to_numpy()},
                                index=BASE.index)], axis=1),
}

ym = y_raw.to_numpy(dtype=float)
res = {}
print()
print("─" * 88)
for name, F in sets.items():
    F = F.loc[:, ~F.columns.duplicated()]
    F = F.fillna(F.median(numeric_only=True)).fillna(0.0)
    X_, y_ = F.to_numpy(dtype=float), ym
    ok = np.isfinite(y_) & keep.to_numpy() & np.isfinite(X_).all(axis=1)
    X_, y_, g_ = X_[ok], y_[ok], groups_mod[ok]
    pred = np.zeros_like(y_)
    for tr, te in GroupKFold(n_splits=5).split(X_, y_, g_):
        m = xgb()
        m.fit(X_[tr], y_[tr])
        pred[te] = m.predict(X_[te])
    r2 = r2_score(y_, pred)
    res[name] = (r2, int(ok.sum()), F.shape[1])
    print(f"  {name:38s} n={int(ok.sum()):5d}  d={F.shape[1]:3d}  GroupKFold R² = {r2:.4f}")

base_r2 = res[list(sets)[0]][0]
print()
print("相对 BASE 的增益:")
for name, (r2, n, d) in res.items():
    print(f"  {name:38s} {r2 - base_r2:+.4f}")
