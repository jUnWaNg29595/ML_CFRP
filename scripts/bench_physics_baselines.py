# -*- coding: utf-8 -*-
"""
物理基线准入评估 (Phase A 前置)
对每个候选高分子物理公式，单测其与实测目标的相关性，决定：
  - Spearman >= 0.4  → 可作硬结构基座（进网络方程）
  - 0.2 ~ 0.4        → 降级为软约束/普通特征
  - < 0.2            → 弃用

覆盖:
  A. ν 基线: 网络节点密度(反应模拟) vs Flory 理论公式 → 实测 ν
  B. 橡胶弹性 E=3νRT → 模量 (尺度偏差 + 秩相关)
  C. Fox-Loshaek Tg = Tg∞ - K/Mc → tg_c (隔离公式质量)
  D. Kissinger 升温速率偏移 → td5 (同配方内 β 斜率验证)
  E. 结构刚性指数 (芳香度/柔性) → tg_c
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import spearmanr
from sklearn.metrics import r2_score

DATA = r"C:/Users/wangj/Desktop/ml_dataset"
R_GAS = 8.314
RHO_KG_M3 = 1200.0  # 环氧网络典型密度


def sp(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20:
        return np.nan, int(m.sum())
    return float(spearmanr(x[m], y[m])[0]), int(m.sum())


def verdict(rho):
    if not np.isfinite(rho):
        return "样本不足"
    if rho >= 0.4:
        return "✅ 可作硬基座"
    if rho >= 0.2:
        return "⚠️ 降级软约束/特征"
    return "❌ 弃用"


print("=" * 78)
print("物理基线准入评估报告")
print("=" * 78)

# ---------------------------------------------------------------- 公共: 哈希关联
from core.formulation_fusion import FormulationFusionEngine
_eng = FormulationFusionEngine(verbose=False)


def hash_series(df):
    return df.apply(_eng.compute_formulation_hash, axis=1)


def load(target):
    return pd.read_csv(f"{DATA}/ml_qspr_model_{target}.csv",
                       encoding="utf-8", encoding_errors="replace", low_memory=False)


def nu_map(target="crosslink_density_mol_m3", col="crosslink_density_mol_m3"):
    """实测 ν: hash -> 中位数"""
    df = load(target)
    v = pd.to_numeric(df[col], errors="coerce")
    h = hash_series(df)
    m = {}
    for hh, vv in zip(h, v):
        if pd.notna(vv) and 100 <= vv <= 1e4:
            m.setdefault(hh, []).append(float(vv))
    return {k: float(np.median(v)) for k, v in m.items()}


# ================================================================ A. ν 基线
print("\n" + "─" * 78)
print("A. 交联密度 ν 的物理基线质量（目标: 实测 ν 中位数）")
print("─" * 78)

nu_tab = load("crosslink_density_mol_m3")
nu_true = pd.to_numeric(nu_tab["crosslink_density_mol_m3"], errors="coerce")
valid = (nu_true >= 100) & (nu_true <= 1e4)
print(f"实测 ν 有效样本: {int(valid.sum())} / {len(nu_tab)}")

# A1. Flory 理论公式基线
from core.crosslink_physics import compute_crosslink_features
xl = compute_crosslink_features(nu_tab)
nu_theory = xl["xl_nu_theory_mol_m3"].to_numpy(dtype=float)
rho_a, n_a = sp(nu_theory, nu_true.to_numpy())
print(f"  A1 Flory 理论公式  (1e6/EEW)·balance·(f-2)/f·dilution")
print(f"     Spearman = {rho_a:.3f}   n = {n_a}   {verdict(rho_a)}")

# A2. 网络节点密度（反应模拟产物结构）
try:
    from core.reaction_simulator import CrosslinkedFeatureExtractor
    _ex = CrosslinkedFeatureExtractor(verbose=False)
    pair_cols = ["resin_1_structure", "curing_agent_1_structure"]
    cpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "cache", "reaction_features_nu.csv")
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    cache = {}
    if os.path.exists(cpath):
        _c = pd.read_csv(cpath, encoding="utf-8-sig")
        for _, r in _c.iterrows():
            cache[(r["ep"], r["cu"])] = r.drop(labels=["ep", "cu"]).to_dict()
        print(f"     [命中磁盘缓存: {len(cache)} 条]")
    uniq = nu_tab[pair_cols].drop_duplicates().head(700)
    todo = [(a, b) for a, b in zip(uniq[pair_cols[0]], uniq[pair_cols[1]])
            if isinstance(a, str) and isinstance(b, str) and (a, b) not in cache]
    if todo:
        t0 = time.time()
        rows = []
        for ep, cu in todo:
            try:
                f = _ex.extract_crosslink_features(ep, cu, target_conversion=0.95,
                                                  curing_temp=170.0, curing_time=2.0)
            except Exception:
                f = {}
            f = f or {}
            cache[(ep, cu)] = f
            rows.append(dict(ep=ep, cu=cu, **{k: v for k, v in f.items()
                                              if isinstance(v, (int, float, bool))}))
        if rows:
            pd.DataFrame(rows).to_csv(cpath, index=False, encoding="utf-8-sig")
        print(f"     [反应模拟: {len(todo)} 个新配方对, {time.time()-t0:.0f}s]")

    def net_feat(key):
        row = nu_tab[pair_cols].iloc[key] if False else None
        return None

    jd = np.full(len(nu_tab), np.nan)
    ard = np.full(len(nu_tab), np.nan)
    red = np.full(len(nu_tab), np.nan)
    for i, (ep, cu) in enumerate(zip(nu_tab[pair_cols[0]], nu_tab[pair_cols[1]])):
        f = cache.get((ep, cu))
        if not f:
            continue
        jd[i] = f.get("product_junction_site_density", np.nan)
        ard[i] = f.get("product_aromatic_ring_density", np.nan)
        red[i] = f.get("product_residual_epoxide_density", np.nan)

    for name, arr in [("A2 网络节点密度 (产物结构直读)", jd),
                      ("A3 网络芳香密度", ard),
                      ("A4 残余环氧密度 (缺陷)", red)]:
        r, n = sp(arr, nu_true.to_numpy())
        print(f"  {name}")
        print(f"     Spearman = {r:.3f}   n = {n}   {verdict(r)}")
except Exception as e:
    print(f"  [反应模拟不可用: {e}]")

# ================================================================ B. 3νRT
print("\n" + "─" * 78)
print("B. 橡胶弹性 E = 3νRT → 模量（关键: 尺度 vs 秩）")
print("─" * 78)
nu_meas = nu_map()
for tgt in ["tensile_modulus_gpa", "storage_modulus_25c_gpa", "flexural_modulus_gpa"]:
    try:
        df = load(tgt)
    except Exception:
        continue
    y = pd.to_numeric(df[tgt], errors="coerce").to_numpy(dtype=float) * 1000.0  # GPa→MPa
    h = hash_series(df)
    nu_v = np.array([nu_meas.get(hh, np.nan) for hh in h], dtype=float)
    # 用实测 ν 算 3νRT（隔离"ν 估计误差"与"公式误差"）
    e_rub = 3.0 * nu_v * R_GAS * 298.0 / 1e6  # MPa
    ok = np.isfinite(e_rub) & np.isfinite(y) & (y > 0) & (e_rub > 0)
    if ok.sum() < 20:
        print(f"  {tgt}: 关联样本不足 ({int(ok.sum())})")
        continue
    ratio = float(np.median(y[ok] / e_rub[ok]))
    r_lin, _ = sp(e_rub, y)
    r_log, n = sp(np.log(e_rub[ok]), np.log(y[ok]))
    print(f"  {tgt}  (n={n})")
    print(f"     尺度比  E_实测 / 3νRT  中位数 = {ratio:6.0f} 倍   (中位 3νRT = {np.median(e_rub[ok]):.1f} MPa)")
    print(f"     秩相关  Spearman(log 3νRT, log E) = {r_log:.3f}   {verdict(r_log)}")

# ================================================================ C. Fox-Loshaek
print("\n" + "─" * 78)
print("C. Fox–Loshaek  Tg = Tg∞ − K/Mc  → tg_c  (用实测 ν 隔离公式质量)")
print("─" * 78)
tg = load("tg_c")
y_tg = pd.to_numeric(tg["tg_c"], errors="coerce").to_numpy(dtype=float)
h_tg = hash_series(tg)
nu_tg = np.array([nu_meas.get(hh, np.nan) for hh in h_tg], dtype=float)
r_c, n_c = sp(nu_tg, y_tg)
print(f"  C1 核心检验: 实测 ν 与实测 Tg 的秩相关")
print(f"     Spearman = {r_c:.3f}   n = {n_c}   {verdict(r_c)}")
mc = RHO_KG_M3 / np.where(nu_tg > 0, nu_tg, np.nan)      # kg/mol
mc = mc * 1000.0                                          # g/mol
okc = np.isfinite(mc) & np.isfinite(y_tg) & (mc > 0)
if okc.sum() >= 30:
    inv_mc = 1.0 / mc[okc]
    yc = y_tg[okc]
    r_fl, _ = sp(inv_mc, yc)
    print(f"  C2 Fox–Loshaek 形式 (1/Mc 线性于 Tg), Mc=ρ/ν, ρ={RHO_KG_M3:.0f} kg/m3")
    print(f"     Spearman = {r_fl:.3f}   {verdict(r_fl)}")
    # 2 参数标定 + 留出评估（A, K 由数据拟合，检验公式形式能否解释方差）
    from sklearn.model_selection import train_test_split
    X = inv_mc.reshape(-1, 1)
    Xtr, Xte, ytr, yte = train_test_split(X, yc, test_size=0.2, random_state=42)
    A = np.hstack([np.ones_like(Xtr), -Xtr])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    pred = coef[0] - coef[1] * Xte.ravel()
    print(f"     2 参数标定 (Tg∞={coef[0]:.1f}°C, K={coef[1]:.0f}): 留出 R² = {r2_score(yte, pred):.3f}")
    print(f"     → 对比常数基线 R²=0.000（若 >0.15 说明 1/Mc 形式有真实解释力）")

# ================================================================ D. Kissinger
print("\n" + "─" * 78)
print("D. Kissinger 升温速率偏移 → td5（同配方内 β 斜率，零标签验证）")
print("─" * 78)
for tgt, beta_col in [("td5_c", "td5_c_heating_rate_c_min"), ("tmax_c", "tmax_c_heating_rate_c_min")]:
    df = load(tgt)
    if beta_col not in df.columns:
        continue
    y = pd.to_numeric(df[tgt], errors="coerce")
    b = pd.to_numeric(df[beta_col], errors="coerce")
    h = hash_series(df)
    g = pd.DataFrame({"h": h, "y": y, "b": b}).dropna()
    g = g[(g["b"] > 0) & (g["y"] > 0)]
    slopes, preds = [], []
    for hh, sub in g.groupby("h"):
        if sub["b"].nunique() < 2:
            continue
        x = np.log(sub["b"].to_numpy(dtype=float))
        yy = sub["y"].to_numpy(dtype=float)
        if len(x) < 2 or np.ptp(x) == 0:
            continue
        k = np.polyfit(x, yy, 1)[0]        # dT/d(ln β) K per ln unit
        slopes.append(k)
        T_K = float(np.median(yy)) + 273.15
        preds.append(R_GAS * T_K ** 2 / 200000.0)   # Ea=200 kJ/mol 的理论斜率
    if len(slopes) < 5:
        print(f"  {tgt}: 同配方多 β 组数不足 ({len(slopes)})，无法验证")
        continue
    slopes = np.array(slopes)
    print(f"  {tgt}  (n_组 = {len(slopes)})")
    print(f"     实测斜率 dT/d(lnβ) 中位数 = {np.median(slopes):7.1f} K   (论文典型 10~40 K)")
    print(f"     Kissinger 理论斜率 (Ea=200kJ/mol) = {np.median(preds):7.1f} K")
    print(f"     正斜率占比 = {float((slopes > 0).mean())*100:.0f}%  (应 >60% 才说明 β 物理真实)")

# ================================================================ E. 结构刚性
print("\n" + "─" * 78)
print("E. 结构刚性指数 → tg_c / 模量（RDKit，无需标签）")
print("─" * 78)
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    import re
    oxirane = Chem.MolFromSmarts("C1OC1")
    nh = Chem.MolFromSmarts("[NX3;H1,H2]")
    oh = Chem.MolFromSmarts("[OX2H]")

    def clean(s):
        return re.sub(r"\{|\[<[^\]]*\]|\[>[^\]]*\]|\}", "", str(s)).strip()

    def idx_of(smiles):
        m = Chem.MolFromSmiles(clean(smiles)) if isinstance(smiles, str) else None
        if m is None:
            return None
        n_heavy = m.GetNumHeavyAtoms()
        if n_heavy == 0:
            return None
        n_ar = sum(1 for a in m.GetAtoms() if a.GetIsAromatic())
        n_rot = Descriptors.NumRotatableBonds(m)
        n_hb = len(m.GetSubstructMatches(nh)) + len(m.GetSubstructMatches(oh))
        return dict(f_ar=n_ar / n_heavy, flex=n_rot / n_heavy, hb=n_hb / n_heavy)

    cache = {}
    for tbl, ycol in [("tg_c", "tg_c"), ("tensile_modulus_gpa", "tensile_modulus_gpa")]:
        df = load(tbl)
        y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
        for key in ["resin_1_structure", "curing_agent_1_structure"]:
            pass
        pairs = df[["resin_1_structure", "curing_agent_1_structure"]]
        far = np.full(len(df), np.nan)
        flx = np.full(len(df), np.nan)
        hbv = np.full(len(df), np.nan)
        for i, (ep, cu) in enumerate(zip(pairs.iloc[:, 0], pairs.iloc[:, 1])):
            vals = []
            for s in (ep, cu):
                k = str(s)
                if k not in cache:
                    cache[k] = idx_of(s)
                if cache[k]:
                    vals.append(cache[k])
            if vals:
                far[i] = np.mean([v["f_ar"] for v in vals])
                flx[i] = np.mean([v["flex"] for v in vals])
                hbv[i] = np.mean([v["hb"] for v in vals])
        for nm, arr in [("芳香碳分数 f_ar", far), ("柔性指数 可旋转键/重原子", flx),
                        ("氢键位点密度", hbv)]:
            r, n = sp(arr, y)
            print(f"  {tbl:24s} {nm:28s} Spearman = {r:6.3f}  n={n:5d}  {verdict(r)}")
except Exception as e:
    print(f"  [结构指数不可用: {e}]")

print("\n" + "=" * 78)
print("准入结论: Spearman>=0.4 进硬结构基座 | 0.2~0.4 软约束/特征 | <0.2 弃用")
print("=" * 78)
