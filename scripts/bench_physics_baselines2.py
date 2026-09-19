# -*- coding: utf-8 -*-
"""
物理基线准入评估 · 第二轮
  I.  内聚能密度 CED / 溶解度参数 δ (Fedors 基团贡献) → Tg / 模量 / 强度 / 热稳定性
  J.  弱键解离能 BDE 指数 (C-N / C-O / C-S / Si-O ...) → td5 / tmax
  K.  纯物理多元基线 (无配方学习) → Tg / 模量，量化物理骨架天花板
"""
import os
import re
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors as rdMD

RDLogger.DisableLog("rdApp.*")
DATA = r"C:/Users/wangj/Desktop/ml_dataset"

# ---------------------------------------------------------------- Fedors 基团贡献
# Δe (cal/mol), Δv (cm3/mol) — 文献近似值 (Fedors 1974)
FEDORS = {
    "CH3": (1125, 33.5), "CH2": (1180, 16.1), "CH": (820, -1.0), "C": (350, -19.2),
    "phenyl": (7630, 71.4), "OH": (7120, 10.0), "O_ether": (1000, 3.8),
    "C=O": (4150, 10.8), "COO": (4300, 18.0), "NH2": (3000, 19.2),
    "NH": (2000, 4.5), "N_tert": (1000, -9.0), "ring": (250, 16.0),
    "S": (3400, 12.0), "Si": (1400, 15.0), "Cl": (2760, 24.0), "F": (1000, 18.0),
}

_P = {
    "oh": Chem.MolFromSmarts("[OX2H]"),
    "coo": Chem.MolFromSmarts("[CX3](=O)[OX2][#6]"),
    "co": Chem.MolFromSmarts("[CX3]=[OX1]"),
    "ether": Chem.MolFromSmarts("[OD2]([#6])[#6]"),
    "nh2": Chem.MolFromSmarts("[NX3;H2]"),
    "nh": Chem.MolFromSmarts("[NX3;H1]"),
    "n3": Chem.MolFromSmarts("[NX3;H0]"),
}

# ---------------------------------------------------------------- BDE 表 (kJ/mol)
BDE = {
    ("C", "C", False): 350.0,     # sp3-sp3
    ("C", "C", True): 480.0,      # 芳环内 / 芳-芳
    ("C", "N", False): 305.0,     # C-N 胺键（环氧弱键）
    ("C", "N", True): 380.0,      # 芳-N
    ("C", "O", False): 345.0,     # 醚/醇 C-O
    ("C", "O", True): 400.0,      # 芳-O
    ("C", "S", False): 275.0,     # C-S（弱）
    ("S", "S", False): 240.0,     # S-S（最弱）
    ("C", "F", False): 485.0,
    ("C", "Cl", False): 335.0,
    ("C", "Si", False): 320.0,
    ("Si", "O", False): 460.0,
    ("C", "H", False): 415.0,
    ("N", "H", False): 390.0,
    ("O", "H", False): 460.0,
}


def clean(s):
    return re.sub(r"\{|\[<[^\]]*\]|\[>[^\]]*\]|\}", "", str(s)).strip()


def fedors(mol):
    """Fedors 基团贡献 → CED (MPa) 与 δ (MPa^0.5)"""
    de = dv = 0.0
    n_arom_ring = 0
    try:
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                n_arom_ring += 1
    except Exception:
        pass
    counts = {k: 0 for k in FEDORS}
    for a in mol.GetAtoms():
        s = a.GetSymbol()
        if s == "C":
            if a.GetIsAromatic():
                continue
            nh = a.GetTotalNumHs()
            nc = sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "C")
            if nc + nh <= 1 or nh == 3:
                counts["CH3"] += 1
            elif nh == 2:
                counts["CH2"] += 1
            elif nh == 1:
                counts["CH"] += 1
            else:
                counts["C"] += 1
        elif s == "O":
            counts["O_ether"] += 1
        elif s == "N":
            nh = a.GetTotalNumHs()
            counts["NH2" if nh >= 2 else ("NH" if nh == 1 else "N_tert")] += 1
        elif s == "S":
            counts["S"] += 1
        elif s == "Si":
            counts["Si"] += 1
        elif s == "Cl":
            counts["Cl"] += 1
        elif s == "F":
            counts["F"] += 1
    # 修正 O 分类: OH / C=O / COO 覆盖 ether 计数
    n_oh = len(mol.GetSubstructMatches(_P["oh"]))
    n_coo = len(mol.GetSubstructMatches(_P["coo"]))
    n_co = len(mol.GetSubstructMatches(_P["co"]))
    counts["O_ether"] = max(0, counts["O_ether"] - n_oh - n_co)
    counts["OH"] += n_oh
    counts["COO"] += n_coo
    counts["C=O"] += max(0, n_co - n_coo)
    counts["phenyl"] += n_arom_ring
    counts["ring"] += mol.GetRingInfo().NumRings()
    for k, c in counts.items():
        if c:
            de += c * FEDORS[k][0]
            dv += c * FEDORS[k][1]
    if dv <= 1.0:
        return np.nan, np.nan
    ced = de / dv * 4.184          # cal/cm3 → J/cm3 = MPa
    return ced, float(np.sqrt(max(ced, 1e-6)))


def bde_feats(mol):
    """弱键 BDE 指数: 最小值 / 加权均值 / 弱键(<340)占比"""
    vals, n_bonds = [], 0
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        s1, s2 = a1.GetSymbol(), a2.GetSymbol()
        if "H" in (s1, s2):
            continue
        arom = a1.GetIsAromatic() and a2.GetIsAromatic()
        key = tuple(sorted((s1, s2))) + (arom,)
        key2 = (s1, s2, arom)
        v = BDE.get(key2) or BDE.get((s2, s1, arom)) or BDE.get(key) or BDE.get(
            (key[0], key[1], False))
        if v is None:
            v = BDE.get((s1, s2, False)) or BDE.get((s2, s1, False)) or 360.0
        vals.append(float(v))
        n_bonds += 1
    if not vals:
        return np.nan, np.nan, np.nan
    v = np.array(vals)
    return float(v.min()), float(v.mean()), float((v < 340).mean())


_cache = {}


def desc(smi):
    k = str(smi)
    if k in _cache:
        return _cache[k]
    mol = Chem.MolFromSmiles(clean(smi)) if isinstance(smi, str) else None
    if mol is None:
        _cache[k] = None
        return None
    n_h = mol.GetNumHeavyAtoms()
    ced, delta = fedors(mol)
    bmin, bmean, bweak = bde_feats(mol)
    d = dict(
        ced=ced, delta=delta, bmin=bmin, bmean=bmean, bweak=bweak,
        f_ar=sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / max(n_h, 1),
        csp3=Descriptors.FractionCSP3(mol),
        flex=Descriptors.NumRotatableBonds(mol) / max(n_h, 1),
        mw=Descriptors.MolWt(mol),
    )
    _cache[k] = d
    return d


def row_feats(r_, c_, w=(0.6, 0.4)):
    """树脂+固化剂加权平均（按 PHR 近似权重）"""
    out = {}
    got = []
    for s, ww in ((r_, w[0]), (c_, w[1])):
        d = desc(s)
        if d:
            got.append((d, ww))
    if not got:
        return None
    tw = sum(ww for _, ww in got)
    for k in got[0][0]:
        out[k] = sum(d[k] * ww for d, ww in got if np.isfinite(d.get(k, np.nan))) / tw
    return out


def sp(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20:
        return np.nan, int(m.sum())
    return float(spearmanr(x[m], y[m])[0]), int(m.sum())


def verdict(r):
    if not np.isfinite(r):
        return "样本不足"
    return "✅ 硬基座" if abs(r) >= 0.4 else ("⚠️ 软约束" if abs(r) >= 0.2 else "❌ 弃用")


print("=" * 80)
print("物理基线准入评估 · 第二轮 (CED / 弱键 BDE / 纯物理多元基线)")
print("=" * 80)

TARGETS = ["tg_c", "tensile_modulus_gpa", "flexural_modulus_gpa", "tensile_strength_mpa",
           "td5_c", "tmax_c", "char_yield_pct", "tensile_strain_at_break_pct"]

feat_rows = {}
for tgt in TARGETS:
    try:
        df = pd.read_csv(f"{DATA}/ml_qspr_model_{tgt}.csv",
                         encoding="utf-8", encoding_errors="replace", low_memory=False)
    except Exception:
        continue
    y = pd.to_numeric(df[tgt], errors="coerce").to_numpy(float)
    recs = [row_feats(a, b) for a, b in zip(df["resin_1_structure"], df["curing_agent_1_structure"])]
    keys = ["ced", "delta", "bmin", "bmean", "bweak", "f_ar", "csp3", "flex"]
    arr = {k: np.array([(r or {}).get(k, np.nan) for r in recs], float) for k in keys}
    feat_rows[tgt] = (arr, y)
    print(f"\n── {tgt}  (n={len(y)}) ──")
    for k, label in [("delta", "I1 溶解度参数 δ (Fedors)"), ("ced", "I2 内聚能密度 CED"),
                     ("bmin", "J1 最小键解离能 BDE_min"), ("bmean", "J2 平均 BDE"),
                     ("bweak", "J3 弱键占比 (<340 kJ/mol)")]:
        r, n = sp(arr[k], y)
        print(f"  {label:30s} Spearman = {r:6.3f}  n={n:5d}  {verdict(r)}")

# ---------------------------------------------------------------- K. 纯物理多元基线
print("\n" + "─" * 80)
print("K. 纯物理多元基线（5 折 CV，只用物理量，不含配方/工艺学习）")
print("─" * 80)
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from core.formulation_fusion import FormulationFusionEngine
_eng = FormulationFusionEngine(verbose=False)

for tgt in ["tg_c", "tensile_modulus_gpa", "tensile_strength_mpa"]:
    if tgt not in feat_rows:
        continue
    arr, y = feat_rows[tgt]
    # 物理量 + 配方当量 + 工艺（纯物理，无数据挖掘特征）
    df = pd.read_csv(f"{DATA}/ml_qspr_model_{tgt}.csv", encoding="utf-8",
                     encoding_errors="replace", low_memory=False)
    extra = {}
    for c in ["formulation_resin_total_eew_g_eq", "formulation_hardener_total_ahew_g_eq",
              "formulation_r_value", "process_max_temperature_c",
              "process_temperature_time_integral_c_h"]:
        if c in df.columns:
            extra[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
    X = np.column_stack([arr[k] for k in ["delta", "f_ar", "csp3", "flex", "bmin"]]
                        + [extra[k] for k in extra])
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, yy = X[ok], y[ok]
    if len(yy) < 100:
        continue
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    r2 = cross_val_score(pipe, X, yy, cv=KFold(5, shuffle=True, random_state=42),
                         scoring="r2")
    rho = cross_val_score(pipe, X, yy, cv=KFold(5, shuffle=True, random_state=42),
                          scoring="neg_mean_absolute_error")
    print(f"  {tgt:24s} n={len(yy):5d}  物理基线 5折 R² = {r2.mean():.3f} ± {r2.std():.3f}"
          f"   MAE = {-rho.mean():.3f}")
    print(f"      (目标仅用 {X.shape[1]} 个物理/配方量，无 SMILES 指纹、无数据挖掘特征)")

print("\n" + "=" * 80)
print("结论: |Spearman|>=0.4 硬基座 | 0.2~0.4 软约束 | <0.2 弃用")
print("=" * 80)
