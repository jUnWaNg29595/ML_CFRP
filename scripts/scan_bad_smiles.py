# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""扫描全量数据的 SMILES/BigSMILES: 无效行 + 超大分子行(3D卡死源)"""
import time, sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

from core.smiles_utils import convert_to_smiles, parse_chemical_string, _strictly_parseable, split_smiles_cell, bigsmiles_to_smiles

DATA = "cache/session_snapshots/latest_data.csv"
MAX_ATOMS = 150  # 重原子上限(AddHs前), 超过则3D嵌入代价不可控

df = pd.read_csv(DATA)
struct_cols = [c for c in df.columns if c.endswith("_structure")]
print(f"数据 {df.shape[0]} 行, structure 列: {struct_cols}")

report = []
t_all = time.time()

for col in struct_cols:
    for i, s in enumerate(df[col].tolist()):
        raw = s
        if s is None or (isinstance(s, float) and np.isnan(s)) or (pd.isna(s) if not isinstance(s, (list, tuple)) else False):
            continue
        s = str(s).strip()
        if not s or s.lower() in {"nan", "none", "na", "<na>"}:
            continue
        t0 = time.time()
        is_big = "{" in s or "[<" in s or "[>" in s
        try:
            conv = convert_to_smiles(s, fmt="auto") or s
        except Exception as e:
            conv = None
        conv_t = time.time() - t0

        # 转换产物解析 + 原子数(不嵌入, 快速)
        entry = {"col": col, "row": i, "raw": s[:80], "bigsmiles": is_big,
                 "conv_time": round(conv_t, 2), "conv": None, "status": "", "atoms": None}
        if conv is None or not str(conv).strip():
            entry["status"] = "CONVERT_EMPTY"
            report.append(entry); continue
        entry["conv"] = str(conv)[:60]
        mol = parse_chemical_string(str(conv), repair=True, keep_largest_frag=False)
        if mol is None:
            entry["status"] = "PARSE_FAIL(修后仍无效)"
            report.append(entry); continue
        n_heavy = mol.GetNumAtoms()
        entry["atoms"] = n_heavy
        if n_heavy > MAX_ATOMS:
            entry["status"] = f"TOO_BIG(>{MAX_ATOMS}原子, 3D将卡死)"
            report.append(entry); continue
        if conv_t > 5:
            entry["status"] = f"SLOW_CONVERT({conv_t:.1f}s)"
            report.append(entry); continue

rep = pd.DataFrame(report)
rep.to_csv("scripts/scan_smiles_report.csv", index=False, encoding="utf-8-sig")
print(f"\n扫描完成, 总耗时 {time.time()-t_all:.1f}s, 问题行 {len(rep)} 条")
if len(rep):
    print(rep["status"].value_counts().to_string())
    print("\n--- 明细(每类最多8条) ---")
    for st, grp in rep.groupby("status"):
        print(f"\n[{st}] {len(grp)} 条:")
        for _, r in grp.head(8).iterrows():
            print(f"  {r['col']} 行{r['row']} atoms={r['atoms']} conv_t={r['conv_time']}s")
            print(f"    raw: {r['raw']}")
            if r["conv"]: print(f"    conv: {r['conv']}")
