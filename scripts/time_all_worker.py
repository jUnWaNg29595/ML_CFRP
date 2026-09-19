# -*- coding: utf-8 -*-
"""全量真实 worker 计时: 验证防卡死 + 找出最慢行"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

from core.molecular_features import _rdkit3d_feature_worker

DATA = "data_fixed/latest_data_fixed.csv"
df = pd.read_csv(DATA)
struct_cols = [c for c in df.columns if c.endswith("_structure")]

slow = []
t_all = time.time()
for col in struct_cols:
    t_col = time.time()
    fail = 0
    for i, s in enumerate(df[col].tolist()):
        if pd.isna(s) or not str(s).strip():
            continue
        t0 = time.time()
        out = _rdkit3d_feature_worker(s, coulomb_top_k=20)
        dt = time.time() - t0
        if out is None:
            fail += 1
        if dt > 1.0:
            slow.append((col, i, round(dt, 1), str(s)[:60]))
    print(f"{col}: 总耗时 {time.time()-t_col:.1f}s, 失败 {fail}")

print(f"\n=== 全部列合计 {time.time()-t_all:.1f}s ===")
print(f">1s 的慢行 {len(slow)} 条:")
for c, i, dt, s in sorted(slow, key=lambda x: -x[2])[:15]:
    print(f"  {c} 行{i} {dt}s: {s}")

# 行505 详细诊断
print("\n=== resin_2_structure 行505 诊断 ===")
from core.smiles_utils import parse_chemical_string
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
conv = "O[Si](CCC1CCC2OC2C1)OB(O)(O)c1ccc(c2ccccc2)cc1"
m = Chem.MolFromSmiles(conv, sanitize=False)
if m:
    try:
        Chem.SanitizeMol(m)
        print("sanitize OK?")
    except Exception as e:
        print("sanitize 失败:", type(e).__name__, e)
