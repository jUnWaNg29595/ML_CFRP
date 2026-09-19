# -*- coding: utf-8 -*-
"""修复数据中的无效 SMILES/BigSMILES:
- 七元芳环 -> 缩成六元芳环(保留 BigSMILES 语法/端基/取代基)
- 无法机械修复的(如硼 4 配位超标) -> 标记人工核对
产出: data_fixed/latest_data_fixed.csv + fix_report.csv
"""
import sys, os, re, shutil, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

from core.smiles_utils import parse_chemical_string, _strictly_parseable, convert_to_smiles

SRC = "cache/session_snapshots/latest_data.csv"
OUT_DIR = "data_fixed"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(SRC)
struct_cols = [c for c in df.columns if c.endswith("_structure")]

def is_bad(s) -> bool:
    """该字符串经转换链后是否无法得到有效分子"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return False
    s = str(s).strip()
    if not s or s.lower() in {"nan", "none", "na", "<na>"}:
        return False
    conv = convert_to_smiles(s, fmt="auto")
    if not conv or not str(conv).strip():
        return True
    return parse_chemical_string(str(conv), repair=True, keep_largest_frag=False) is None

def shrink_seven_aromatic(s: str):
    """七元芳环修复: 枚举删除裸芳碳(缩环), 返回第一个严格可解析的候选。
    只删 'ccN'/'cccN' 闭环尾部区域的裸 c, 不动取代基与 BigSMILES 语法。"""
    cands = set()
    # c1 ... cc1 闭环: 'ccN' 删其中一个 c
    for m in re.finditer(r'(?<![\w\[|\]])cc(?=[0-9])', s):
        i = m.start()
        cands.add(s[:i+1] + s[i+2:])   # 删第 2 个 c
        cands.add(s[:i] + s[i+1:])     # 删第 1 个 c
    # 'cccN' 再补一档
    for m in re.finditer(r'(?<![\w\[|\]])ccc(?=[0-9])', s):
        i = m.start()
        cands.add(s[:i+1] + s[i+2:])
        cands.add(s[:i+2] + s[i+3:])
    for c in cands:
        # 注意: 候选仍是 BigSMILES 原串(含{}), 必须走转换链校验, 不能直接 RDKit 解析
        if not is_bad(c):
            return c
    return None

def fix_boron_valence(s: str):
    """砸 4 配位中性硼 -> [B-] 硼酸酯负离子(化学上合法的最小语义改动)"""
    if "[B](" in s:
        cand = s.replace("[B](", "[B-](")
        if not is_bad(cand):
            return cand
    return None

fix_report = []
n_fixed = n_manual = 0
for col in struct_cols:
    for i in df.index:
        raw = df.at[i, col]
        if not is_bad(raw):
            continue
        raw_str = str(raw).strip()
        entry = {"col": col, "row": int(i), "raw": raw_str, "fixed": "", "status": ""}
        fixed = shrink_seven_aromatic(raw_str)
        if fixed and not is_bad(fixed):
            df.at[i, col] = fixed
            entry["fixed"] = fixed
            entry["status"] = "已修复(七元芳环缩环为六元, 保留BigSMILES语法)"
            n_fixed += 1
        else:
            # 兑底: 用转换产物做缩环修复, 直接替换为普通 SMILES(丢聚合物语法但保特征可用)
            conv = convert_to_smiles(raw_str, fmt="auto") or ""
            fixed_conv = shrink_seven_aromatic(str(conv)) if conv else None
            if fixed_conv and not is_bad(fixed_conv):
                df.at[i, col] = fixed_conv
                entry["fixed"] = fixed_conv
                entry["status"] = "已修复(转换为普通SMILES后缩环; 聚合物端基语法丢失)"
                n_fixed += 1
            else:
                cand_b = fix_boron_valence(raw_str)
                if cand_b:
                    df.at[i, col] = cand_b
                    entry["fixed"] = cand_b
                    entry["status"] = "已修复(砸4配位中性B改为[B-]砸酸酯负离子)"
                    n_fixed += 1
                else:
                    entry["fixed"] = ""
                    entry["status"] = "无法自动修复(建议人工核对; 提取时该行特征将为NaN)"
                    n_manual += 1
        fix_report.append(entry)

rep = pd.DataFrame(fix_report)
out_csv = os.path.join(OUT_DIR, "latest_data_fixed.csv")
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
rep.to_csv(os.path.join(OUT_DIR, "fix_report.csv"), index=False, encoding="utf-8-sig")

print(f"修复完成: 自动修复 {n_fixed} 条, 需人工核对 {n_manual} 条")
print(f"修复后数据: {out_csv}")
if len(rep):
    for _, r in rep.iterrows():
        print(f"\n[{r['status']}] {r['col']} 行{r['row']}")
        print(f"  原: {r['raw']}")
        if r["fixed"]:
            print(f"  新: {r['fixed']}")
