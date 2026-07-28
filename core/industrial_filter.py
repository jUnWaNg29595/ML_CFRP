# -*- coding: utf-8 -*-
"""工业级固化剂过滤：类别识别、类别感知过滤、训练数据相似度评分。

二阶段策略：
  阶段1 = 类别感知宽松过滤（保留20-30%）
  阶段2 = 训练数据相似度评分排序（保留Top N）

类别自动识别：胺/酸酐/酚/硫醇/咪唑 五类 + 未知
"""

from __future__ import annotations
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, DataStructs
    RDLogger.logger().setLevel(RDLogger.ERROR)
    RDKIT_AVAILABLE = True
except Exception:
    Chem = None
    Descriptors = None
    rdMolDescriptors = None
    AllChem = None
    DataStructs = None
    RDKIT_AVAILABLE = False

# ─── 固化剂类别 SMARTS 定义 ───────────────────────────────────────────
# 每类多个 SMARTS 以覆盖不同结构变体
HARDENER_CLASS_SMARTS: Dict[str, List[str]] = {
    "胺": [
        "[NX3;H2]([#6])[#6]",           # 伯胺
        "[NX3;H1]([#6])([#6])",          # 仲胺
        "[NX3]([#6])([#6])[#6]",         # 叔胺
        "[NX3;H2]C",                     # 脂肪伯胺
        "[NX3;H1]([#6])C",              # 脂肪仲胺
        "N([#6])([#6])[#6]",            # 脂肪叔胺
    ],
    "酸酐": [
        "O=C1OC(=O)C1",                  # 琥珀酸酐环
        "O=C1OC(=O)c2ccccc21",           # 邻苯二甲酸酐
        "O=C1OC(=O)C=C1",                # 马来酸酐
        "[CX3](=[OX1])[OX2][CX3](=[OX1])",  # 线性酸酐
        "O=C1OC(=O)C2CC1C2",            # 降莰烷二酸酐
    ],
    "酚": [
        "[OX2H]c1ccccc1",                # 酚羟基
        "[OX2H]c1ccc([OX2H])cc1",        # 对苯二酚
        "c1ccc([OX2H])cc1",              # 苯酚
        "[OX2H]c1cc([OX2H])cc([OX2H])c1",  # 多酚
    ],
    "硫醇": [
        "[SH]",                          # 巯基
        "[SX2H]",                        # 硫醇
    ],
    "咪唑": [
        "c1cnc[nH]1",                    # 咪唑
        "c1cncn1",                       # 咪唑环（无H）
        "n1ccnc1",                       # 咪唑变体
    ],
}

# 类别典型物化范围（用于类别感知阈值）
CLASS_PROP_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "胺":   {"mol_wt": (60, 600), "logp": (-1.0, 5.0), "heavy": (4, 45)},
    "酸酐": {"mol_wt": (80, 500), "logp": (0.0, 4.5), "heavy": (5, 35)},
    "酚":   {"mol_wt": (80, 500), "logp": (1.0, 6.0), "heavy": (5, 40)},
    "硫醇": {"mol_wt": (100, 800), "logp": (1.0, 5.0), "heavy": (5, 55)},
    "咪唑": {"mol_wt": (60, 300), "logp": (-0.5, 3.0), "heavy": (4, 25)},
}

# 未知类别的保守范围
UNKNOWN_RANGES: Dict[str, Tuple[float, float]] = {
    "mol_wt": (80, 500), "logp": (-1.0, 5.0), "heavy": (4, 40),
}

# 类别优先级（匹配顺序，酸酐优先以避免被胺误匹配）
_CLASS_PRIORITY = ["酸酐", "咪唑", "酚", "硫醇", "胺"]

# 预编译SMARTS模式（延迟初始化）
_COMPILED_SMARTS: Optional[Dict[str, List]] = None


def _get_compiled_smarts() -> Dict[str, List]:
    """获取预编译的SMARTS模式，延迟初始化。"""
    global _COMPILED_SMARTS
    if _COMPILED_SMARTS is not None:
        return _COMPILED_SMARTS
    if not RDKIT_AVAILABLE:
        return {}
    compiled = {}
    for cls_name, smarts_list in HARDENER_CLASS_SMARTS.items():
        patterns = []
        for sma in smarts_list:
            try:
                pat = Chem.MolFromSmarts(sma)
                if pat is not None:
                    patterns.append(pat)
            except Exception:
                pass
        compiled[cls_name] = patterns
    _COMPILED_SMARTS = compiled
    return compiled


def classify_hardener(smiles: str) -> Optional[str]:
    """识别固化剂类别，返回类别名称或 None。

    按优先级匹配：酸酐 > 咪唑 > 酚 > 硫醇 > 胺。
    酸酐优先是因为酸酐结构中可能含有类似胺的氮，但酸酐是更特异的类别。
    """
    if not RDKIT_AVAILABLE or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
    except Exception:
        return None

    compiled = _get_compiled_smarts()
    for cls_name in _CLASS_PRIORITY:
        patterns = compiled.get(cls_name, [])
        for pat in patterns:
            try:
                if mol.HasSubstructMatch(pat):
                    return cls_name
            except Exception:
                continue
    return None


def _classify_single(smiles: str) -> Tuple[str, Optional[str]]:
    """单条分类，返回 (smiles, class_name)。"""
    return (smiles, classify_hardener(smiles))


def batch_classify(
    smiles_list: List[str],
    workers: int = 128,
) -> Dict[str, List[str]]:
    """批量分类固化剂，返回 {类别: [smiles列表]}。

    Args:
        smiles_list: SMILES列表
        workers: 并行线程数

    Returns:
        字典，键为类别名（含"未知"），值为该类别的SMILES列表
    """
    result: Dict[str, List[str]] = {}
    if not smiles_list:
        return result

    if not RDKIT_AVAILABLE:
        result["未知"] = list(smiles_list)
        return result

    if len(smiles_list) < 100:
        for smi in smiles_list:
            cls = classify_hardener(smi)
            key = cls if cls is not None else "未知"
            result.setdefault(key, []).append(smi)
        return result

    actual_workers = min(workers, max(1, len(smiles_list) // 10), os.cpu_count() or 128)
    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        futures = {pool.submit(_classify_single, smi): smi for smi in smiles_list}
        for fut in as_completed(futures):
            try:
                smi, cls = fut.result()
                key = cls if cls is not None else "未知"
                result.setdefault(key, []).append(smi)
            except Exception:
                smi = futures[fut]
                result.setdefault("未知", []).append(smi)

    return result

# ─── 熔点估算（改进的多描述符模型）───────────────────────────────────

def estimate_melting_point(mol) -> float:
    """估算熔点（°C），使用改进的多描述符线性模型。

    基于经验公式，结合分子量、LogP、芳香性、氢键等描述符。
    注意：这是粗略估算，真实熔点需要实验测定。
    """
    if not RDKIT_AVAILABLE or mol is None:
        return 999.0

    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        rot = Descriptors.NumRotatableBonds(mol)
        aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)

        # 改进的线性模型（基于实验数据的粗略拟合）
        # 芳香性提升熔点，氢键提升熔点，可旋转键降低熔点
        est_tm = (
            50.0
            + 0.35 * mw
            + 15.0 * aromatic
            + 20.0 * rings
            - 40.0 * logp
            - 8.0 * rot
            + 5.0 * hbd
            + 3.0 * hba
        )
        return max(-100.0, min(500.0, est_tm))
    except Exception:
        return 999.0


def get_mol_properties(mol) -> Optional[Dict[str, float]]:
    """获取分子的关键物化属性。"""
    if not RDKIT_AVAILABLE or mol is None:
        return None
    try:
        return {
            "mol_wt": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "heavy": mol.GetNumHeavyAtoms(),
            "rotatable": Descriptors.NumRotatableBonds(mol),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "aromatic": rdMolDescriptors.CalcNumAromaticRings(mol),
            "rings": rdMolDescriptors.CalcNumRings(mol),
        }
    except Exception:
        return None


# ─── 类别感知阈值检查 ───────────────────────────────────────────────

def check_category_constraints(
    mol,
    cls_name: Optional[str],
    props: Optional[Dict[str, float]],
    stage1_max_mp: float = 130.0,
) -> Tuple[bool, List[str]]:
    """检查分子是否符合类别感知的物化约束。

    Args:
        mol: RDKit分子对象
        cls_name: 类别名称（胺/酸酐/酚/硫醇/咪唑/None）
        props: 分子属性字典
        stage1_max_mp: 最大熔点阈值

    Returns:
        (是否通过, 失败原因列表)
    """
    if props is None:
        return False, ["无法解析分子属性"]

    # 获取类别对应的阈值范围
    if cls_name and cls_name in CLASS_PROP_RANGES:
        ranges = CLASS_PROP_RANGES[cls_name]
    else:
        ranges = UNKNOWN_RANGES

    failures = []

    # 分子量检查
    mw = props["mol_wt"]
    if not (ranges["mol_wt"][0] <= mw <= ranges["mol_wt"][1]):
        failures.append(f"分子量{mw:.0f}超出{cls_name or '未知'}范围")

    # LogP检查
    logp = props["logp"]
    if not (ranges["logp"][0] <= logp <= ranges["logp"][1]):
        failures.append(f"LogP{logp:.1f}超出{cls_name or '未知'}范围")

    # 重原子数检查
    heavy = props["heavy"]
    if not (ranges["heavy"][0] <= heavy <= ranges["heavy"][1]):
        failures.append(f"重原子数{heavy}超出{cls_name or '未知'}范围")

    # 熔点估算检查
    est_mp = estimate_melting_point(mol)
    if est_mp > stage1_max_mp:
        failures.append(f"估算熔点{est_mp:.0f}°C超过阈值{stage1_max_mp}°C")

    return len(failures) == 0, failures

# ─── PAINS 过滤（酸酐豁免）───────────────────────────────────────────

# 简化版PAINS子结构（排除酸酐）
_PAINS_SMARTS = [
    "[#6]=[#6]-[#6]=[#6]-[#6]=[#6]",   # 过长共轭
    "[#7][#7]=[#6]",                    # 腙
    "[#16][#6](=[#8])[#6]",             # 硫酯
    "[#6](=[#8])[#6](=[#8])",           # 二酮
    "[#7]=[#6]-[#6]#[#7]",              # 烯腈
    "[#16]-[#16]",                      # 二硫键
    "[#6]=[#6]-[#6]#[#6]",              # 烯炔
    "[#7]-[#7]=[#6]",                   # 偶氮
    "[#6]#[#6]",                        # 累积烯烃
]

_COMPILED_PAINS: Optional[List] = None


def _get_pains_patterns() -> List:
    """获取预编译的PAINS模式。"""
    global _COMPILED_PAINS
    if _COMPILED_PAINS is not None:
        return _COMPILED_PAINS
    if not RDKIT_AVAILABLE:
        return []
    patterns = []
    for sma in _PAINS_SMARTS:
        try:
            pat = Chem.MolFromSmarts(sma)
            if pat is not None:
                patterns.append(pat)
        except Exception:
            pass
    _COMPILED_PAINS = patterns
    return patterns


def check_pains(mol, cls_name: Optional[str]) -> bool:
    """检查分子是否含有PAINS子结构（酸酐类豁免）。

    Args:
        mol: RDKit分子对象
        cls_name: 类别名称

    Returns:
        True = 含有PAINS结构，应排除
        False = 无PAINS结构，可保留
    """
    # 酸酐类是有效固化剂，豁免PAINS检查
    if cls_name == "酸酐":
        return False

    patterns = _get_pains_patterns()
    for pat in patterns:
        try:
            if mol.HasSubstructMatch(pat):
                return True
        except Exception:
            continue
    return False


# ─── 阶段1：类别感知宽松过滤 ─────────────────────────────────────────

def stage1_industrial_filter(
    smiles_list: List[str],
    stage1_max_mp: float = 130.0,
    workers: int = 128,
    label: str = "分子",
) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    """阶段1：类别感知宽松过滤。

    类别自动识别 → 类别感知阈值检查 → PAINS排除（酸酐豁免）。

    Args:
        smiles_list: 待过滤的SMILES列表
        stage1_max_mp: 最大熔点阈值
        workers: 并行线程数
        label: 标签（仅用于日志）

    Returns:
        (通过列表, 过滤统计, 类别统计)
    """
    stats: Dict[str, int] = {
        "total": len(smiles_list), "passed": 0,
        "failed_parse": 0, "failed_mol_wt": 0, "failed_logp": 0,
        "failed_heavy": 0, "failed_melting": 0, "failed_pains": 0,
        "failed_unknown": 0,
    }
    class_counts: Dict[str, int] = {}
    passed: List[str] = []

    if not RDKIT_AVAILABLE:
        stats["passed"] = len(smiles_list)
        return list(smiles_list), stats, class_counts

    def _process_one(smi: str) -> Optional[Tuple[str, str, int, int, int, int, int]]:
        """返回 (smi, cls, failed_mol_wt, failed_logp, failed_heavy, failed_melting, failed_pains) 或 None。"""
        if not smi or not str(smi).strip():
            return None
        try:
            mol = Chem.MolFromSmiles(str(smi).strip())
            if mol is None:
                return None
        except Exception:
            return None

        cls_name = classify_hardener(str(smi).strip()) or "未知"
        props = get_mol_properties(mol)
        if props is None:
            return None

        # 类别感知阈值检查
        passed_flag, failures = check_category_constraints(mol, cls_name, props, stage1_max_mp)

        # PAINS检查（酸酐豁免）
        pains_flag = check_pains(mol, cls_name)

        # 计数失败类型
        f_mw = 1 if "分子量" in str(failures) else 0
        f_logp = 1 if "LogP" in str(failures) else 0
        f_heavy = 1 if "重原子" in str(failures) else 0
        f_mp = 1 if "熔点" in str(failures) else 0
        f_pains = 1 if pains_flag else 0

        if passed_flag and not pains_flag:
            return (str(smi).strip(), cls_name, 0, 0, 0, 0, 0)
        else:
            return (None, cls_name, f_mw, f_logp, f_heavy, f_mp, f_pains)

    actual_workers = min(workers, max(1, len(smiles_list) // 10), os.cpu_count() or 128)

    if len(smiles_list) < 200:
        for smi in smiles_list:
            r = _process_one(smi)
            if r is None:
                stats["failed_parse"] += 1
            elif r[0] is not None:
                passed.append(r[0])
                class_counts[r[1]] = class_counts.get(r[1], 0) + 1
                stats["passed"] += 1
            else:
                class_counts[r[1]] = class_counts.get(r[1], 0) + 1
                stats["failed_mol_wt"] += r[2]
                stats["failed_logp"] += r[3]
                stats["failed_heavy"] += r[4]
                stats["failed_melting"] += r[5]
                stats["failed_pains"] += r[6]
    else:
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            fut_map = {pool.submit(_process_one, smi): smi for smi in smiles_list}
            for fut in as_completed(fut_map):
                try:
                    r = fut.result()
                    if r is None:
                        stats["failed_parse"] += 1
                    elif r[0] is not None:
                        passed.append(r[0])
                        class_counts[r[1]] = class_counts.get(r[1], 0) + 1
                        stats["passed"] += 1
                    else:
                        class_counts[r[1]] = class_counts.get(r[1], 0) + 1
                        stats["failed_mol_wt"] += r[2]
                        stats["failed_logp"] += r[3]
                        stats["failed_heavy"] += r[4]
                        stats["failed_melting"] += r[5]
                        stats["failed_pains"] += r[6]
                except Exception:
                    stats["failed_parse"] += 1

    # 去重
    seen = set()
    deduped = []
    for s in passed:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    stats["passed"] = len(deduped)
    stats["dedup_removed"] = len(passed) - len(deduped)

    return deduped, stats, class_counts

# ─── 训练数据相似度评分 ─────────────────────────────────────────────

def extract_known_hardeners_from_training_data(
    df: pd.DataFrame,
    hardener_cols: List[str],
) -> Set[str]:
    """从训练数据中提取已知固化剂的SMILES集合。

    从多列（如 curing_agent_smiles_1 ~ curing_agent_smiles_10）中
    提取所有非空SMILES，去重后返回。

    Args:
        df: 训练数据DataFrame
        hardener_cols: 固化剂SMILES列的列名列表

    Returns:
        已知固化剂SMILES的集合
    """
    if df is None or df.empty:
        return set()
    known_set: Set[str] = set()
    for col in hardener_cols:
        if col in df.columns:
            vals = df[col].dropna().astype(str).str.strip()
            known_set.update(vals[vals != "nan"].values)
    return known_set


def _compute_fingerprints(smiles_list: List[str], workers: int = 128) -> List:
    """批量计算Morgan指纹。"""
    if not RDKIT_AVAILABLE or not smiles_list:
        return []

    fps = []
    n_workers = min(workers, max(1, len(smiles_list) // 10), os.cpu_count() or 128)

    def _fp(smi: str):
        try:
            mol = Chem.MolFromSmiles(str(smi).strip())
            if mol is None:
                return None
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        except Exception:
            return None

    if len(smiles_list) < 200:
        for smi in smiles_list:
            fp = _fp(smi)
            if fp is not None:
                fps.append(fp)
        return fps

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        fut_list = [pool.submit(_fp, smi) for smi in smiles_list]
        for fut in as_completed(fut_list):
            try:
                fp = fut.result()
                if fp is not None:
                    fps.append(fp)
            except Exception:
                continue
    return fps


def compute_similarity_scores(
    candidates: List[str],
    known_set: Set[str],
    workers: int = 128,
    batch_size: int = 500,
) -> pd.DataFrame:
    """计算候选分子与已知固化剂的平均Tanimoto相似度。

    使用ECFP4 Morgan指纹（半径2, 2048位），计算每个候选分子与
    所有已知固化剂的平均Tanimoto相似度。

    Args:
        candidates: 候选SMILES列表
        known_set: 已知固化剂SMILES集合
        workers: 并行线程数
        batch_size: 每批处理的分子数

    Returns:
        DataFrame，列: ["smiles", "avg_similarity", "max_similarity", "min_similarity", "class"]
    """
    if not RDKIT_AVAILABLE or not candidates or not known_set:
        return pd.DataFrame({"smiles": candidates, "avg_similarity": 0.0, "max_similarity": 0.0, "min_similarity": 0.0, "class": ""})

    # 预计算已知固化剂的指纹
    known_fps = _compute_fingerprints(list(known_set), workers)
    if not known_fps:
        return pd.DataFrame({"smiles": candidates, "avg_similarity": 0.0, "max_similarity": 0.0, "min_similarity": 0.0, "class": ""})

    # 分批处理候选分子
    results = []
    n_workers = min(workers, max(1, len(candidates) // 10), os.cpu_count() or 128)

    def _process_batch(batch_smiles: List[str]) -> List[Dict]:
        batch_result = []
        for smi in batch_smiles:
            try:
                mol = Chem.MolFromSmiles(str(smi).strip())
                if mol is None:
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                similarities = [DataStructs.TanimotoSimilarity(fp, kfp) for kfp in known_fps]
                if not similarities:
                    continue
                batch_result.append({
                    "smiles": str(smi).strip(),
                    "avg_similarity": float(np.mean(similarities)),
                    "max_similarity": float(np.max(similarities)),
                    "min_similarity": float(np.min(similarities)),
                })
            except Exception:
                continue
        return batch_result

    # 分批
    all_batches = [candidates[i:i+batch_size] for i in range(0, len(candidates), batch_size)]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        fut_list = [pool.submit(_process_batch, batch) for batch in all_batches]
        for fut in as_completed(fut_list):
            try:
                results.extend(fut.result())
            except Exception:
                continue

    if not results:
        return pd.DataFrame({"smiles": candidates, "avg_similarity": 0.0, "max_similarity": 0.0, "min_similarity": 0.0, "class": ""})

    df = pd.DataFrame(results)
    # 添加类别信息
    df["class"] = df["smiles"].apply(lambda s: classify_hardener(s) or "未知")
    return df

# ─── 二阶段 Pipeline ──────────────────────────────────────────────

@dataclass
class IndustrialFilterResult:
    """工业过滤结果。"""
    passed: List[str]
    stats: Dict[str, int]
    score_df: pd.DataFrame
    class_counts: Dict[str, int]


def pipeline_industrial_filter(
    smiles_list: List[str],
    known_set: Set[str],
    stage1_max_mp: float = 130.0,
    stage2_top_n: int = 2000,
    workers: int = 128,
    label: str = "分子",
) -> Tuple[List[str], Dict[str, int], pd.DataFrame]:
    """二阶段工业级过滤Pipeline。

    阶段1: 类别感知宽松过滤（保留20-30%）
    阶段2: 训练数据相似度评分排序（保留Top N）

    Args:
        smiles_list: 待过滤的SMILES列表
        known_set: 已知固化剂SMILES集合（从训练数据提取）
        stage1_max_mp: 最大熔点阈值
        stage2_top_n: 阶段2保留的Top N数量
        workers: 并行线程数
        label: 标签（仅用于日志）

    Returns:
        (最终通过列表, 过滤统计, 相似度评分DataFrame)
    """
    if not smiles_list:
        return [], {"total": 0, "passed": 0, "stage1_passed": 0}, pd.DataFrame()

    # 阶段1：类别感知宽松过滤
    stage1_passed, stats, class_counts = stage1_industrial_filter(
        smiles_list, stage1_max_mp=stage1_max_mp, workers=workers, label=label,
    )
    stats["stage1_passed"] = len(stage1_passed)

    if not stage1_passed:
        return [], stats, pd.DataFrame()

    # 阶段2：相似度评分排序（仅当有已知数据时）
    if known_set and len(known_set) >= 3:
        score_df = compute_similarity_scores(
            stage1_passed, known_set, workers=workers,
        )
        if not score_df.empty:
            score_df = score_df.sort_values("avg_similarity", ascending=False)
            score_df["rank"] = range(1, len(score_df) + 1)
            top_n = min(stage2_top_n, len(score_df))
            top_smiles = set(score_df.head(top_n)["smiles"].tolist())
            final_passed = [s for s in stage1_passed if s in top_smiles]
            stats["stage2_kept"] = len(final_passed)
            stats["stage2_removed"] = len(stage1_passed) - len(final_passed)
            return final_passed, stats, score_df
        else:
            stats["stage2_skipped"] = len(stage1_passed)
            return stage1_passed, stats, pd.DataFrame()
    else:
        stats["stage2_skipped_no_data"] = len(stage1_passed)
        return stage1_passed, stats, pd.DataFrame()


# ─── UI 渲染 ──────────────────────────────────────────────────────

def render_industrial_filter_ui(st_module, label: str = "固化剂"):
    """在Streamlit中渲染工业过滤控件，返回过滤配置。

    与旧版 `render_industrial_filter_ui` 接口兼容，但内部已更新。
    """
    with st_module.expander(f"工业级候选过滤（{label}）", expanded=False):
        st_module.caption("二阶段智能过滤：类别识别 → 类别感知阈值 → PAINS排除（酸酐豁免）→ 训练数据相似度评分排序")
        col1, col2, col3 = st_module.columns(3)
        with col1:
            enable_filter = st_module.checkbox(f"启用{label}工业过滤", value=True, key=f"vs_ind_filter_enable_{label}")
            max_mp = st_module.number_input(f"最高估算熔点(°C)", -100, 500, 130, key=f"vs_ind_max_mp_{label}")
        with col2:
            min_mw = st_module.number_input(f"最小分子量", 20, 500, 80, key=f"vs_ind_min_mw_{label}")
            max_mw = st_module.number_input(f"最大分子量", 100, 2000, 800, key=f"vs_ind_max_mw_{label}")
        with col3:
            min_logp = st_module.number_input(f"最小LogP", -5, 5, -1, key=f"vs_ind_min_logp_{label}")
            max_logp = st_module.number_input(f"最大LogP", 0, 15, 6, key=f"vs_ind_max_logp_{label}")
        st_module.caption("已启用：类别自动识别 + 类别感知阈值 + PAINS排除（酸酐豁免）+ 相似度评分排序")
    return enable_filter, {
        "max_melting_point": float(max_mp),
        "min_mol_wt": float(min_mw),
        "max_mol_wt": float(max_mw),
        "min_logp": float(min_logp),
        "max_logp": float(max_logp),
    }


def filter_industrial_candidates(
    smiles_list,
    min_mol_wt=80, max_mol_wt=800,
    min_logp=-1, max_logp=6,
    min_heavy_atoms=4, max_heavy_atoms=60,
    max_rotatable_bonds=15,
    max_melting_point=130, min_melting_point=-50,
    reject_lipinski_violators=True,
    reject_pains=True,
    label="分子",
):
    """兼容旧版接口的过滤函数，内部调用新的二阶段pipeline。

    当不需要相似度评分时使用此函数。
    """
    if not smiles_list:
        return list(smiles_list or []), {"total": len(smiles_list or []), "passed": 0}

    # 使用阶段1过滤（无相似度评分）
    passed, stats, class_counts = stage1_industrial_filter(
        list(smiles_list),
        stage1_max_mp=float(max_melting_point),
        workers=12,
        label=label,
    )
    return passed, stats
