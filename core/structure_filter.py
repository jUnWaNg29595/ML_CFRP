# -*- coding: utf-8 -*-
"""结构合理性过滤模块。

在 PubChem 拉取后、工业级过滤前，排除高反应性基团、高张力结构、
异常杂原子等不合理的候选分子。

阶段1：硬过滤（快速排除）
阶段2：结构合理性评分（SA Score + 张力能 + 环复杂度）
阶段3：综合排序与输出
"""

from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDLogger.logger().setLevel(RDLogger.ERROR)
    RDKIT_AVAILABLE = True
except Exception:
    Chem = None
    Descriptors = None
    rdMolDescriptors = None
    RDKIT_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# 元素白名单
# ─────────────────────────────────────────────────────────────

ALLOWED_ELEMENTS: Set[int] = {1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
# H, C, N, O, F, Si, P, S, Cl, Br, I

# ─────────────────────────────────────────────────────────────
# 高反应性基团 SMARTS
# ─────────────────────────────────────────────────────────────

REACTIVE_GROUP_SMARTS: Dict[str, str] = {
    # 基础危险基团
    "过氧化物": "OO",
    "重氮": "[N]=[N+]",
    "叠氮": "[N-]=[N+]=[N-]",
    "酰卤": "C(=O)[Cl,Br,I]",
    "异氰酸酯": "N=C=O",
    # 扩展危险基团
    "醛基": "[CH]=O",
    "硫醚(非硫醇)": "S[#6]~[#6]",
    "迈克尔受体": "C=CC(=O)",
    # 不稳定结构
    "烯醇": "C=C-O",
    "缩醛": "[#6]O[#6]O[#6]",
    "缩酮": "[#6]O[#6](O[#6])[#6]",
}

_COMPILED_REACTIVE: Optional[Dict[str, List]] = None


def _get_reactive_patterns() -> Dict[str, List]:
    """获取预编译的高反应性基团模式。"""
    global _COMPILED_REACTIVE
    if _COMPILED_REACTIVE is not None:
        return _COMPILED_REACTIVE
    if not RDKIT_AVAILABLE:
        return {}
    compiled = {}
    for name, sma in REACTIVE_GROUP_SMARTS.items():
        try:
            pat = Chem.MolFromSmarts(sma)
            if pat is not None:
                compiled[name] = [pat]
        except Exception:
            pass
    _COMPILED_REACTIVE = compiled
    return compiled

# ─────────────────────────────────────────────────────────────
# 高张力结构 SMARTS
# ─────────────────────────────────────────────────────────────

HIGH_STRAIN_SMARTS: Dict[str, str] = {
    "累积烯": "C=C=C",
    "环炔(小环)": "C1C#CC1",
}

_COMPILED_STRAIN: Optional[Dict[str, List]] = None


def _get_strain_patterns() -> Dict[str, List]:
    """获取预编译的高张力结构模式。"""
    global _COMPILED_STRAIN
    if _COMPILED_STRAIN is not None:
        return _COMPILED_STRAIN
    if not RDKIT_AVAILABLE:
        return {}
    compiled = {}
    for name, sma in HIGH_STRAIN_SMARTS.items():
        try:
            pat = Chem.MolFromSmarts(sma)
            if pat is not None:
                compiled[name] = [pat]
        except Exception:
            pass
    _COMPILED_STRAIN = compiled
    return compiled

# ─────────────────────────────────────────────────────────────
# 官能团数量限制
# ─────────────────────────────────────────────────────────────

FUNCTIONAL_GROUP_LIMITS: Dict[str, Dict] = {
    "羟基": {"smarts": "[OH]", "max": 4},
    "羧基": {"smarts": "C(=O)[OH]", "max": 2},
    "氨基": {"smarts": "[NX3;H2,H1]", "max": 4},
    "酯基": {"smarts": "C(=O)O[#6]", "max": 3},
    "环氧基": {"smarts": "[O;r3]1[C;r3][C;r3]1", "max": 8},
}

_COMPILED_FG: Optional[Dict[str, List]] = None


def _get_fg_patterns() -> Dict[str, List]:
    """获取预编译的官能团模式。"""
    global _COMPILED_FG
    if _COMPILED_FG is not None:
        return _COMPILED_FG
    if not RDKIT_AVAILABLE:
        return {}
    compiled = {}
    for name, cfg in FUNCTIONAL_GROUP_LIMITS.items():
        try:
            pat = Chem.MolFromSmarts(cfg["smarts"])
            if pat is not None:
                compiled[name] = [pat]
        except Exception:
            pass
    _COMPILED_FG = compiled
    return compiled

# ─────────────────────────────────────────────────────────────
# 杂原子比例限制
# ─────────────────────────────────────────────────────────────

MAX_HETEROATOM_RATIO = 1.5  # 杂原子数/碳原子数


# ─────────────────────────────────────────────────────────────
# 检查函数
# ─────────────────────────────────────────────────────────────

def check_allowed_elements(mol) -> Tuple[bool, List[str]]:
    """检查分子是否只包含允许的元素。"""
    if not RDKIT_AVAILABLE or mol is None:
        return True, []
    
    forbidden = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num not in ALLOWED_ELEMENTS:
            forbidden.append(Chem.GetPeriodicTable().GetElementSymbol(atomic_num))
    
    return len(forbidden) == 0, forbidden


def check_reactive_groups(mol) -> List[str]:
    """检查分子是否含有高反应性基团，返回发现的基团名称列表。"""
    if not RDKIT_AVAILABLE or mol is None:
        return []
    
    found = []
    patterns = _get_reactive_patterns()
    for name, pats in patterns.items():
        for pat in pats:
            try:
                if mol.HasSubstructMatch(pat):
                    found.append(name)
                    break
            except Exception:
                continue
    return found


def check_high_strain(mol) -> Tuple[bool, List[str]]:
    """检查分子是否含有高张力结构，返回 (是否高张力, 结构列表)。"""
    if not RDKIT_AVAILABLE or mol is None:
        return False, []
    
    found = []
    patterns = _get_strain_patterns()
    for name, pats in patterns.items():
        for pat in pats:
            try:
                if mol.HasSubstructMatch(pat):
                    found.append(name)
                    break
            except Exception:
                continue
    
    return len(found) > 0, found


def check_functional_group_limits(mol) -> Tuple[bool, Dict[str, int]]:
    """检查官能团数量是否超限，返回 (是否通过, 各官能团计数)。"""
    if not RDKIT_AVAILABLE or mol is None:
        return True, {}
    
    counts = {}
    patterns = _get_fg_patterns()
    
    for name, pats in patterns.items():
        count = 0
        for pat in pats:
            try:
                count = len(mol.GetSubstructMatches(pat))
            except Exception:
                count = 0
        counts[name] = count
        
        max_allowed = FUNCTIONAL_GROUP_LIMITS[name]["max"]
        if count > max_allowed:
            return False, counts
    
    return True, counts


def check_heteroatom_ratio(mol) -> Tuple[bool, float]:
    """检查杂原子比例，返回 (是否通过, 比例值)。"""
    if not RDKIT_AVAILABLE or mol is None:
        return True, 0.0
    
    c_count = 0
    hetero_count = 0
    
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 6:
            c_count += 1
        elif atomic_num not in {1}:  # 排除氢
            hetero_count += 1
    
    if c_count == 0:
        return False, 999.0
    
    ratio = hetero_count / c_count
    return ratio <= MAX_HETEROATOM_RATIO, ratio


# ─────────────────────────────────────────────────────────────
# 硬过滤主函数
# ─────────────────────────────────────────────────────────────

def hard_filter_structure(
    smiles: str,
    check_elements: bool = True,
    check_reactive: bool = True,
    check_strain: bool = True,
    check_fg_limits: bool = True,
    check_hetero_ratio: bool = True,
) -> Tuple[bool, Dict]:
    """对单个 SMILES 执行硬过滤检查。
    
    Returns:
        (是否通过, 详细信息字典)
    """
    if not RDKIT_AVAILABLE or not smiles:
        return True, {"skipped": True}
    
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return False, {"parse_failed": True}
    except Exception:
        return False, {"parse_failed": True}
    
    info = {}
    
    # 1. 元素白名单
    if check_elements:
        elem_ok, forbidden = check_allowed_elements(mol)
        info["forbidden_elements"] = forbidden
        if not elem_ok:
            info["reject_reason"] = "forbidden_elements"
            return False, info
    
    # 2. 高反应性基团
    if check_reactive:
        reactive = check_reactive_groups(mol)
        info["reactive_groups"] = reactive
        if reactive:
            info["reject_reason"] = "reactive_groups"
            return False, info
    
    # 3. 高张力结构（check_high_strain返回True表示有高张力）
    if check_strain:
        has_high_strain, strain_list = check_high_strain(mol)
        info["strain_structures"] = strain_list
        if has_high_strain:
            info["reject_reason"] = "high_strain"
            return False, info
    
    # 4. 官能团数量限制
    if check_fg_limits:
        fg_ok, fg_counts = check_functional_group_limits(mol)
        info["functional_group_counts"] = fg_counts
        if not fg_ok:
            info["reject_reason"] = "fg_limit_exceeded"
            return False, info
    
    # 5. 杂原子比例
    if check_hetero_ratio:
        ratio_ok, ratio = check_heteroatom_ratio(mol)
        info["heteroatom_ratio"] = ratio
        if not ratio_ok:
            info["reject_reason"] = "heteroatom_ratio"
            return False, info
    
    info["reject_reason"] = None
    return True, info


def batch_hard_filter(
    smiles_list: List[str],
    check_elements: bool = True,
    check_reactive: bool = True,
    check_strain: bool = True,
    check_fg_limits: bool = True,
    check_hetero_ratio: bool = True,
    workers: int = 128,
) -> Tuple[List[str], Dict[str, int], List[Dict]]:
    """批量硬过滤。
    
    Returns:
        (通过列表, 统计字典, 详细信息列表)
    """
    if not smiles_list:
        return [], {"total": 0, "passed": 0}, []
    
    stats = {
        "total": len(smiles_list),
        "passed": 0,
        "failed_parse": 0,
        "failed_elements": 0,
        "failed_reactive": 0,
        "failed_strain": 0,
        "failed_fg": 0,
        "failed_hetero": 0,
    }
    
    passed = []
    details = []
    
    def _filter_one(smi: str):
        ok, info = hard_filter_structure(
            smi,
            check_elements=check_elements,
            check_reactive=check_reactive,
            check_strain=check_strain,
            check_fg_limits=check_fg_limits,
            check_hetero_ratio=check_hetero_ratio,
        )
        return (smi, ok, info)
    
    if len(smiles_list) < 100:
        for smi in smiles_list:
            smi_clean, ok, info = _filter_one(smi)
            details.append({"smiles": smi_clean, "passed": ok, **info})
            if ok:
                passed.append(smi_clean)
                stats["passed"] += 1
            else:
                reason = info.get("reject_reason", "unknown")
                if reason == "parse_failed":
                    stats["failed_parse"] += 1
                elif reason == "forbidden_elements":
                    stats["failed_elements"] += 1
                elif reason == "reactive_groups":
                    stats["failed_reactive"] += 1
                elif reason == "high_strain":
                    stats["failed_strain"] += 1
                elif reason == "fg_limit_exceeded":
                    stats["failed_fg"] += 1
                elif reason == "heteroatom_ratio":
                    stats["failed_hetero"] += 1
    else:
        actual_workers = min(workers, max(1, len(smiles_list) // 10), os.cpu_count() or 128)
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            fut_map = {pool.submit(_filter_one, smi): smi for smi in smiles_list}
            for fut in as_completed(fut_map):
                try:
                    smi_clean, ok, info = fut.result()
                    details.append({"smiles": smi_clean, "passed": ok, **info})
                    if ok:
                        passed.append(smi_clean)
                        stats["passed"] += 1
                    else:
                        reason = info.get("reject_reason", "unknown")
                        if reason == "parse_failed":
                            stats["failed_parse"] += 1
                        elif reason == "forbidden_elements":
                            stats["failed_elements"] += 1
                        elif reason == "reactive_groups":
                            stats["failed_reactive"] += 1
                        elif reason == "high_strain":
                            stats["failed_strain"] += 1
                        elif reason == "fg_limit_exceeded":
                            stats["failed_fg"] += 1
                        elif reason == "heteroatom_ratio":
                            stats["failed_hetero"] += 1
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
    
    return deduped, stats, details

# ─────────────────────────────────────────────────────────────
# SA Score 计算
# ─────────────────────────────────────────────────────────────

def calculate_sa_score(mol) -> float:
    """计算合成可及性评分 (1-10)。

    简化版 SA Score 估算，基于分子复杂度指标。

    Args:
        mol: RDKit 分子对象

    Returns:
        SA Score 值 (1-10)
    """
    if not RDKIT_AVAILABLE or mol is None:
        return 10.0
    
    try:
        n_heavy = mol.GetNumHeavyAtoms()
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        n_hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
        n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
        
        score = (
            1.0
            + 0.05 * n_heavy
            + 0.5 * n_rings
            + 0.3 * n_stereo
            + 0.1 * n_hetero
            + 0.8 * n_bridgehead
            + 0.5 * n_spiro
            + 0.05 * n_rotatable
        )
        return max(1.0, min(10.0, score))
    except Exception:
        return 10.0


def batch_calculate_sa_scores(
    smiles_list: List[str],
    workers: int = 128,
) -> Dict[str, float]:
    """批量计算 SA Score。
    
    Args:
        smiles_list: SMILES 列表
        workers: 并行线程数
    
    Returns:
        {smiles: sa_score} 字典
    """
    if not RDKIT_AVAILABLE or not smiles_list:
        return {s: 10.0 for s in smiles_list}
    
    results = {}
    
    def _calc(smi: str):
        try:
            mol = Chem.MolFromSmiles(str(smi).strip())
            if mol is None:
                return (smi, 10.0)
            score = calculate_sa_score(mol)
            return (smi, score)
        except Exception:
            return (smi, 10.0)
    
    if len(smiles_list) < 200:
        for smi in smiles_list:
            s, score = _calc(smi)
            results[s] = score
    else:
        actual_workers = min(workers, max(1, len(smiles_list) // 10), os.cpu_count() or 128)
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            fut_list = [pool.submit(_calc, smi) for smi in smiles_list]
            for fut in as_completed(fut_list):
                try:
                    s, score = fut.result()
                    results[s] = score
                except Exception:
                    pass
    
    return results


def filter_by_sa_score(
    smiles_list: List[str],
    threshold: float = 7.0,
    workers: int = 128,
) -> Tuple[List[str], Dict[str, float], Dict[str, int]]:
    """按 SA Score 阈值过滤并排序。
    
    Args:
        smiles_list: SMILES 列表
        threshold: SA Score 阈值（分子评分 > 此值被排除）
        workers: 并行线程数
    
    Returns:
        (通过列表, {smiles: score} 字典, 统计字典)
    """
    if not smiles_list:
        return [], {}, {"total": 0, "passed": 0}
    
    scores = batch_calculate_sa_scores(smiles_list, workers=workers)
    
    stats = {
        "total": len(smiles_list),
        "passed": 0,
        "failed_sa_threshold": 0,
    }
    
    passed = []
    for smi in smiles_list:
        score = scores.get(smi, 10.0)
        if score <= threshold:
            passed.append(smi)
            stats["passed"] += 1
        else:
            stats["failed_sa_threshold"] += 1
    
    # 按 SA Score 升序排序
    passed.sort(key=lambda s: scores.get(s, 10.0))
    
    return passed, scores, stats

# ─────────────────────────────────────────────────────────────
# 张力能估算
# ─────────────────────────────────────────────────────────────

def estimate_strain_energy(mol) -> float:
    """估算分子张力能 (kcal/mol)。

    通过 MMFF94 力场优化前后能量差估算。
    
    Args:
        mol: RDKit 分子对象
    
    Returns:
        张力能估算值 (kcal/mol)，无法计算时返回 0.0
    """
    if not RDKIT_AVAILABLE or mol is None:
        return 0.0
    
    try:
        from rdkit.Chem import rdForceFieldHelpers
        
        # 检查是否有 MMFF 参数
        if not rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
            return 0.0
        
        # 计算初始能量
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol)
        if ff is None:
            return 0.0
        
        e_initial = ff.CalcEnergy()
        
        # 复制分子并优化
        mol2 = Chem.Mol(mol)
        ff2 = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol2)
        if ff2 is None:
            return 0.0
        
        ff2.Minimize(maxIts=200)
        e_final = ff2.CalcEnergy()
        
        # 张力能 = 初始能量 - 优化后能量
        strain = e_initial - e_final
        return max(0.0, strain)
    except Exception:
        return 0.0


def calc_ring_complexity(mol) -> float:
    """计算环复杂度评分（越高越复杂）。
    
    Args:
        mol: RDKit 分子对象
    
    Returns:
        环复杂度评分
    """
    if not RDKIT_AVAILABLE or mol is None:
        return 0.0
    
    try:
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        
        # 环贡献 + 桥头原子贡献(高) + 螺原子贡献
        return n_rings * 2.0 + n_bridgehead * 3.0 + n_spiro * 4.0
    except Exception:
        return 0.0


def calculate_composite_score(
    mol,
    sa_weight: float = 0.5,
    strain_weight: float = 0.15,
    ring_weight: float = 0.15,
) -> float:
    """计算综合结构合理性评分。
    
    公式: COMPOSITE = SA * 0.5 + STRAIN * 0.15 + RING * 0.15
    
    Args:
        mol: RDKit 分子对象
        sa_weight: SA Score 权重
        strain_weight: 张力能权重
        ring_weight: 环复杂度权重
    
    Returns:
        综合评分 (越高越难合成/不合理)
    """
    sa = calculate_sa_score(mol)
    strain = estimate_strain_energy(mol) / 10.0  # 归一化
    ring = calc_ring_complexity(mol) / 5.0  # 归一化
    
    return sa * sa_weight + strain * strain_weight + ring * ring_weight


def batch_calculate_scores(
    smiles_list: List[str],
    workers: int = 128,
    include_strain: bool = True,
) -> pd.DataFrame:
    """批量计算结构合理性评分。
    
    Args:
        smiles_list: SMILES 列表
        workers: 并行线程数
        include_strain: 是否包含张力能计算（较慢）
    
    Returns:
        DataFrame 包含: smiles, sa_score, ring_complexity, [strain_energy], composite_score
    """
    if not smiles_list:
        return pd.DataFrame()
    
    results = []
    
    def _calc(smi: str):
        try:
            mol = Chem.MolFromSmiles(str(smi).strip())
            if mol is None:
                return None
            
            sa = calculate_sa_score(mol)
            ring = calc_ring_complexity(mol)
            strain = estimate_strain_energy(mol) if include_strain else 0.0
            composite = calculate_composite_score(mol)
            
            return {
                "smiles": str(smi).strip(),
                "sa_score": sa,
                "ring_complexity": ring,
                "strain_energy": strain,
                "composite_score": composite,
            }
        except Exception:
            return None
    
    if len(smiles_list) < 200:
        for smi in smiles_list:
            r = _calc(smi)
            if r:
                results.append(r)
    else:
        actual_workers = min(workers, max(1, len(smiles_list) // 10), os.cpu_count() or 128)
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            fut_list = [pool.submit(_calc, smi) for smi in smiles_list]
            for fut in as_completed(fut_list):
                try:
                    r = fut.result()
                    if r:
                        results.append(r)
                except Exception:
                    pass
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────
# 预设配置
# ─────────────────────────────────────────────────────────────

PRESET_CONFIGS: Dict[str, Dict] = {
    "环氧树脂": {
        "sa_threshold": 7.0,
        "strain_threshold": 30.0,
        "check_elements": True,
        "check_reactive": True,
        "check_strain": True,
        "check_fg_limits": True,
        "check_hetero_ratio": True,
        "epoxide_range": (1, 8),
    },
    "固化剂-胺类": {
        "sa_threshold": 6.0,
        "strain_threshold": 25.0,
        "check_elements": True,
        "check_reactive": True,
        "check_strain": True,
        "check_fg_limits": True,
        "check_hetero_ratio": True,
        "amine_range": (1, 4),
    },
    "固化剂-酸酐类": {
        "sa_threshold": 6.0,
        "strain_threshold": 30.0,
        "check_elements": True,
        "check_reactive": False,  # 酸酐是有效固化剂，豁免部分反应性检查
        "check_strain": True,
        "check_fg_limits": True,
        "check_hetero_ratio": True,
    },
    "固化剂-酚类": {
        "sa_threshold": 6.0,
        "strain_threshold": 25.0,
        "check_elements": True,
        "check_reactive": True,
        "check_strain": True,
        "check_fg_limits": True,
        "check_hetero_ratio": True,
        "hydroxyl_range": (1, 3),
    },
    "固化剂-硫醇": {
        "sa_threshold": 6.0,
        "strain_threshold": 20.0,
        "check_elements": True,
        "check_reactive": True,
        "check_strain": True,
        "check_fg_limits": True,
        "check_hetero_ratio": True,
    },
    "固化剂-咪唑": {
        "sa_threshold": 7.0,
        "strain_threshold": 25.0,
        "check_elements": True,
        "check_reactive": True,
        "check_strain": True,
        "check_fg_limits": True,
        "check_hetero_ratio": True,
    },
}


# ─────────────────────────────────────────────────────────────
# Pipeline 函数
# ─────────────────────────────────────────────────────────────

def pipeline_structure_filter(
    smiles_list: List[str],
    preset: str = "default",
    custom_config: Optional[Dict] = None,
    workers: int = 128,
) -> Tuple[List[str], Dict[str, int], pd.DataFrame]:
    """结构合理性过滤 Pipeline。
    
    阶段1: 硬过滤（快速排除）
    阶段2: SA Score 计算 + 阈值过滤
    阶段3: 综合评分排序
    
    Args:
        smiles_list: SMILES 列表
        preset: 预设名称（环氧树脂、固化剂-胺类 等）
        custom_config: 自定义配置（覆盖预设）
        workers: 并行线程数
    
    Returns:
        (通过列表, 统计字典, 评分DataFrame)
    """
    if not smiles_list:
        return [], {"total": 0, "passed": 0}, pd.DataFrame()
    
    # 合并配置
    if preset == "default":
        preset = "环氧树脂"  # 默认预设
    
    config = PRESET_CONFIGS.get(preset, PRESET_CONFIGS["环氧树脂"]).copy()
    if custom_config:
        config.update(custom_config)
    
    stats = {
        "total": len(smiles_list),
        "passed": 0,
        "stage1_passed": 0,
        "stage2_passed": 0,
    }
    
    # 阶段1：硬过滤
    passed_stage1, stats_stage1, details_stage1 = batch_hard_filter(
        smiles_list,
        check_elements=config.get("check_elements", True),
        check_reactive=config.get("check_reactive", True),
        check_strain=config.get("check_strain", True),
        check_fg_limits=config.get("check_fg_limits", True),
        check_hetero_ratio=config.get("check_hetero_ratio", True),
        workers=workers,
    )
    stats["stage1_passed"] = len(passed_stage1)
    stats.update({f"stage1_{k}": v for k, v in stats_stage1.items() if k not in ["total", "passed"]})
    
    if not passed_stage1:
        return [], stats, pd.DataFrame()
    
    # 阶段2：SA Score 计算 + 阈值过滤
    scores_df = batch_calculate_scores(
        passed_stage1,
        workers=workers,
        include_strain=True,
    )
    
    if scores_df.empty:
        return [], stats, pd.DataFrame()
    
    sa_threshold = config.get("sa_threshold", 7.0)
    scores_df = scores_df[scores_df["sa_score"] <= sa_threshold]
    stats["stage2_passed"] = len(scores_df)
    
    if scores_df.empty:
        return [], stats, pd.DataFrame()
    
    # 阶段3：按综合评分排序
    scores_df = scores_df.sort_values("composite_score", ascending=True)
    scores_df["rank"] = range(1, len(scores_df) + 1)
    
    final_passed = scores_df["smiles"].tolist()
    stats["passed"] = len(final_passed)
    
    return final_passed, stats, scores_df
