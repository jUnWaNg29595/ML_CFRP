# -*- coding: utf-8 -*-
"""分子特征工程模块 - 完整5种提取方法 + 分子指纹 (高性能优化版)"""

# ============================================
# 重要：必须在导入任何库之前设置环境变量！
# ============================================
import os
# 禁用 TensorFlow，避免与 transformers 冲突
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_TORCH', '1')
# 设置 Hugging Face 镜像源
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
# 禁用 Hugging Face Hub 的在线检查
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('HF_HUB_DISABLE_EXPERIMENTAL_WARNING', '1')
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')

# ============================================
# 重要：必须在导入 RDKit 之前导入线程配置！
# ============================================
from . import thread_config

import pandas as pd
import numpy as np
import builtins
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from rdkit.Chem import MACCSkeys
from tqdm import tqdm
import warnings
from collections import OrderedDict, Counter
import re  # 新增: 用于分割多组分 SMILES
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from functools import partial  # 新增


def _safe_console_print(*args, **kwargs):
    """Keep diagnostic logging from breaking extraction on GBK Windows consoles."""
    try:
        return builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(kwargs.get("file", sys.stdout), "encoding", None) or "ascii"
        safe_args = [
            str(arg).encode(encoding, errors="replace").decode(encoding, errors="replace")
            for arg in args
        ]
        return builtins.print(*safe_args, **kwargs)


print = _safe_console_print

# PyTorch 是可选依赖 (用于 ANI2x 力场计算)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# [新增] 后台任务管理器 - 支持任务取消
from .task_manager import (
    get_task_manager, 
    is_cancelled, 
    CancellableProcessPoolExecutor
)

# [新增] 支持 SMILES / SELFIES / BigSMILES 输入
from .smiles_utils import (
    convert_to_smiles,
    normalize_chemical_string,
    parse_chemical_string,
    parse_smiles_quiet,
    split_smiles_cell,
    canonicalize_smiles,
    detect_chem_string_format,
    diagnose_chemical_string,
)

try:
    from .bigsmiles_stochastic_graph import (
        parse_bigsmiles_stochastic_graph,
        sample_bigsmiles_realizations,
    )
    BIGSMILES_STOCHASTIC_AVAILABLE = True
except Exception:
    parse_bigsmiles_stochastic_graph = None
    sample_bigsmiles_realizations = None
    BIGSMILES_STOCHASTIC_AVAILABLE = False

warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import AllChem
    from rdkit.Chem import Descriptors3D, rdMolDescriptors

    def _embed_molecule_compat(mol, params):
        """Try both EmbedMolecule calling conventions across RDKit versions."""
        # prefer keyword if supported, otherwise fall back
        try:
            return AllChem.EmbedMolecule(mol, params=params), None
        except TypeError as e_kw:
            try:
                return AllChem.EmbedMolecule(mol, params), None
            except Exception as e_pos:
                return -1, f"EmbedMolecule failed (kw={e_kw}; pos={e_pos})"
        except Exception as e:
            return -1, f"EmbedMolecule failed ({e})"


    def rdkit3d_debug_one(smiles: str, coulomb_top_k: int = 10) -> dict:
        """Run one-sample 3D pipeline and return detailed diagnostics (no silent swallowing)."""
        info = {"input": str(smiles)[:300]}
        try:
            if smiles is None or (hasattr(pd, "isna") and pd.isna(smiles)):
                info["stage"] = "input"
                info["error"] = "smiles is NA"
                return info
            s = str(smiles).strip()
            if (not s) or (s.lower() in {"nan", "none", "<na>", "na"}):
                info["stage"] = "input"
                info["error"] = "smiles is empty-like"
                return info

            # split like worker (支持 SMILES / SELFIES / BigSMILES)
            s = convert_to_smiles(s, fmt="auto") or s
            frags = split_smiles_cell(s)
            if not frags:
                info["stage"] = "split"
                info["error"] = "no fragments after split"
                return info

            ok_frags = 0
            for frag in frags[:5]:
                info.setdefault("fragments", []).append(frag[:120])
                mol = parse_chemical_string(
                    frag,
                    repair=True,
                    keep_largest_frag=False,
                )
                if mol is None:
                    info.setdefault("frag_parse_failed", []).append(frag[:120])
                    continue
                if mol.GetNumAtoms() < 2:
                    info.setdefault("frag_too_small", []).append(frag[:120])
                    continue

                mol = Chem.AddHs(mol)
                params = _get_etkdg_params() if "_get_etkdg_params" in globals() else (AllChem.ETKDGv3() if hasattr(AllChem,"ETKDGv3") else AllChem.ETKDG())
                # best-effort params
                for attr, val in [("useRandomCoords", True), ("numThreads", 1), ("maxAttempts", 50)]:
                    try:
                        setattr(params, attr, val)
                    except Exception:
                        pass

                res, err = _embed_molecule_compat(mol, params)
                info.setdefault("embed_results", []).append(int(res))
                if err:
                    info.setdefault("embed_errors", []).append(err[:200])

                if res != 0:
                    # try fallback
                    try:
                        res2 = AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=100)
                        info.setdefault("embed_fallback_results", []).append(int(res2))
                        if res2 != 0:
                            continue
                    except Exception as e2:
                        info.setdefault("embed_fallback_errors", []).append(str(e2)[:200])
                        continue

                ok_frags += 1
                # optimize best-effort
                try:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=100)
                except Exception:
                    pass

            info["stage"] = "embed"
            info["ok_fragments"] = ok_frags
            info["total_fragments_checked"] = min(len(frags), 5)

            if ok_frags == 0:
                info["error"] = "all fragments failed to embed"
            return info

        except Exception as e:
            info["stage"] = "exception"
            info["error"] = str(e)[:300]
            return info

    def _get_etkdg_params():
        """Return the best-available ETKDG params across RDKit versions."""
        if hasattr(AllChem, "ETKDGv3"):
            params = AllChem.ETKDGv3()
        elif hasattr(AllChem, "ETKDGv2"):
            params = AllChem.ETKDGv2()
        else:
            params = AllChem.ETKDG()
        # robust defaults (best-effort across versions)
        try:
            params.useRandomCoords = True
        except Exception:
            pass
        try:
            params.maxAttempts = 50
        except Exception:
            pass
        try:
            params.numThreads = 1
        except Exception:
            pass
        return params

    from rdkit.Chem import MACCSkeys

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from mordred import Calculator, descriptors

    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False


_POLYMER_BLOCK_RE = re.compile(r"\{([^{}]*)\}")
_POLYMER_BOND_DESC_RE = re.compile(r"\[\s*(<|>|\*|\$)([^\]]*)\]")
_POLYMER_RING_TOKEN_RE = re.compile(r"%\d{2}|\d")
_POLYMER_CHARGE_RE = re.compile(r"\[[^\]]*[+-][^\]]*\]")
_POLYMER_SPECIAL_CHAR_RE = re.compile(r"[{}\[\]<>*|;,]")
_POLYMER_ALT_SEP_RE = re.compile(r"\s*[;,|]\s*|\s+\+\s+")
_POLYMER_AROMATIC_CHAR_RE = re.compile(r"[bcnosp]")


def _safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def _safe_std(values):
    return float(np.std(values)) if values else 0.0


def _safe_max(values):
    return float(np.max(values)) if values else 0.0


def _safe_min(values):
    return float(np.min(values)) if values else 0.0


def _string_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    probs = np.array([c / total for c in counts.values()], dtype=float)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


def _extract_polymer_unit_candidates(text) -> list[str]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    s = str(text).strip()
    if not s or s.lower() in {"nan", "none", "<na>", "na", "null"}:
        return []

    s = re.sub(r"^(SMILES|SELFIES|BIGSMILES)\s*[:：]\s*", "", s, flags=re.I).strip()
    blocks = _POLYMER_BLOCK_RE.findall(s)
    seeds = blocks if blocks else [s]
    candidates: list[str] = []

    for seed in seeds:
        cleaned = _POLYMER_BOND_DESC_RE.sub("", str(seed))
        cleaned = cleaned.replace("{", "").replace("}", "")
        parts = [p.strip() for p in _POLYMER_ALT_SEP_RE.split(cleaned) if p and str(p).strip()]
        if not parts:
            parts = [cleaned]

        expanded_parts: list[str] = []
        for part in parts:
            sub_frags = [frag.strip() for frag in str(part).split(".") if frag and str(frag).strip()]
            expanded_parts.extend(sub_frags if sub_frags else [part])

        for frag in expanded_parts:
            normalized = normalize_chemical_string(
                frag,
                canonicalize=True,
                repair=True,
                keep_largest_frag=False,
            )
            if normalized:
                normalized_frags = [x.strip() for x in normalized.split(".") if x.strip()]
                candidates.extend(normalized_frags if normalized_frags else [normalized])
                continue

            converted = convert_to_smiles(frag, fmt="auto")
            if converted:
                converted = str(converted).strip()
                if converted:
                    candidates.extend([x.strip() for x in converted.split(".") if x.strip()] or [converted])

    deduped = list(OrderedDict((c, None) for c in candidates if c).keys())
    return deduped


def _empty_polymer_graph_sample_features() -> dict:
    return {
        "polymer_graph_parse_success": 0.0,
        "polymer_graph_segment_count": 0.0,
        "polymer_graph_block_count": 0.0,
        "polymer_graph_repeat_candidate_count": 0.0,
        "polymer_graph_end_group_candidate_count": 0.0,
        "polymer_graph_edge_count": 0.0,
        "polymer_graph_free_fragment_count": 0.0,
        "polymer_sample_count": 0.0,
        "polymer_sample_unique_count": 0.0,
        "polymer_sample_unique_ratio": 0.0,
        "polymer_sample_valid_count": 0.0,
        "polymer_sample_valid_ratio": 0.0,
        "polymer_sample_char_len_mean": 0.0,
        "polymer_sample_char_len_std": 0.0,
        "polymer_sample_fragment_count_mean": 0.0,
        "polymer_sample_heavy_atoms_mean": 0.0,
        "polymer_sample_heavy_atoms_std": 0.0,
        "polymer_sample_hetero_atoms_mean": 0.0,
        "polymer_sample_ring_count_mean": 0.0,
        "polymer_sample_exact_mw_mean": 0.0,
    }


def _extract_bigsmiles_graph_sample_features(
    text,
    n_samples: int = 4,
    min_repeat_units: int = 1,
    max_repeat_units: int = 2,
    random_state: int = 17,
) -> dict:
    feat = _empty_polymer_graph_sample_features()
    if not BIGSMILES_STOCHASTIC_AVAILABLE or not text:
        return feat

    s = str(text).strip()
    if not s or s.lower() in {"nan", "none", "<na>", "na", "null"}:
        return feat

    try:
        graph = parse_bigsmiles_stochastic_graph(s)
    except Exception:
        graph = None

    if graph is not None:
        try:
            summary = graph.summary()
        except Exception:
            summary = {}
        feat["polymer_graph_parse_success"] = 1.0
        feat["polymer_graph_segment_count"] = float(summary.get("n_segments", 0.0) or 0.0)
        feat["polymer_graph_block_count"] = float(summary.get("n_blocks", 0.0) or 0.0)
        feat["polymer_graph_repeat_candidate_count"] = float(summary.get("n_repeat_unit_candidates", 0.0) or 0.0)
        feat["polymer_graph_end_group_candidate_count"] = float(summary.get("n_end_group_candidates", 0.0) or 0.0)
        feat["polymer_graph_edge_count"] = float(summary.get("n_edges", 0.0) or 0.0)
        feat["polymer_graph_free_fragment_count"] = float(summary.get("n_free_fragments", 0.0) or 0.0)

    try:
        sampled = sample_bigsmiles_realizations(
            s,
            n_samples=int(n_samples),
            min_repeat_units=int(min_repeat_units),
            max_repeat_units=int(max_repeat_units),
            random_state=int(random_state),
        )
    except Exception:
        sampled = []

    sampled = [str(item).strip() for item in (sampled or []) if str(item).strip()]
    if not sampled:
        return feat

    feat["polymer_sample_count"] = float(len(sampled))
    unique_sampled = list(OrderedDict((item, None) for item in sampled).keys())
    feat["polymer_sample_unique_count"] = float(len(unique_sampled))
    feat["polymer_sample_unique_ratio"] = float(len(unique_sampled) / max(len(sampled), 1))

    char_lens = [len(item) for item in sampled]
    fragment_counts = [max(1, len([frag for frag in item.split(".") if frag.strip()])) for item in sampled]
    feat["polymer_sample_char_len_mean"] = _safe_mean(char_lens)
    feat["polymer_sample_char_len_std"] = _safe_std(char_lens)
    feat["polymer_sample_fragment_count_mean"] = _safe_mean(fragment_counts)

    normalized_samples = []
    for item in sampled:
        normalized = normalize_chemical_string(
            item,
            fmt="smiles",
            canonicalize=True,
            repair=True,
            keep_largest_frag=False,
        )
        if normalized:
            normalized_samples.append(normalized)

    feat["polymer_sample_valid_count"] = float(len(normalized_samples))
    feat["polymer_sample_valid_ratio"] = float(len(normalized_samples) / max(len(sampled), 1))

    if not RDKIT_AVAILABLE or not normalized_samples:
        return feat

    heavy_atoms = []
    hetero_atoms = []
    ring_counts = []
    exact_mws = []

    for item in normalized_samples:
        try:
            mol = parse_smiles_quiet(item)
        except Exception:
            mol = None
        if mol is None:
            continue
        heavy_atoms.append(float(mol.GetNumHeavyAtoms()))
        hetero_atoms.append(float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6})))
        try:
            ring_counts.append(float(rdMolDescriptors.CalcNumRings(mol)))
        except Exception:
            pass
        try:
            exact_mws.append(float(Descriptors.ExactMolWt(mol)))
        except Exception:
            pass

    if heavy_atoms:
        feat["polymer_sample_heavy_atoms_mean"] = _safe_mean(heavy_atoms)
        feat["polymer_sample_heavy_atoms_std"] = _safe_std(heavy_atoms)
    if hetero_atoms:
        feat["polymer_sample_hetero_atoms_mean"] = _safe_mean(hetero_atoms)
    if ring_counts:
        feat["polymer_sample_ring_count_mean"] = _safe_mean(ring_counts)
    if exact_mws:
        feat["polymer_sample_exact_mw_mean"] = _safe_mean(exact_mws)

    return feat


def extract_polymer_string_features(smiles_like_list, prefix=None, include_bigsmiles_graph_stats: bool = True) -> pd.DataFrame:
    """Extract polymer-notation features from SMILES / BigSMILES / gBigSMILES-like strings.

    This complements RDKit descriptors with syntax-level polymer information that would
    otherwise be lost when BigSMILES is heuristically flattened into a representative SMILES.
    """
    rows = []
    feature_cache = {}
    for raw in smiles_like_list:
        cache_key = None if raw is None else str(raw)
        if cache_key is not None and cache_key in feature_cache:
            rows.append(dict(feature_cache[cache_key]))
            continue

        feat = {
            "polymer_missing": 1.0,
            "polymer_is_bigsmiles": 0.0,
            "polymer_is_selfies": 0.0,
            "polymer_has_polymer_syntax": 0.0,
            "polymer_char_len": 0.0,
            "polymer_token_entropy": 0.0,
            "polymer_num_repeat_blocks": 0.0,
            "polymer_num_bond_descriptors": 0.0,
            "polymer_num_left_bond_desc": 0.0,
            "polymer_num_right_bond_desc": 0.0,
            "polymer_num_star_bond_desc": 0.0,
            "polymer_num_star_tokens": 0.0,
            "polymer_num_fragments_est": 0.0,
            "polymer_num_separator_tokens": 0.0,
            "polymer_num_dot_tokens": 0.0,
            "polymer_num_branch_tokens": 0.0,
            "polymer_num_ring_tokens": 0.0,
            "polymer_num_charge_tokens": 0.0,
            "polymer_num_aromatic_chars": 0.0,
            "polymer_special_char_ratio": 0.0,
            "polymer_num_unit_candidates": 0.0,
            "polymer_num_unique_unit_candidates": 0.0,
            "polymer_unit_char_len_mean": 0.0,
            "polymer_unit_char_len_std": 0.0,
            "polymer_unit_char_len_max": 0.0,
            "polymer_unit_valid_mol_count": 0.0,
            "polymer_unit_valid_ratio": 0.0,
            "polymer_unit_heavy_atoms_mean": 0.0,
            "polymer_unit_heavy_atoms_max": 0.0,
            "polymer_unit_hetero_atoms_mean": 0.0,
            "polymer_unit_ring_count_mean": 0.0,
            "polymer_unit_exact_mw_mean": 0.0,
        }
        feat.update(_empty_polymer_graph_sample_features())

        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            rows.append(feat)
            continue

        s = str(raw).strip()
        if not s or s.lower() in {"nan", "none", "<na>", "na", "null"}:
            rows.append(feat)
            continue

        fmt = detect_chem_string_format(s)
        blocks = _POLYMER_BLOCK_RE.findall(s)
        bond_desc = _POLYMER_BOND_DESC_RE.findall(s)
        candidates = _extract_polymer_unit_candidates(s)
        try:
            fragment_est = split_smiles_cell(s)
        except Exception:
            fragment_est = []

        feat["polymer_missing"] = 0.0
        feat["polymer_is_bigsmiles"] = 1.0 if fmt == "bigsmiles" else 0.0
        feat["polymer_is_selfies"] = 1.0 if fmt == "selfies" else 0.0
        feat["polymer_has_polymer_syntax"] = 1.0 if (blocks or bond_desc or "*" in s or "BIGSMILES" in s.upper()) else 0.0
        feat["polymer_char_len"] = float(len(s))
        feat["polymer_token_entropy"] = _string_entropy(s)
        feat["polymer_num_repeat_blocks"] = float(len(blocks))
        feat["polymer_num_bond_descriptors"] = float(len(bond_desc))
        feat["polymer_num_left_bond_desc"] = float(sum(1 for sym, _ in bond_desc if sym == "<"))
        feat["polymer_num_right_bond_desc"] = float(sum(1 for sym, _ in bond_desc if sym == ">"))
        feat["polymer_num_star_bond_desc"] = float(sum(1 for sym, _ in bond_desc if sym == "*"))
        feat["polymer_num_star_tokens"] = float(s.count("*"))
        feat["polymer_num_fragments_est"] = float(len(fragment_est) if fragment_est else max(1, s.count(".") + 1))
        feat["polymer_num_separator_tokens"] = float(s.count(";") + s.count("；") + s.count("|") + s.count(","))
        feat["polymer_num_dot_tokens"] = float(s.count("."))
        feat["polymer_num_branch_tokens"] = float(s.count("(") + s.count(")"))
        feat["polymer_num_ring_tokens"] = float(len(_POLYMER_RING_TOKEN_RE.findall(s)))
        feat["polymer_num_charge_tokens"] = float(len(_POLYMER_CHARGE_RE.findall(s)))
        feat["polymer_num_aromatic_chars"] = float(len(_POLYMER_AROMATIC_CHAR_RE.findall(s)))
        feat["polymer_special_char_ratio"] = float(len(_POLYMER_SPECIAL_CHAR_RE.findall(s)) / max(len(s), 1))
        feat["polymer_num_unit_candidates"] = float(len(candidates))
        feat["polymer_num_unique_unit_candidates"] = float(len(set(candidates)))

        if include_bigsmiles_graph_stats and (fmt == "bigsmiles" or feat["polymer_has_polymer_syntax"] > 0):
            feat.update(_extract_bigsmiles_graph_sample_features(s))

        if candidates:
            lens = [len(x) for x in candidates]
            feat["polymer_unit_char_len_mean"] = _safe_mean(lens)
            feat["polymer_unit_char_len_std"] = _safe_std(lens)
            feat["polymer_unit_char_len_max"] = _safe_max(lens)

        valid_mols = []
        if RDKIT_AVAILABLE and candidates:
            for cand in candidates:
                try:
                    mol = parse_chemical_string(
                        cand,
                        repair=True,
                        keep_largest_frag=False,
                    )
                    if mol is not None:
                        valid_mols.append(mol)
                except Exception:
                    continue

        if candidates:
            feat["polymer_unit_valid_mol_count"] = float(len(valid_mols))
            feat["polymer_unit_valid_ratio"] = float(len(valid_mols) / max(len(candidates), 1))

        if valid_mols:
            heavy_atoms = [float(mol.GetNumHeavyAtoms()) for mol in valid_mols]
            hetero_atoms = [
                float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6}))
                for mol in valid_mols
            ]
            ring_counts = [float(rdMolDescriptors.CalcNumRings(mol)) for mol in valid_mols]
            exact_mws = [float(Descriptors.ExactMolWt(mol)) for mol in valid_mols]
            feat["polymer_unit_heavy_atoms_mean"] = _safe_mean(heavy_atoms)
            feat["polymer_unit_heavy_atoms_max"] = _safe_max(heavy_atoms)
            feat["polymer_unit_hetero_atoms_mean"] = _safe_mean(hetero_atoms)
            feat["polymer_unit_ring_count_mean"] = _safe_mean(ring_counts)
            feat["polymer_unit_exact_mw_mean"] = _safe_mean(exact_mws)

        if cache_key is not None:
            feature_cache[cache_key] = dict(feat)
        rows.append(feat)

    df = pd.DataFrame(rows)
    if prefix:
        df = _add_prefix_to_columns(df, prefix)
    return df


_ION_ALIAS_PATTERNS = [
    (re.compile(r"\[BF4-\]", flags=re.I), "F[B-](F)(F)F"),
    (re.compile(r"\[PF6-\]", flags=re.I), "F[P-](F)(F)(F)(F)F"),
    (re.compile(r"\[ClO4-\]", flags=re.I), "[O-][Cl](=O)(=O)=O"),
    (re.compile(r"\[NO3-\]", flags=re.I), "[O-][N+](=O)[O-]"),
    (re.compile(r"\[OTf-\]", flags=re.I), "OS(=O)(=O)C(F)(F)F"),
    (re.compile(r"\[TfO-\]", flags=re.I), "OS(=O)(=O)C(F)(F)F"),
]

_COMMON_METAL_SYMBOLS = {
    "Li", "Na", "K", "Rb", "Cs",
    "Mg", "Ca", "Sr", "Ba",
    "Al", "Ga", "In", "Sn", "Pb",
    "Ti", "Zr", "Hf", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}


def _replace_common_ion_aliases(text: str) -> str:
    updated = str(text or "")
    for pattern, replacement in _ION_ALIAS_PATTERNS:
        updated = pattern.sub(replacement, updated)
    return updated


def _safe_fragment_mol_from_text(fragment: str):
    if not RDKIT_AVAILABLE:
        return None
    frag = str(fragment or "").strip()
    if not frag:
        return None
    try:
        mol = parse_chemical_string(
            frag,
            repair=True,
            keep_largest_frag=False,
        )
        if mol is not None:
            return mol
    except Exception:
        pass
    try:
        alt = _replace_common_ion_aliases(frag)
        if alt != frag:
            mol = parse_chemical_string(
                alt,
                repair=True,
                keep_largest_frag=False,
            )
            if mol is not None:
                return mol
    except Exception:
        pass
    return None


def extract_ionic_semantic_features(smiles_like_list, prefix=None) -> pd.DataFrame:
    rows = []
    for raw in smiles_like_list:
        feat = {
            "ionic_missing": 1.0,
            "ionic_has_any_charge": 0.0,
            "ionic_has_cation": 0.0,
            "ionic_has_anion": 0.0,
            "ionic_has_metal": 0.0,
            "ionic_cation_count": 0.0,
            "ionic_anion_count": 0.0,
            "ionic_metal_atom_count": 0.0,
            "ionic_fragment_count": 0.0,
            "ionic_net_formal_charge": 0.0,
            "ionic_counterion_balance_abs": 0.0,
            "ionic_alias_replacement_used": 0.0,
            "ionic_proxy_parse_ok": 0.0,
            "ionic_direct_parse_ok": 0.0,
            "ionic_normalized_ok": 0.0,
            "ionic_contains_bf4": 0.0,
            "ionic_contains_pf6": 0.0,
            "ionic_contains_clo4": 0.0,
            "ionic_contains_nitrate": 0.0,
            "ionic_contains_triflate": 0.0,
            "ionic_metal_species_count": 0.0,
        }

        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            rows.append(feat)
            continue

        s = str(raw).strip()
        if not s or s.lower() in {"nan", "none", "<na>", "na", "null"}:
            rows.append(feat)
            continue

        feat["ionic_missing"] = 0.0
        feat["ionic_contains_bf4"] = 1.0 if re.search(r"\[BF4-\]", s, flags=re.I) else 0.0
        feat["ionic_contains_pf6"] = 1.0 if re.search(r"\[PF6-\]", s, flags=re.I) else 0.0
        feat["ionic_contains_clo4"] = 1.0 if re.search(r"\[ClO4-\]", s, flags=re.I) else 0.0
        feat["ionic_contains_nitrate"] = 1.0 if re.search(r"\[NO3-\]", s, flags=re.I) else 0.0
        feat["ionic_contains_triflate"] = 1.0 if re.search(r"\[(?:OTf|TfO)-\]", s, flags=re.I) else 0.0

        replaced = _replace_common_ion_aliases(s)
        feat["ionic_alias_replacement_used"] = 1.0 if replaced != s else 0.0

        try:
            diag = diagnose_chemical_string(s)
        except Exception:
            diag = {}
        feat["ionic_proxy_parse_ok"] = 1.0 if diag.get("proxy_smiles_ok") else 0.0
        feat["ionic_direct_parse_ok"] = 1.0 if diag.get("rdkit_direct_ok") else 0.0
        feat["ionic_normalized_ok"] = 1.0 if diag.get("normalized_ok") else 0.0

        try:
            fragments = split_smiles_cell(replaced)
        except Exception:
            fragments = [replaced]
        fragments = [str(f).strip() for f in (fragments or []) if str(f).strip()]
        feat["ionic_fragment_count"] = float(len(fragments))

        cation_count = 0
        anion_count = 0
        metal_atom_count = 0
        net_charge = 0
        metal_species = set()

        for frag in fragments:
            mol = _safe_fragment_mol_from_text(frag)
            if mol is None:
                continue
            frag_charge = 0
            frag_has_pos = False
            frag_has_neg = False
            for atom in mol.GetAtoms():
                fc = int(atom.GetFormalCharge())
                frag_charge += fc
                if fc > 0:
                    frag_has_pos = True
                elif fc < 0:
                    frag_has_neg = True
                symbol = str(atom.GetSymbol() or "")
                if symbol in _COMMON_METAL_SYMBOLS:
                    metal_atom_count += 1
                    metal_species.add(symbol)
            net_charge += frag_charge
            if frag_has_pos:
                cation_count += 1
            if frag_has_neg:
                anion_count += 1

        feat["ionic_cation_count"] = float(cation_count)
        feat["ionic_anion_count"] = float(anion_count)
        feat["ionic_has_cation"] = 1.0 if cation_count > 0 else 0.0
        feat["ionic_has_anion"] = 1.0 if anion_count > 0 else 0.0
        feat["ionic_has_any_charge"] = 1.0 if (cation_count > 0 or anion_count > 0 or net_charge != 0) else 0.0
        feat["ionic_metal_atom_count"] = float(metal_atom_count)
        feat["ionic_has_metal"] = 1.0 if metal_atom_count > 0 else 0.0
        feat["ionic_net_formal_charge"] = float(net_charge)
        feat["ionic_counterion_balance_abs"] = float(abs(cation_count - anion_count))
        feat["ionic_metal_species_count"] = float(len(metal_species))
        rows.append(feat)

    df = pd.DataFrame(rows)
    if prefix:
        df = _add_prefix_to_columns(df, prefix)
    return df


def extract_bigsmiles_ensemble_features(
    smiles_like_list,
    prefix=None,
    n_samples: int = 8,
    min_repeat_units: int = 1,
    max_repeat_units: int = 4,
    random_state: int = 17,
) -> pd.DataFrame:
    rows = []
    for raw in smiles_like_list:
        feat = {
            "bigsmiles_ensemble_used": 0.0,
            "bigsmiles_ensemble_sample_count": 0.0,
            "bigsmiles_ensemble_unique_count": 0.0,
            "bigsmiles_ensemble_valid_ratio": 0.0,
            "bigsmiles_ensemble_heavy_atoms_mean": 0.0,
            "bigsmiles_ensemble_heavy_atoms_std": 0.0,
            "bigsmiles_ensemble_exact_mw_mean": 0.0,
            "bigsmiles_ensemble_exact_mw_std": 0.0,
            "bigsmiles_ensemble_ring_count_mean": 0.0,
            "bigsmiles_ensemble_hetero_atoms_mean": 0.0,
            "bigsmiles_ensemble_char_len_mean": 0.0,
            "bigsmiles_ensemble_fragment_count_mean": 0.0,
        }
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            rows.append(feat)
            continue
        s = str(raw).strip()
        if not s or detect_chem_string_format(s) != "bigsmiles" or not BIGSMILES_STOCHASTIC_AVAILABLE:
            rows.append(feat)
            continue

        try:
            sampled = sample_bigsmiles_realizations(
                s,
                n_samples=int(n_samples),
                min_repeat_units=int(min_repeat_units),
                max_repeat_units=int(max_repeat_units),
                random_state=int(random_state),
            )
        except Exception:
            sampled = []

        sampled = [str(item).strip() for item in (sampled or []) if str(item).strip()]
        if not sampled:
            rows.append(feat)
            continue

        feat["bigsmiles_ensemble_used"] = 1.0
        feat["bigsmiles_ensemble_sample_count"] = float(len(sampled))
        feat["bigsmiles_ensemble_unique_count"] = float(len(OrderedDict((x, None) for x in sampled).keys()))
        feat["bigsmiles_ensemble_char_len_mean"] = _safe_mean([len(x) for x in sampled])
        feat["bigsmiles_ensemble_fragment_count_mean"] = _safe_mean(
            [max(1, len([frag for frag in x.split(".") if frag.strip()])) for x in sampled]
        )

        valid_mols = []
        for item in sampled:
            normalized = normalize_chemical_string(
                item,
                fmt="smiles",
                canonicalize=True,
                repair=True,
                keep_largest_frag=False,
            )
            if not normalized:
                continue
            mol = _safe_fragment_mol_from_text(normalized)
            if mol is not None:
                valid_mols.append(mol)

        feat["bigsmiles_ensemble_valid_ratio"] = float(len(valid_mols) / max(len(sampled), 1))
        if valid_mols:
            heavy_atoms = [float(mol.GetNumHeavyAtoms()) for mol in valid_mols]
            exact_mws = []
            ring_counts = []
            hetero_atoms = []
            for mol in valid_mols:
                try:
                    exact_mws.append(float(Descriptors.ExactMolWt(mol)))
                except Exception:
                    pass
                try:
                    ring_counts.append(float(rdMolDescriptors.CalcNumRings(mol)))
                except Exception:
                    pass
                hetero_atoms.append(float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6})))
            feat["bigsmiles_ensemble_heavy_atoms_mean"] = _safe_mean(heavy_atoms)
            feat["bigsmiles_ensemble_heavy_atoms_std"] = _safe_std(heavy_atoms)
            feat["bigsmiles_ensemble_exact_mw_mean"] = _safe_mean(exact_mws)
            feat["bigsmiles_ensemble_exact_mw_std"] = _safe_std(exact_mws)
            feat["bigsmiles_ensemble_ring_count_mean"] = _safe_mean(ring_counts)
            feat["bigsmiles_ensemble_hetero_atoms_mean"] = _safe_mean(hetero_atoms)

        rows.append(feat)

    df = pd.DataFrame(rows)
    if prefix:
        df = _add_prefix_to_columns(df, prefix)
    return df


def extract_configured_semantic_features(
    smiles_like_list,
    params=None,
    *,
    prefix=None,
    preserve_duplicate_columns=False,
) -> pd.DataFrame:
    """Build every configured notation/ionic feature with one shared contract."""
    params = dict(params or {})
    preserve_duplicate_columns = bool(
        params.get("preserve_duplicate_columns", preserve_duplicate_columns)
    )
    frames = []
    append_polymer_ensemble = bool(params.get("append_polymer_semantic_features", False))
    append_polymer_string = bool(params.get("append_polymer_string_features", False))
    append_ionic = bool(params.get("append_ionic_semantic_features", False))

    # Training historically included string features whenever ensemble features
    # were enabled, even if the explicit string flag was absent.
    if append_polymer_string or append_polymer_ensemble:
        frames.append(extract_polymer_string_features(smiles_like_list))
    if append_polymer_ensemble:
        frames.append(
            extract_bigsmiles_ensemble_features(
                smiles_like_list,
                n_samples=int(params.get("bigsmiles_semantic_num_samples", 8)),
                min_repeat_units=int(params.get("bigsmiles_semantic_min_repeat_units", 1)),
                max_repeat_units=int(params.get("bigsmiles_semantic_max_repeat_units", 4)),
                random_state=int(params.get("bigsmiles_semantic_random_state", 17)),
            )
        )
    if append_ionic:
        frames.append(extract_ionic_semantic_features(smiles_like_list))

    frames = [
        frame.reset_index(drop=True)
        for frame in frames
        if isinstance(frame, pd.DataFrame) and len(frame) == len(smiles_like_list)
    ]
    if not frames:
        return pd.DataFrame(index=range(len(smiles_like_list)))

    result = pd.concat(frames, axis=1)
    if not preserve_duplicate_columns:
        result = result.loc[:, ~result.columns.duplicated()]
    if prefix:
        result = _add_prefix_to_columns(result, prefix)
    return result


def append_configured_semantic_features(
    features_df,
    valid_indices,
    smiles_like_list,
    params=None,
    *,
    preserve_duplicate_columns=False,
) -> tuple[pd.DataFrame, list[int]]:
    """Append configured semantic features while preserving extractor row alignment."""
    params = dict(params or {})
    preserve_duplicate_columns = bool(
        params.get("preserve_duplicate_columns", preserve_duplicate_columns)
    )
    indices = [int(i) for i in (valid_indices or [])]
    semantic_full = extract_configured_semantic_features(
        smiles_like_list,
        params,
        preserve_duplicate_columns=preserve_duplicate_columns,
    )
    base = features_df if isinstance(features_df, pd.DataFrame) else pd.DataFrame()
    if semantic_full.empty and len(semantic_full.columns) == 0:
        return base, indices

    if not base.empty and indices:
        base = base.copy()
        semantic_subset = semantic_full.iloc[indices].reset_index(drop=True)
        base = pd.concat([base.reset_index(drop=True), semantic_subset], axis=1)
        if not preserve_duplicate_columns:
            base = base.loc[:, ~base.columns.duplicated()]
        return base, indices

    return semantic_full.reset_index(drop=True), list(range(len(semantic_full)))


# =============================================================================
# 辅助函数：3D 构象生成 (用于多进程)
# =============================================================================
def _generate_3d_data_worker(smiles):
    """
    单个样本的 3D 构象生成工作函数（供多进程调用）

    - 支持多组分/多片段 SMILES：会自动按 ';'、'；'、'|'、带空格的 ' + '、以及 '.' 进行分割
    - 对每个片段分别生成 3D（ETKDGv3）并做轻量优化（MMFF / UFF）
    - 仅保留 ANI2x 支持的元素：H,C,N,O,F,S,Cl

    返回:
        list[tuple[list[int], np.ndarray]]  # [(atomic_numbers, coordinates), ...]
        或 None（任一片段失败则返回 None，保证数据质量）
        或 dict with 'error' key（用于调试）
    """
    if not RDKIT_AVAILABLE:
        return {'error': 'rdkit_unavailable'}

    # 抑制RDKit警告
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    try:
        # [修复] 更强的类型检查
        if smiles is None:
            return {'error': 'invalid_smiles', 'message': 'smiles is None'}

        # 检查是否是 NaN（支持多种 NaN 类型）
        if isinstance(smiles, float):
            if np.isnan(smiles):
                return {'error': 'invalid_smiles', 'message': 'smiles is NaN (float)'}
            # 如果是普通浮点数，转为字符串
            smiles = str(smiles)

        # pandas NA 检查
        if hasattr(pd, 'isna') and pd.isna(smiles):
            return {'error': 'invalid_smiles', 'message': 'smiles is pandas NA'}

        # 转为字符串并清理
        try:
            s = str(smiles).strip()
        except Exception as e:
            return {'error': 'invalid_smiles', 'message': f'cannot convert to string: {e}'}

        if (not s) or (s.lower() in {"nan", "none", "<na>", "na", "null"}):
            return {'error': 'empty_smiles', 'message': f'empty or invalid string: {s}'}

        # --- 预处理：处理聚合物中的 * / [*] 占位符 ---
        # RDKit/ETKDG 不支持 '*' 原子；用 'C' 作为占位（与 3D 构象描述符逻辑保持一致）
        if '*' in s:
            try:
                s = re.sub(r"\[\s*\*\s*\]", "C", s)  # 替换 [*] 或 [ * ]
            except Exception:
                pass
            s = s.replace('*', 'C')

        # 1) 智能分割多组分（支持 SMILES / SELFIES / BigSMILES）
        try:
            s = convert_to_smiles(s, fmt="auto") or s
        except Exception as e:
            return {'error': 'convert_failed', 'message': f'convert_to_smiles failed: {e}'}

        try:
            frags = split_smiles_cell(s)
        except Exception as e:
            return {'error': 'split_failed', 'message': f'split_smiles_cell failed: {e}'}

        if not frags:
            return {'error': 'no_fragments', 'message': f'no fragments after split: {s[:100]}'}

        frag_data = []
        fail_reasons = []

        supported_species = {1, 6, 7, 8, 9, 16, 17}  # H,C,N,O,F,S,Cl (ANI2x)

        for frag in frags:
            mol = parse_chemical_string(
                frag,
                repair=True,
                keep_largest_frag=False,
            )
            if mol is None:
                # 片段解析失败：跳过该片段（不放弃整个样本）
                fail_reasons.append('parse_failed')
                continue

            # ✅ 检查分子是否有原子
            if mol.GetNumAtoms() == 0:
                fail_reasons.append('no_atoms')
                continue

            # ✅ 提前检查元素是否被ANI2x支持（在加氢前检查重原子）
            heavy_atoms = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
            # ANI2x支持的重原子: C(6), N(7), O(8), F(9), S(16), Cl(17)
            supported_heavy = {6, 7, 8, 9, 16, 17}
            if not heavy_atoms.issubset(supported_heavy):
                # 含有不支持的元素（如Si, Fe, Zr等金属），跳过该片段
                unsupported = heavy_atoms - supported_heavy
                fail_reasons.append(f'unsupported_elements:{unsupported}')
                continue

            mol = Chem.AddHs(mol)  # 力场/ANI 计算建议加氢

            # ✅ 再次检查（AddHs后）
            if mol.GetNumAtoms() == 0:
                fail_reasons.append('no_atoms_after_addh')
                continue

            # 2) 生成 3D 构象（ETKDGv3）
            params = _get_etkdg_params()
            # RDKit 版本差异：部分属性可能不存在/只读，使用 best-effort 设置
            for _attr, _val in [("useRandomCoords", True), ("numThreads", 1), ("maxAttempts", 50)]:
                try:
                    setattr(params, _attr, _val)
                except Exception:
                    pass

            res, _err = _embed_molecule_compat(mol, params)
            res = int(res) if res is not None else -1
            if res != 0:
                # 兜底：再试一次
                try:
                    res = AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=100)
                except TypeError:
                    # 某些 RDKit 版本不支持这些关键字
                    try:
                        res = AllChem.EmbedMolecule(mol)
                    except Exception:
                        res = -1
                except Exception:
                    res = -1
                if res != 0:
                    # 该片段 3D 生成失败：跳过该片段
                    fail_reasons.append('embed_failed')
                    continue

            # 3) 快速几何优化：优先 MMFF，否则 UFF（减少迭代次数避免卡住）
            try:
                # MMFF更稳定，减少迭代次数
                AllChem.MMFFOptimizeMolecule(mol, maxIters=20)
            except Exception:
                try:
                    # UFF作为备选，同样减少迭代
                    AllChem.UFFOptimizeMolecule(mol, maxIters=50)
                except Exception:
                    # 如果力场优化失败，直接使用未优化的构象
                    pass

            # 4) 提取数据（此时元素已确认支持）
            atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            coords = mol.GetConformer().GetPositions().astype(np.float32)

            frag_data.append((atoms, coords))

        if not frag_data:
            return {'error': 'all_fragments_failed', 'reasons': fail_reasons}
        return frag_data

    except Exception as e:
        return {'error': 'exception', 'message': str(e)[:200]}



# =============================================================================
# 3D 描述符：RDKit3D + Coulomb Matrix (可选更前沿的构象表征)
# =============================================================================
def _rdkit3d_feature_worker(smiles, coulomb_top_k: int = 10):
    """
    计算单个样本的 3D 构象描述符（修复版）
    """
    if not RDKIT_AVAILABLE:
        return None

    try:
        if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)) or (hasattr(pd, 'isna') and pd.isna(smiles)):
            return None
        s = str(smiles).strip()
        if (not s) or (s.lower() in {"nan", "none", "<na>", "na"}):
            return None


        # --- 预处理：处理聚合物中的 * 号 ---
        # 3D 构象生成不支持 *，将其替换为 C (甲基) 以模拟占位
        if '*' in s:
            s = s.replace('*', 'C')

        # 分割多组分（支持 SMILES / SELFIES / BigSMILES）
        s = convert_to_smiles(s, fmt="auto") or s
        frags = split_smiles_cell(s)
        if not frags:
            return None

        total_atoms = 0
        n_frags = 0
        d3_weighted = {}
        eig_all = []

        for frag in frags:
            mol = parse_chemical_string(
                frag,
                repair=True,
                keep_largest_frag=False,
            )
            if mol is None:
                continue  # 解析失败跳过该片段，不要直接返回 None

            # 过滤掉单原子或太小的碎片（通常是离子或杂质），它们很难生成有意义的 3D
            if mol.GetNumAtoms() < 2:
                continue

            mol = Chem.AddHs(mol)

            # --- 生成 3D 构象 (放宽参数) ---
            params = _get_etkdg_params()
            # RDKit 版本差异：部分属性可能不存在/只读，使用 best-effort 设置
            for _attr, _val in [("useRandomCoords", True), ("numThreads", 1), ("maxAttempts", 50)]:
                try:
                    setattr(params, _attr, _val)
                except Exception:
                    pass

            # 尝试嵌入
            res, _err = _embed_molecule_compat(mol, params)
            res = int(res) if res is not None else -1

            # 如果失败，尝试更激进的随机坐标
            if res != 0:
                try:
                    # 兼容不同 RDKit：有的版本支持这些关键字，有的会报 TypeError
                    res2, _err2 = _embed_molecule_compat(mol, params)
                    res2 = int(res2) if res2 is not None else -1
                except Exception:
                    res2 = -1
                if res2 != 0:
                    try:
                        res2 = AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=100)
                    except Exception:
                        res2 = -1
                if res2 != 0:
                    # [修改] 如果该片段生成失败，仅跳过该片段，不放弃整个样本
                    continue

                    # 优化
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=100)
            except Exception:
                pass

            n_atoms = int(mol.GetNumAtoms())
            if n_atoms <= 0:
                continue

            n_frags += 1
            total_atoms += n_atoms

            # RDKit 3D descriptors
            try:
                d3 = Descriptors3D.CalcMolDescriptors3D(mol)  # dict
                for k, v in d3.items():
                    val = float(v)
                    if np.isfinite(val):
                        d3_weighted[k] = d3_weighted.get(k, 0.0) + val * n_atoms
            except Exception:
                pass

            # Coulomb matrix
            try:
                cm = rdMolDescriptors.CalcCoulombMat(mol)
                cm_arr = np.array([list(row) for row in cm], dtype=float)
                eig = np.linalg.eigvalsh(cm_arr)
                eig_all.append(eig)
            except Exception:
                pass

        # [修改] 如果所有片段都失败了，才返回 None
        if total_atoms <= 0:
            # 打开下面的注释可以调试具体是哪个 SMILES 失败了
            # print(f"❌ 所有片段3D生成均失败: {s}")
            return None

        out = {
            "rdkit3d_n_atoms": int(total_atoms),
            "rdkit3d_n_fragments": int(n_frags),
        }

        # 加权平均
        for k, v in d3_weighted.items():
            out[f"rdkit3d_{k}"] = float(v) / float(total_atoms)

        # Coulomb Matrix 处理
        if eig_all:
            eig_concat = np.concatenate(eig_all).astype(float)
            if eig_concat.size > 0:
                eig_sorted = np.sort(eig_concat)[::-1]  # desc
                for i in range(int(coulomb_top_k)):
                    out[f"coulomb_eig_{i + 1}"] = float(eig_sorted[i]) if i < len(eig_sorted) else 0.0
                out["coulomb_eig_mean"] = float(np.mean(eig_concat))
                out["coulomb_eig_std"] = float(np.std(eig_concat))
                out["coulomb_eig_max"] = float(np.max(eig_concat))
                out["coulomb_eig_min"] = float(np.min(eig_concat))
            else:
                _fill_nan(out, coulomb_top_k)
        else:
            _fill_nan(out, coulomb_top_k)

        return out

    except Exception as e:
        # print(f"❌ 3D Worker 异常: {e}") # 调试用
        return None


def _fill_nan(out, k):
    for i in range(int(k)):
        out[f"coulomb_eig_{i + 1}"] = np.nan
    out["coulomb_eig_mean"] = np.nan
    out["coulomb_eig_std"] = np.nan
    out["coulomb_eig_max"] = np.nan
    out["coulomb_eig_min"] = np.nan


class RDKit3DDescriptorExtractor:
    """RDKit 3D 构象描述符提取器（可选更前沿的几何表征）"""

    def __init__(self, coulomb_top_k: int = 10):
        self.coulomb_top_k = int(coulomb_top_k)
        self.feature_names = []  # 运行后才知道完整列名

    def smiles_to_3d_descriptors(self, smiles_list, n_jobs: int | None = None):
        if not RDKIT_AVAILABLE:
            raise ImportError("需要安装 RDKit 才能使用 3D 描述符。")

        if n_jobs is None:
            n_jobs = 1 if os.name == 'nt' else max(1, (mp.cpu_count() or 1) - 1)

        # 避免 CUDA + fork 多进程导致崩溃/报错：在 CUDA 下强制 3D 生成单进程更稳
        try:
            if getattr(self, 'device', None) is not None and getattr(self.device, 'type', '') == 'cuda' and n_jobs and int(n_jobs) > 1:
                print('⚠️ 检测到 CUDA 环境：为避免多进程 fork/CUDA 问题，已将 3D 生成 n_jobs 自动降为 1')
                n_jobs = 1
        except Exception:
            pass

        feats = []
        valid_indices = []

        print(f"\n🧊 3D 构象描述符提取 (n_jobs={n_jobs}, coulomb_top_k={self.coulomb_top_k})")

        worker = partial(_rdkit3d_feature_worker, coulomb_top_k=self.coulomb_top_k)

        if n_jobs == 1:
            for idx, s in enumerate(tqdm(smiles_list, desc="3D Descriptors")):
                # [新增] 检查是否请求取消
                if is_cancelled():
                    print("⏹️ 任务已取消")
                    break
                out = worker(s)
                if out is not None:
                    feats.append(out)
                    valid_indices.append(idx)
        else:
            try:
                # ✅ 修复：使用 submit + wait 替代 map，添加超时机制避免卡死
                from concurrent.futures import wait, TimeoutError as FuturesTimeoutError
                
                per_molecule_timeout = 30  # 单分子超时时间（秒）
                results_dict = {}  # {index: result}
                total = len(smiles_list)
                timeout_count = 0
                
                # 分批处理
                batch_submit_size = n_jobs * 2
                pbar = tqdm(total=total, desc=f"3D Descriptors ({n_jobs} workers)")
                
                with CancellableProcessPoolExecutor(max_workers=n_jobs, task_name="3D描述符提取") as executor:
                    for batch_start in range(0, total, batch_submit_size):
                        # 检查是否请求取消
                        if is_cancelled():
                            print("⏹️ 任务已取消")
                            pbar.close()
                            break
                            
                        batch_end = min(batch_start + batch_submit_size, total)
                        batch_smiles = smiles_list[batch_start:batch_end]
                        
                        # 提交这一批任务
                        futures = {
                            executor.submit(worker, s): batch_start + j 
                            for j, s in enumerate(batch_smiles)
                        }
                        
                        # 等待这批任务完成，设置超时
                        batch_timeout = per_molecule_timeout * len(batch_smiles) / max(1, n_jobs) + 10
                        done, not_done = wait(futures.keys(), timeout=batch_timeout)
                        
                        # 处理完成的任务
                        for future in done:
                            idx = futures[future]
                            try:
                                out = future.result(timeout=1)
                                if out is not None:
                                    results_dict[idx] = out
                            except Exception:
                                pass
                        
                        # 取消超时的任务
                        for future in not_done:
                            future.cancel()
                            timeout_count += 1
                        
                        pbar.update(len(batch_smiles))
                
                pbar.close()
                
                if timeout_count > 0:
                    print(f"⚠️ {timeout_count} 个分子处理超时，已跳过")
                
                # 按索引顺序排列结果
                for idx in sorted(results_dict.keys()):
                    feats.append(results_dict[idx])
                    valid_indices.append(idx)
                    
            except Exception as e:
                if "取消" in str(e) or is_cancelled():
                    print("⏹️ 任务已取消")
                else:
                    print(f"⚠️ 3D 并行提取失败，回退单进程：{e}")
                    for idx, s in enumerate(tqdm(smiles_list, desc="3D Descriptors (fallback)")):
                        if is_cancelled():
                            print("⏹️ 任务已取消")
                            break
                        out = worker(s)
                        if out is not None:
                            feats.append(out)
                            valid_indices.append(idx)

        if not feats:
            return pd.DataFrame(), []

        df = pd.DataFrame(feats)
        df = df.apply(pd.to_numeric, errors='coerce')
        self.feature_names = df.columns.tolist()

        return df, valid_indices



# =============================================================================
# 预训练 SMILES Transformer Embedding（可选：需要 transformers）
# =============================================================================
class SmilesTransformerEmbeddingExtractor:
    """
    预训练 SMILES Transformer 表征（例如 ChemBERTa 等）

    - 适合做“前沿特征工程”：不依赖手工描述符，能学习到更抽象的分子语义表示
    - 注意：首次运行会从 HuggingFace 下载模型权重（需要联网）
    """

    _CACHE = {}  # (model_name, device_str) -> (tokenizer, model, hidden_size)

    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        pooling: str = "cls",
        max_length: int = 128,
        device=None,
        trust_remote_code: bool = False
    ):
        # 禁用 huggingface_hub 的在线元数据检查
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

        self.model_name = model_name
        self.pooling = (pooling or "cls").lower()
        self.max_length = int(max_length)
        self.trust_remote_code = bool(trust_remote_code)

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            self.torch = torch
            self.AutoTokenizer = AutoTokenizer
            self.AutoModel = AutoModel
            self.AVAILABLE = True
        except Exception as e:
            print(f"Warning: Failed to import torch/transformers: {e}")
            self.AVAILABLE = False
            self.feature_names = []
            self.torch = None
            return

        if device is None:
            self.device = self.torch.device('cuda' if self.torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        cache_key = (self.model_name, str(self.device), self.trust_remote_code)
        if cache_key in self._CACHE:
            self.tokenizer, self.model, self.hidden_size = self._CACHE[cache_key]
        else:
            # 智能路径解析：如果本地有缓存，直接使用本地路径
            import os
            import glob

            resolved_model_path = self.model_name

            # 检查是否是 HuggingFace 模型名称格式
            if '/' in self.model_name and not os.path.exists(self.model_name):
                # 尝试在本地缓存中查找
                cache_base = os.path.expanduser("~/.cache/huggingface/hub")
                model_cache_name = f"models--{self.model_name.replace('/', '--')}"
                model_cache_dir = os.path.join(cache_base, model_cache_name)

                if os.path.exists(model_cache_dir):
                    # 查找 refs/main 指向的快照
                    refs_main = os.path.join(model_cache_dir, "refs", "main")
                    if os.path.exists(refs_main):
                        with open(refs_main, 'r') as f:
                            snapshot_id = f.read().strip()
                        snapshot_path = os.path.join(model_cache_dir, "snapshots", snapshot_id)
                        if os.path.exists(snapshot_path):
                            resolved_model_path = snapshot_path
                            print(f"[INFO] 使用本地缓存: {snapshot_path}")

            os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

            # 使用解析后的路径加载
            try:
                self.tokenizer = self.AutoTokenizer.from_pretrained(
                    resolved_model_path,
                    trust_remote_code=self.trust_remote_code,
                    local_files_only=True  # 强制使用本地
                )
            except (OSError, ValueError) as e:
                # 本地没有，再从镜像源下载
                print(f"[INFO] 本地未找到模型，从镜像源下载...")
                try:
                    self.tokenizer = self.AutoTokenizer.from_pretrained(
                        self.model_name,  # 使用原始名称下载
                        trust_remote_code=self.trust_remote_code,
                        local_files_only=False
                    )
                except TypeError:
                    self.tokenizer = self.AutoTokenizer.from_pretrained(
                        self.model_name,
                        local_files_only=False
                    )
            # 某些 tokenizer 可能没有 pad_token，做个兜底
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.cls_token

            def _load_model(
                low_cpu_mem_usage=None,
                device_map=None,
                use_safetensors=None,
                fast_init=None,
                torch_dtype=None,
                local_files_only=True,  # 默认优先使用本地
            ):
                kwargs = {
                    "trust_remote_code": self.trust_remote_code,
                    "local_files_only": local_files_only
                }
                if low_cpu_mem_usage is not None:
                    kwargs["low_cpu_mem_usage"] = bool(low_cpu_mem_usage)
                if device_map is not None:
                    kwargs["device_map"] = device_map
                if use_safetensors is not None:
                    kwargs["use_safetensors"] = bool(use_safetensors)
                if fast_init is not None:
                    kwargs["_fast_init"] = bool(fast_init)
                if torch_dtype is not None:
                    kwargs["torch_dtype"] = torch_dtype
                try:
                    return self.AutoModel.from_pretrained(resolved_model_path, **kwargs)
                except TypeError:
                    kwargs.pop("low_cpu_mem_usage", None)
                    kwargs.pop("device_map", None)
                    kwargs.pop("use_safetensors", None)
                    kwargs.pop("_fast_init", None)
                    kwargs.pop("torch_dtype", None)
                    kwargs.pop("local_files_only", None)
                    return self.AutoModel.from_pretrained(resolved_model_path, **kwargs)

            self.model = None
            # 先尝试从本地加载
            load_attempts = [
                {"low_cpu_mem_usage": None, "local_files_only": True},
                {"low_cpu_mem_usage": False, "local_files_only": True},
                {"low_cpu_mem_usage": False, "use_safetensors": False, "local_files_only": True},
            ]

            # 如果本地加载失败，再尝试联网下载
            fallback_attempts = [
                {"low_cpu_mem_usage": None, "local_files_only": False},
                {"low_cpu_mem_usage": False, "local_files_only": False},
                {"low_cpu_mem_usage": False, "use_safetensors": False, "local_files_only": False},
                {"low_cpu_mem_usage": False, "use_safetensors": False, "fast_init": False, "local_files_only": False},
                {
                    "low_cpu_mem_usage": False,
                    "use_safetensors": False,
                    "fast_init": False,
                    "torch_dtype": self.torch.float32,
                    "local_files_only": False,
                },
            ]

            # 先尝试本地
            for opts in load_attempts:
                try:
                    self.model = _load_model(**opts)
                    if not any(getattr(p, "is_meta", False) for p in self.model.parameters()):
                        break
                except (OSError, ValueError):
                    continue

            # 本地失败，尝试联网
            if self.model is None or any(getattr(p, "is_meta", False) for p in self.model.parameters()):
                for opts in fallback_attempts:
                    self.model = _load_model(**opts)
                    if not any(getattr(p, "is_meta", False) for p in self.model.parameters()):
                        break

            if any(getattr(p, "is_meta", False) for p in self.model.parameters()):
                # Manual fallback: load state_dict without accelerate/device_map
                try:
                    from transformers import AutoConfig
                    from transformers.utils import cached_file
                    from transformers.modeling_utils import load_sharded_checkpoint
                    config = AutoConfig.from_pretrained(
                        self.model_name,
                        trust_remote_code=self.trust_remote_code,
                    )
                    self.model = self.AutoModel.from_config(config)
                    index_files = [
                        "model.safetensors.index.json",
                        "pytorch_model.bin.index.json",
                    ]
                    idx_path = None
                    for name in index_files:
                        try:
                            idx_path = cached_file(
                                self.model_name,
                                name,
                                _raise_exceptions_for_missing_entries=False,
                                _raise_exceptions_for_connection_errors=True,
                            )
                        except Exception:
                            idx_path = None
                        if idx_path:
                            break
                    if idx_path:
                        folder = os.path.dirname(idx_path)
                        load_sharded_checkpoint(
                            self.model,
                            folder,
                            strict=False,
                            prefer_safe=bool("safetensors" in str(idx_path)),
                        )
                    else:
                        weight_files = ["model.safetensors", "pytorch_model.bin"]
                        weight_path = None
                        for name in weight_files:
                            try:
                                weight_path = cached_file(
                                    self.model_name,
                                    name,
                                    _raise_exceptions_for_missing_entries=False,
                                    _raise_exceptions_for_connection_errors=True,
                                )
                            except Exception:
                                weight_path = None
                            if weight_path:
                                break
                        if not weight_path:
                            raise RuntimeError("model weight file not found in cache")
                        if str(weight_path).endswith(".safetensors"):
                            try:
                                from safetensors.torch import load_file as safe_load
                            except Exception:
                                safe_load = None
                            if safe_load is None:
                                raise RuntimeError("safetensors not available to load weights")
                            state_dict = safe_load(weight_path)
                        else:
                            state_dict = self.torch.load(weight_path, map_location="cpu")
                        self.model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    raise RuntimeError(
                        "transformer model loaded on meta device; "
                        "manual weight loading failed"
                    ) from e

            try:
                self.model.to(self.device)
            except NotImplementedError:
                self.device = self.torch.device("cpu")
                self.model.to(self.device)
            self.model.eval()

            # hidden size
            self.hidden_size = int(getattr(self.model.config, "hidden_size", 0) or 0)

            self._CACHE[cache_key] = (self.tokenizer, self.model, self.hidden_size)

        # feature names 运行后根据 hidden_size 生成
        self.feature_names = [f"lm_emb_{i}" for i in range(self.hidden_size)] if self.hidden_size else []

    def _pool(self, last_hidden_state, attention_mask):
        # last_hidden_state: (B, L, H)
        if self.pooling == "mean":
            # mean pooling with mask
            mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            summed = (last_hidden_state * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return summed / denom
        # default: cls pooling (take first token)
        return last_hidden_state[:, 0, :]

    def smiles_to_embeddings(self, smiles_list, batch_size: int = 32):
        if not self.AVAILABLE:
            raise ImportError("需要 transformers：pip install transformers")

        # 过滤空值
        valid_indices = []
        texts = []
        for i, s in enumerate(smiles_list):
            if s is None or (isinstance(s, float) and np.isnan(s)):
                continue
            ss = str(s).strip()
            if not ss:
                continue
            valid_indices.append(i)
            texts.append(ss)

        if not texts:
            return pd.DataFrame(), []

        embs = []

        for start in tqdm(range(0, len(texts), batch_size), desc="Transformer Embedding"):
            batch = texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self.torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                pooled = self._pool(last_hidden, inputs.get("attention_mask"))
                embs.append(pooled.detach().cpu().numpy().astype(np.float32))

        emb_mat = np.vstack(embs)
        # 生成列名
        if not self.feature_names or len(self.feature_names) != emb_mat.shape[1]:
            self.feature_names = [f"lm_emb_{i}" for i in range(emb_mat.shape[1])]

        df = pd.DataFrame(emb_mat, columns=self.feature_names)
        return df, valid_indices

class RDKitFeatureExtractor:
    """RDKit基础提取器"""

    def __init__(self):
        self.feature_names = None

    def smiles_to_rdkit_features(self, smiles_list):
        if not RDKIT_AVAILABLE:
            raise ImportError("需要安装rdkit")

        features_list, valid_indices = [], []
        descriptor_funcs = dict(Descriptors.descList)

        for idx, smiles in enumerate(tqdm(smiles_list, desc="RDKit提取")):
            try:
                mol = parse_chemical_string(
                    smiles,
                    repair=True,
                    keep_largest_frag=False,
                )
                if mol is None:
                    continue
                features = {}
                for name, func in descriptor_funcs.items():
                    try:
                        val = func(mol)
                        # [修复] 确保值是标量
                        if isinstance(val, (list, tuple, np.ndarray)):
                            continue
                        features[name] = val if (isinstance(val, (int, float)) and np.isfinite(val)) else np.nan
                    except:
                        features[name] = np.nan
                features_list.append(features)
                valid_indices.append(idx)
            except:
                continue

        if not features_list:
            return pd.DataFrame(), []

        df = pd.DataFrame(features_list)
        df = df.select_dtypes(include=[np.number])
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, df.var() > 0]
        df = df.fillna(df.median())
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep='first')]

        self.feature_names = df.columns.tolist()
        return df, valid_indices


# =============================================================================
# 顶级 Worker 函数（用于并行处理）
# =============================================================================
def _rdkit_single_smiles_worker(smiles, start_idx):
    """
    处理单个 SMILES 的 worker 函数
    使用 joblib 时，处理单个样本比处理批次更灵活
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        import numpy as np
    except ImportError:
        return None, start_idx
    
    try:
        if smiles is None:
            return None, start_idx
        if isinstance(smiles, float) and np.isnan(smiles):
            return None, start_idx
        s = str(smiles).strip()
        if not s or s.lower() in ('nan', 'none', ''):
            return None, start_idx
        
        # 快速清理，但保留多片段聚合物/配方信息
        s = s.strip('"\'')
        s = normalize_chemical_string(
            s,
            canonicalize=False,
            repair=True,
            keep_largest_frag=False,
        ) or s

        mol = parse_smiles_quiet(s)
        if mol is None:
            s_simple = normalize_chemical_string(
                s,
                canonicalize=False,
                repair=False,
                keep_largest_frag=False,
            ) or s.replace('@', '').replace('/', '').replace('\\', '')
            mol = parse_chemical_string(
                s_simple,
                repair=True,
                keep_largest_frag=False,
            )
            if mol is None:
                return None, start_idx
        
        descriptor_funcs = dict(Descriptors.descList)
        features = {}
        for name, func in descriptor_funcs.items():
            try:
                val = func(mol)
                # [修复] 确保值是标量，如果是序列则跳过
                if isinstance(val, (list, tuple, np.ndarray)):
                    continue
                features[name] = val if (isinstance(val, (int, float)) and np.isfinite(val)) else np.nan
            except:
                features[name] = np.nan
        return features, start_idx
    except:
        return None, start_idx


class OptimizedRDKitFeatureExtractor:
    """
    并行版 RDKit 提取器 (V4 - 使用 joblib 解决 Streamlit 死锁)
    
    关键修复：
    1. 使用 joblib 替代 multiprocessing（专门解决 fork 死锁问题）
    2. joblib 使用 loky 后端，自动处理 Streamlit/Jupyter 环境
    3. 提供多种后端选择：joblib > threading > 单进程
    """

    def __init__(self, n_jobs=-1, batch_size=500, fast_mode=True, backend='auto', max_workers=None):
        """
        Args:
            n_jobs: 并行进程数（-1表示使用所有核心）
            batch_size: 批大小（joblib 模式下此参数用于进度显示）
            fast_mode: 快速模式（始终为 True）
            backend: 'auto', 'joblib', 'threading', 'single'
            max_workers: 最大工作进程数（None表示不限制）
        """
        cpu_count = mp.cpu_count() or 4

        # [性能修复] 解除线程限制，使用全部CPU核心
        if n_jobs == -1:
            if max_workers is None:
                # 使用全部核心（不再限制为8）
                self.n_jobs = cpu_count
            else:
                self.n_jobs = max(1, min(max_workers, cpu_count))
        else:
            self.n_jobs = max(1, min(n_jobs, cpu_count))

        self.batch_size = batch_size
        self.fast_mode = fast_mode
        self.backend = backend
        self.feature_names = None

    def smiles_to_rdkit_features(self, smiles_list):
        """并行提取 RDKit 描述符"""
        import sys
        
        total_samples = len(smiles_list)
        print(f"\n🚀 RDKit 并行版提取启动")
        print(f"   📊 样本: {total_samples}, 进程: {self.n_jobs}")
        sys.stdout.flush()
        
        # 如果只有1个进程，直接单进程处理
        if self.n_jobs == 1:
            print("   ℹ️ 单进程模式")
            sys.stdout.flush()
            return self._single_process_extract(smiles_list)
        
        # 尝试使用 joblib
        if self.backend in ('auto', 'joblib'):
            result = self._joblib_extract(smiles_list)
            if result is not None:
                return result
        
        # 尝试使用 threading
        if self.backend in ('auto', 'threading'):
            result = self._threading_extract(smiles_list)
            if result is not None:
                return result
        
        # 回退到单进程
        print("   🔄 回退到单进程模式")
        sys.stdout.flush()
        return self._single_process_extract(smiles_list)
    
    def _joblib_extract(self, smiles_list):
        """使用 joblib 进行并行处理"""
        import sys
        
        try:
            from joblib import Parallel, delayed
            print(f"   🔧 使用 joblib (loky 后端)")
            sys.stdout.flush()
        except ImportError:
            print("   ⚠️ joblib 未安装，尝试其他方法...")
            sys.stdout.flush()
            return None
        
        total_samples = len(smiles_list)
        all_features = []
        all_indices = []
        
        try:
            # joblib 的 Parallel 会自动处理 fork 问题
            # backend='loky' 是默认的，使用独立的进程池
            results = Parallel(
                n_jobs=self.n_jobs,
                backend='loky',  # loky 后端专门解决 fork 问题
                verbose=10,      # 显示进度
                pre_dispatch='2*n_jobs'  # 减少内存占用
            )(
                delayed(_rdkit_single_smiles_worker)(smi, idx) 
                for idx, smi in enumerate(smiles_list)
            )
            
            # 收集结果
            for features, idx in results:
                if features is not None:
                    all_features.append(features)
                    all_indices.append(idx)
            
            print(f"\n✅ joblib 并行完成: {len(all_indices)}/{total_samples} 样本")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"\n❌ joblib 失败: {e}")
            sys.stdout.flush()
            return None
        
        return self._build_dataframe(all_features, all_indices)
    
    def _threading_extract(self, smiles_list):
        """使用多线程进行处理（备选方案）"""
        import sys
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        print(f"   🔧 使用 ThreadPoolExecutor")
        sys.stdout.flush()
        
        total_samples = len(smiles_list)
        all_features = []
        all_indices = []
        
        try:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(_rdkit_single_smiles_worker, smi, idx): idx
                    for idx, smi in enumerate(smiles_list)
                }
                
                completed = 0
                for future in tqdm(as_completed(futures), total=total_samples, 
                                   desc=f"RDKit ({self.n_jobs}线程)"):
                    try:
                        features, idx = future.result(timeout=60)
                        if features is not None:
                            all_features.append(features)
                            all_indices.append(idx)
                    except:
                        pass
                    completed += 1
            
            print(f"\n✅ 线程池完成: {len(all_indices)}/{total_samples} 样本")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"\n❌ 线程池失败: {e}")
            sys.stdout.flush()
            return None
        
        return self._build_dataframe(all_features, all_indices)
    
    def _single_process_extract(self, smiles_list):
        """单进程提取"""
        descriptor_funcs = dict(Descriptors.descList)
        all_features, all_indices = [], []
        
        for idx, smiles in enumerate(tqdm(smiles_list, desc="RDKit(单进程)")):
            try:
                if smiles is None:
                    continue
                if isinstance(smiles, float) and np.isnan(smiles):
                    continue
                s = str(smiles).strip()
                if not s or s.lower() in ('nan', 'none', ''):
                    continue
                
                s = s.strip('"\'')
                s = s.replace(';', '.').replace('；', '.').replace('|', '.')
                if '.' in s:
                    frags = [f.strip() for f in s.split('.') if f.strip()]
                    s = max(frags, key=len) if frags else s
                
                mol = parse_chemical_string(
                    s,
                    repair=True,
                    keep_largest_frag=False,
                )
                if mol is None:
                    s = s.replace('@', '').replace('/', '').replace('\\', '')
                    mol = parse_chemical_string(
                        s,
                        repair=True,
                        keep_largest_frag=False,
                    )
                
                if mol is None:
                    continue
                    
                features = {}
                for name, func in descriptor_funcs.items():
                    try:
                        val = func(mol)
                        # [修复] 确保值是标量
                        if isinstance(val, (list, tuple, np.ndarray)):
                            continue
                        features[name] = val if (isinstance(val, (int, float)) and np.isfinite(val)) else np.nan
                    except:
                        features[name] = np.nan
                all_features.append(features)
                all_indices.append(idx)
            except:
                continue
        
        return self._build_dataframe(all_features, all_indices)
    
    def _build_dataframe(self, all_features, all_indices):
        """构建结果 DataFrame"""
        if not all_features:
            print("⚠️ 无有效特征")
            return pd.DataFrame(), []
            
        df = pd.DataFrame(all_features)
        df = df.select_dtypes(include=[np.number])
        df = df.dropna(axis=1, how='all')
        if len(df) > 0:
            df = df.loc[:, df.var() > 0]
        df = df.fillna(df.median())
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
        
        self.feature_names = df.columns.tolist()
        print(f"📈 特征数: {len(self.feature_names)}")
        return df, all_indices


class MemoryEfficientRDKitExtractor:
    """内存优化版提取器"""

    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        self.feature_names = None

    def smiles_to_rdkit_features(self, smiles_list):
        if not RDKIT_AVAILABLE:
            raise ImportError("需要安装rdkit")

        all_features, all_indices = [], []
        descriptor_funcs = dict(Descriptors.descList)

        for batch_start in tqdm(range(0, len(smiles_list), self.batch_size), desc="内存优化提取"):
            batch = smiles_list[batch_start:batch_start + self.batch_size]
            for i, smiles in enumerate(batch):
                try:
                    mol = parse_chemical_string(
                        smiles,
                        repair=True,
                        keep_largest_frag=False,
                    )
                    if mol is None:
                        continue
                    features = {}
                    for name, func in descriptor_funcs.items():
                        try:
                            val = func(mol)
                            features[name] = val if np.isfinite(val) else np.nan
                        except:
                            features[name] = np.nan
                    all_features.append(features)
                    all_indices.append(batch_start + i)
                except:
                    continue

        if not all_features:
            return pd.DataFrame(), []

        df = pd.DataFrame(all_features)
        df = df.select_dtypes(include=[np.number])
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, df.var() > 0]
        df = df.fillna(df.median())

        self.feature_names = df.columns.tolist()
        return df, all_indices


class AdvancedMolecularFeatureExtractor:
    """高级分子特征提取器"""

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("需要安装rdkit")
        self.descriptor_names = []

    def _smiles_to_mol(self, smiles, lightweight=False):
        """将 SMILES 转为 RDKit Mol 对象
        lightweight=True 时跳过昂贵的 normalize/repair，直接用 RDKit 解析
        （适用于 Mordred 等批量场景，SMILES 通常已经是合法的）
        """
        try:
            if pd.isna(smiles):
                return None
            if lightweight:
                s = str(smiles).strip()
                mol = parse_smiles_quiet(s)
                if mol is not None:
                    return mol
                # 轻量回退：只做基础清理，不做 repair
                cleaned = normalize_chemical_string(s, repair=False, canonicalize=False, keep_largest_frag=False)
                if cleaned:
                    return parse_smiles_quiet(cleaned)
                return None
            return parse_chemical_string(
                smiles,
                repair=True,
                keep_largest_frag=False,
            )
        except:
            return None

    def _process_result(self, features, indices, is_df=False):
        # 检查是否为空
        if is_df:
            if features is None or (isinstance(features, pd.DataFrame) and features.empty):
                return pd.DataFrame(), []
            df = features
        else:
            if not features:
                return pd.DataFrame(), []
            df = pd.DataFrame(features)

        df = df.select_dtypes(include=[np.number])
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, df.var() > 0] if len(df) > 0 else df
        df = df.fillna(df.median())
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep='first')]

        return df, indices

    def smiles_to_rdkit_features(self, smiles_list):
        all_features, valid_indices = [], []
        descriptor_funcs = {name: func for name, func in Descriptors.descList}

        print(f"\n🧬 RDKit特征提取")
        for idx, smiles in enumerate(tqdm(smiles_list, desc="提取中")):
            mol = self._smiles_to_mol(smiles)
            if mol is None:
                continue
            features = {}
            for name, func in descriptor_funcs.items():
                try:
                    val = func(mol)
                    # [修复] 确保值是标量
                    if isinstance(val, (list, tuple, np.ndarray)):
                        continue
                    features[name] = val if (isinstance(val, (int, float)) and np.isfinite(val)) else np.nan
                except:
                    features[name] = np.nan
            all_features.append(features)
            valid_indices.append(idx)

        return self._process_result(all_features, valid_indices)

    def smiles_to_mordred(self, smiles_list, batch_size=1000, ignore_3D: bool = True,
                          n_jobs: int | None = None, progress_callback=None):
        """
        Mordred特征提取 - 高性能优化版
        优化点：
        1. 去重计算：相同 SMILES 只算一次 Mordred
        2. 轻量预处理：跳过昂贵的 normalize/repair
        3. 智能并行：Windows 下限制合理进程数，避免 IPC 开销
        4. 进度回调：支持 Streamlit 进度条
        5. 描述符预筛选：跳过耗时且低价值的描述符
        6. 更细粒度的批次 + 实时进度

        Args:
            progress_callback: 可选回调函数 callback(progress_float, status_text)
                               progress_float: 0.0~1.0
                               status_text: 当前状态描述
        """
        if not MORDRED_AVAILABLE:
            raise ImportError("需要安装mordred")

        def _report(pct, msg):
            """统一进度上报"""
            print(msg)
            if progress_callback:
                try:
                    progress_callback(min(pct, 1.0), msg)
                except Exception:
                    pass

        _report(0.0, "🔬 Mordred特征提取（优化模式）")
        t_start = time.time()

        # 1. 预处理分子 - 使用轻量模式，跳过昂贵的 normalize
        mols = []
        valid_indices = []
        smiles_canonical = []  # 用于去重
        total = len(smiles_list)
        for idx, smiles in enumerate(smiles_list):
            mol = self._smiles_to_mol(smiles, lightweight=True)
            if mol:
                mols.append(mol)
                valid_indices.append(idx)
                smiles_canonical.append(Chem.MolToSmiles(mol))
            if idx % 500 == 0:
                _report(0.05 * (idx / max(total, 1)),
                        f"  预处理分子 {idx}/{total}...")

        if not mols:
            return pd.DataFrame(), []

        _report(0.05, f"✓ 预处理完成：{len(mols)}/{total} 个有效分子")

        # 2. 去重：相同 SMILES 只计算一次 Mordred
        unique_smiles = {}  # canonical_smiles -> index in unique list
        unique_mols = []
        mol_to_unique = []  # maps each mol index -> unique index

        for i, csmi in enumerate(smiles_canonical):
            if csmi not in unique_smiles:
                unique_smiles[csmi] = len(unique_mols)
                unique_mols.append(mols[i])
            mol_to_unique.append(unique_smiles[csmi])

        dedup_ratio = len(unique_mols) / len(mols) * 100
        _report(0.08, f"✓ 去重：{len(mols)} → {len(unique_mols)} 个唯一分子（{dedup_ratio:.1f}%）")

        # 3. 初始化计算器 - 可选跳过耗时描述符
        from mordred import Calculator, descriptors as mordred_descriptors

        calc = Calculator(mordred_descriptors, ignore_3D=bool(ignore_3D))

        # 移除已知的超慢描述符（对 CFRP 预测价值低但计算极慢）
        slow_desc_names = {
            'InformationContent', 'TotalInformationContent',
            'StructuralInformationContent', 'ComplementaryInformationContent',
            'ModifiedInformationContent', 'BondInformationContent',
        }
        original_n_desc = len(calc.descriptors)
        calc.descriptors = [d for d in calc.descriptors
                            if type(d).__name__ not in slow_desc_names]
        removed_n = original_n_desc - len(calc.descriptors)
        if removed_n > 0:
            _report(0.09, f"✓ 跳过 {removed_n} 个慢速描述符，保留 {len(calc.descriptors)} 个")

        # 4. 智能选择进程数 - 优化 Windows 下的并行策略
        is_windows = os.name == 'nt'
        cpu_count = mp.cpu_count() or 1

        try:
            n_jobs_int = int(n_jobs) if n_jobs is not None else -1
        except (TypeError, ValueError):
            n_jobs_int = -1

        if n_jobs_int <= 0:
            # 自动模式：Windows 下保守选择，避免 IPC 开销
            if is_windows:
                n_proc = min(cpu_count - 2, 16)  # Windows 最多 16 进程
            else:
                n_proc = min(cpu_count - 1, 8)
            n_proc = max(1, n_proc)
        else:
            n_proc = max(1, min(n_jobs_int, cpu_count))
            # Windows: 限制合理上限，61 进程 IPC 开销太大
            if is_windows and n_proc > 16:
                print(f"⚠️ Windows 下 {n_proc} 进程 IPC 开销过大，自动降至 16")
                n_proc = 16

        # 分子太少时不值得并行
        if len(unique_mols) < n_proc * 10:
            n_proc = max(1, len(unique_mols) // 10)

        _report(0.10, f"✓ 使用 {n_proc} 个进程进行 Mordred 计算")

        # 5. 分批计算 - 使用更小的批次以获得更细粒度的进度
        all_dfs = []
        total_mols = len(unique_mols)
        n_proc = max(1, min(n_proc, total_mols))

        # 自适应 batch_size：服务器上用更小的批次以便更频繁地更新进度
        if total_mols <= 500:
            effective_batch = total_mols  # 小数据集一次算完
        elif total_mols <= 2000:
            effective_batch = min(batch_size, 200)
        else:
            effective_batch = min(batch_size, 500)

        _report(0.10, f"开始计算 {total_mols} 个唯一分子的 Mordred 描述符（batch_size={effective_batch}）...")
        n_batches = (total_mols + effective_batch - 1) // effective_batch

        # 进度区间：0.10 ~ 0.90 用于 Mordred 计算
        progress_base = 0.10
        progress_range = 0.80

        for batch_idx, i in enumerate(range(0, total_mols, effective_batch), 1):
            batch_mols = unique_mols[i: i + effective_batch]

            batch_progress = progress_base + progress_range * (batch_idx / n_batches)
            _report(batch_progress,
                    f"  批次 {batch_idx}/{n_batches} - 处理 {len(batch_mols)} 个分子 ({batch_idx * 100 // n_batches}%)...")

            try:
                if n_proc > 1:
                    try:
                        df_batch = calc.pandas(batch_mols, nproc=n_proc, quiet=True)
                    except TypeError:
                        try:
                            df_batch = calc.pandas(batch_mols, n_proc=n_proc, quiet=True)
                        except TypeError:
                            if i == 0:
                                print("  ⚠️ Mordred 版本不支持并行参数，切换至默认模式...")
                            n_proc = 1
                            df_batch = calc.pandas(batch_mols, quiet=True)
                    except Exception as e:
                        if i == 0:
                            print(f"  ⚠️ 并行计算出错 ({str(e)})，自动切换回单进程模式...")
                        n_proc = 1
                        df_batch = calc.pandas(batch_mols, quiet=True)
                else:
                    df_batch = calc.pandas(batch_mols, quiet=True)

                if type(df_batch).__name__ == 'MordredDataFrame':
                    df_batch = pd.DataFrame(df_batch)

                all_dfs.append(df_batch)

            except Exception as e:
                print(f"  ❌ 批次 {batch_idx} 计算失败: {str(e)}")
                empty_df = pd.DataFrame(index=range(len(batch_mols)), columns=[str(d) for d in calc.descriptors])
                all_dfs.append(empty_df)

        elapsed = time.time() - t_start
        _report(0.92, f"✓ Mordred 计算完成！{total_mols} 个唯一分子，耗时 {elapsed:.1f}s")

        if not all_dfs:
            return pd.DataFrame(), []

        # 6. 合并去重结果
        _report(0.93, "  合并结果...")
        unique_df = pd.concat(all_dfs, ignore_index=True)

        # 7. 还原：将去重结果映射回原始顺序
        final_df = unique_df.iloc[mol_to_unique].reset_index(drop=True)

        # 8. 后处理 - 使用向量化操作替代 apply
        _report(0.95, "  后处理：类型转换与清洗...")
        for col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

        # 9. 移除全 NaN 列和零方差列（减少后续计算量）
        before_cols = final_df.shape[1]
        final_df = final_df.dropna(axis=1, how='all')
        # 移除零方差列
        var = final_df.var(numeric_only=True)
        zero_var_cols = var[var < 1e-10].index.tolist()
        if zero_var_cols:
            final_df = final_df.drop(columns=zero_var_cols)
        after_cols = final_df.shape[1]
        if before_cols != after_cols:
            _report(0.98, f"  ✓ 清洗：{before_cols} → {after_cols} 个有效描述符（移除 {before_cols - after_cols} 个无效列）")

        total_elapsed = time.time() - t_start
        _report(1.0, f"✅ Mordred 提取完成！{after_cols} 个描述符，总耗时 {total_elapsed:.1f}s")

        return self._process_result(final_df, valid_indices, is_df=True)

    def smiles_to_graph_features(self, smiles_list):
        all_features, valid_indices = [], []

        print(f"\n🕸️ 图特征提取")
        for idx, smiles in enumerate(tqdm(smiles_list, desc="构建图")):
            mol = self._smiles_to_mol(smiles)
            if mol is None:
                continue

            try:
                num_atoms = mol.GetNumAtoms()
                num_bonds = mol.GetNumBonds()
                features = {
                    'graph_num_nodes': num_atoms,
                    'graph_num_edges': num_bonds,
                    'graph_avg_degree': 2 * num_bonds / num_atoms if num_atoms > 0 else 0,
                    'graph_density': num_bonds / (num_atoms * (num_atoms - 1) / 2) if num_atoms > 1 else 0,
                    'num_rings': Chem.GetSSSR(mol).__len__(),
                    'num_aromatic_atoms': sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()),
                    'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'mol_weight': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'tpsa': Descriptors.TPSA(mol),
                }
                all_features.append(features)
                valid_indices.append(idx)
            except:
                continue

        return self._process_result(all_features, valid_indices)


class MLForceFieldExtractor:
    """
    机器学习力场特征提取器（TorchANI / ANI2x）

    ✅ 修复点（对应“力场特征总是 0”的常见原因）：
    1) 旧版在 batch padding 后尝试用“拆包”获取 atomic_energies，易与 TorchANI 输出结构不匹配，
       导致能量被错误计算为接近 0（甚至变成全 0）。
    2) 旧版将 padding 原子当作真实原子（或错误 mask），会污染能量/力。
    3) 多组分/多片段 SMILES（A.B 或 A;B）若直接作为一个体系计算，片段间非物理近距离会导致异常。

    本实现策略：
    - 先多进程生成 3D 构象（每个片段独立）
    - 按 “原子数相同” 分组做 batch 推理（无需 padding）
    - 对每个样本把各片段的结果聚合为一个特征向量
    """

    SUPPORTED_SPECIES = {1, 6, 7, 8, 9, 16, 17}  # H,C,N,O,F,S,Cl (ANI2x)

    _HARTREE_TO_KJ_MOL = 2625.499638
    _HARTREE_TO_KCAL_MOL = 627.509474

    def __init__(self, device=None, energy_unit: str = "hartree"):
        """
        Args:
            device: torch.device 或 None（自动选择 cuda/cpu）
            energy_unit: 'hartree' | 'kJ/mol' | 'kcal/mol'
        """
        try:
            import torchani
            import torch
            self.torch = torch
            self.torchani = torchani
            self.AVAILABLE = True
            self.IMPORT_ERROR = None
        except Exception as e:
            # torchani 可能”已安装但无法导入”（例如版本/依赖不匹配、CUDA/GLIBC 问题等）
            print(f"Warning: Failed to import torch/torchani: {e}")
            self.AVAILABLE = False
            self.feature_names = []
            self.IMPORT_ERROR = repr(e)
            self.MODEL_ERROR = None
            self.torch = None
            return


        if device is None:
            self.device = self.torch.device('cuda' if self.torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.energy_unit = (energy_unit or "hartree").lower()

        # ✅ CPU 性能优化：让 Torch 在 CPU 上充分使用线程
        # 注意：在多进程 3D 生成时，ANI 推理通常在主进程执行，因此这里多线程能明显加速
        try:
            if self.device.type == "cpu":
                import os as _os
                n_cpu = _os.cpu_count() or 1
                # 计算线程：尽量用满 CPU；Interop 线程保持较小以减少调度开销
                self.torch.set_num_threads(n_cpu)
                try:
                    self.torch.set_num_interop_threads(min(4, n_cpu))
                except Exception:
                    pass
        except Exception:
            pass

        try:
            self.model = self.torchani.models.ANI2x().to(self.device)
            self.model.eval()
            self.MODEL_ERROR = None
        except Exception as e:
            # 可能原因：首次实例化需要联网下载权重；或 torch/torchani 版本不兼容
            self.MODEL_ERROR = repr(e)
            print(f"ANI Model load error: {e}")
            self.AVAILABLE = False
            self.feature_names = []
            return

        # 保留旧列名，避免下游逻辑/历史模型不兼容
        self.feature_names = [
            'ani_energy',
            'ani_energy_per_atom',
            'ani_max_force',
            'ani_mean_force',
            'ani_force_std',
            # 新增诊断/结构信息
            'ani_n_atoms',
            'ani_n_fragments',
            'ani_success'
        ]

    def _convert_energy(self, e_hartree: float) -> float:
        if e_hartree is None or (isinstance(e_hartree, float) and (np.isnan(e_hartree) or np.isinf(e_hartree))):
            return np.nan
        if self.energy_unit in ["hartree", "ha"]:
            return float(e_hartree)
        if self.energy_unit in ["kj/mol", "kjmol", "kj"]:
            return float(e_hartree) * self._HARTREE_TO_KJ_MOL
        if self.energy_unit in ["kcal/mol", "kcalmol", "kcal"]:
            return float(e_hartree) * self._HARTREE_TO_KCAL_MOL
        # 未知单位：不转换
        return float(e_hartree)

    def _infer_batch(self, species_np: np.ndarray, coords_np: np.ndarray):
        """
        对同原子数的一组分子做 batch 推理（无 padding）
        species_np: (B, N) int64 原子序数
        coords_np: (B, N, 3) float32 3D 坐标
        返回:
            energies: (B,) float
            forces: (B, N, 3) float
        """
        species = self.torch.tensor(species_np, dtype=self.torch.long, device=self.device)
        coords = self.torch.tensor(coords_np, dtype=self.torch.float32, device=self.device)
        coords.requires_grad_(True)

        energy = self.model((species, coords)).energies  # (B,)
        forces = -self.torch.autograd.grad(
            energy.sum(), coords, create_graph=False, retain_graph=False
        )[0]  # (B, N, 3)

        return (
            energy.detach().cpu().numpy().astype(np.float64),
            forces.detach().cpu().numpy().astype(np.float64)
        )

    def smiles_to_ani_features(self, smiles_list, batch_size: int = 256, n_jobs: int | None = None):
        if not self.AVAILABLE:
            raise ImportError("请先安装 torchani: pip install torchani")

        # -------- 1) 多进程生成 3D 构象（每个样本可能含多个片段）--------
        print(f"\n⚛️ 正在生成 3D 构象（多组分将按片段分别生成）...")

        # Windows 下多进程可能不稳定，默认降为单进程
        if n_jobs is None:
            n_jobs = 1 if os.name == 'nt' else max(1, (mp.cpu_count() or 1) - 1)

        # ✅ 修复：检测 CUDA 环境并限制 worker 数量
        # spawn 模式下每个子进程都要重新加载 RDKit/PyTorch 等大型库，内存消耗巨大
        use_spawn = False
        try:
            if self.device is not None and getattr(self.device, 'type', '') == 'cuda' and n_jobs > 1:
                use_spawn = True
                # ✅ 关键修复：限制 spawn 模式下的最大 worker 数量
                # 每个 spawn 子进程大约消耗 500MB-1GB 内存（RDKit + NumPy + 基础库）
                MAX_SPAWN_WORKERS = 8  # 最多 8 个 worker，避免 OOM
                original_n_jobs = n_jobs
                n_jobs = min(n_jobs, MAX_SPAWN_WORKERS)
                if original_n_jobs != n_jobs:
                    print(f'⚠️ CUDA + spawn 模式：为避免内存溢出，worker 数从 {original_n_jobs} 降至 {n_jobs}')
                print(f'✅ 检测到 CUDA 环境，使用 spawn 模式进行 3D 生成（{n_jobs} workers）')
        except Exception:
            pass

        valid_indices = []
        sample_frags = []  # list[list[(atoms, coords)]]
        
        # 单分子超时时间（秒）
        per_molecule_timeout = 30

        try:
            if n_jobs == 1:
                # 单进程（更稳）
                for i, s in enumerate(tqdm(smiles_list, desc="3D Generation")):
                    # [新增] 检查是否请求取消
                    if is_cancelled():
                        print("⏹️ 任务已取消")
                        break
                    res = _generate_3d_data_worker(s)
                    if res is not None:
                        valid_indices.append(i)
                        sample_frags.append(res)
            elif use_spawn:
                # ✅ 修复：使用 mp.Pool + 分批提交，避免内存堆积
                # 关键改进：
                # 1. maxtasksperchild=50 定期重启 worker 释放内存
                # 2. 分批提交任务，每批只提交 batch_chunk 个
                # 3. 使用 apply_async + timeout 处理单个任务超时
                
                results_dict = {}  # {index: result}
                total = len(smiles_list)
                timeout_count = 0
                error_count = 0
                error_samples = []  # 记录前几个错误样本
                fail_stats = {}  # 统计失败原因

                ctx = mp.get_context('spawn')
                pbar = tqdm(total=total, desc=f"3D Generation (spawn, {n_jobs} workers)")

                # 每批提交的任务数（控制内存峰值）
                batch_chunk = n_jobs * 4

                try:
                    with ctx.Pool(processes=n_jobs, maxtasksperchild=50) as pool:
                        for batch_start in range(0, total, batch_chunk):
                            if is_cancelled():
                                print("⏹️ 任务已取消")
                                break

                            batch_end = min(batch_start + batch_chunk, total)

                            # 提交这一批任务
                            async_results = []
                            for idx in range(batch_start, batch_end):
                                smi = smiles_list[idx]
                                ar = pool.apply_async(_generate_3d_data_worker, (smi,))
                                async_results.append((idx, ar))

                            # 获取这一批的结果
                            for idx, ar in async_results:
                                if is_cancelled():
                                    break
                                try:
                                    res = ar.get(timeout=per_molecule_timeout)
                                    # 检查结果类型
                                    if isinstance(res, dict) and 'error' in res:
                                        # 失败，统计原因
                                        error_type = res.get('error', 'unknown')
                                        fail_stats[error_type] = fail_stats.get(error_type, 0) + 1
                                        if len(error_samples) < 5:
                                            error_samples.append({
                                                'idx': idx,
                                                'smiles': smiles_list[idx][:100] if idx < len(smiles_list) else 'N/A',
                                                'error': error_type,
                                                'details': res.get('reasons') or res.get('message', '')
                                            })
                                    elif res is not None and isinstance(res, list):
                                        # 成功
                                        results_dict[idx] = res
                                except mp.TimeoutError:
                                    timeout_count += 1
                                    fail_stats['timeout'] = fail_stats.get('timeout', 0) + 1
                                except Exception as e:
                                    error_count += 1
                                    fail_stats['exception'] = fail_stats.get('exception', 0) + 1
                                    # 记录前5个错误样本用于调试
                                    if len(error_samples) < 5:
                                        error_samples.append({
                                            'idx': idx,
                                            'smiles': smiles_list[idx][:100] if idx < len(smiles_list) else 'N/A',
                                            'error': 'exception',
                                            'details': str(e)[:200]
                                        })
                                pbar.update(1)

                        # 优雅关闭进程池
                        pool.close()
                        pool.join()

                except Exception as e:
                    print(f"⚠️ spawn 模式进程池错误: {e}")
                finally:
                    pbar.close()

                # 打印详细统计
                total_failed = total - len(results_dict)
                if total_failed > 0:
                    print(f"\n⚠️ {total_failed}/{total} 个分子处理失败 ({total_failed/total*100:.1f}%)")
                    print(f"   失败原因统计:")
                    for reason, count in sorted(fail_stats.items(), key=lambda x: -x[1]):
                        print(f"   - {reason}: {count} ({count/total_failed*100:.1f}%)")

                    if error_samples:
                        print(f"\n   前几个失败样本:")
                        for sample in error_samples[:3]:
                            print(f"   - 索引 {sample['idx']}: {sample['smiles'][:50]}...")
                            print(f"     原因: {sample['error']}")
                            if sample['details']:
                                print(f"     详情: {sample['details']}")
                
                # 按索引顺序排列结果
                for i in sorted(results_dict.keys()):
                    valid_indices.append(i)
                    sample_frags.append(results_dict[i])
            else:
                # 普通多进程（非CUDA环境）- ✅ 修复：添加超时机制，避免任务卡死
                from concurrent.futures import wait, FIRST_COMPLETED, TimeoutError as FuturesTimeoutError
                
                results_dict = {}  # {index: result}
                total = len(smiles_list)
                timeout_count = 0
                
                # 分批处理，每批大小 = workers * 2
                batch_submit_size = n_jobs * 2
                pbar = tqdm(total=total, desc=f"3D Generation ({n_jobs} workers)")
                
                with CancellableProcessPoolExecutor(max_workers=n_jobs, task_name="ANI-3D构象生成") as executor:
                    for batch_start in range(0, total, batch_submit_size):
                        # 检查是否请求取消
                        if is_cancelled():
                            print("⏹️ 任务已取消")
                            pbar.close()
                            break
                            
                        batch_end = min(batch_start + batch_submit_size, total)
                        batch_smiles = smiles_list[batch_start:batch_end]
                        
                        # 提交这一批任务
                        futures = {
                            executor.submit(_generate_3d_data_worker, s): batch_start + j 
                            for j, s in enumerate(batch_smiles)
                        }
                        
                        # 等待这批任务完成，设置总超时
                        batch_timeout = per_molecule_timeout * len(batch_smiles) / max(1, n_jobs) + 10
                        done, not_done = wait(futures.keys(), timeout=batch_timeout)
                        
                        # 处理完成的任务
                        for future in done:
                            i = futures[future]
                            try:
                                res = future.result(timeout=1)
                                if res is not None:
                                    results_dict[i] = res
                            except Exception:
                                pass  # 单个任务失败，继续处理其他任务
                        
                        # 取消超时的任务
                        for future in not_done:
                            future.cancel()
                            timeout_count += 1
                        
                        pbar.update(len(batch_smiles))
                
                pbar.close()
                
                if timeout_count > 0:
                    print(f"⚠️ {timeout_count} 个分子处理超时，已跳过")
                
                # 按索引顺序排列结果
                for i in sorted(results_dict.keys()):
                    valid_indices.append(i)
                    sample_frags.append(results_dict[i])
        except Exception as e:
            if "取消" in str(e) or is_cancelled():
                print("⏹️ 任务已取消")
            else:
                print(f"⚠️ 3D 并行生成失败，回退到单进程：{e}")
                valid_indices = []
                sample_frags = []
                for i, s in enumerate(tqdm(smiles_list, desc="3D Generation (fallback)")):
                    if is_cancelled():
                        print("⏹️ 任务已取消")
                        break
                    res = _generate_3d_data_worker(s)
                    if res is not None:
                        valid_indices.append(i)
                        sample_frags.append(res)

        if not sample_frags:
            return pd.DataFrame(), []

        # -------- 2) 展平片段，按原子数分组 batch 推理（无 padding）--------
        from collections import defaultdict

        frag_records = []  # 每个元素对应一个片段
        for orig_i, frags in zip(valid_indices, sample_frags):
            for atoms, coords in frags:
                frag_records.append({
                    'orig_index': orig_i,
                    'n_atoms': int(len(atoms)),
                    'atoms': atoms,
                    'coords': coords,
                    'energy': np.nan,
                    'forces': None,
                    'failed': False
                })

        groups = defaultdict(list)
        for r in frag_records:
            groups[r['n_atoms']].append(r)

        print(f"⚛️ 开始 ANI 推理（按原子数分组批处理，Batch Size={batch_size}, Device={self.device}）...")
        
        # 统计总片段数
        total_frags = len(frag_records)
        processed_frags = 0
        failed_frags = 0
        
        # 按原子数排序，优先处理小分子（更快）
        sorted_groups = sorted(groups.items(), key=lambda x: x[0])
        
        # 显示分组信息
        print(f"   共 {total_frags} 个片段，分为 {len(groups)} 个原子数组")
        
        # 创建总进度条
        pbar = tqdm(total=total_frags, desc="ANI Inference")
        
        for n_atoms, recs in sorted_groups:
            group_size = len(recs)
            
            for start in range(0, group_size, batch_size):
                batch = recs[start:start + batch_size]
                batch_success = False
                
                try:
                    species_np = np.asarray([b['atoms'] for b in batch], dtype=np.int64)
                    coords_np = np.stack([b['coords'] for b in batch]).astype(np.float32)

                    energies, forces = self._infer_batch(species_np, coords_np)
                    for k, b in enumerate(batch):
                        b['energy'] = float(energies[k])
                        b['forces'] = forces[k]
                    batch_success = True
                    processed_frags += len(batch)
                except Exception as e:
                    # 批次失败：标记所有为失败，不再逐个重试（太慢）
                    for b in batch:
                        b['failed'] = True
                        b['energy'] = np.nan
                        b['forces'] = None
                    failed_frags += len(batch)
                    processed_frags += len(batch)
                
                pbar.update(len(batch))
        
        pbar.close()
        
        if failed_frags > 0:
            print(f"⚠️ {failed_frags}/{total_frags} 个片段推理失败（{failed_frags/total_frags*100:.1f}%）")

        # -------- 3) 按样本聚合片段结果，生成特征 --------
        sample_acc = {}
        for idx in valid_indices:
            sample_acc[idx] = {
                'energies': [],
                'force_norms': [],
                'n_atoms': 0,
                'n_frags': 0,
                'failed': False
            }

        for r in frag_records:
            acc = sample_acc.get(r['orig_index'])
            if acc is None:
                continue

            if r.get('failed') or r.get('forces') is None or (not np.isfinite(r.get('energy', np.nan))):
                acc['failed'] = True
                continue

            acc['energies'].append(float(r['energy']))
            norms = np.linalg.norm(np.asarray(r['forces'], dtype=np.float64), axis=1)
            acc['force_norms'].append(norms)
            acc['n_atoms'] += int(r['n_atoms'])
            acc['n_frags'] += 1

        features_list = []
        final_indices = []

        for idx in valid_indices:
            acc = sample_acc[idx]
            if acc['failed'] or acc['n_atoms'] <= 0 or len(acc['energies']) == 0:
                continue

            e_total = float(np.sum(acc['energies']))
            e_total_conv = self._convert_energy(e_total)
            e_per_atom = e_total_conv / acc['n_atoms'] if acc['n_atoms'] > 0 else np.nan

            if acc['force_norms']:
                fn = np.concatenate(acc['force_norms'])
                f_max = float(np.max(fn)) if fn.size else np.nan
                f_mean = float(np.mean(fn)) if fn.size else np.nan
                f_std = float(np.std(fn)) if fn.size else np.nan
            else:
                f_max = f_mean = f_std = np.nan

            feats = {
                'ani_energy': e_total_conv,
                'ani_energy_per_atom': e_per_atom,
                'ani_max_force': f_max,
                'ani_mean_force': f_mean,
                'ani_force_std': f_std,
                'ani_n_atoms': int(acc['n_atoms']),
                'ani_n_fragments': int(acc['n_frags']),
                'ani_success': 1
            }
            features_list.append(feats)
            final_indices.append(idx)

        if not features_list:
            return pd.DataFrame(), []

        df = pd.DataFrame(features_list)
        return df, final_indices


def _quick_ff_embed_mol(mol):
    mol = Chem.AddHs(mol)
    params = _get_etkdg_params()
    for _attr, _val in [("useRandomCoords", True), ("numThreads", 1), ("maxAttempts", 50)]:
        try:
            setattr(params, _attr, _val)
        except Exception:
            pass
    res, _err = _embed_molecule_compat(mol, params)
    res = int(res) if res is not None else -1
    if res != 0:
        try:
            res = AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=100)
        except Exception:
            res = -1
    return mol if res == 0 else None


def _quick_ff_mmff_energy(mol, max_iters: int, do_minimize: bool):
    try:
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            return None
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        if ff is None:
            return None
        if do_minimize:
            ff.Minimize(maxIts=int(max_iters))
        return float(ff.CalcEnergy())
    except Exception:
        return None


def _quick_ff_uff_energy(mol, max_iters: int, do_minimize: bool):
    try:
        ff = AllChem.UFFGetMoleculeForceField(mol)
        if ff is None:
            return None
        if do_minimize:
            ff.Minimize(maxIts=int(max_iters))
        return float(ff.CalcEnergy())
    except Exception:
        return None


def _quick_ff_calc_features(
    smiles: str,
    ff_mode: str = "auto",
    max_iters: int = 200,
    minimize: bool = True,
    max_heavy_atoms: int | None = None,
    max_fragments: int | None = None,
    keep_largest_fragment: bool = False,
    skip_optimize_above_atoms: int | None = None,
):
    if smiles is None or (hasattr(pd, "isna") and pd.isna(smiles)):
        return None
    s = str(smiles).strip()
    if not s or s.lower() in {"nan", "none", "<na>", "na"}:
        return None
    if "*" in s:
        try:
            s = re.sub(r"\[\s*\*\s*\]", "C", s)
        except Exception:
            pass
        s = s.replace("*", "C")

    s = convert_to_smiles(s, fmt="auto") or s
    frags = split_smiles_cell(s)
    if not frags:
        return None

    frag_mols = []
    for frag in frags:
        mol = parse_chemical_string(
            frag,
            repair=True,
            keep_largest_frag=False,
        )
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        hac = int(mol.GetNumHeavyAtoms())
        frag_mols.append((frag, mol, hac))

    if not frag_mols:
        return None

    if keep_largest_fragment:
        frag_mols = [max(frag_mols, key=lambda x: x[2])]
    elif max_fragments is not None and int(max_fragments) > 0 and len(frag_mols) > int(max_fragments):
        frag_mols = sorted(frag_mols, key=lambda x: x[2], reverse=True)[: int(max_fragments)]

    total_heavy = sum(int(hac) for _f, _m, hac in frag_mols)
    if max_heavy_atoms is not None and int(max_heavy_atoms) > 0 and total_heavy > int(max_heavy_atoms):
        return None

    mols = []
    for _frag, mol, _hac in frag_mols:
        mol = _quick_ff_embed_mol(mol)
        if mol is not None:
            mols.append(mol)

    if not mols:
        return None

    mode = str(ff_mode or "auto").lower()
    use_mmff = False
    if mode == "auto":
        use_mmff = all(AllChem.MMFFHasAllMoleculeParams(m) for m in mols)
    elif mode == "mmff":
        use_mmff = True
    elif mode == "uff":
        use_mmff = False

    energies = []
    for mol in mols:
        n_atoms = int(mol.GetNumAtoms())
        do_minimize = bool(minimize)
        if skip_optimize_above_atoms is not None and int(skip_optimize_above_atoms) > 0:
            if n_atoms > int(skip_optimize_above_atoms):
                do_minimize = False
        if use_mmff:
            e = _quick_ff_mmff_energy(mol, max_iters=int(max_iters), do_minimize=do_minimize)
        else:
            e = _quick_ff_uff_energy(mol, max_iters=int(max_iters), do_minimize=do_minimize)
        if e is None:
            return None
        energies.append(e)

    total_atoms = sum(int(m.GetNumAtoms()) for m in mols)
    total_energy = float(np.sum(energies)) if energies else np.nan
    per_atom = total_energy / total_atoms if total_atoms > 0 else np.nan

    return {
        "ff_energy": total_energy,
        "ff_energy_per_atom": per_atom,
        "ff_used_mmff": 1 if use_mmff else 0,
        "ff_used_uff": 0 if use_mmff else 1,
        "ff_n_atoms": int(total_atoms),
        "ff_n_fragments": int(len(mols)),
        "ff_success": 1,
    }


def _quick_ff_worker(
    smiles: str,
    ff_mode: str,
    max_iters: int,
    minimize: bool,
    max_heavy_atoms: int | None,
    max_fragments: int | None,
    keep_largest_fragment: bool,
    skip_optimize_above_atoms: int | None,
):
    try:
        return _quick_ff_calc_features(
            smiles=smiles,
            ff_mode=ff_mode,
            max_iters=max_iters,
            minimize=minimize,
            max_heavy_atoms=max_heavy_atoms,
            max_fragments=max_fragments,
            keep_largest_fragment=keep_largest_fragment,
            skip_optimize_above_atoms=skip_optimize_above_atoms,
        )
    except Exception:
        return None


class QuickForceFieldFeatureExtractor:
    """Fast force-field features using RDKit MMFF/UFF."""

    def __init__(
        self,
        ff_mode: str = "auto",
        max_iters: int = 200,
        minimize: bool = True,
        per_mol_timeout_s: int | None = None,
        max_heavy_atoms: int | None = None,
        max_fragments: int | None = None,
        keep_largest_fragment: bool = False,
        skip_optimize_above_atoms: int | None = None,
    ):
        if not RDKIT_AVAILABLE:
            raise ImportError("需要安装 rdkit")
        self.ff_mode = str(ff_mode or "auto").lower()
        self.max_iters = int(max_iters) if max_iters is not None else 200
        self.minimize = bool(minimize)
        if per_mol_timeout_s is None:
            self.per_mol_timeout_s = None
        else:
            per_mol_timeout_s = int(per_mol_timeout_s)
            self.per_mol_timeout_s = per_mol_timeout_s if per_mol_timeout_s > 0 else None
        self.max_heavy_atoms = int(max_heavy_atoms) if max_heavy_atoms not in (None, "", 0) else None
        self.max_fragments = int(max_fragments) if max_fragments not in (None, "", 0) else None
        self.keep_largest_fragment = bool(keep_largest_fragment)
        self.skip_optimize_above_atoms = int(skip_optimize_above_atoms) if skip_optimize_above_atoms not in (None, "", 0) else None
        self.feature_names = [
            "ff_energy",
            "ff_energy_per_atom",
            "ff_used_mmff",
            "ff_used_uff",
            "ff_n_atoms",
            "ff_n_fragments",
            "ff_success",
        ]

    def _embed_mol(self, mol):
        return _quick_ff_embed_mol(mol)

    def _mmff_energy(self, mol):
        return _quick_ff_mmff_energy(mol, max_iters=int(self.max_iters), do_minimize=bool(self.minimize))

    def _uff_energy(self, mol):
        return _quick_ff_uff_energy(mol, max_iters=int(self.max_iters), do_minimize=bool(self.minimize))

    def _calc_features(self, smiles: str):
        return _quick_ff_calc_features(
            smiles=smiles,
            ff_mode=self.ff_mode,
            max_iters=int(self.max_iters),
            minimize=bool(self.minimize),
            max_heavy_atoms=self.max_heavy_atoms,
            max_fragments=self.max_fragments,
            keep_largest_fragment=bool(self.keep_largest_fragment),
            skip_optimize_above_atoms=self.skip_optimize_above_atoms,
        )

    def smiles_to_ff_features(self, smiles_list, n_jobs: int | None = None):
        features_list = []
        valid_indices = []
        if self.per_mol_timeout_s is not None and self.per_mol_timeout_s > 0:
            if n_jobs is None:
                n_jobs = 1 if os.name == "nt" else max(1, (mp.cpu_count() or 1) // 2)
            n_jobs = max(1, int(n_jobs))
            per_timeout = float(self.per_mol_timeout_s)
            total = len(smiles_list)
            timeout_count = 0
            error_count = 0
            batch_size = max(1, n_jobs * 8)
            ctx = mp.get_context("spawn") if os.name == "nt" else mp.get_context("fork")
            pbar = tqdm(total=total, desc=f"Quick FF ({n_jobs} workers)")
            for batch_start in range(0, total, batch_size):
                if is_cancelled():
                    break
                batch_end = min(batch_start + batch_size, total)
                batch_smiles = smiles_list[batch_start:batch_end]
                pool = ctx.Pool(processes=n_jobs, maxtasksperchild=50)
                async_results = []
                for j, smi in enumerate(batch_smiles):
                    ar = pool.apply_async(
                        _quick_ff_worker,
                        (
                            smi,
                            self.ff_mode,
                            int(self.max_iters),
                            bool(self.minimize),
                            self.max_heavy_atoms,
                            self.max_fragments,
                            bool(self.keep_largest_fragment),
                            self.skip_optimize_above_atoms,
                        ),
                    )
                    async_results.append((batch_start + j, ar))
                timed_out = False
                for idx, ar in async_results:
                    if is_cancelled():
                        break
                    try:
                        feats = ar.get(timeout=per_timeout)
                        if feats is not None:
                            features_list.append(feats)
                            valid_indices.append(idx)
                    except mp.TimeoutError:
                        timeout_count += 1
                        timed_out = True
                    except Exception:
                        error_count += 1
                    pbar.update(1)
                if timed_out:
                    pool.terminate()
                else:
                    pool.close()
                pool.join()
            pbar.close()
            if timeout_count > 0:
                print(f"⚠️ Quick FF: {timeout_count} 个分子超时，已跳过")
            if error_count > 0:
                print(f"⚠️ Quick FF: {error_count} 个分子出错，已跳过")
        else:
            for idx, smi in enumerate(tqdm(smiles_list, desc="Quick FF")):
                feats = self._calc_features(smi)
                if feats is None:
                    continue
                features_list.append(feats)
                valid_indices.append(idx)
        if not features_list:
            return pd.DataFrame(), []
        return pd.DataFrame(features_list), valid_indices


class XTBFeatureExtractor:
    """xTB semi-empirical features via external xtb binary."""

    def __init__(
        self,
        xtb_path: str | None = None,
        method: str = "gfn2",
        run_mode: str = "sp",
        charge: int = 0,
        uhf: int = 0,
        timeout_s: int = 300,
        max_iters: int = 200,
        per_mol_timeout_s: int | None = None,
        max_heavy_atoms: int | None = None,
        max_fragments: int | None = None,
        keep_largest_fragment: bool = False,
        cache_size: int = 10000,
        random_state: int = 42,
    ):
        if not RDKIT_AVAILABLE:
            raise ImportError("需要安装 rdkit")
        requested_xtb_path = str(xtb_path or "xtb").strip()
        resolved_xtb_path = shutil.which(requested_xtb_path)
        if resolved_xtb_path is None and os.path.basename(requested_xtb_path).lower() in {"xtb", "xtb.exe"}:
            executable_name = "xtb.exe" if os.name == "nt" else "xtb"
            env_candidates = [
                os.path.join(sys.prefix, "Library", "bin", executable_name),
                os.path.join(sys.prefix, "bin", executable_name),
                os.path.join(sys.prefix, "Scripts", executable_name),
            ]
            resolved_xtb_path = next((p for p in env_candidates if os.path.isfile(p)), None)
        self.xtb_path = resolved_xtb_path or requested_xtb_path
        self.method = str(method or "gfn2").lower()
        self.run_mode = str(run_mode or "sp").lower()
        self.charge = int(charge)
        self.uhf = int(uhf)
        self.timeout_s = int(timeout_s) if timeout_s is not None else 300
        self.max_iters = int(max_iters) if max_iters is not None else 200
        if per_mol_timeout_s is None:
            self.per_mol_timeout_s = int(self.timeout_s)
        else:
            per_mol_timeout_s = int(per_mol_timeout_s)
            self.per_mol_timeout_s = per_mol_timeout_s if per_mol_timeout_s > 0 else None
        self.max_heavy_atoms = int(max_heavy_atoms) if max_heavy_atoms not in (None, "", 0) else None
        self.max_fragments = int(max_fragments) if max_fragments not in (None, "", 0) else None
        self.keep_largest_fragment = bool(keep_largest_fragment)
        self.cache_size = int(cache_size) if cache_size is not None else 0
        self.random_state = int(random_state)
        self.AVAILABLE = bool(resolved_xtb_path or shutil.which(self.xtb_path))
        self.feature_names = [
            "xtb_total_energy",
            "xtb_energy_per_atom",
            "xtb_homo",
            "xtb_lumo",
            "xtb_gap",
            "xtb_dipole",
            "xtb_n_atoms",
            "xtb_n_fragments",
            "xtb_success",
        ]
        self._cache = OrderedDict()

    def _cache_get(self, key: str | None):
        if not key or self.cache_size <= 0:
            return None
        val = self._cache.get(key)
        if val is not None:
            try:
                self._cache.move_to_end(key)
            except Exception:
                pass
        return val

    def _cache_set(self, key: str | None, val):
        if not key or self.cache_size <= 0 or val is None:
            return
        self._cache[key] = val
        try:
            self._cache.move_to_end(key)
            if len(self._cache) > int(self.cache_size):
                self._cache.popitem(last=False)
        except Exception:
            pass

    def _embed_mol(self, mol):
        mol = Chem.AddHs(mol)
        params = _get_etkdg_params()
        for _attr, _val in [
            ("useRandomCoords", True),
            ("numThreads", 1),
            ("maxAttempts", 50),
            ("randomSeed", self.random_state),
        ]:
            try:
                setattr(params, _attr, _val)
            except Exception:
                pass
        res, _err = _embed_molecule_compat(mol, params)
        res = int(res) if res is not None else -1
        if res != 0:
            try:
                res = AllChem.EmbedMolecule(
                    mol,
                    useRandomCoords=True,
                    maxAttempts=100,
                    randomSeed=self.random_state,
                )
            except Exception:
                res = -1
        if res != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=int(self.max_iters))
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=int(self.max_iters))
            except Exception:
                pass
        return mol

    def _mol_to_xyz(self, mol):
        conf = mol.GetConformer()
        lines = [str(mol.GetNumAtoms()), "xtb input"]
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            lines.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
        return "\n".join(lines)

    def _last_float(self, text: str):
        vals = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", text)
        if not vals:
            vals = re.findall(r"[-+]?\d+(?:[eE][-+]?\d+)?", text)
        if not vals:
            return None
        try:
            return float(vals[-1])
        except Exception:
            return None

    def _parse_xtb_output(self, text: str):
        out = {}
        for line in text.splitlines():
            low = line.lower().strip()
            if "total energy" in low and "free energy" not in low:
                val = self._last_float(line)
                if val is not None:
                    out["total_energy"] = val
            if "homo-lumo gap" in low or "hl-gap" in low:
                val = self._last_float(line)
                if val is not None:
                    out["gap"] = val
            # 修复：匹配包含(HOMO)或HOMO的行，不仅仅是开头
            if "(homo)" in low or ("homo" in low and "lumo" not in low and "gap" not in low):
                val = self._last_float(line)
                if val is not None:
                    out["homo"] = val
            # 修复：匹配包含(LUMO)或LUMO的行
            if "(lumo)" in low or ("lumo" in low and "homo" not in low and "gap" not in low):
                val = self._last_float(line)
                if val is not None:
                    out["lumo"] = val
            if "dipole moment" in low or "molecular dipole" in low:
                val = self._last_float(line)
                if val is not None:
                    out["dipole"] = val
        return out

    def _run_xtb(self, xyz_text: str, timeout_s: int | None = None):
        if not self.AVAILABLE:
            return None
        method = "2" if "gfn2" in self.method else "1"
        mode_flag = "--sp" if self.run_mode in {"sp", "singlepoint"} else "--opt"
        if timeout_s is None:
            timeout_s = int(self.timeout_s)
        else:
            timeout_s = int(timeout_s)
        with tempfile.TemporaryDirectory() as td:
            xyz_path = os.path.join(td, "input.xyz")
            with open(xyz_path, "w", encoding="utf-8") as f:
                f.write(xyz_text)
            cmd = [
                self.xtb_path,
                xyz_path,
                "--gfn",
                method,
                "--chrg",
                str(int(self.charge)),
                "--uhf",
                str(int(self.uhf)),
                mode_flag,
            ]
            out, err = None, None  # [修复] 初始化变量
            try:
                if os.name == "nt":
                    # [修复] Windows 下使用 utf-8 编码，避免 gbk 解码错误
                    proc = subprocess.Popen(
                        cmd,
                        cwd=td,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding='utf-8',
                        errors='replace',  # 遇到无法解码的字符用 ? 替换
                    )
                else:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=td,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        start_new_session=True,
                    )
                out, err = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    if os.name == "nt":
                        proc.kill()
                    else:
                        os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                return None
            except Exception:
                return None

            # [修复] 检查 out 和 err 是否为 None
            if out is None or err is None:
                return None
            if proc.returncode != 0:
                return None
            return self._parse_xtb_output(out + "\n" + err)

    def _calc_features(self, smiles: str):
        if smiles is None or (hasattr(pd, "isna") and pd.isna(smiles)):
            return None
        s = str(smiles).strip()
        if not s or s.lower() in {"nan", "none", "<na>", "na"}:
            return None
        if "*" in s:
            try:
                s = re.sub(r"\[\s*\*\s*\]", "C", s)
            except Exception:
                pass
            s = s.replace("*", "C")

        s = convert_to_smiles(s, fmt="auto") or s
        frags = split_smiles_cell(s)
        if not frags:
            return None

        start_time = time.time()
        deadline = None
        if self.per_mol_timeout_s is not None:
            deadline = start_time + float(self.per_mol_timeout_s)

        frag_mols = []
        for frag in frags:
            mol = parse_chemical_string(
                frag,
                repair=True,
                keep_largest_frag=False,
            )
            if mol is None or mol.GetNumAtoms() == 0:
                continue
            hac = int(mol.GetNumHeavyAtoms())
            frag_mols.append((frag, mol, hac))

        if not frag_mols:
            return None

        if self.keep_largest_fragment:
            frag_mols = [max(frag_mols, key=lambda x: x[2])]
        elif self.max_fragments is not None and len(frag_mols) > int(self.max_fragments):
            frag_mols = sorted(frag_mols, key=lambda x: x[2], reverse=True)[: int(self.max_fragments)]

        total_atoms = 0
        n_frags = 0
        energy_sum = 0.0
        weighted = {"homo": 0.0, "lumo": 0.0, "gap": 0.0, "dipole": 0.0}
        weight_sum = {"homo": 0.0, "lumo": 0.0, "gap": 0.0, "dipole": 0.0}
        timed_out = False

        for frag, mol, hac in frag_mols:
            if deadline is not None and time.time() > deadline:
                timed_out = True
                break
            if self.max_heavy_atoms is not None and hac > int(self.max_heavy_atoms):
                continue
            cache_key = None
            if self.cache_size > 0:
                try:
                    canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                    cache_key = f"{canon}|{self.method}|{self.run_mode}|{self.charge}|{self.uhf}"
                except Exception:
                    cache_key = None

            cached = self._cache_get(cache_key)
            if cached is not None:
                res = cached
                try:
                    n_atoms = int(Chem.AddHs(mol).GetNumAtoms())
                except Exception:
                    n_atoms = int(mol.GetNumAtoms())
            else:
                mol = self._embed_mol(mol)
                if mol is None:
                    continue
                n_atoms = int(mol.GetNumAtoms())
                if self.max_heavy_atoms is not None and int(mol.GetNumHeavyAtoms()) > int(self.max_heavy_atoms):
                    continue
                if deadline is not None:
                    remaining = float(deadline - time.time())
                    if remaining <= 0:
                        timed_out = True
                        break
                    run_timeout = min(float(self.timeout_s), remaining)
                else:
                    run_timeout = float(self.timeout_s)
                xyz = self._mol_to_xyz(mol)
                res = self._run_xtb(xyz, timeout_s=run_timeout)
                if res is None:
                    continue
                self._cache_set(cache_key, res)

            if res is None:
                continue
            energy = res.get("total_energy")
            if energy is None:
                continue
            energy_sum += float(energy)
            total_atoms += int(n_atoms)
            n_frags += 1
            for key in ["homo", "lumo", "gap", "dipole"]:
                if key in res and res[key] is not None:
                    weighted[key] += float(res[key]) * int(n_atoms)
                    weight_sum[key] += int(n_atoms)

        if timed_out:
            return None
        if n_frags <= 0 or total_atoms <= 0:
            return None

        def _avg(k):
            if weight_sum[k] <= 0:
                return np.nan
            return weighted[k] / weight_sum[k]

        return {
            "xtb_total_energy": float(energy_sum),
            "xtb_energy_per_atom": float(energy_sum) / float(total_atoms) if total_atoms > 0 else np.nan,
            "xtb_homo": _avg("homo"),
            "xtb_lumo": _avg("lumo"),
            "xtb_gap": _avg("gap"),
            "xtb_dipole": _avg("dipole"),
            "xtb_n_atoms": int(total_atoms),
            "xtb_n_fragments": int(n_frags),
            "xtb_success": 1,
        }

    def featurize(self, smiles_list, n_jobs=1):
        """提取 xTB 特征

        Args:
            smiles_list: SMILES 列表
            n_jobs: 并行进程数，1 表示单进程，-1 表示使用所有 CPU 核心

        Returns:
            (DataFrame, valid_indices)
        """
        if not self.AVAILABLE:
            return pd.DataFrame(), []

        features_list = []
        valid_indices = []

        # 确定并行进程数
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        elif n_jobs <= 0:
            n_jobs = 1

        # [修复] Windows 限制处理
        original_n_jobs = n_jobs
        use_pool = False
        if os.name == 'nt':
            if n_jobs > 61:
                print(f"⚠️ Windows 环境：multiprocessing.Pool 最多支持 61 个进程，将 n_jobs 从 {n_jobs} 降为 61")
                n_jobs = 61
            use_pool = False
        else:
            # Linux 无限制，优先使用 ProcessPoolExecutor（更现代）
            use_pool = False

        if n_jobs == 1:
            # 单进程模式
            for idx, smi in enumerate(tqdm(smiles_list, desc="xTB")):
                feats = self._calc_features(smi)
                if feats is None:
                    continue
                features_list.append(feats)
                valid_indices.append(idx)
        else:
            # 多进程模式
            print(f"🚀 使用 {n_jobs} 个进程并行计算 xTB 特征...")

            if use_pool:
                # 使用 multiprocessing.Pool（仅Linux，Windows限制61）
                try:
                    with mp.Pool(processes=n_jobs) as pool:
                        results = []
                        for idx, smi in enumerate(smiles_list):
                            results.append((idx, pool.apply_async(self._calc_features, (smi,))))

                        # 收集结果
                        for idx, async_result in tqdm(results, desc="xTB"):
                            try:
                                feats = async_result.get(timeout=self.timeout_s + 10)
                                if feats is not None:
                                    features_list.append((idx, feats))
                            except Exception:
                                pass  # 跳过失败的分子
                except Exception as e:
                    print(f"⚠️ multiprocessing.Pool 失败: {e}")
                    print(f"   降级到 ProcessPoolExecutor (最多 61 进程)")
                    n_jobs = min(n_jobs, 61)
                    use_pool = False

            if not use_pool:
                # 使用 ProcessPoolExecutor（标准方式）
                from concurrent.futures import ProcessPoolExecutor, as_completed

                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    # 提交所有任务
                    future_to_idx = {
                        executor.submit(self._calc_features, smi): idx
                        for idx, smi in enumerate(smiles_list)
                    }

                    # 收集结果
                    for future in tqdm(as_completed(future_to_idx), total=len(smiles_list), desc="xTB"):
                        idx = future_to_idx[future]
                        try:
                            feats = future.result()
                            if feats is not None:
                                features_list.append((idx, feats))
                        except Exception:
                            pass  # 跳过失败的分子

            # 按原始索引排序
            features_list.sort(key=lambda x: x[0])
            valid_indices = [idx for idx, _ in features_list]
            features_list = [feats for _, feats in features_list]

        if not features_list:
            return pd.DataFrame(), []
        return pd.DataFrame(features_list), valid_indices


class ExternalMDFeatureExtractor:
    """External MD feature loader (e.g., LAMMPS outputs)."""

    def __init__(
        self,
        results_path: str | None = None,
        results_df: pd.DataFrame | None = None,
        key_col: str | None = None,
        resin_col: str = "resin_smiles",
        hardener_col: str = "hardener_smiles",
        key_sep: str = "||",
        key_mode: str = "pair",
        canonicalize: bool = True,
        feature_cols: list | None = None,
        drop_missing: bool = False,
    ):
        self.results_path = results_path
        self.results_df = results_df
        self.key_col = key_col
        self.resin_col = resin_col
        self.hardener_col = hardener_col
        self.key_sep = key_sep
        self.key_mode = (key_mode or "pair").lower()
        self.canonicalize = bool(canonicalize)
        self.feature_cols = feature_cols or None
        self.drop_missing = bool(drop_missing)
        self.feature_cols_ = None
        self._table = None

    def _canon(self, s: str) -> str | None:
        if s is None:
            return None
        v = str(s).strip()
        if not v or v.lower() in {"nan", "none", "<na>", "na"}:
            return None
        if not self.canonicalize:
            return v
        c = canonicalize_smiles(v)
        return c or v

    def _make_key(self, resin: str | None, hardener: str | None) -> str | None:
        r = self._canon(resin)
        if self.key_mode in {"resin", "resin_only"}:
            return r
        h = self._canon(hardener)
        if r and h:
            return f"{r}{self.key_sep}{h}"
        return r or h

    def _load_results(self) -> pd.DataFrame:
        if self.results_df is not None:
            return self.results_df.copy()
        if not self.results_path:
            return pd.DataFrame()
        if not os.path.isfile(self.results_path):
            return pd.DataFrame()
        try:
            return pd.read_csv(self.results_path)
        except Exception:
            return pd.DataFrame()

    def _prepare_table(self):
        if self._table is not None:
            return
        df = self._load_results()
        if df is None or df.empty:
            self._table = pd.DataFrame()
            self.feature_cols_ = []
            return

        if self.key_col and self.key_col in df.columns:
            keys = df[self.key_col].astype(str)
        else:
            resin_vals = df[self.resin_col] if self.resin_col in df.columns else pd.Series([None] * len(df))
            hard_vals = df[self.hardener_col] if self.hardener_col in df.columns else pd.Series([None] * len(df))
            keys = []
            for r, h in zip(resin_vals, hard_vals):
                keys.append(self._make_key(r, h))
            keys = pd.Series(keys)

        df = df.copy()
        df["_md_key"] = keys
        df = df[df["_md_key"].notna()].copy()
        if self.feature_cols:
            cols = [c for c in self.feature_cols if c in df.columns]
        else:
            cols = [
                c for c in df.select_dtypes(include=np.number).columns
                if c not in {self.key_col, self.resin_col, self.hardener_col}
            ]
        if not cols:
            self._table = pd.DataFrame()
            self.feature_cols_ = []
            return

        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.set_index("_md_key")[cols]
        if df.index.duplicated().any():
            df = df.groupby(level=0).mean(numeric_only=True)

        self._table = df
        self.feature_cols_ = cols

    def featurize(self, resin_smiles_list, hardener_smiles_list=None):
        self._prepare_table()
        if self._table is None or self._table.empty:
            return pd.DataFrame(), []

        keys = []
        hard_list = hardener_smiles_list or [None] * len(resin_smiles_list)
        for r, h in zip(resin_smiles_list, hard_list):
            keys.append(self._make_key(r, h))

        features_df = self._table.reindex(keys)
        if features_df is None:
            return pd.DataFrame(), []

        features_df = features_df.reset_index(drop=True)
        if self.drop_missing:
            mask = features_df.notna().any(axis=1)
            valid_indices = [i for i, keep in enumerate(mask.tolist()) if keep]
            features_df = features_df.loc[mask].reset_index(drop=True)
        else:
            valid_indices = list(range(len(features_df)))
        return features_df, valid_indices

class EpoxyDomainFeatureExtractor:
    """环氧树脂领域知识特征提取器 (增强版：加入电子效应模拟 + 反应产物模拟)"""

    def __init__(self, enable_reaction_simulation: bool = True, target_conversion: float = 0.5):
        """
        Args:
            enable_reaction_simulation: 是否启用环氧-固化剂反应模拟（生成交联产物特征）
            target_conversion: 目标转化率（0-1），用于模拟不同固化程度
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("需要安装 rdkit")
        
        self.enable_reaction_simulation = enable_reaction_simulation
        self.target_conversion = target_conversion
        
        # 尝试加载反应模拟模块
        self._reaction_simulator = None
        self._crosslink_extractor = None
        if enable_reaction_simulation:
            try:
                from .reaction_simulator import EpoxyReactionSimulator, CrosslinkedFeatureExtractor
                self._reaction_simulator = EpoxyReactionSimulator(verbose=False)
                self._crosslink_extractor = CrosslinkedFeatureExtractor(verbose=False)
            except ImportError:
                print("⚠️ reaction_simulator 模块未找到，反应模拟功能将被禁用")
                self.enable_reaction_simulation = False

    def _get_epoxide_count(self, mol):
        patt = Chem.MolFromSmarts("[C]1[O][C]1")
        matches = mol.GetSubstructMatches(patt)
        return len(matches)

    def _get_active_hydrogen_count(self, mol):
        count = 0
        for atom in mol.GetAtoms():
            # 计算与氮原子相连的氢原子数 (胺类固化剂)
            if atom.GetAtomicNum() == 7:
                count += atom.GetTotalNumHs()
        return count
    
    def _get_anhydride_count(self, mol):
        """计算酸酐基团数量"""
        patt = Chem.MolFromSmarts("[CX3](=[OX1])[OX2][CX3](=[OX1])")
        matches = mol.GetSubstructMatches(patt)
        return len(matches)
    
    def _get_thiol_count(self, mol):
        """计算硫醇基团数量"""
        patt = Chem.MolFromSmarts("[SX2H]")
        matches = mol.GetSubstructMatches(patt)
        return len(matches)
    
    def _detect_curer_type(self, mol):
        """检测固化剂类型"""
        amine_count = self._get_active_hydrogen_count(mol)
        anhydride_count = self._get_anhydride_count(mol)
        thiol_count = self._get_thiol_count(mol)
        
        if anhydride_count > 0:
            return "anhydride", anhydride_count * 2  # 酸酐开环后有2个反应位点
        elif thiol_count > 0:
            return "thiol", thiol_count
        elif amine_count > 0:
            return "amine", amine_count
        else:
            return "unknown", 0

    def _calc_electronic_props(self, mol):
        """计算电子性质 (作为DFT的低成本替代)"""
        try:
            # 计算 Gasteiger 部分电荷
            AllChem.ComputeGasteigerCharges(mol)
            charges = []
            for atom in mol.GetAtoms():
                # 获取计算出的电荷
                c = atom.GetProp('_GasteigerCharge')
                # 有些原子可能无法计算，返回inf或nan
                if c and not c.lower().startswith('nan') and not c.lower().startswith('inf'):
                    charges.append(float(c))

            if not charges:
                return 0.0, 0.0, 0.0

            max_pos_charge = max(charges)  # 亲电性指标
            max_neg_charge = min(charges)  # 亲核性指标

            # 拓扑极性表面积 (TPSA) - 表征分子极性
            tpsa = Descriptors.TPSA(mol)

            return max_pos_charge, max_neg_charge, tpsa
        except Exception:
            return 0.0, 0.0, 0.0
    
    def _extract_crosslink_features(self, smi_r, smi_h):
        """提取交联反应产物特征"""
        if not self.enable_reaction_simulation or self._crosslink_extractor is None:
            return {}
        
        try:
            features = self._crosslink_extractor.extract_crosslink_features(
                smi_r, smi_h, target_conversion=self.target_conversion
            )
            # 添加前缀以区分
            return {f"Crosslink_{k}": v for k, v in features.items()}
        except Exception:
            return {}

    def extract_features(self, resin_smiles_list, hardener_smiles_list, stoichiometry_list=None, stoich_mode: str = 'Resin/Hardener (总质量比, R/H)'):
        features_list = []
        valid_indices = []
        error_count = 0
        error_samples = []  # 记录前几个错误样本

        if len(resin_smiles_list) != len(hardener_smiles_list):
            print(f"❌ 错误：树脂列表长度 ({len(resin_smiles_list)}) 与固化剂列表长度 ({len(hardener_smiles_list)}) 不匹配")
            return pd.DataFrame(), []

        print(f"🔍 开始提取环氧树脂反应特征，共 {len(resin_smiles_list)} 个样本")

        # 遍历每对样本
        for idx, (smi_r, smi_h) in enumerate(zip(resin_smiles_list, hardener_smiles_list)):
            try:
                # 检查输入是否为空
                if pd.isna(smi_r) or pd.isna(smi_h):
                    error_count += 1
                    if len(error_samples) < 5:
                        error_samples.append(f"样本 {idx}: 树脂或固化剂 SMILES 为空")
                    continue

                mol_r = parse_chemical_string(
                    smi_r,
                    repair=True,
                    keep_largest_frag=False,
                )
                mol_h = parse_chemical_string(
                    smi_h,
                    repair=True,
                    keep_largest_frag=False,
                )

                if mol_r is None or mol_h is None:
                    error_count += 1
                    if len(error_samples) < 5:
                        error_samples.append(f"样本 {idx}: 无法解析 SMILES (树脂={smi_r[:50] if isinstance(smi_r, str) else smi_r}, 固化剂={smi_h[:50] if isinstance(smi_h, str) else smi_h})")
                    continue

                # 1. 基础化学计量特征 (原有功能)
                mw_r = Descriptors.MolWt(mol_r)
                mw_h = Descriptors.MolWt(mol_h)
                f_epoxy = self._get_epoxide_count(mol_r)
                f_amine = self._get_active_hydrogen_count(mol_h)

                eew = mw_r / f_epoxy if f_epoxy > 0 else mw_r
                ahew = mw_h / f_amine if f_amine > 0 else mw_h

                # 计算理论配比 (phr)
                theo_phr = (ahew / eew) * 100 if eew > 0 else 0


                # 用户提供的配比（可选）
                # 说明：
                # - stoich_mode = "Resin/Hardener (总质量比, R/H)"：列值为 树脂总量/固化剂总量 (R/H)
                #   则可换算为实际 PHR = 100 / (R/H)
                # - stoich_mode = "PHR (Hardener per 100 Resin)"：列值即为 PHR
                # - stoich_mode = "Equiv Ratio (当量比, H/R)"：列值为 固化剂当量/树脂当量
                # - stoich_mode = "Equiv Ratio (当量比, R/H)"：列值为 树脂当量/固化剂当量
                actual_phr = theo_phr
                if stoichiometry_list is not None and idx < len(stoichiometry_list):
                    try:
                        v = float(stoichiometry_list[idx])
                        if v > 0:
                            if stoich_mode.startswith("Resin/Hardener"):
                                # R/H -> PHR = 100 * H/R = 100 / (R/H)
                                actual_phr = 100.0 / v
                            elif stoich_mode.startswith("PHR"):
                                actual_phr = v
                            elif stoich_mode.startswith("Equiv Ratio (当量比, H/R)"):
                                # H/R = (actual_phr / AHEW) / (100 / EEW)
                                actual_phr = (v * 100.0 * ahew / eew) if eew > 0 and ahew > 0 else 0.0
                            elif stoich_mode.startswith("Equiv Ratio (当量比, R/H)"):
                                # R/H = (100 / EEW) / (actual_phr / AHEW)
                                actual_phr = (100.0 * ahew / (eew * v)) if eew > 0 and ahew > 0 else 0.0
                            else:
                                actual_phr = v
                    except Exception:
                        pass

                # 与理论配比的偏离（用于反映固化欠量/过量）
                stoich_ratio = (actual_phr / theo_phr) if theo_phr > 0 else 0.0
                stoich_delta = actual_phr - theo_phr

                # 当量比（基于树脂/固化剂等效重量）
                # resin_eq: 每100份树脂的环氧当量；hardener_eq: 实际配比下的固化剂当量
                resin_eq = (100.0 / eew) if eew > 0 else 0.0
                hardener_eq = (actual_phr / ahew) if ahew > 0 else 0.0
                equiv_ratio_h_to_r = (hardener_eq / resin_eq) if resin_eq > 0 else 0.0
                equiv_ratio_r_to_h = (resin_eq / hardener_eq) if hardener_eq > 0 else 0.0
                # 2. 电子性质特征 (新增功能 - 模拟DFT)
                r_pos_chg, r_neg_chg, r_tpsa = self._calc_electronic_props(mol_r)
                h_pos_chg, h_neg_chg, h_tpsa = self._calc_electronic_props(mol_h)

                features = {
                    'EEW': eew,
                    'AHEW': ahew,
                    'Resin_Functionality': f_epoxy,
                    'Hardener_Functionality': f_amine,
                    'Theoretical_PHR': theo_phr,
                    'Actual_PHR': actual_phr,
                    'Stoich_Ratio': stoich_ratio,
                    'Stoich_Delta': stoich_delta,
                    'Resin_Eq_100': resin_eq,
                    'Hardener_Eq': hardener_eq,
                    'Equiv_Ratio_H_to_R': equiv_ratio_h_to_r,
                    'Equiv_Ratio_R_to_H': equiv_ratio_r_to_h,
                    # 新增特征列
                    'Resin_Max_Pos_Charge': r_pos_chg,
                    'Resin_Max_Neg_Charge': r_neg_chg,
                    'Resin_TPSA': r_tpsa,
                    'Hardener_Max_Pos_Charge': h_pos_chg,
                    'Hardener_TPSA': h_tpsa
                }
                
                # 3. 检测固化剂类型并添加相关特征
                curer_type, curer_func = self._detect_curer_type(mol_h)
                features['Curer_Type_Amine'] = 1 if curer_type == 'amine' else 0
                features['Curer_Type_Anhydride'] = 1 if curer_type == 'anhydride' else 0
                features['Curer_Type_Thiol'] = 1 if curer_type == 'thiol' else 0
                features['Curer_Functionality_Detected'] = curer_func
                
                # 4. 计算理论最大转化率 (基于化学计量比)
                if f_epoxy > 0 and curer_func > 0:
                    r_value = curer_func / f_epoxy  # 活性氢/环氧比
                    features['Stoichiometry_r'] = r_value
                    features['Theoretical_Alpha_Max'] = min(1.0, r_value, 1.0/r_value) if r_value > 0 else 0.0
                else:
                    features['Stoichiometry_r'] = 0.0
                    features['Theoretical_Alpha_Max'] = 0.0
                
                # 5. 交联反应模拟特征 (如果启用)
                if self.enable_reaction_simulation:
                    crosslink_features = self._extract_crosslink_features(smi_r, smi_h)
                    features.update(crosslink_features)

                features_list.append(features)
                valid_indices.append(idx)

            except Exception as e:
                error_count += 1
                if len(error_samples) < 5:
                    error_samples.append(f"样本 {idx}: {type(e).__name__}: {str(e)[:100]}")
                continue

        # 输出统计信息
        print(f"✅ 成功提取: {len(features_list)} 个样本")
        print(f"❌ 失败: {error_count} 个样本")
        if error_samples:
            print(f"⚠️ 前 {len(error_samples)} 个错误样本:")
            for err in error_samples:
                print(f"   - {err}")

        if not features_list:
            print("❌ 没有成功提取任何特征！请检查输入数据。")
            return pd.DataFrame(), []

        return pd.DataFrame(features_list), valid_indices


class FingerprintExtractor:
    """分子指纹提取器：支持 MACCS Keys 和 Morgan Fingerprints (支持双组分拼接)"""

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("需要安装 rdkit")

    def _gen_fp_array(self, mol, fp_type, n_bits, radius, use_chirality: bool = False, use_features: bool = False):
        """辅助函数：生成单个分子的指纹数组"""
        if fp_type == 'MACCS':
            return np.array(MACCSkeys.GenMACCSKeys(mol))
        elif fp_type == 'Morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=use_chirality, useFeatures=use_features)
            return np.array(fp)
        return np.array([])

    def smiles_to_fingerprints(self, smiles_list, smiles_list_2=None, fp_type='MACCS', n_bits=2048, radius=2, use_chirality: bool = False, use_features: bool = False, drop_all_zero_bits: bool = False):
        """
        提取分子指纹。
        Args:
            smiles_list: 树脂/第一组分 SMILES
            smiles_list_2: (可选) 固化剂/第二组分 SMILES。如果提供，将拼接两个指纹。
        """
        all_fps = []
        valid_indices = []

        # 判断是否需要双组分拼接
        is_dual = smiles_list_2 is not None and len(smiles_list_2) == len(smiles_list)

        desc_str = f"提取 {fp_type} 指纹"
        if is_dual:
            desc_str += " (双组分拼接: Resin + Hardener)"

        print(f"\n👆 {desc_str}")

        # 同一批次中常见重复分子；缓存解析和指纹，避免重复调用 RDKit。
        fp_cache = {}

        def _cache_key(raw):
            if raw is None:
                return ("none", "")
            try:
                return (type(raw).__name__, str(raw).strip())
            except Exception:
                return (type(raw).__name__, repr(raw))

        def _get_cached_fp(raw):
            key = _cache_key(raw)
            if key not in fp_cache:
                mol = parse_chemical_string(
                    raw,
                    repair=True,
                    keep_largest_frag=False,
                )
                if mol is None:
                    fp_cache[key] = None
                else:
                    fp_cache[key] = np.asarray(
                        self._gen_fp_array(
                            mol,
                            fp_type,
                            n_bits,
                            radius,
                            use_chirality=use_chirality,
                            use_features=use_features,
                        ),
                        dtype=np.uint8,
                    )
            return fp_cache[key]

        for idx, smi1 in enumerate(tqdm(smiles_list, desc="指纹提取")):
            try:
                fp1_arr = _get_cached_fp(smi1)
                if fp1_arr is None:
                    continue

                # 2. 处理第二个分子 (如果有)
                if is_dual:
                    fp2_arr = _get_cached_fp(smiles_list_2[idx])
                    if fp2_arr is None:
                        continue
                    row = np.concatenate((fp1_arr, fp2_arr))
                else:
                    row = fp1_arr

                all_fps.append(row)
                valid_indices.append(idx)

            except Exception as e:
                continue

        if not all_fps:
            return pd.DataFrame(), []

        print(f"[DEBUG] 开始转换为 DataFrame，共 {len(all_fps)} 个样本")

        try:
            data_array = np.asarray(all_fps, dtype=np.uint8)
            fp_width = int(data_array.shape[1])
            resin_width = int(len(all_fps[0]) if not is_dual else len(all_fps[0]) // 2)
            columns = [f"Resin_{fp_type}_{i}" for i in range(resin_width)]
            if is_dual:
                columns.extend(
                    f"Hardener_{fp_type}_{i}"
                    for i in range(fp_width - resin_width)
                )
            df = pd.DataFrame(data_array, columns=columns)
        except Exception as e:
            print(f"[DEBUG] 指纹数组转换失败: {e}")
            return pd.DataFrame(), []

        # 可选：移除全为0的列（会导致不同数据集列数不一致；用于模型复用时建议关闭）
        if drop_all_zero_bits:
            print(f"[DEBUG] 开始移除全零列...")
            df = df.loc[:, (df != 0).any(axis=0)]
            print(f"[DEBUG] 移除全零列完成，剩余列数: {df.shape[1]}")

        print(f"[DEBUG] 返回结果: df.shape={df.shape}, valid_indices数量={len(valid_indices)}")
        return df, valid_indices

# =============================================================================
# [新增] MACCS 键定义字典 (用于解释器)
# =============================================================================
MACCS_DEFINITIONS = {
    1: "ISOTOPE", 2: "Atomic no > 103", 3: "Group IVa,Va,VIa Rows 4-6", 4: "Actinides", 
    5: "Group IIIA,IVA", 6: "Lanthanides", 7: "Group VA,VIA Rows 4-6", 8: "QAAA@1", 
    9: "Group VIII (Fe...)", 10: "Group IIA", 11: "4M Ring", 12: "Group IB,IIB", 
    13: "ON(C)C", 14: "S-S", 15: "OC(O)O", 16: "Q:Q", 17: "C#C", 18: "Group IIIA", 
    19: "7M Ring", 20: "Si", 21: "C=C(Q)Q", 22: "3M Ring", 23: "NC(O)O", 24: "N-O", 
    25: "NC(N)N", 26: "C$=C($)C($)C", 27: "I", 28: "QCH2Q", 29: "P", 30: "CQ(C)(C)A", 
    31: "QX", 32: "CSN", 33: "NS", 34: "CH2=A", 35: "Group IA", 36: "S Heterocycle", 
    37: "NC(O)N", 38: "NC(C)N", 39: "OS(O)O", 40: "S-O", 41: "C#N", 42: "F", 43: "QHAQH", 
    44: "Other", 45: "C=CN", 46: "Br", 47: "SAN", 48: "OQ(O)O", 49: "C=C", 50: "C=C(C)C", 
    51: "CSO", 52: "NN", 53: "CN(C)C", 54: "C=C(O)C", 55: "OSO", 56: "ON(O)C", 
    57: "O Heterocycle", 58: "QSQ", 59: "Snot%A%A", 60: "S=O", 61: "AS(A)A", 
    62: "A$A!A$A", 63: "N=O", 64: "A-S", 65: "C%N", 66: "CC(C)(C)C", 67: "QSQ", 
    68: "QHQH (&...)", 69: "QQH", 70: "Q-N-Q", 71: "NO", 72: "O-A", 73: "S=A", 
    74: "CH3ACH3", 75: "A!N$A", 76: "C=C(O)O", 77: "NAN", 78: "C=N", 79: "N$A$N", 
    80: "NAAAN", 81: "SA(A)A", 82: "ACH2QA", 83: "QAA@1", 84: "NH2", 85: "CN(C)Q", 
    86: "CH2QCH2", 87: "X!A$A", 88: "S", 89: "OAAAO", 90: "QHAAQH", 91: "QHAAQH", 
    92: "OC(N)C", 93: "QCH3", 94: "QN", 95: "NAAO", 96: "5M Ring", 97: "N A A O", 
    98: "QAAAA@1", 99: "C=C", 100: "ACH2N", 101: "8M Ring", 102: "QO", 103: "Cl", 
    104: "QA(Q)Q", 105: "A$A($)A", 106: "QA(Q)Q", 107: "X (Halogen)", 108: "CH3AAACH2", 
    109: "ACH2O", 110: "NCO", 111: "NAAOH", 112: "AA(A)(A)A", 113: "Onot%A%A", 
    114: "CH3CH2A", 115: "CH3ACH2", 116: "CH3AAO", 117: "NAO", 118: "ACH2CH2A > 1", 
    119: "N=A", 120: "Heterocyclic atom > 1", 121: "N Heterocycle", 122: "AN(A)A", 
    123: "OCO", 124: "QQ", 125: "Aromatic Ring > 1", 126: "A!O!A", 127: "A$A!O > 1", 
    128: "ACH2A > 1", 129: "ACH2A", 130: "QQ > 1", 131: "QH > 1", 132: "OH > 1", 
    133: "A@A!A", 134: "X (Halogen)", 135: "Nnot%A%A", 136: "O=A > 1", 137: "Heterocycle", 
    138: "QCH2Q > 1", 139: "OH", 140: "O > 3", 141: "CH3 > 2", 142: "N > 1", 
    143: "A$A!A$A", 144: "Anot%A%A", 145: "6M ring > 1", 146: "O > 2", 147: "ACH2CH2A", 
    148: "AQ(A)A", 149: "CH3 > 1", 150: "A!A$A!A", 151: "NH", 152: "OC(C)C", 
    153: "QCH2Q", 154: "C=O", 155: "A!CH2!A", 156: "NA(A)A", 157: "C-O", 158: "C-N", 
    159: "O > 1", 160: "CH3", 161: "N", 162: "Aromatic", 163: "6M Ring", 164: "O", 
    165: "Ring", 166: "Fragments"
}

def get_maccs_description(key_idx):
    """根据键索引获取 MACCS 描述"""
    try:
        idx = int(key_idx)
        return MACCS_DEFINITIONS.get(idx, "Unknown Fragment")
    except:
        return "Invalid Key"


# =============================================================================
# [便捷包装函数] 供 app.py 批量处理模式调用
# =============================================================================

def _add_prefix_to_columns(df, prefix):
    """给 DataFrame 的列名添加前缀

    Args:
        df: DataFrame
        prefix: 前缀字符串，可以是 None

    Returns:
        DataFrame: 列名添加前缀后的 DataFrame
    """
    if prefix and len(df.columns) > 0:
        # [修复] 确保 prefix 是字符串
        prefix_str = str(prefix) if prefix is not None else ""
        if prefix_str:  # 只有非空字符串才添加前缀
            df.columns = [f"{prefix_str}_{col}" for col in df.columns]
    return df


def extract_fingerprints(smiles_list, fp_type='MACCS', n_bits=2048, radius=2, 
                         use_chirality=False, use_features=False, prefix=None):
    """
    便捷函数：提取分子指纹
    
    Args:
        smiles_list: SMILES 列表
        fp_type: 指纹类型 ('MACCS' 或 'Morgan')
        n_bits: Morgan 指纹位数
        radius: Morgan 指纹半径
        use_chirality: 是否使用手性信息
        use_features: 是否使用特征信息
        prefix: 特征名前缀
    
    Returns:
        DataFrame: 指纹特征
    """
    extractor = FingerprintExtractor()
    df, valid_indices = extractor.smiles_to_fingerprints(
        smiles_list, 
        fp_type=fp_type, 
        n_bits=n_bits, 
        radius=radius,
        use_chirality=use_chirality,
        use_features=use_features
    )
    if prefix:
        df = _add_prefix_to_columns(df, prefix)
    return df


def extract_rdkit_descriptors(smiles_list, prefix=None):
    """
    便捷函数：提取 RDKit 标准描述符（单进程）
    
    Args:
        smiles_list: SMILES 列表
        prefix: 特征名前缀
    
    Returns:
        DataFrame: RDKit 描述符特征
    """
    extractor = RDKitFeatureExtractor()
    df, valid_indices = extractor.smiles_to_rdkit_features(smiles_list)
    if prefix:
        df = _add_prefix_to_columns(df, prefix)
    return df


def extract_rdkit_descriptors_parallel(smiles_list, n_jobs=-1, batch_size=500, 
                                        fast_mode=True, prefix=None):
    """
    便捷函数：并行提取 RDKit 描述符
    
    Args:
        smiles_list: SMILES 列表
        n_jobs: 并行进程数（-1 表示自动）
        batch_size: 批处理大小
        fast_mode: 快速模式
        prefix: 特征名前缀
    
    Returns:
        DataFrame: RDKit 描述符特征
    """
    extractor = OptimizedRDKitFeatureExtractor(
        n_jobs=n_jobs, 
        batch_size=batch_size, 
        fast_mode=fast_mode
    )
    df, valid_indices = extractor.smiles_to_rdkit_features(smiles_list)
    if prefix:
        df = _add_prefix_to_columns(df, prefix)
    return df


def extract_rdkit_descriptors_lowmem(smiles_list, batch_size=100, prefix=None):
    """
    便捷函数：内存优化版 RDKit 描述符提取
    
    Args:
        smiles_list: SMILES 列表
        batch_size: 批处理大小
        prefix: 特征名前缀
    
    Returns:
        DataFrame: RDKit 描述符特征
    """
    extractor = MemoryEfficientRDKitExtractor(batch_size=batch_size)
    df, valid_indices = extractor.smiles_to_rdkit_features(smiles_list)
    if prefix:
        df = _add_prefix_to_columns(df, prefix)
    return df


def extract_mordred_descriptors(smiles_list, ignore_3D=True, batch_size=1000, n_jobs=None, prefix=None,
                                progress_callback=None):
    """
    便捷函数：提取 Mordred 描述符

    Args:
        smiles_list: SMILES 列表
        ignore_3D: 是否忽略 3D 描述符
        batch_size: 批处理大小
        n_jobs: 并行进程数（None/-1 表示自动）
        prefix: 特征名前缀
        progress_callback: 进度回调 callback(progress_float, status_text)

    Returns:
        DataFrame: Mordred 描述符特征
    """
    if not MORDRED_AVAILABLE:
        raise ImportError("需要安装 mordred: pip install mordred")

    extractor = AdvancedMolecularFeatureExtractor()
    df, valid_indices = extractor.smiles_to_mordred(
        smiles_list,
        batch_size=batch_size,
        ignore_3D=ignore_3D,
        n_jobs=n_jobs,
        progress_callback=progress_callback
    )
    if prefix:
        df = _add_prefix_to_columns(df, prefix)
    return df


class FGDFeatureExtractor:
    """
    [增强版] FGD (Functional Group Distinction) 特征提取器
    针对用户数据集进行了定制优化：增加了硫醇、酰肼、二苯甲酮等识别规则。
    """

    def __init__(self, cache_size: int = 10000):
        if not RDKIT_AVAILABLE:
            raise ImportError("FGD 提取需要 RDKit 支持。")

        # 1. 定义骨架 (Substrates) - 优先级：结构越特异，越靠前
        self.substrates = {
            # --- [新增] 针对您数据中的二苯甲酮环氧 ---
            "Benzophenone": "c1ccc(cc1)C(=O)c2ccc(cc2)",

            "DGEBA": "c1ccc(cc1)C(C)(C)c2ccc(cc2)",  # 双酚A型
            "DGEBF": "c1ccc(cc1)Cc2ccc(cc2)",  # 双酚F型 (也匹配 DDM 固化剂骨架)
            "Novolac": "c1ccc(O)c(c1)Cc2ccccc2",  # 酚醛骨架
            "TDE-85 (Ester)": "C(=O)OC",  # 酯环族/通用酯键
            "Cycloaliphatic": "C1CCCCC1",  # 脂环族 (六元环)
            "Isocyanurate": "N1C(=O)NC(=O)NC1=O",  # 异氰尿酸酯 (TGIC等)
            "Aliphatic Chain": "[CX4,CX3]~[CX4,CX3]~[CX4,CX3]~[CX4,CX3]",  # 长链脂肪族
            "Benzene Ring": "c1ccccc1"  # 简单苯环 (兜底)
        }

        # 2. 定义官能团 (Groups) - 决定反应机理
        self.groups = {
            "Epoxide": "C1OC1",  # 环氧基
            "Anhydride": "C(=O)OC(=O)",  # 酸酐 (如 MTHPA)

            # --- [新增] 针对您数据中的 NNC(=O) ---
            "Hydrazide": "[NX3][NX3]C(=O)",  # 酰肼 (潜伏性固化剂)

            # --- [新增] 针对您数据中的 SCC... ---
            "Thiol": "[#16X2H]",  # 巯基/硫醇 (-SH)

            "Methacrylate": "CC(=C)C(=O)O",  # 甲基丙烯酸酯
            "Acrylate": "C=CC(=O)O",  # 丙烯酸酯
            "Amine (Primary)": "[NX3;H2]",  # 伯胺 (如 DDM)
            "Amine (Secondary)": "[NX3;H1]",  # 仲胺
            "Hydroxyl": "[OX2H]",  # 羟基
            "Vinyl": "C=C",  # 乙烯基 (兜底)
        }

        # 预编译 pattern
        self._sub_pats = {}
        for k, v in self.substrates.items():
            try:
                self._sub_pats[k] = Chem.MolFromSmarts(v)
            except:
                pass

        self._grp_pats = {}
        for k, v in self.groups.items():
            try:
                self._grp_pats[k] = Chem.MolFromSmarts(v)
            except:
                pass

        self.substrate_names = list(self.substrates.keys()) + ["Other_Substrate"]
        self.group_names = list(self.groups.keys()) + ["Other_Group"]
        self.feature_names_ = (
            [f"Substrate_{name}" for name in self.substrate_names]
            + [f"Group_{name}" for name in self.group_names]
        )

        self.cache_size = int(cache_size) if cache_size is not None else 0
        self._mol_cache = OrderedDict()

    def _clean_smiles(self, text):
        """清洗混合物SMILES，处理分号等非标准分隔符"""
        if pd.isna(text):
            return None
        s = str(text).strip()
        # 支持输入为 SELFIES / BigSMILES：先转换为 SMILES 再清洗
        s = convert_to_smiles(s, fmt="auto") or s
        # 将分隔符统一为 dot (表示非键连混合物)
        frags = split_smiles_cell(s)
        if frags:
            s = ".".join(frags)
        s = s.replace(';', '.').replace('；', '.')
        return s

    def _unique_preserve(self, items):
        seen = set()
        out = []
        for x in items:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    def _get_mol(self, smi: str):
        if not smi:
            return None
        if self.cache_size > 0 and smi in self._mol_cache:
            self._mol_cache.move_to_end(smi)
            return self._mol_cache[smi]
        mol = parse_chemical_string(
            smi,
            repair=True,
            keep_largest_frag=False,
        )
        if mol is None:
            return None
        if self.cache_size > 0:
            self._mol_cache[smi] = mol
            self._mol_cache.move_to_end(smi)
            while len(self._mol_cache) > self.cache_size:
                self._mol_cache.popitem(last=False)
        return mol

    def _iter_fragments(self, mol, keep_largest_frag: bool = True):
        if mol is None:
            return []
        try:
            frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
        except Exception:
            frags = [mol]
        if not frags:
            frags = [mol]
        if keep_largest_frag and len(frags) > 1:
            frags = [max(frags, key=lambda m: m.GetNumHeavyAtoms())]
        return frags

    def _match_mol(self, mol, keep_largest_frag: bool = True, count_features: bool = False):
        sub_counts = {name: 0 for name in self.substrate_names}
        grp_counts = {name: 0 for name in self.group_names}
        matched_subs = []
        matched_grps = []

        for frag in self._iter_fragments(mol, keep_largest_frag=keep_largest_frag):
            for name, pat in self._sub_pats.items():
                if pat and frag.HasSubstructMatch(pat):
                    matched_subs.append(name)
                    if count_features:
                        sub_counts[name] += len(frag.GetSubstructMatches(pat))
                    else:
                        sub_counts[name] = 1
            for name, pat in self._grp_pats.items():
                if pat and frag.HasSubstructMatch(pat):
                    matched_grps.append(name)
                    if count_features:
                        grp_counts[name] += len(frag.GetSubstructMatches(pat))
                    else:
                        grp_counts[name] = 1

        matched_subs = self._unique_preserve(matched_subs)
        matched_grps = self._unique_preserve(matched_grps)

        if not matched_subs:
            sub_counts["Other_Substrate"] = 1
        if not matched_grps:
            grp_counts["Other_Group"] = 1

        return matched_subs, matched_grps, sub_counts, grp_counts

    def categorize_smiles(self, smiles_list, multi_label: bool = False, keep_largest_frag: bool = True):
        """
        输入 SMILES 列表，返回 DataFrame 包含 'FGD_Substrate' 和 'FGD_Group'
        """
        results = []
        valid_indices = []

        print(f"\n📑 正在执行 FGD 官能团分类 (增强版)...")

        for idx, raw_smi in enumerate(tqdm(smiles_list, desc="FGD Classification")):
            try:
                smi = self._clean_smiles(raw_smi)
                if not smi:
                    continue

                mol = self._get_mol(smi)
                if mol is None:
                    continue

                matched_subs, matched_grps, _, _ = self._match_mol(
                    mol, keep_largest_frag=keep_largest_frag, count_features=False
                )
                sub_type = matched_subs[0] if matched_subs else "Other_Substrate"
                func_group = matched_grps[0] if matched_grps else "Other_Group"

                row = {"FGD_Substrate": sub_type, "FGD_Group": func_group}
                if multi_label:
                    row["FGD_Substrate_All"] = ";".join(matched_subs) if matched_subs else "Other_Substrate"
                    row["FGD_Group_All"] = ";".join(matched_grps) if matched_grps else "Other_Group"
                results.append(row)
                valid_indices.append(idx)

            except Exception:
                continue

        if not results:
            return pd.DataFrame(), []

        df = pd.DataFrame(results)
        return df, valid_indices

    def featurize(
        self,
        smiles_list,
        multi_label: bool = True,
        keep_largest_frag: bool = True,
        count_features: bool = False,
    ):
        """Return multi-hot/count FGD features with fixed column space."""
        results = []
        valid_indices = []

        print(f"\n📑 正在执行 FGD 官能团分类 (特征版)...")

        for idx, raw_smi in enumerate(tqdm(smiles_list, desc="FGD Featurize")):
            try:
                smi = self._clean_smiles(raw_smi)
                if not smi:
                    continue

                mol = self._get_mol(smi)
                if mol is None:
                    continue

                matched_subs, matched_grps, sub_counts, grp_counts = self._match_mol(
                    mol, keep_largest_frag=keep_largest_frag, count_features=count_features
                )

                if not multi_label:
                    if matched_subs:
                        keep = matched_subs[0]
                        for name in self.substrate_names:
                            if name not in {keep, "Other_Substrate"}:
                                sub_counts[name] = 0
                        if keep != "Other_Substrate":
                            sub_counts["Other_Substrate"] = 0
                    if matched_grps:
                        keep = matched_grps[0]
                        for name in self.group_names:
                            if name not in {keep, "Other_Group"}:
                                grp_counts[name] = 0
                        if keep != "Other_Group":
                            grp_counts["Other_Group"] = 0

                row = {}
                for name in self.substrate_names:
                    row[f"Substrate_{name}"] = sub_counts.get(name, 0)
                for name in self.group_names:
                    row[f"Group_{name}"] = grp_counts.get(name, 0)

                results.append(row)
                valid_indices.append(idx)
            except Exception:
                continue

        if not results:
            return pd.DataFrame(), []

        df = pd.DataFrame(results, columns=self.feature_names_)
        return df, valid_indices
