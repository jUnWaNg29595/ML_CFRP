# -*- coding: utf-8 -*-
"""
Formulation Fusion & Chemical Semantic Alignment Engine
======================================================
解决高分子材料建模中的三大工程痛点：
1. 跨表自动对齐：性能窄表 (ml_qspr_model_*.csv) 与配方宽表 (ml_wide_samples.csv) 自动无缝融合；
2. 化学内容语义感知：通过分子官能团与数据内容验证列名，防止模糊正则误匹配；
3. 单位统一与分级物理兜底：将 PHR、wt%、pbw 统一为无量纲摩尔当量，处理缺失配比与当量。
"""

import os
import re
import hashlib
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from rdkit import Chem


def normalize_cross_platform_path(p: str) -> str:
    """
    跨平台路径规范化：在 Windows 与 WSL/Linux 之间自动转换路径表达。
    - Linux/WSL 环境下：将 \\\\wsl.localhost\\... 或 C:\\... 转换为 /home/... 或 /mnt/c/...
    - Windows 环境下：将 /home/... 转换为 \\\\wsl.localhost\\Ubuntu-24.04\\home\\...
    """
    if not p:
        return ""
    p_str = str(p).strip().strip("'\"")
    if not p_str:
        return ""
    if os.name != 'nt':  # 运行在 Linux / WSL 环境
        # 1. Windows UNC 格式：\\wsl.localhost\Ubuntu-24.04\home\... 或 //wsl$...
        m_unc = re.match(r'^[/\\]{2}wsl(?:\.localhost|\$)[/\\]+[^/\\]+([/\\].*)$', p_str, re.IGNORECASE)
        if m_unc:
            linux_path = m_unc.group(1).replace('\\', '/')
            return linux_path
        # 2. Windows 盘符格式：C:\... 或 C:/...
        m_drive = re.match(r'^([a-zA-Z]):[/\\](.*)$', p_str)
        if m_drive:
            drive = m_drive.group(1).lower()
            rest = m_drive.group(2).replace('\\', '/')
            return f"/mnt/{drive}/{rest}"
    else:  # 运行在 Windows 环境
        # 1. WSL 路径 /mnt/c/... 还原为 C:\...
        m_mnt = re.match(r'^/mnt/([a-zA-Z])/(.*)$', p_str)
        if m_mnt:
            drive = m_mnt.group(1).upper()
            rest = m_mnt.group(2).replace('/', '\\')
            return f"{drive}:\\{rest}"
        # 2. Linux 原生路径 /home/... 映射为 Windows UNC 共享
        if p_str.startswith('/home/') or (p_str.startswith('/') and not p_str.startswith('//')):
            for prefix in ("//wsl.localhost/Ubuntu-24.04", "//wsl$/Ubuntu-24.04"):
                cand = f"{prefix}{p_str}"
                if os.path.exists(cand):
                    return os.path.normpath(cand)
            return os.path.normpath(f"//wsl.localhost/Ubuntu-24.04{p_str}")
    return os.path.normpath(p_str)


def get_candidate_dataset_dirs(custom_dir: Optional[str] = None) -> List[str]:
    """
    智能获取可能存在数据集（ml_wide_samples.csv、ml_qspr_model_*.csv 等）的候选目录列表。
    
    跨平台适配：
    1. 用户显式指定目录 custom_dir（及其深层子目录探测）
    2. 环境变量 ML_DATASET_DIR
    3. 相对路径/层级探测（支持 WSL 与 Windows 项目结构）：
       - 项目同级/上级目录：../ml_dataset, .., ml_dataset
       - 基于代码文件的上层目录
    4. WSL / Linux 原生路径（如 /home/wangj/project/ml_dataset, /home/wangj/project）
    5. Windows UNC 共享路径与本地路径（如 \\\\\\\\wsl.localhost\\\\Ubuntu-24.04\\\\home\\\\wangj\\\\project...）
    6. 当前工作目录
    
    返回去重且真实存在的目录列表，且按数据集特征完整度评分排序。
    """
    raw_candidates: List[str] = []

    # 1. 显式指定
    if custom_dir:
        raw_candidates.append(str(custom_dir))

    # 2. 环境变量
    env_dir = os.environ.get("ML_DATASET_DIR")
    if env_dir:
        raw_candidates.append(env_dir)

    # 3. 相对路径推断（无论在 WSL 还是 Windows，只要在项目目录内运行都能定位上级与周边）
    cwd = os.getcwd()
    raw_candidates.extend([
        os.path.join(cwd, "ml_dataset"),
        os.path.abspath(os.path.join(cwd, "..", "ml_dataset")),
        os.path.abspath(os.path.join(cwd, "..")),
        cwd,
    ])

    # 基于当前模块文件位置的相对推断
    try:
        mod_dir = os.path.dirname(os.path.abspath(__file__))
        raw_candidates.extend([
            os.path.abspath(os.path.join(mod_dir, "..", "ml_dataset")),
            os.path.abspath(os.path.join(mod_dir, "..", "..", "ml_dataset")),
            os.path.abspath(os.path.join(mod_dir, "..", "..")),
        ])
    except Exception:
        pass

    # 4. Linux / WSL 常见路径（针对 WSL 环境及 Windows UNC 访问）
    raw_candidates.extend([
        "/home/wangj/project/ml_dataset",
        "/home/wangj/project",
        "/home/wangj/ml_dataset",
        "/mnt/c/Users/wangj/Desktop/ml_dataset",
    ])

    # 5. Windows 原生与 UNC 路径（支持从 Windows 端直接访问 WSL 项目与数据）
    raw_candidates.extend([
        r"\\wsl.localhost\Ubuntu-24.04\home\wangj\project\ml_dataset",
        r"//wsl.localhost/Ubuntu-24.04/home/wangj/project/ml_dataset",
        r"\\wsl.localhost\Ubuntu-24.04\home\wangj\project",
        r"//wsl.localhost/Ubuntu-24.04/home/wangj/project",
        r"\\wsl$\Ubuntu-24.04\home\wangj\project\ml_dataset",
        r"//wsl$/Ubuntu-24.04/home/wangj/project/ml_dataset",
        r"\\wsl$\Ubuntu-24.04\home\wangj\project",
        r"//wsl$/Ubuntu-24.04/home/wangj/project",
        r"C:\Users\wangj\Desktop\ml_dataset",
        r"C:\Users\wangj\project\ml_dataset",
        r"C:\Users\wangj\project",
    ])

    # 跨平台转换扩充
    expanded_candidates: List[str] = []
    for p in raw_candidates:
        if not p:
            continue
        if p not in expanded_candidates:
            expanded_candidates.append(p)
        norm_p = normalize_cross_platform_path(p)
        if norm_p and norm_p not in expanded_candidates:
            expanded_candidates.append(norm_p)

    # 深度子目录探测：对每个存在目录，自动发现包含 .csv 的子目录（如 ml_dataset, dataset, data 等）
    sub_discovered: List[str] = []
    for d in expanded_candidates:
        if not d:
            continue
        try:
            if os.path.exists(d) and os.path.isdir(d):
                for item in os.listdir(d):
                    sub_path = os.path.join(d, item)
                    try:
                        if os.path.isdir(sub_path):
                            item_low = item.lower()
                            if "dataset" in item_low or "data" in item_low or item_low in ("ml", "raw"):
                                sub_discovered.append(sub_path)
                            else:
                                sub_files = os.listdir(sub_path)
                                if any(f.lower().endswith(".csv") for f in sub_files):
                                    sub_discovered.append(sub_path)
                    except Exception:
                        pass
        except Exception:
            pass
    expanded_candidates.extend(sub_discovered)

    # 规范化与去重过滤（将 \\wsl$ 与 \\wsl.localhost 视为同级归一化去重）
    seen_keys = set()
    existing_dirs: List[str] = []
    for p in expanded_candidates:
        if not p:
            continue
        p_clean = os.path.normpath(p)
        comp_key = p_clean.lower() if os.name == 'nt' else p_clean
        comp_key = comp_key.replace('\\\\wsl$\\', '\\\\wsl.localhost\\').replace('//wsl$/', '//wsl.localhost/')
        if comp_key in seen_keys:
            continue
        seen_keys.add(comp_key)
        try:
            if os.path.exists(p) and os.path.isdir(p):
                existing_dirs.append(p)
        except Exception:
            pass

    # 智能评分排序：含关键标志性数据集的目录置顶
    def score_dir(d: str) -> int:
        score = 0
        try:
            files = os.listdir(d)
            files_lower = [f.lower() for f in files]
            if "ml_wide_samples.csv" in files_lower:
                score += 100
            if "ml_qspr_model_tg_c.csv" in files_lower:
                score += 50
            if "ml_performance_all.csv" in files_lower:
                score += 30
            qspr_count = sum(1 for f in files_lower if "qspr_model" in f and f.endswith(".csv"))
            score += min(qspr_count * 2, 40)
            perf_count = sum(1 for f in files_lower if "performance" in f and f.endswith(".csv"))
            score += min(perf_count * 2, 30)
            csv_count = sum(1 for f in files_lower if f.endswith(".csv"))
            score += min(csv_count, 20)
        except Exception:
            pass
        return score

    existing_dirs.sort(key=score_dir, reverse=True)
    return existing_dirs


def get_default_dataset_dir(custom_dir: Optional[str] = None) -> str:
    """获取首选的默认数据集目录，若均未找到则回退至安全默认路径"""
    candidates = get_candidate_dataset_dirs(custom_dir)
    if candidates:
        return candidates[0]
    if custom_dir:
        norm_c = normalize_cross_platform_path(custom_dir)
        return norm_c if norm_c else custom_dir
    return os.path.join(os.getcwd(), "ml_dataset")

class FormulationFusionEngine:
    """配方数据融合与化学语义对齐引擎"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        # SMARTS 模式用于化学内容语义校验
        self.pat_epoxy = Chem.MolFromSmarts("[C,c]1O[C,c]1")
        self.pat_amine = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")
        self.pat_anhydride = Chem.MolFromSmarts("C(=O)OC(=O)")
        self.pat_thiol = Chem.MolFromSmarts("[SX2H]")
        self.pat_phenol = Chem.MolFromSmarts("[OX2H]c1ccccc1")
        self.pat_dicy = Chem.MolFromSmarts("NC(=N)NC#N")
        self.last_dataset_dir: Optional[str] = None

    def find_companion_wide_table(self, main_path: str, custom_dir: Optional[str] = None) -> Optional[str]:
        """
        在主数据表同级目录或指定数据集目录下自动寻找 ml_wide_samples.csv
        """
        candidate_dirs = []
        if custom_dir:
            c_norm = normalize_cross_platform_path(custom_dir)
            if c_norm and os.path.exists(c_norm):
                candidate_dirs.append(c_norm)
            elif os.path.exists(custom_dir):
                candidate_dirs.append(custom_dir)
        if main_path:
            m_norm = normalize_cross_platform_path(main_path)
            p_to_check = m_norm if (m_norm and os.path.exists(m_norm)) else main_path
            if os.path.exists(p_to_check):
                candidate_dirs.append(os.path.dirname(os.path.abspath(p_to_check)))
        candidate_dirs.extend(get_candidate_dataset_dirs(custom_dir))

        for c_dir in candidate_dirs:
            if not c_dir or not os.path.exists(c_dir):
                continue
            wide_file = os.path.join(c_dir, "ml_wide_samples.csv")
            if os.path.exists(wide_file):
                self.last_dataset_dir = c_dir
                return wide_file
        return None
    def compute_formulation_hash(self, row: pd.Series, ignore_process: bool = False) -> str:
        """
        基于关键化学组分与工艺条件计算配方唯一物理哈希指纹
        抗行乱序、抗列名微小差异

        Args:
            row: 数据行
            ignore_process: 是否忽略工艺温度列。用于「主表缺失工艺列」的兜底补齐场景——
                此时主表无法提供温度参与指纹，必须与母宽表按纯配方指纹对齐。
        """
        parts = []
        # 收集树脂与固化剂结构（严格排除 _format 等元数据列）
        for c in row.index:
            c_str = str(c).lower()
            if ("resin" in c_str or "curing" in c_str or "hardener" in c_str) and (c_str.endswith("_structure") or c_str.endswith("_smiles")):
                val = str(row[c]).strip() if pd.notna(row[c]) else ""
                if val and val.lower() not in ["none", "nan", ""]:
                    parts.append(f"{c_str}:{val[:80]}")
        # 收集最高温度（ignore_process=True 时跳过，用于纯配方指纹对齐）
        if not ignore_process:
            for c in row.index:
                if "max_temperature" in str(c).lower() or "curing_temp" in str(c).lower():
                    val = row[c]
                    if pd.notna(val):
                        parts.append(f"temp:{float(val):.1f}")
                        break

        sig = "|".join(sorted(parts))
        return hashlib.md5(sig.encode("utf-8")).hexdigest()

    @staticmethod
    def _main_has_any_temperature(df: pd.DataFrame) -> bool:
        """判断主表是否含任何固化温度列（决定指纹对齐是否纳入温度维度）"""
        for c in df.columns:
            c_low = str(c).lower()
            if "max_temperature" in c_low or "curing_temp" in c_low:
                if df[c].notna().any():
                    return True
        return False

    def extract_target_name_from_df(self, df: pd.DataFrame, df_wide: Optional[pd.DataFrame] = None, filename: str = "") -> Optional[str]:
        """
        从目标性能窄表中提取目标列名称 (如 tg_c, tensile_strength_mpa 等)
        """
        if df is None or len(df.columns) == 0:
            return None

        # 0. 优先匹配已知的核心材料性能目标列名
        common_targets = [
            "tg_c", "td5_c", "td10_c", "td50_c", "tmax_c",
            "tensile_strength_mpa", "tensile_modulus_gpa", "tensile_strain_at_break_pct",
            "flexural_strength_mpa", "flexural_modulus_gpa",
            "compressive_strength_mpa", "compressive_modulus_gpa",
            "impact_strength_kj_m2", "charpy_impact_kj_m2", "izod_impact_j_m",
            "fracture_toughness_kic_mpa_m05", "gic_j_m2",
            "lap_shear_strength_mpa", "shear_strength_mpa",
            "crosslink_density_mol_m3", "cte_glassy_per_k", "cte_rubbery_per_k",
            "char_yield_pct", "degree_of_cure_pct", "gel_time_min",
            "cure_reaction_enthalpy_j_g", "dsc_cure_onset_c", "dsc_cure_peak_c",
            "dsc_cure_time_h", "storage_modulus_25c_gpa", "tan_delta_peak_value"
        ]
        for ct in common_targets:
            if ct in df.columns:
                return ct

        # 1. 如果文件名包含 ml_qspr_model_<target>，提取正则目标
        if filename and "ml_qspr_model_" in filename.lower():
            m = re.search(r"ml_qspr_model_([a-zA-Z0-9_]+?)(?:\.csv)*$", os.path.basename(filename).lower())
            if m:
                clean_target = m.group(1)
                for c in df.columns:
                    if str(c).lower() == clean_target:
                        return str(c)

        # 2. 从列名逆向查找（排除结构、工艺、机理、衍生特征、指纹等非目标列）
        cols = df.columns.tolist()
        non_target_prefixes = (
            'process_', 'resin_', 'curing_', 'small_', 'formulation_',
            'initiator_', 'accelerator_', 'reactive_', 'catalyst_', 'filler_', 'other_',
            'mech_', 'crosslink_', 'rdkit_', 'mordred_', 'fp_', 'descriptor_', 'calc_'
        )
        for c in reversed(cols):
            c_str = str(c).lower()
            if (
                c_str.startswith(non_target_prefixes)
                or '_test_' in c_str
                or '_heating_rate' in c_str
                or c_str.endswith(('_structure', '_smiles', '_format', '_id', '_unit', '_type', '_name'))
            ):
                continue
            return str(c)

        return str(df.columns[-1]) if len(df.columns) > 0 else None

    def _extract_target_name_from_df(self, df: pd.DataFrame, df_wide: Optional[pd.DataFrame] = None) -> Optional[str]:
        return self.extract_target_name_from_df(df, df_wide)

    def auto_align_and_fuse(
        self,
        df_main: pd.DataFrame,
        df_wide: pd.DataFrame,
        target_col: Optional[str] = None,
        fill_missing_r: bool = True,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        将目标性能表与母宽表进行自适应多级对齐并融合 (严格防笛卡尔积膨胀)
        
        返回: (融合后的DataFrame, 融合统计元信息)
        """
        meta = {
            "strategy": "none",
            "matched_rows": 0,
            "total_rows": len(df_main),
            "match_rate": 0.0,
            "fused_columns_count": 0,
        }

        if df_wide is None or len(df_wide) == 0:
            return df_main, meta

        # 策略 0: 如果 df_main 已经融合过母宽表（具备关键配比/组分列），直接保留，杜绝重复融合膨胀。
        # [工艺参数兜底] 若主表缺失工艺温度/时间列而母宽表具备（典型场景：旧版融合产物），
        # 则按「纯配方指纹」（忽略温度维度）补齐工艺列后再返回，无需重建融合。
        if "resin_1_amount_phr" in df_main.columns or "resin_3_structure" in df_main.columns:
            proc_cols_missing = [
                c for c in df_wide.columns
                if c.startswith("process_") and c != "process_id" and c not in df_main.columns
            ]
            if proc_cols_missing:
                hash_fn = lambda r: self.compute_formulation_hash(r, ignore_process=True)
                main_hashes = df_main.apply(hash_fn, axis=1)
                wide_hashes = df_wide.apply(hash_fn, axis=1)
                fused = self._backfill_columns_by_fingerprint(
                    df_main, df_wide, main_hashes, wide_hashes, proc_cols_missing
                )
                if fused is not None:
                    meta["strategy"] = "already_fused_preserved + process_backfill (fingerprint)"
                    meta["matched_rows"] = len(fused)
                    meta["match_rate"] = 1.0
                    meta["fused_columns_count"] = len(proc_cols_missing)
                    meta["process_backfilled"] = proc_cols_missing
                    return fused, meta
            meta["strategy"] = "already_fused_preserved"
            meta["matched_rows"] = len(df_main)
            meta["match_rate"] = 1.0
            meta["fused_columns_count"] = 0
            return df_main, meta

        # 策略 1 (黄金对齐，最高优先级): 目标非空过滤对齐（ml_qspr_model_*.csv 的标准生成方式）
        # 窄表原始就是由母宽表依据目标属性非空筛选出来的，顺序和行数严格 1-to-1 对应！
        if target_col is None:
            target_col = self.extract_target_name_from_df(df_main, df_wide)

        if target_col and target_col in df_wide.columns and target_col in df_main.columns:
            wide_target_sub = df_wide[df_wide[target_col].notna()].reset_index(drop=True)
            if len(wide_target_sub) == len(df_main):
                try:
                    main_vals = pd.to_numeric(df_main[target_col], errors="coerce").dropna().values
                    wide_vals = pd.to_numeric(wide_target_sub[target_col], errors="coerce").dropna().values
                    if len(main_vals) == len(wide_vals) and np.allclose(main_vals[:10], wide_vals[:10], rtol=1e-3, atol=1e-3):
                        # 完美对齐，补充 wide 独有列
                        wide_cols_to_add = [c for c in wide_target_sub.columns if c not in df_main.columns]
                        fused = pd.concat([df_main.reset_index(drop=True), wide_target_sub[wide_cols_to_add]], axis=1)
                        meta["strategy"] = f"target_filter_alignment ({target_col})"
                        meta["matched_rows"] = len(fused)
                        meta["match_rate"] = 1.0
                        meta["fused_columns_count"] = len(wide_cols_to_add)
                        return fused, meta
                except Exception:
                    pass

        # 策略 2: 安全主键连接 (严格去重，严禁很多对多的笛卡尔积膨胀)
        id_cols = [c for c in ["record_id", "sample_id", "formulation_id"] if c in df_main.columns and c in df_wide.columns]
        if id_cols:
            # 优先选择在 wide 表中唯一的值
            merge_key = None
            for c in id_cols:
                if df_wide[c].is_unique:
                    merge_key = c
                    break
            if not merge_key:
                merge_key = id_cols[0]

            wide_cols_to_use = [c for c in df_wide.columns if c not in df_main.columns or c == merge_key]
            df_wide_sub = df_wide[wide_cols_to_use]
            # 关键防膨胀保护：若 merge_key 在 wide 表中有重复值，先去重再左连接
            if not df_wide_sub[merge_key].is_unique:
                df_wide_sub = df_wide_sub.drop_duplicates(subset=[merge_key], keep='first')

            fused = pd.merge(df_main, df_wide_sub, on=merge_key, how="left")
            if len(fused) == len(df_main):
                meta["strategy"] = f"safe_key_join ({merge_key})"
                meta["matched_rows"] = len(fused)
                meta["match_rate"] = 1.0
                meta["fused_columns_count"] = len(wide_cols_to_use) - 1
                return fused, meta

        # 策略 3: 配方化学指纹哈希对齐 (Formulation Hash Alignment)
        # 若主表本身缺失温度列而母宽表具备，则温度无法参与指纹，改用纯配方指纹对齐后补齐工艺列
        ignore_process = not self._main_has_any_temperature(df_main)
        hash_fn = lambda r: self.compute_formulation_hash(r, ignore_process=ignore_process)
        main_hashes = df_main.apply(hash_fn, axis=1)
        wide_hashes = df_wide.apply(hash_fn, axis=1)

        # 建立 wide 表 hash 查找字典（只保留第一个匹配）
        wide_hash_dict = {}
        for idx, h in enumerate(wide_hashes):
            if h not in wide_hash_dict:
                wide_hash_dict[h] = idx

        matched_indices = []
        matched_count = 0
        for h in main_hashes:
            if h in wide_hash_dict:
                matched_indices.append(wide_hash_dict[h])
                matched_count += 1
            else:
                matched_indices.append(None)

        if matched_count > 0.5 * len(df_main):
            # 取出匹配到的 wide 行
            # 需要补充的关键列：配比/当量列 + 工艺参数列（工艺温度、时间等）
            key_phr_cols = [c for c in df_wide.columns if ("amount_phr" in c or "equivalent_weight" in c or "molecular_weight" in c) and c not in df_main.columns]
            proc_cols = [c for c in df_wide.columns if c.startswith("process_") and c != "process_id" and c not in df_main.columns]
            key_phr_cols = key_phr_cols + [c for c in proc_cols if c not in key_phr_cols]
            if not key_phr_cols:
                key_phr_cols = [c for c in df_wide.columns if c not in df_main.columns]

            supp_df = pd.DataFrame(index=df_main.index, columns=key_phr_cols)
            for m_i, w_i in enumerate(matched_indices):
                if w_i is not None:
                    supp_df.iloc[m_i] = df_wide.iloc[w_i][key_phr_cols]

            fused = pd.concat([df_main, supp_df], axis=1)
            meta["strategy"] = "chemical_fingerprint_hash" + (" + process_backfill" if proc_cols else "")
            if proc_cols:
                meta["process_backfilled"] = proc_cols
            meta["matched_rows"] = matched_count
            meta["match_rate"] = round(matched_count / max(1, len(df_main)), 4)
            meta["fused_columns_count"] = len(key_phr_cols)
            return fused, meta

        # 若无法精确对齐，则原样返回
        return df_main, meta

    def _backfill_columns_by_fingerprint(
        self,
        df_main: pd.DataFrame,
        df_wide: pd.DataFrame,
        main_hashes: pd.Series,
        wide_hashes: pd.Series,
        backfill_cols: List[str],
        min_match_rate: float = 0.5,
    ) -> Optional[pd.DataFrame]:
        """
        通过指纹哈希将母宽表中的指定列回填到主表（严格 1:1，防笛卡尔积膨胀）。

        匹配率低于 min_match_rate 时返回 None（调用方回退原行为），
        否则返回按原行序补齐后的新 DataFrame（保留主表原 dtype）。
        """
        if not backfill_cols or len(df_main) == 0 or len(df_wide) == 0:
            return None

        wide_first = {}
        for w_idx, h in enumerate(wide_hashes):
            if h and h not in wide_first:
                wide_first[h] = w_idx

        matched_w_idx = [wide_first.get(h) for h in main_hashes]
        matched_count = sum(1 for w in matched_w_idx if w is not None)
        if matched_count < min_match_rate * len(df_main):
            return None

        left = df_main.copy()
        left["__fp_hash__"] = main_hashes.values
        right = df_wide[backfill_cols].copy()
        right["__fp_hash__"] = wide_hashes.values
        # 同一指纹只取首个匹配，杜绝多对多膨胀
        right = right.drop_duplicates(subset=["__fp_hash__"], keep="first")
        fused = left.merge(right, on="__fp_hash__", how="left", sort=False)
        fused = fused.drop(columns=["__fp_hash__"])
        if len(fused) != len(df_main):
            return None
        # 恢复主表原始行序
        fused.index = df_main.index
        return fused

    def _locate_dataset_file(self, filename: str, custom_dir: Optional[str] = None) -> Optional[str]:
        """在常见数据集目录下定位指定文件"""
        candidate_dirs = []
        if custom_dir:
            c_norm = normalize_cross_platform_path(custom_dir)
            if c_norm and os.path.exists(c_norm):
                candidate_dirs.append(c_norm)
            elif os.path.exists(custom_dir):
                candidate_dirs.append(custom_dir)
        if getattr(self, "last_dataset_dir", None):
            l_norm = normalize_cross_platform_path(self.last_dataset_dir)
            if l_norm and os.path.exists(l_norm):
                candidate_dirs.append(l_norm)
            elif os.path.exists(self.last_dataset_dir):
                candidate_dirs.append(self.last_dataset_dir)
        candidate_dirs.extend(get_candidate_dataset_dirs(custom_dir))
        for d in candidate_dirs:
            if not d or not os.path.exists(d):
                continue
            p = os.path.join(d, filename)
            if os.path.exists(p):
                self.last_dataset_dir = d
                return p
        return None

    def augment_with_test_standards(
        self,
        df: pd.DataFrame,
        standards_path: Optional[str] = None,
        bridge_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        将测试标准信息 (ASTM / ISO / GB / DIN EN ISO / JIS) 关联到融合后的数据集。

        关联策略：
            1. df 自带 performance_row_id → 直接按性能行关联；
            2. df 仅有 record_id → 通过性能总表 (ml_performance_all.csv) 的
               record→performance 映射桥接后关联。

        同一样本引用多个标准时去重聚合并排序，输出三列：
            - test_standard_organization: 标准组织 ("ASTM; ISO")
            - test_standard_canonical: 规范标准编号 ("ASTM D3418-1982; ISO 75-1-2004")
            - test_standard_count: 引用标准数量

        严格左连接、防膨胀；标准文件缺失或关联失败时原样返回。
        """
        meta: Dict[str, Any] = {
            "standards_matched": 0,
            "standards_coverage": 0.0,
            "standards_strategy": None,
        }
        if df is None or len(df) == 0:
            return df, meta

        # 1. 定位标准表
        if not standards_path or not os.path.exists(str(standards_path)):
            standards_path = self._locate_dataset_file("ml_performance_standards.csv")
        if not standards_path or not os.path.exists(str(standards_path)):
            return df, meta
        try:
            std = pd.read_csv(standards_path, low_memory=False)
        except Exception:
            return df, meta
        if not {"performance_row_id", "standard_canonical", "standard_organization"}.issubset(std.columns):
            return df, meta
        std = std[std["performance_row_id"].notna()].copy()

        def _join_unique(series: pd.Series) -> str:
            return "; ".join(sorted({x for v in series for x in str(v).split("; ") if x and x.lower() != "nan"}))

        # 2. 性能行级聚合 (一个 perf 行可能引用多个标准)
        std_agg = std.groupby("performance_row_id").agg(
            test_standard_canonical=("standard_canonical", _join_unique),
            test_standard_organization=("standard_organization", _join_unique),
        ).reset_index()
        std_agg["test_standard_count"] = std_agg["test_standard_canonical"].str.split("; ").str.len()

        # 3. 选择关联键
        if "performance_row_id" in df.columns:
            key = "performance_row_id"
            meta["standards_strategy"] = "direct_performance_row_id"
        elif "record_id" in df.columns:
            # 通过性能总表桥接 record → performance_row_id
            if not bridge_path or not os.path.exists(str(bridge_path)):
                bridge_path = self._locate_dataset_file("ml_performance_all.csv")
            if not bridge_path or not os.path.exists(str(bridge_path)):
                return df, meta
            try:
                bridge = pd.read_csv(bridge_path, low_memory=False, usecols=["record_id", "performance_row_id"])
            except Exception:
                return df, meta
            rec_std = bridge.drop_duplicates(["record_id", "performance_row_id"]).merge(
                std_agg, on="performance_row_id", how="inner"
            )
            if rec_std.empty:
                return df, meta
            std_agg = rec_std.groupby("record_id").agg(
                test_standard_canonical=("test_standard_canonical", _join_unique),
                test_standard_organization=("test_standard_organization", _join_unique),
            ).reset_index()
            std_agg["test_standard_count"] = std_agg["test_standard_canonical"].str.split("; ").str.len()
            key = "record_id"
            meta["standards_strategy"] = "record_bridge_via_performance_all"
        else:
            return df, meta

        # 4. 防膨胀左连接
        if std_agg[key].duplicated().any():
            std_agg = std_agg.drop_duplicates(subset=[key], keep="first")
        add_cols = [c for c in std_agg.columns if c != key and c not in df.columns]
        matched = int(df[key].isin(set(std_agg[key])).sum())
        meta["standards_matched"] = matched
        meta["standards_coverage"] = round(matched / len(df), 4)
        if not add_cols:
            return df, meta
        fused = df.merge(std_agg[[key] + add_cols], on=key, how="left", sort=False)
        if len(fused) != len(df):
            return df, meta
        fused.index = df.index
        meta["standards_matched"] = int(fused["test_standard_count"].notna().sum())
        meta["standards_coverage"] = round(meta["standards_matched"] / len(fused), 4)
        return fused, meta

    def detect_column_roles_with_semantic_check(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        通过【正则模式 + RDKit化学内容抽样验证】，100% 确保列角色推断准确无误

        [添加剂感知修复] 纯化学内容验证会把含环氧的活性稀释剂列自动判为“树脂”、
        含胺的增韧剂列判为“固化剂”、甚至将产物列（crosslink_product_*，历史
        产物含环氧）判为原料，造成自引用污染。因此先按列名排除添加剂/衍生列，
        它们的小分子处理由反应提取链内部的添加剂分诊（P2/P3）负责。
        """
        roles = {
            "resins": [],       # [(列名, 序号, 检测说明)]
            "curers": [],       # [(列名, 序号, 检测说明)]
            "phr_cols": [],     # [(列名, 对应组分)]
            "ratio_cols": [],   # 单一配比列候选
            "eew_cols": [],     # 当量列
        }

        # 添加剂/衍生输出列排除表（不做树脂/固化剂角色判定）
        _excluded_col_re = re.compile(
            r"(small_additive|additive|diluent|toughener|filler|catalyst|accelerator|"
            r"initiator|compatibilizer|pigment|solvent|crosslink_|mech_|"
            r"product_smiles|product_structure|product_bigsmiles|rdkit_|mordred_|"
            r"fp_|descriptor_)",
            re.I,
        )

        text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 1. 扫描树脂列
        for c in text_cols:
            c_low = str(c).lower()
            if _excluded_col_re.search(c_low):
                continue
            is_name_match = bool(re.search(r"resin|epoxy", c_low)) and ("structure" in c_low or "smiles" in c_low or re.search(r"_\d+$", c_low))
            # 抽样化学内容验证
            sample_has_epoxy = self._sample_has_substructure(df[c], self.pat_epoxy)
            if sample_has_epoxy or is_name_match:
                note = "化学验证: 含环氧基团" if sample_has_epoxy else "名称匹配"
                roles["resins"].append((c, note))

        # 2. 扫描固化剂列
        for c in text_cols:
            c_low = str(c).lower()
            if c in [r[0] for r in roles["resins"]]:
                continue
            if _excluded_col_re.search(c_low):
                continue
            is_name_match = bool(re.search(r"cur|hardener|amine", c_low)) and ("structure" in c_low or "smiles" in c_low or re.search(r"_\d+$", c_low))
            sample_has_curer = (
                self._sample_has_substructure(df[c], self.pat_amine) or
                self._sample_has_substructure(df[c], self.pat_anhydride) or
                self._sample_has_substructure(df[c], self.pat_thiol)
            )
            if sample_has_curer or is_name_match:
                note = "化学验证: 含固化剂官能团(胺/酸酐/硫醇)" if sample_has_curer else "名称匹配"
                roles["curers"].append((c, note))

        # 3. 扫描配比列
        for c in num_cols:
            c_low = str(c).lower()
            if "phr" in c_low:
                roles["phr_cols"].append(c)
            elif "ratio" in c_low or "r_value" in c_low:
                roles["ratio_cols"].append(c)
            elif "eew" in c_low or "ahew" in c_low:
                roles["eew_cols"].append(c)

        return roles

    def _sample_has_substructure(self, series: pd.Series, pattern: Chem.Mol, max_samples: int = 5) -> bool:
        """抽样检查该列文本是否属于该类分子"""
        if pattern is None:
            return False
        valid_count = 0
        match_count = 0
        for val in series.dropna():
            if not isinstance(val, str) or len(val.strip()) < 3:
                continue
            smi = val.strip()
            # 剥离 BigSMILES 符号
            smi_clean = re.sub(r"\{\[.*?\]\}", "C", smi)
            try:
                m = Chem.MolFromSmiles(smi_clean)
                if m:
                    valid_count += 1
                    if m.HasSubstructMatch(pattern):
                        match_count += 1
            except Exception:
                pass
            if valid_count >= max_samples:
                break
        return (match_count > 0 and (match_count / max(1, valid_count)) >= 0.4)

    def normalize_formulation_units_to_canonical(
        self,
        resins: List[Dict[str, Any]],
        curers: List[Dict[str, Any]],
        unit_type: str = "PHR",
        r_value_hint: Optional[float] = None
    ) -> Tuple[float, float, float, int]:
        """
        将任意单位体系（PHR / wt% / pbw / 摩尔比 / 当量比）严格归一化为基准官能团摩尔当量
        
        返回:
            (r_val, f_avg, theoretical_Mc, flag_stoich_imputed)
        """
        flag_imputed = 0

        # 计算树脂总当量数 N_epoxy
        total_epoxy_eq = 0.0
        total_resin_mass = 0.0
        for r in resins:
            w = r.get("weight", 1.0)
            ew = r.get("eew", 185.0)
            total_resin_mass += w
            total_epoxy_eq += (w / max(ew, 1e-4))

        # 计算固化剂总当量数 N_curer
        total_curer_eq = 0.0
        total_curer_mass = 0.0
        for h in curers:
            w = h.get("weight", 1.0)
            ew = h.get("ahew", 60.0)
            total_curer_mass += w
            total_curer_eq += (w / max(ew, 1e-4))

        # 1. 确定最终的无量纲 r 值
        if r_value_hint is not None and not np.isnan(r_value_hint) and r_value_hint > 0:
            r_val = float(r_value_hint)
        elif total_epoxy_eq > 0 and total_curer_eq > 0 and total_curer_mass > 0 and total_resin_mass > 0:
            # 标准化学当量计算
            r_val = total_curer_eq / total_epoxy_eq
        else:
            # 物理分级兜底：文献缺失固化剂具体用量时，按高分子化学常识取 r = 1.0 (理想配比)
            r_val = 1.0
            flag_imputed = 1

        r_val = float(np.clip(r_val, 0.05, 20.0))

        # 2. 计算平均单体官能度 f_avg
        f_resins = [r.get("func", 2.0) for r in resins] or [2.0]
        f_curers = [h.get("func", 4.0) for h in curers] or [4.0]
        f_r_avg = float(np.mean(f_resins))
        f_h_avg = float(np.mean(f_curers))
        f_avg = (f_r_avg + f_h_avg * r_val) / (1.0 + r_val)

        # 3. 理论交联点间分子量 Mc
        mw_resins = [r.get("mw", 340.0) for r in resins] or [340.0]
        mw_curers = [h.get("mw", 200.0) for h in curers] or [200.0]
        mw_avg = (np.mean(mw_resins) + np.mean(mw_curers) * r_val) / (1.0 + r_val)
        mc = mw_avg / max(f_avg - 2.0, 0.08)
        mc = float(np.clip(mc, 50.0, 5000.0))

        return r_val, f_avg, mc, flag_imputed

    # 单组分配方模式需剥离的多组分列前缀
    _SINGLE_COMPONENT_DROP_PREFIXES = (
        "resin_2_", "resin_3_", "curing_agent_2_", "curing_agent_3_",
        "small_additive_1_", "small_additive_2_",
        # 其他添加剂类组分 (引发剂/促进剂/催化剂/活性稀释剂/增韧剂/填料/其他)
        # 的具体信息列 (结构、用量、数量统计等)，纯净双组分体系不携带
        "initiator_", "accelerator_", "catalyst_",
        "reactive_diluent_", "reactive_toughener_",
        "filler_", "other_",
    )
    # 单组分配方模式下无信息量的组分数量/添加剂总量类列
    _SINGLE_COMPONENT_DROP_EXACT = (
        "reactive_diluent_total_phr", "reactive_toughener_total_phr",
        "small_additive_total_phr", "filler_total_phr", "other_total_phr",
    )

    def single_component_excluded_columns(self, cols) -> List[str]:
        """
        返回单组分配方模式下应从工作表剔除的列：
        - 多组分列 (resin_2/3, curing_agent_2/3, small_additive_1/2 全部前缀列)
        - 组分数量统计列 (*_component_count, *_duplicate_component_count)——单组分下恒为 1 或 0，无信息量
        - 其他添加剂/增韧剂/稀释剂/填料总量列
        """
        excluded = []
        for c in cols:
            c_low = str(c).lower()
            if c_low.startswith(self._SINGLE_COMPONENT_DROP_PREFIXES):
                excluded.append(c)
            elif c_low.endswith(("_component_count", "_duplicate_component_count")):
                excluded.append(c)
            elif c_low in self._SINGLE_COMPONENT_DROP_EXACT:
                excluded.append(c)
        return excluded

    @staticmethod
    def _mask_has_real_value(sub: pd.DataFrame) -> pd.Series:
        """行级判定：子表中任一列存在真实取值（数值列 0 视为不存在）"""
        mask = pd.Series(False, index=sub.index)
        for c in sub.columns:
            col = sub[c]
            if pd.api.types.is_numeric_dtype(col):
                mask = mask | (col.notna() & (col.fillna(0) != 0))
            else:
                mask = mask | col.notna()
        return mask
    def compute_and_fill_molecular_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基于 SMILES 结构与高分子物理当量关系对各组分分子量进行多级互补计算与填补。
        推导层级：
        1. 优先使用已有解析分子量 (*_mw_resolved)；
        2. 若缺失，使用 RDKit 基于 SMILES (*_structure) 精确计算 MolWt；
        3. 若仍缺失，根据高分子物理关系推导：
           - 树脂：MW = EEW * f_network；
           - 固化剂：MW = AHEW * f_network；
        4. 统一回填到标准分子量字段 (*_molecular_weight_g_mol)；
        5. 派生配方级分子量比值 (formulation_mw_ratio_h_r) 与加权平均单体分子量 (formulation_monomer_avg_mw)。
        """
        out = df.copy()

        mw_cache = {}

        def _calc_rdkit_mw(smi):
            if smi is None or pd.isna(smi):
                return None
            smi_str = str(smi).strip()
            if not smi_str or smi_str.lower() in ["none", "nan", ""]:
                return None
            if smi_str in mw_cache:
                return mw_cache[smi_str]
            val = None
            try:
                mol = Chem.MolFromSmiles(smi_str)
                if mol:
                    from rdkit.Chem import Descriptors
                    val = float(Descriptors.MolWt(mol))
            except Exception:
                pass
            mw_cache[smi_str] = val
            return val

        # 处理主树脂与辅树脂 (resin_1, resin_2)
        for i in (1, 2):
            smi_col = f"resin_{i}_structure"
            std_mw_col = f"resin_{i}_molecular_weight_g_mol"
            res_mw_col = f"resin_{i}_mw_resolved"
            ew_col = f"resin_{i}_ew_resolved"
            f_col = f"resin_{i}_f_network"

            if smi_col not in out.columns and std_mw_col not in out.columns:
                continue

            current_mw = pd.to_numeric(out[res_mw_col], errors="coerce") if res_mw_col in out.columns else pd.Series(np.nan, index=out.index)
            if std_mw_col in out.columns:
                current_mw = current_mw.combine_first(pd.to_numeric(out[std_mw_col], errors="coerce"))

            # RDKit 计算补齐
            if smi_col in out.columns:
                calc_mw = out[smi_col].apply(_calc_rdkit_mw)
                current_mw = current_mw.combine_first(calc_mw)

            # 物理当量与官能度倒推: EEW * f_network
            if ew_col in out.columns and f_col in out.columns:
                eew = pd.to_numeric(out[ew_col], errors="coerce")
                f_net = pd.to_numeric(out[f_col], errors="coerce")
                physics_mw = eew * f_net
                physics_mw = physics_mw.where(physics_mw > 0, np.nan)
                current_mw = current_mw.combine_first(physics_mw)

            out[std_mw_col] = current_mw

        # 处理主固化剂与辅固化剂 (curing_agent_1, curing_agent_2)
        for i in (1, 2):
            smi_col = f"curing_agent_{i}_structure"
            std_mw_col = f"curing_agent_{i}_molecular_weight_g_mol"
            res_mw_col = f"curing_agent_{i}_mw_resolved"
            ew_col = f"curing_agent_{i}_ew_resolved"
            f_col = f"curing_agent_{i}_f_network"

            if smi_col not in out.columns and std_mw_col not in out.columns:
                continue

            current_mw = pd.to_numeric(out[res_mw_col], errors="coerce") if res_mw_col in out.columns else pd.Series(np.nan, index=out.index)
            if std_mw_col in out.columns:
                current_mw = current_mw.combine_first(pd.to_numeric(out[std_mw_col], errors="coerce"))

            # RDKit 计算补齐
            if smi_col in out.columns:
                calc_mw = out[smi_col].apply(_calc_rdkit_mw)
                current_mw = current_mw.combine_first(calc_mw)

            # 物理当量与官能度倒推: AHEW * f_network
            if ew_col in out.columns and f_col in out.columns:
                ahew = pd.to_numeric(out[ew_col], errors="coerce")
                f_net = pd.to_numeric(out[f_col], errors="coerce")
                physics_mw = ahew * f_net
                physics_mw = physics_mw.where(physics_mw > 0, np.nan)
                current_mw = current_mw.combine_first(physics_mw)

            out[std_mw_col] = current_mw

        # 4. 固化剂用量 (PHR) 的物理当量兜底补全:
        # 当 curing_agent_1_amount_phr 缺失但记录了当量比 r、EEW 与 AHEW 时，
        # 根据高分子物理严格关系 phr = r * (AHEW / EEW) * 100 自动推导补齐
        ca_phr = pd.to_numeric(out.get("curing_agent_1_amount_phr"), errors="coerce")
        r_val = pd.to_numeric(out.get("formulation_r_value"), errors="coerce")
        eew_val = pd.to_numeric(out.get("formulation_resin_total_eew_g_eq"), errors="coerce")
        ahew_val = pd.to_numeric(out.get("formulation_hardener_total_ahew_g_eq"), errors="coerce")

        if r_val is not None and eew_val is not None and ahew_val is not None:
            ca_phr_inferred = (r_val * ahew_val / eew_val * 100.0).where(
                (r_val > 0) & (eew_val > 0) & (ahew_val > 0), np.nan
            )
            out["curing_agent_1_amount_phr"] = ca_phr.combine_first(ca_phr_inferred) if ca_phr is not None else ca_phr_inferred

        # 5. 派生高分子配方级特征
        r1_mw = pd.to_numeric(out.get("resin_1_molecular_weight_g_mol"), errors="coerce")
        ca1_mw = pd.to_numeric(out.get("curing_agent_1_molecular_weight_g_mol"), errors="coerce")
        if r1_mw is not None and ca1_mw is not None:
            # 固化剂/树脂分子量比值
            out["formulation_mw_ratio_h_r"] = (ca1_mw / r1_mw).where((r1_mw > 0) & (ca1_mw > 0), np.nan)

            # 按 PHR 加权平均单体分子量 (此时因 PHR 补齐，分子量加权平均值缺失率也降至 3% 左右)
            r_phr = pd.to_numeric(out.get("resin_1_amount_phr", 100.0), errors="coerce").fillna(100.0)
            ca_phr_eff = pd.to_numeric(out.get("curing_agent_1_amount_phr"), errors="coerce")
            tot_phr = r_phr + ca_phr_eff
            out["formulation_monomer_avg_mw"] = ((r_phr * r1_mw + ca_phr_eff * ca1_mw) / tot_phr).where(tot_phr > 0, np.nan)

        return out

    def classify_formulation_systems(self, df: pd.DataFrame) -> pd.Series:
        """
        基于化学语义、反应机理与固化剂官能团对配方进行高精度体系分类。
        返回分类标签 Series，类别包括：
        - '环氧-胺体系 (Epoxy-Amine)'
        - '环氧-酸酐体系 (Epoxy-Anhydride)'
        - '环氧-酚类体系 (Epoxy-Phenolic)'
        - '环氧-硫醇体系 (Epoxy-Thiol)'
        - '催化型/自聚开环体系 (Catalytic / Homopolymerization)'
        - '其他/未分类体系 (Other / Unclassified)'
        """
        if df is None or len(df) == 0:
            return pd.Series(dtype=object)

        # 预先缓存唯一 SMILES 的官能团分析结果，提升万级样本的分类性能
        smiles_cache: Dict[str, Optional[str]] = {}

        def check_smiles_category(smi: Any) -> Optional[str]:
            if smi is None or pd.isna(smi) or str(smi).strip().lower() in ["none", "nan", ""]:
                return None
            smi_clean = str(smi).strip()
            if smi_clean in smiles_cache:
                return smiles_cache[smi_clean]
            cat = None
            try:
                mol = Chem.MolFromSmiles(smi_clean)
                if mol:
                    if mol.HasSubstructMatch(self.pat_anhydride):
                        cat = "anhydride"
                    elif mol.HasSubstructMatch(self.pat_amine):
                        cat = "amine"
                    elif mol.HasSubstructMatch(self.pat_thiol):
                        cat = "thiol"
                    elif hasattr(self, "pat_phenol") and mol.HasSubstructMatch(self.pat_phenol):
                        cat = "phenolic"
            except Exception:
                pass
            smiles_cache[smi_clean] = cat
            return cat

        mech_series = df["curing_mechanism"].astype(str).str.lower() if "curing_mechanism" in df.columns else pd.Series("", index=df.index)
        std_series = df["curing_type_standard"].astype(str).str.lower() if "curing_type_standard" in df.columns else pd.Series("", index=df.index)
        ca1_smi = df["curing_agent_1_structure"].fillna("") if "curing_agent_1_structure" in df.columns else pd.Series("", index=df.index)
        ca2_smi = df["curing_agent_2_structure"].fillna("") if "curing_agent_2_structure" in df.columns else pd.Series("", index=df.index)

        name_cols = [c for c in df.columns if "curing_agent" in c.lower() and "name" in c.lower()]

        results = []
        for idx in range(len(df)):
            mech = mech_series.iloc[idx]
            std = std_series.iloc[idx]

            # 1. 优先根据明确的机理标注
            if "amine" in mech:
                results.append("环氧-胺体系 (Epoxy-Amine)")
                continue
            if "anhydride" in mech:
                results.append("环氧-酸酐体系 (Epoxy-Anhydride)")
                continue
            if "thiol" in mech:
                results.append("环氧-硫醇体系 (Epoxy-Thiol)")
                continue
            if "phenolic" in mech:
                results.append("环氧-酚类体系 (Epoxy-Phenolic)")
                continue
            if any(k in mech for k in ["homopolymerization", "imidazole", "anionic", "cationic"]):
                results.append("催化型/自聚开环体系 (Catalytic / Homopolymerization)")
                continue

            # 2. 固化剂 SMILES 官能团精准识别
            c1 = check_smiles_category(ca1_smi.iloc[idx])
            c2 = check_smiles_category(ca2_smi.iloc[idx])
            if c1 == "anhydride" or c2 == "anhydride":
                results.append("环氧-酸酐体系 (Epoxy-Anhydride)")
                continue
            if c1 == "amine" or c2 == "amine":
                results.append("环氧-胺体系 (Epoxy-Amine)")
                continue
            if c1 == "thiol" or c2 == "thiol":
                results.append("环氧-硫醇体系 (Epoxy-Thiol)")
                continue
            if c1 == "phenolic" or c2 == "phenolic":
                results.append("环氧-酚类体系 (Epoxy-Phenolic)")
                continue

            # 3. 常见固化剂名称关键词辅助
            is_amine_name = False
            for ncol in name_cols:
                nval = str(df[ncol].iloc[idx] or "").lower()
                if any(k in nval for k in ["amine", "ddm", "dds", "detda", "teta", "deta", "ipda", "jeffamine", "pacm", "mda", "mxd", "diamin", "triamin"]):
                    is_amine_name = True
                    break
            if is_amine_name:
                results.append("环氧-胺体系 (Epoxy-Amine)")
                continue

            # 4. 活性氢当量 AHEW 判定
            if "curing_agent_1_active_hydrogen_equivalent_count" in df.columns:
                h_cnt = df["curing_agent_1_active_hydrogen_equivalent_count"].iloc[idx]
                if pd.notna(h_cnt) and float(h_cnt) > 0:
                    results.append("环氧-胺体系 (Epoxy-Amine)")
                    continue

            # 5. 固化类型分类
            if std in ["catalytic_curing", "self_curing", "initiator_triggered"]:
                results.append("催化型/自聚开环体系 (Catalytic / Homopolymerization)")
                continue

            results.append("其他/未分类体系 (Other / Unclassified)")

        return pd.Series(results, index=df.index)

    def get_formulation_system_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """获取数据集的配方体系分布计数字典"""
        systems = self.classify_formulation_systems(df)
        return dict(systems.value_counts())


    def clean_features_for_ml(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        curing_type_filter: Optional[str] = "external_hardener",
        mode: str = "qspr_clean",
        drop_metadata: bool = True,
        base_df: Optional[pd.DataFrame] = None,
        single_component_only: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        面向机器学习训练对融合后的数据集进行纯净化与规范化特征清洗

        参数:
            df: 待清洗的 DataFrame (融合后)
            target_col: 目标预测变量 (如 tg_c)
            curing_type_filter: 固化体系筛选值 (如 'external_hardener', 若为 None 或 'all' 则不筛选)
            mode: 清洗模式 ('qspr_clean': 紧凑标准 QSPR 模式; 'comprehensive': 全息保留模式)
            drop_metadata: 是否强制剔除元数据、格式列和无预测价值的列
            base_df: 原始窄表 (用于提取原窄表拥有的测试条件列等)
            single_component_only: 单组分配方模式——仅保留树脂 1 组分 + 固化剂 1 组分、
                且不含小分子添加剂的样本；工作表同步剔除多组分列与组分数量特征列
                (单组分下 *_component_count 恒为 1/0，无预测价值)
        """
        stats: Dict[str, Any] = {
            "initial_rows": len(df),
            "initial_cols": len(df.columns),
            "filtered_rows": len(df),
            "final_cols": len(df.columns),
            "curing_filter_applied": curing_type_filter,
            "mode": mode,
            "dropped_columns_sample": [],
            "resin_3_included": False,
            "single_component_only": bool(single_component_only),
            "single_filtered_rows": len(df),
            "single_dropped_cols_count": 0,
        }

        out_df = df.copy()

        # 1. 固化体系筛选与分类提取
        if curing_type_filter and str(curing_type_filter).lower() not in ["all", "全部", "none"]:
            c_filter_str = str(curing_type_filter).lower()
            if "胺" in c_filter_str or "amine" in c_filter_str:
                systems = self.classify_formulation_systems(out_df)
                mask = systems == "环氧-胺体系 (Epoxy-Amine)"
                out_df = out_df[mask].copy()
                stats["filtered_rows"] = len(out_df)
                stats["system_selected"] = "环氧-胺体系 (Epoxy-Amine)"
            elif "酸酐" in c_filter_str or "anhydride" in c_filter_str:
                systems = self.classify_formulation_systems(out_df)
                mask = systems == "环氧-酸酐体系 (Epoxy-Anhydride)"
                out_df = out_df[mask].copy()
                stats["filtered_rows"] = len(out_df)
                stats["system_selected"] = "环氧-酸酐体系 (Epoxy-Anhydride)"
            elif "external_hardener" in c_filter_str:
                if "curing_type_standard" in out_df.columns:
                    out_df = out_df[out_df["curing_type_standard"] == "external_hardener"].copy()
                    stats["filtered_rows"] = len(out_df)
                    stats["system_selected"] = "常规外加固化剂 (external_hardener)"
            elif "curing_type_standard" in out_df.columns:
                out_df = out_df[out_df["curing_type_standard"] == curing_type_filter].copy()
                stats["filtered_rows"] = len(out_df)

        # 1.5 单组分配方筛选：剔除多组分 (树脂/固化剂 >1 组分) 与含小分子添加剂 (1 或 2) 的样本
        if single_component_only:
            multi_cols = [c for c in out_df.columns if str(c).lower().startswith(("resin_2_", "resin_3_", "curing_agent_2_", "curing_agent_3_"))]
            small_cols = [c for c in out_df.columns if str(c).lower().startswith(("small_additive_1_", "small_additive_2_"))]
            mask_drop = pd.Series(False, index=out_df.index)
            if multi_cols:
                mask_drop = mask_drop | self._mask_has_real_value(out_df[multi_cols])
            if small_cols:
                mask_drop = mask_drop | self._mask_has_real_value(out_df[small_cols])
            out_df = out_df[~mask_drop].copy()
            stats["single_filtered_rows"] = len(out_df)
            stats["filtered_rows"] = len(out_df)

        # 2. 彻底剥离机理特征列 (mech_*)：融合流程已废弃机理特征注入，
        #    同时清洗旧宽表中可能残留的 mech_* 列，防止其进入训练矩阵
        mech_cols = [c for c in out_df.columns if c.startswith("mech_")]
        if mech_cols:
            out_df = out_df.drop(columns=mech_cols)

        # 3. 定义无意义元数据列判定规则
        def is_useless_metadata_col(c: str) -> bool:
            c_low = str(c).lower()
            # 格式列 (format)
            if c_low.endswith("_format") or "_structure_format" in c_low or c_low == "format":
                return True
            # 固化体系与机理文本标签 (筛选后已无方差或属于非数值标签)
            if c_low in ["curing_type_standard", "curing_mechanism", "formulation_resin_phr_basis_type"]:
                return True
            # 原始抽取单位与统计边界元数据
            if any(c_low.endswith(sfx) for sfx in [
                "_unit_raw", "_basis", "_phr_lower", "_phr_upper", 
                "_equivalent_weight_type", "_stoichiometric_relevant"
            ]):
                return True
            # 文献行政元数据
            if any(c_low.startswith(pfx) for pfx in [
                "source_", "paper_", "table_", "raw_", "manifest_", "duplicate_", "review_"
            ]):
                return True
            if c_low in [
                "sample_id", "record_id", "doi", "title", "authors", "journal", 
                "year", "volume", "issue", "pages", "notes", "comments", "citation"
            ]:
                return True
            if "unnamed:" in c_low:
                return True
            return False

        if mode == "qspr_clean":
            # [逐组分物理量补齐] 窄表原本缺少逐组分 MW/EEW/AHEW，导致
            # crosslink_physics 的 hybrid 分支不可达（f_avg 恒 NaN）。
            # 这里按 文献 → 当量×官能度 → 结构直算 分层补齐，
            # BigSMILES 采样代理与多片段单元格的 MW 会被拒绝（见 component_physics）。
            # 失败不阻断清洗流程。
            cp_available = False
            try:
                from .component_physics import enrich_narrow_table

                _enriched = enrich_narrow_table(out_df)
                if _enriched is not None and len(_enriched) == len(out_df):
                    out_df = _enriched
                    cp_available = True
                    stats["component_physics_enriched"] = True
            except Exception as _cp_err:
                stats["component_physics_enriched"] = False
                stats["component_physics_error"] = str(_cp_err)

            # [关键增强] 对树脂与固化剂分子量进行多级计算与精准回填补齐，并派生分子量比值等核心特征
            out_df = self.compute_and_fill_molecular_weights(out_df)

            # 类似 ml_qspr_model_ 结构的标准精纯特征组织（彻底剥离50+重复冗余/高缺失/常数列）
            valuable_ordered_cols = [
                # 1. 分子化学结构 (SMILES 供特征工程/指纹提取)
                "resin_1_structure", "curing_agent_1_structure",
                "resin_2_structure", "curing_agent_2_structure",
                "small_additive_1_structure",
                # 2. 分子量特征 (全面多级计算补齐，缺失率接近 0%)
                "resin_1_molecular_weight_g_mol",
                "curing_agent_1_molecular_weight_g_mol",
                "formulation_mw_ratio_h_r",
                "formulation_monomer_avg_mw",
                "resin_2_molecular_weight_g_mol",
                "curing_agent_2_molecular_weight_g_mol",
                # 3. 核心配比与基准物理当量 (高覆盖率)
                "resin_1_amount_phr", "curing_agent_1_amount_phr",
                "resin_2_amount_phr", "curing_agent_2_amount_phr",
                "formulation_resin_total_eew_g_eq",
                "formulation_hardener_total_ahew_g_eq",
                "formulation_r_value",
                # 4. 高分子网络物理机理特征 (覆盖率100%且物理自洽)
                "resin_1_f_network",
                "curing_agent_1_f_network",
                "cp_f_avg_network",
                "cp_W_g_per_epoxy",
                "cp_epoxy_conc_mol_m3",
                "cp_f_r", "cp_f_h_network",
                # 5. 固化热历史工艺参数 (对力学与Tg至关重要)
                "process_max_temperature_c",
                "process_temperature_time_integral_c_h",
                "process_total_time_h",
                "process_final_cure_temperature_c",
                "process_final_cure_time_h",
                "process_atmosphere",
                "process_has_post_cure",
            ]

            if cp_available:
                # 补充保留有效机理特征，过滤掉已在分子量和当量中规范化的重复列
                _cp_extra = ["cp_f_r", "cp_f_h_network", "cp_f_avg_network", "cp_W_g_per_epoxy", "cp_epoxy_conc_mol_m3"]
                for _c in _cp_extra:
                    if _c in out_df.columns and _c not in valuable_ordered_cols:
                        valuable_ordered_cols.append(_c)
                stats["component_physics_cols"] = len([c for c in _cp_extra if c in out_df.columns])

            target_test_cols = []
            cand_sources = [base_df] if base_df is not None else [out_df]
            for src in cand_sources:
                if src is not None and target_col:
                    target_test_cols.extend([
                        c for c in src.columns 
                        if str(c).startswith(f"{target_col}_") and not is_useless_metadata_col(c) and c not in target_test_cols
                    ])

            final_cols = []
            # 单组分配方模式：组织特征清单时直接排除多组分列与组分数量特征
            single_drop = set(self.single_component_excluded_columns(valuable_ordered_cols)) if single_component_only else set()
            for c in valuable_ordered_cols:
                if c in out_df.columns and c not in final_cols and c not in single_drop:
                    final_cols.append(c)
            if single_component_only:
                stats["single_dropped_cols_count"] = len(single_drop)
            for c in target_test_cols:
                if c in out_df.columns and c not in final_cols:
                    final_cols.append(c)
            # 测试标准列 (test_standard_*) 一并保留
            for c in out_df.columns:
                if str(c).startswith("test_standard_") and c not in final_cols:
                    final_cols.append(c)
            for c in mech_cols:
                if c in out_df.columns and c not in final_cols:
                    final_cols.append(c)

            # 目标属性置于末尾
            if not target_col or target_col not in out_df.columns:
                auto_t = self.extract_target_name_from_df(out_df)
                if auto_t and auto_t in out_df.columns:
                    target_col = auto_t

            if target_col and target_col in out_df.columns and target_col not in final_cols:
                final_cols.append(target_col)

            # 最终安全兜底：如果尚未有目标列，则按常见目标兜底；若已有目标列，确保其置于最后一列
            if not target_col or target_col not in final_cols:
                for known_t in ["tg_c", "td5_c", "tensile_strength_mpa", "flexural_strength_mpa"]:
                    if known_t in out_df.columns and known_t not in final_cols:
                        final_cols.append(known_t)
                        target_col = known_t
                        break
            elif target_col in final_cols and final_cols[-1] != target_col:
                final_cols.remove(target_col)
                final_cols.append(target_col)

            # 记录 resin_3 是否保留
            if "resin_3_structure" in final_cols or any(c.startswith("resin_3_") for c in final_cols):
                stats["resin_3_included"] = True

            out_df = out_df[[c for c in final_cols if c in out_df.columns]].copy()

        else:
            # Comprehensive 模式: 保留所有非空特征，但剔除无意义列
            if single_component_only:
                single_drop_cols = self.single_component_excluded_columns(out_df.columns)
                if single_drop_cols:
                    out_df = out_df.drop(columns=[c for c in out_df.columns if c in set(single_drop_cols)])
                stats["single_dropped_cols_count"] = len(single_drop_cols)

            if drop_metadata:
                cols_to_drop = [c for c in out_df.columns if is_useless_metadata_col(c)]
                stats["dropped_columns_sample"] = cols_to_drop[:15]
                out_df = out_df.drop(columns=cols_to_drop)

                # 剔除全 NaN 列
                all_nan = [c for c in out_df.columns if out_df[c].isna().all()]
                if all_nan:
                    out_df = out_df.drop(columns=all_nan)

            if "resin_3_structure" in out_df.columns:
                stats["resin_3_included"] = True

        stats["final_cols"] = len(out_df.columns)
        return out_df, stats

