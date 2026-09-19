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


class FormulationFusionEngine:
    """配方数据融合与化学语义对齐引擎"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        # SMARTS 模式用于化学内容语义校验
        self.pat_epoxy = Chem.MolFromSmarts("[C,c]1O[C,c]1")
        self.pat_amine = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")
        self.pat_anhydride = Chem.MolFromSmarts("C(=O)OC(=O)")
        self.pat_thiol = Chem.MolFromSmarts("[SX2H]")

    def find_companion_wide_table(self, main_path: str, custom_dir: Optional[str] = None) -> Optional[str]:
        """
        在主数据表同级目录或指定数据集目录下自动寻找 ml_wide_samples.csv
        """
        candidate_dirs = []
        if custom_dir and os.path.exists(custom_dir):
            candidate_dirs.append(custom_dir)
        if main_path and os.path.exists(main_path):
            candidate_dirs.append(os.path.dirname(os.path.abspath(main_path)))
        # 默认常见路径
        candidate_dirs.extend([
            r"C:\Users\wangj\Desktop\ml_dataset",
            os.path.join(os.getcwd(), "ml_dataset"),
            os.getcwd(),
        ])

        for c_dir in candidate_dirs:
            if not c_dir or not os.path.exists(c_dir):
                continue
            wide_file = os.path.join(c_dir, "ml_wide_samples.csv")
            if os.path.exists(wide_file):
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
        if custom_dir and os.path.exists(custom_dir):
            candidate_dirs.append(custom_dir)
        candidate_dirs.extend([
            r"C:\Users\wangj\Desktop\ml_dataset",
            os.path.join(os.getcwd(), "ml_dataset"),
            os.getcwd(),
        ])
        for d in candidate_dirs:
            if not d or not os.path.exists(d):
                continue
            p = os.path.join(d, filename)
            if os.path.exists(p):
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

        # 1. 固化体系筛选 (如用户常用的 external_hardener)
        if curing_type_filter and str(curing_type_filter).lower() not in ["all", "全部", "none"]:
            if "curing_type_standard" in out_df.columns:
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
            # 类似 ml_qspr_model_ 结构的标准特征组织
            valuable_ordered_cols = [
                # 结构列 (包含三组分)
                "resin_1_structure", "resin_2_structure", "resin_3_structure",
                "curing_agent_1_structure", "curing_agent_2_structure", "curing_agent_3_structure",
                "small_additive_1_structure", "small_additive_2_structure",
                # 单体精确物理量 (包含 resin_3 与各组分真实 PHR / MW / 官能度)
                "resin_1_amount_phr", "resin_2_amount_phr", "resin_3_amount_phr",
                "resin_1_molecular_weight_g_mol", "resin_2_molecular_weight_g_mol", "resin_3_molecular_weight_g_mol",
                "resin_1_epoxy_group_count", "resin_2_epoxy_group_count", "resin_3_epoxy_group_count",
                "resin_3_equivalent_group_count", "resin_3_equivalent_weight_g_eq",
                "curing_agent_1_amount_phr", "curing_agent_2_amount_phr", "curing_agent_3_amount_phr",
                "curing_agent_1_molecular_weight_g_mol", "curing_agent_2_molecular_weight_g_mol", "curing_agent_3_molecular_weight_g_mol",
                "curing_agent_1_active_hydrogen_equivalent_count", "curing_agent_2_active_hydrogen_equivalent_count",
                # 配方宏观基准当量
                "initiator_present", "formulation_resin_total_eew_g_eq", "formulation_hardener_total_ahew_g_eq",
                "formulation_resin_hardener_equivalent_ratio", "formulation_r_value", "formulation_epoxy_binder_total_phr",
                # 固化工艺参数
                "process_max_temperature_c", "process_temperature_time_integral_c_h", "process_total_time_h",
                "process_final_cure_temperature_c", "process_final_cure_time_h", "process_atmosphere", "process_has_post_cure",
                # 体系组分宏观统计量
                "resin_component_count", "resin_total_phr", "resin_epoxy_group_total", "resin_equivalent_group_total",
                "curing_agent_component_count", "curing_agent_total_phr", "curing_agent_active_hydrogen_total", "curing_agent_equivalent_group_total",
                "small_additive_component_count", "initiator_component_count", "accelerator_component_count", "catalyst_component_count",
                "reactive_diluent_component_count", "reactive_toughener_component_count", "reactive_toughener_total_phr",
                "filler_component_count", "other_component_count",
            ]

            # 提取与当前目标相关的测试条件列 (如 tg_c_test_method 等)
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

