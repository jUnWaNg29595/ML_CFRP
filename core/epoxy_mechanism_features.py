# -*- coding: utf-8 -*-
"""
core/epoxy_mechanism_features.py

环氧-固化剂高分子交联机理与动力学特征计算引擎：
1. 聚合物三维交联网络参数：
   - 体系化学计量比 r (AHEW/EEW)
   - 对称偏离度 |r - 1.0| (交联网络缺陷度)
   - 对数计量比 ln(r)
   - 理论最大固化度 alpha_max
   - 理论凝胶点转化率 alpha_gel (Flory-Stockmayer 凝胶理论)
   - 平均官能度 f_avg
   - 理论交联点间分子量 Mc = M_bar / (f_avg - 2)
   - 理论交联密度代理 rho_c = 1000 / Mc
2. 多组分前线轨道能差矩阵与动力学指标 (Delta E):
   - delta_E_min: 先发活性通道 (决定凝胶与起始温度)
   - delta_E_max: 最钝迟钝通道 (决定未反应缺陷与深层固化风险)
   - delta_E_span: 动力学色散度 (反映分步固化与相分离倾向)
   - delta_E_weighted: 体系加权宏观活化能垒
3. 多组分配方物理混合加权特征 (分子量、极性TPSA、芳环密度等)
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class EpoxyMechanismEngine:
    """环氧交联网络机理与动力学特征计算引擎"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if RDKIT_AVAILABLE:
            self._epoxy_pattern = Chem.MolFromSmarts('[C]1[O][C]1')
            self._pri_amine_pattern = Chem.MolFromSmarts('[NX3;H2]')
            self._sec_amine_pattern = Chem.MolFromSmarts('[NX3;H1]')
            self._anhydride_pattern = Chem.MolFromSmarts('C(=O)OC(=O)')
            self._thiol_pattern = Chem.MolFromSmarts('[SX2;H1]')
            self._aromatic_ring_pattern = Chem.MolFromSmarts('a1aaaaa1')
        else:
            self._epoxy_pattern = None

    @staticmethod
    def clean_structure_string(text: Any) -> str:
        """清洗结构字符串，去除非法空白与 NaN"""
        if text is None or pd.isna(text):
            return ""
        s = str(text).strip()
        if s.lower() in ["nan", "none", "null", ""]:
            return ""
        return s

    def parse_molecule_safe(self, smi_or_bigsmi: str) -> Optional[Any]:
        """安全解析分子，支持普通 SMILES 和 BigSMILES 低聚物/骨架提取"""
        if not RDKIT_AVAILABLE or not smi_or_bigsmi:
            return None

        clean_str = self.clean_structure_string(smi_or_bigsmi)
        if not clean_str:
            return None

        # 1. 尝试直接解析
        try:
            mol = Chem.MolFromSmiles(clean_str)
            if mol is not None:
                return mol
        except Exception:
            pass

        # 2. 如果包含 BigSMILES 语法，剥离重复单元抽取端基骨架
        if "{" in clean_str and "}" in clean_str:
            try:
                # 剥离 {[...]}
                stripped = re.sub(r'\{.*?\}', '', clean_str)
                stripped = stripped.replace("..", ".").strip(".")
                if stripped:
                    mol = Chem.MolFromSmiles(stripped)
                    if mol is not None:
                        return mol
            except Exception:
                pass

            # 3. 尝试提取重复单元作为骨架代表
            try:
                match = re.search(r'\{.*?[<>\d]*\](.*?)(?:\[[<>\d]*\]).*?\}', clean_str)
                if match:
                    rep_smi = match.group(1).strip()
                    mol = Chem.MolFromSmiles(rep_smi)
                    if mol is not None:
                        return mol
            except Exception:
                pass

        return None

    def get_epoxide_count(self, smi: str, mol: Optional[Any] = None) -> int:
        """获取环氧基数量"""
        if mol is None:
            mol = self.parse_molecule_safe(smi)
        if mol is not None and self._epoxy_pattern is not None:
            try:
                matches = mol.GetSubstructMatches(self._epoxy_pattern)
                return len(matches)
            except Exception:
                pass

        # 字符串正则降级兜底
        s = self.clean_structure_string(smi)
        if not s:
            return 0
        epoxy_matches = len(re.findall(r'O1CC1|C1OC1|1OC1|C1CO1', s))
        return max(epoxy_matches, 2 if "dgeba" in s.lower() or "epoxy" in s.lower() else 0)

    def get_active_hydrogen_count(self, smi: str, mol: Optional[Any] = None) -> Tuple[int, str]:
        """获取固化剂活性氢数量及类型"""
        if mol is None:
            mol = self.parse_molecule_safe(smi)

        curer_type = "other"
        if mol is not None and RDKIT_AVAILABLE:
            try:
                n_pri = len(mol.GetSubstructMatches(self._pri_amine_pattern))
                n_sec = len(mol.GetSubstructMatches(self._sec_amine_pattern))
                n_anh = len(mol.GetSubstructMatches(self._anhydride_pattern))
                n_sh = len(mol.GetSubstructMatches(self._thiol_pattern))

                if n_pri > 0 or n_sec > 0:
                    curer_type = "amine"
                    return (n_pri * 2 + n_sec), curer_type
                elif n_anh > 0:
                    curer_type = "anhydride"
                    return (n_anh * 2), curer_type
                elif n_sh > 0:
                    curer_type = "thiol"
                    return n_sh, curer_type
            except Exception:
                pass

        # 字符串正则降级
        s = self.clean_structure_string(smi)
        if not s:
            return 0, "unknown"

        # 胺类正则估计 (N, NH2)
        n_count = len(re.findall(r'N', s))
        if n_count > 0:
            # 芳香二胺通常有 2 个 -NH2，共 4 个活性氢
            if n_count >= 2:
                return 4, "amine"
            return 2, "amine"
        return 1, "other"

    def calc_single_molecule_properties(
        self,
        smi: str,
        is_resin: bool = True,
        fallback_ew: Optional[float] = None,
        fallback_mw: Optional[float] = None
    ) -> Dict[str, Any]:
        """计算单分子的物化参数与前线轨道代理值"""
        mol = self.parse_molecule_safe(smi)

        props = {
            "smi": smi,
            "mw": np.nan,
            "tpsa": np.nan,
            "aromatic_rings": 0,
            "rotatable_bonds": 0,
            "h_donors": 0,
            "h_acceptors": 0,
            "functionality": 0,
            "ew": np.nan,  # EEW or AHEW
            "curer_type": "resin" if is_resin else "unknown",
            "homo_proxy": np.nan,
            "lumo_proxy": np.nan,
        }

        if mol is not None and RDKIT_AVAILABLE:
            try:
                props["mw"] = float(Descriptors.MolWt(mol))
                props["tpsa"] = float(Descriptors.TPSA(mol))
                props["aromatic_rings"] = int(Descriptors.NumAromaticRings(mol))
                props["rotatable_bonds"] = int(Descriptors.NumRotatableBonds(mol))
                props["h_donors"] = int(Descriptors.NumHDonors(mol))
                props["h_acceptors"] = int(Descriptors.NumHAcceptors(mol))

                # Gasteiger Partial Charges 估算轨道代理值
                try:
                    AllChem.ComputeGasteigerCharges(mol)
                    charges = [
                        float(atom.GetProp('_GasteigerCharge'))
                        for atom in mol.GetAtoms()
                        if atom.HasProp('_GasteigerCharge')
                        and not atom.GetProp('_GasteigerCharge').lower().startswith(('nan', 'inf'))
                    ]
                    if charges:
                        max_pos = max(charges)
                        max_neg = min(charges)
                        # 亲电性 (LUMO代理：正电荷越集中，LUMO能级越低)
                        props["lumo_proxy"] = -1.0 * max_pos * 5.0 - 0.5
                        # 亲核性 (HOMO代理：负电荷越集中，HOMO能级越高)
                        props["homo_proxy"] = max_neg * 4.0 - 6.0
                except Exception:
                    pass
            except Exception:
                pass

        # 官能度与当量计算
        if is_resin:
            func = self.get_epoxide_count(smi, mol)
            props["functionality"] = func if func > 0 else 2
            if not np.isnan(props["mw"]) and props["functionality"] > 0:
                props["ew"] = props["mw"] / props["functionality"]
        else:
            func, c_type = self.get_active_hydrogen_count(smi, mol)
            props["functionality"] = func if func > 0 else 4
            props["curer_type"] = c_type
            if not np.isnan(props["mw"]) and props["functionality"] > 0:
                props["ew"] = props["mw"] / props["functionality"]

        # 外部 fallback 补充
        if fallback_mw is not None and not pd.isna(fallback_mw) and float(fallback_mw) > 0:
            if np.isnan(props["mw"]):
                props["mw"] = float(fallback_mw)
        if fallback_ew is not None and not pd.isna(fallback_ew) and float(fallback_ew) > 0:
            props["ew"] = float(fallback_ew)

        # 默认兜底
        if np.isnan(props["ew"]):
            props["ew"] = 180.0 if is_resin else 50.0
        if np.isnan(props["mw"]):
            props["mw"] = props["ew"] * max(props["functionality"], 1)

        return props

    def compute_formulation_mechanism_features(
        self,
        resin_list: List[Dict[str, Any]],
        curer_list: List[Dict[str, Any]],
        given_r_value: Optional[float] = None,
        given_resin_total_phr: Optional[float] = None,
        given_curer_total_phr: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        计算多组分配方的综合机理特征
        resin_list: [{'smi': str, 'weight': float, 'lumo': Optional[float], ...}]
        curer_list: [{'smi': str, 'weight': float, 'homo': Optional[float], ...}]
        """
        # 1. 过滤空组分并归一化组内权重
        valid_resins = [r for r in resin_list if r.get("smi")]
        valid_curers = [c for c in curer_list if c.get("smi")]

        if not valid_resins or not valid_curers:
            return {}

        w_r_sum = sum(max(r.get("weight", 1.0), 0.0) for r in valid_resins)
        if w_r_sum <= 0:
            w_r_sum = float(len(valid_resins))
        for r in valid_resins:
            r["norm_weight"] = max(r.get("weight", 1.0), 0.0) / w_r_sum

        w_h_sum = sum(max(c.get("weight", 1.0), 0.0) for c in valid_curers)
        if w_h_sum <= 0:
            w_h_sum = float(len(valid_curers))
        for c in valid_curers:
            c["norm_weight"] = max(c.get("weight", 1.0), 0.0) / w_h_sum

        # 2. 计算加权官能度与当量
        weighted_f_r = sum(r["functionality"] * r["norm_weight"] for r in valid_resins)
        weighted_f_h = sum(c["functionality"] * c["norm_weight"] for c in valid_curers)

        # 加权 EEW 与 AHEW
        inv_eew = sum((r["norm_weight"] / r["ew"]) for r in valid_resins if r["ew"] > 0)
        weighted_eew = (1.0 / inv_eew) if inv_eew > 0 else 180.0

        inv_ahew = sum((c["norm_weight"] / c["ew"]) for c in valid_curers if c["ew"] > 0)
        weighted_ahew = (1.0 / inv_ahew) if inv_ahew > 0 else 50.0

        # 3. 决定化学计量比 r (AHEW/EEW 当量比)
        r_val = 1.0
        if given_r_value is not None and not pd.isna(given_r_value) and float(given_r_value) > 0:
            r_val = float(given_r_value)
        elif given_resin_total_phr and given_curer_total_phr:
            try:
                r_phr = float(given_resin_total_phr)
                h_phr = float(given_curer_total_phr)
                if r_phr > 0 and h_phr > 0 and weighted_eew > 0 and weighted_ahew > 0:
                    # r = (h_phr / ahew) / (r_phr / eew)
                    r_val = (h_phr / weighted_ahew) / (r_phr / weighted_eew)
            except Exception:
                pass

        # 4. 高分子物理交联特征
        stoich_deviation = abs(r_val - 1.0)
        stoich_log_ratio = float(np.log(np.clip(r_val, 1e-4, 1e4)))

        # 理论最大转化率 (Carothers/Flory 凝胶极限)
        if r_val > 0:
            theoretical_alpha_max = min(1.0, r_val, 1.0 / r_val)
        else:
            theoretical_alpha_max = 0.0

        # 凝胶点转化率 alpha_gel
        if weighted_f_r > 1.0 and weighted_f_h > 1.0:
            alpha_gel = 1.0 / math.sqrt(max((weighted_f_r - 1.0) * (weighted_f_h - 1.0), 1e-5))
            alpha_gel = min(1.0, alpha_gel)
        else:
            alpha_gel = 1.0

        # 混合体系平均分子官能度 f_avg
        # 摩尔分数加权
        mol_r = sum(r["norm_weight"] / max(r["mw"], 10.0) for r in valid_resins)
        mol_h = sum(c["norm_weight"] / max(c["mw"], 10.0) for c in valid_curers) * r_val
        total_mols = mol_r + mol_h
        if total_mols > 0:
            f_avg = (mol_r * weighted_f_r + mol_h * weighted_f_h) / total_mols
        else:
            f_avg = (weighted_f_r + weighted_f_h) / 2.0

        # 平均配方分子量与理论 Mc
        mw_r_avg = sum(r["mw"] * r["norm_weight"] for r in valid_resins)
        mw_h_avg = sum(c["mw"] * c["norm_weight"] for c in valid_curers)
        total_phr_est = 100.0 + 100.0 * (weighted_ahew / weighted_eew) * r_val
        formula_mw = (100.0 * mw_r_avg + (total_phr_est - 100.0) * mw_h_avg) / total_phr_est

        # 交联点间分子量 Mc
        # Flory-Stockmayer: Mc = M_unit / (f_avg - 2)
        theoretical_Mc = formula_mw / max(f_avg - 2.0, 0.08)
        theoretical_Mc = float(np.clip(theoretical_Mc, 50.0, 5000.0))

        # 理论交联密度代理 (mmol/g 或相对指数)
        crosslink_density_proxy = 1000.0 / theoretical_Mc

        # 5. 多组分成对前线轨道能差 (Delta E)
        delta_e_list = []
        delta_e_weights = []

        for r in valid_resins:
            lumo = r.get("lumo")
            if lumo is None or pd.isna(lumo):
                lumo = r.get("lumo_proxy", -1.5)

            for c in valid_curers:
                homo = c.get("homo")
                if homo is None or pd.isna(homo):
                    homo = c.get("homo_proxy", -7.5)

                gap = abs(float(lumo) - float(homo))
                p_ij = r["norm_weight"] * c["norm_weight"]

                delta_e_list.append(gap)
                delta_e_weights.append(p_ij)

        if delta_e_list:
            delta_e_min = float(min(delta_e_list))
            delta_e_max = float(max(delta_e_list))
            delta_e_span = delta_e_max - delta_e_min
            delta_e_weighted = float(sum(g * w for g, w in zip(delta_e_list, delta_e_weights)))
        else:
            delta_e_min = 6.0
            delta_e_max = 6.0
            delta_e_span = 0.0
            delta_e_weighted = 6.0

        # 6. 配方加权物理混合物特征
        weighted_tpsa = sum(r["tpsa"] * r["norm_weight"] for r in valid_resins if not np.isnan(r["tpsa"])) + \
                        sum(c["tpsa"] * c["norm_weight"] for c in valid_curers if not np.isnan(c["tpsa"]))
        weighted_aromatic_rings = sum(r["aromatic_rings"] * r["norm_weight"] for r in valid_resins) + \
                                 sum(c["aromatic_rings"] * c["norm_weight"] for c in valid_curers)
        weighted_rotatable_bonds = sum(r["rotatable_bonds"] * r["norm_weight"] for r in valid_resins) + \
                                  sum(c["rotatable_bonds"] * c["norm_weight"] for c in valid_curers)

        return {
            "mech_stoichiometry_r": round(r_val, 4),
            "mech_stoich_deviation": round(stoich_deviation, 4),
            "mech_stoich_log_ratio": round(stoich_log_ratio, 4),
            "mech_theoretical_alpha_max": round(theoretical_alpha_max, 4),
            "mech_theoretical_alpha_gel": round(alpha_gel, 4),
            "mech_weighted_epoxy_func": round(weighted_f_r, 3),
            "mech_weighted_curer_func": round(weighted_f_h, 3),
            "mech_average_functionality": round(f_avg, 3),
            "mech_theoretical_Mc": round(theoretical_Mc, 2),
            "mech_crosslink_density_proxy": round(crosslink_density_proxy, 4),
            "mech_delta_E_min": round(delta_e_min, 4),
            "mech_delta_E_max": round(delta_e_max, 4),
            "mech_delta_E_span": round(delta_e_span, 4),
            "mech_delta_E_weighted": round(delta_e_weighted, 4),
            "mech_weighted_formula_mw": round(formula_mw, 2),
            "mech_weighted_tpsa": round(weighted_tpsa, 2),
            "mech_weighted_aromatic_rings": round(weighted_aromatic_rings, 2),
            "mech_weighted_rotatable_bonds": round(weighted_rotatable_bonds, 2),
        }

    def enrich_dataframe(
        self,
        df: pd.DataFrame,
        wide_df: Optional[pd.DataFrame] = None,
        progress_callback: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        对输入的 DataFrame (如 ml_qspr_model_tg_c.csv) 进行一键机理特征衍生与增强
        """
        df_out = df.copy()

        # 检查是否可以与 wide_df 对齐补充 phr/ew
        has_aligned_wide = False
        if wide_df is not None and len(wide_df) == len(df_out):
            has_aligned_wide = True

        results = []
        total = len(df_out)

        # 组分列模式识别
        resin_smi_cols = [c for c in df_out.columns if re.match(r'^resin_\d+_structure$', c)]
        if not resin_smi_cols:
            resin_smi_cols = [c for c in df_out.columns if "resin" in c and "smiles" in c or "structure" in c]

        curer_smi_cols = [c for c in df_out.columns if re.match(r'^curing_agent_\d+_structure$', c)]
        if not curer_smi_cols:
            curer_smi_cols = [c for c in df_out.columns if "cur" in c and "smiles" in c or "structure" in c]

        r_value_col = "formulation_r_value" if "formulation_r_value" in df_out.columns else None
        r_phr_col = "resin_total_phr" if "resin_total_phr" in df_out.columns else None
        h_phr_col = "curing_agent_total_phr" if "curing_agent_total_phr" in df_out.columns else None

        # 缓存单分子计算结果以极大加速计算
        mol_cache: Dict[str, Dict[str, Any]] = {}

        for i in range(total):
            row = df_out.iloc[i]
            wide_row = wide_df.iloc[i] if has_aligned_wide else None

            # 1. 收集树脂组分
            resins = []
            for col_idx, col in enumerate(resin_smi_cols, start=1):
                smi = self.clean_structure_string(row.get(col))
                if smi:
                    weight = 1.0
                    fb_ew = None
                    fb_mw = None

                    # 从 wide_df 或当前行获取精确 PHR 与 EEW
                    if wide_row is not None:
                        phr_val = wide_row.get(f"resin_{col_idx}_amount_phr")
                        if phr_val is not None and not pd.isna(phr_val) and float(phr_val) > 0:
                            weight = float(phr_val)
                        fb_ew = wide_row.get(f"resin_{col_idx}_equivalent_weight_g_eq")
                        fb_mw = wide_row.get(f"resin_{col_idx}_molecular_weight_g_mol")

                    cache_key = f"R|{smi}|{fb_ew}|{fb_mw}"
                    if cache_key in mol_cache:
                        p = dict(mol_cache[cache_key])
                    else:
                        p = self.calc_single_molecule_properties(smi, is_resin=True, fallback_ew=fb_ew, fallback_mw=fb_mw)
                        mol_cache[cache_key] = p

                    # 检查是否有显式 xTB 列
                    lumo_col = f"resin_{col_idx}_xtb_lumo"
                    if lumo_col in df_out.columns and not pd.isna(row[lumo_col]):
                        p["lumo"] = float(row[lumo_col])
                    p["weight"] = weight
                    resins.append(p)

            # 2. 收集固化剂组分
            curers = []
            for col_idx, col in enumerate(curer_smi_cols, start=1):
                smi = self.clean_structure_string(row.get(col))
                if smi:
                    weight = 1.0
                    fb_ew = None
                    fb_mw = None

                    if wide_row is not None:
                        phr_val = wide_row.get(f"curing_agent_{col_idx}_amount_phr")
                        if phr_val is not None and not pd.isna(phr_val) and float(phr_val) > 0:
                            weight = float(phr_val)
                        fb_ew = wide_row.get(f"curing_agent_{col_idx}_equivalent_weight_g_eq")
                        fb_mw = wide_row.get(f"curing_agent_{col_idx}_molecular_weight_g_mol")

                    cache_key = f"H|{smi}|{fb_ew}|{fb_mw}"
                    if cache_key in mol_cache:
                        p = dict(mol_cache[cache_key])
                    else:
                        p = self.calc_single_molecule_properties(smi, is_resin=False, fallback_ew=fb_ew, fallback_mw=fb_mw)
                        mol_cache[cache_key] = p

                    homo_col = f"curing_agent_{col_idx}_xtb_homo"
                    if homo_col in df_out.columns and not pd.isna(row[homo_col]):
                        p["homo"] = float(row[homo_col])
                    p["weight"] = weight
                    curers.append(p)

            # 3. 计算配方机理特征
            given_r = row.get(r_value_col) if r_value_col else None
            given_r_phr = row.get(r_phr_col) if r_phr_col else None
            given_h_phr = row.get(h_phr_col) if h_phr_col else None

            feat = self.compute_formulation_mechanism_features(
                resins,
                curers,
                given_r_value=given_r,
                given_resin_total_phr=given_r_phr,
                given_curer_total_phr=given_h_phr,
            )
            results.append(feat)

            if progress_callback and (i + 1) % 500 == 0:
                progress_callback(i + 1, total)

        feat_df = pd.DataFrame(results)
        for col in feat_df.columns:
            df_out[col] = feat_df[col].values

        return df_out
