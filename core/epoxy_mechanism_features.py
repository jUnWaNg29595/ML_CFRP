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
   - 理论交联密度（**主口径 mol/m³**；另附 mmol/g 兼容口径）
2. 多组分前线轨道能差矩阵与动力学指标 (Delta E):
   - delta_E_min: 先发活性通道 (决定凝胶与起始温度)
   - delta_E_max: 最钝迟钝通道 (决定未反应缺陷与深层固化风险)
   - delta_E_span: 动力学色散度 (反映分步固化与相分离倾向)
   - delta_E_weighted: 体系加权宏观活化能垒
3. 多组分配方物理混合加权特征 (分子量、极性TPSA、芳环密度等)
"""

import os
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .reaction_core import (
    DEFAULT_RHO_G_CM3,
    inspect_resin_functionality,
    inspect_curer_functionality,
    compute_molar_average_functionality,
    compute_gel_point_conversion,
)
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
            self._pri_amine_pattern = Chem.MolFromSmarts('[NX3;H2;!$(NC=O);!$(NS(=O)=O)]')
            self._sec_amine_pattern = Chem.MolFromSmarts('[NX3;H1;!$(NC=O);!$(NS(=O)=O)]')
            # 环状酸酐通用模式：兼容 RDKit 芳构化感知（PMDA/BTDA 型稠环芳酐
            # 的羰基骨架会被感知为芳香体系，传统 C(=O)OC(=O) 模式完全匹配不上）
            self._anhydride_pattern = Chem.MolFromSmarts('[o,OX2]1~[#6](=[OX1])~[#6]~[#6]~[#6](=[OX1])~1')
            self._thiol_pattern = Chem.MolFromSmarts('[SX2;H1]')
            # [R 修复] 扩展固化剂类型识别：异氰酸酯 / 酚羟基 / 醇羟基 / 羧基
            self._isocyanate_pattern = Chem.MolFromSmarts('[NX2]=[CX2]=[OX1]')
            self._phenol_oh_pattern = Chem.MolFromSmarts('[OX2H]-c')
            self._alcohol_oh_pattern = Chem.MolFromSmarts('[OX2H]-[CX4]')
            self._carboxyl_pattern = Chem.MolFromSmarts('[OX2H]-[CX3]=[OX1]')
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
        """获取环氧基数量（优先使用 reaction_core 高精度统一检测）。"""
        info = inspect_resin_functionality(smi, mol)
        if info.is_valid and info.n_groups > 0:
            return info.n_groups

        # 字符串正则降级兜底
        s = self.clean_structure_string(smi)
        if not s:
            return 0
        epoxy_matches = len(re.findall(r'O1CC1|C1OC1|1OC1|C1CO1', s))
        if epoxy_matches > 0:
            return epoxy_matches
        if "dgeba" in s.lower():
            return 2
        return 0

    def get_active_hydrogen_count(self, smi: str, mol: Optional[Any] = None) -> Tuple[int, str]:
        """获取固化剂活性氢（当量位点）数量及类型（优先使用 reaction_core 统一规范）。"""
        info = inspect_curer_functionality(smi, mol)
        if info.is_valid:
            return int(info.f_stoich), info.mechanism
        elif info.mechanism == "non_reactive":
            return 0, "non_reactive"

        # 字符串正则降级 (仅在 mol 为 None 无法被 RDKit 解析时提供保底)
        s = self.clean_structure_string(smi)
        if not s:
            return 0, "unknown"

        nco = len(re.findall(r'N=C=O|N\\?C=O|NCO', s))
        if nco > 0:
            return nco, "isocyanate"
        oh = len(re.findall(r'(?:^|[^A-Za-z])O(?![A-Za-z])|\[OH\]|cO', s))
        amine_h = len(re.findall(r'N(?![A-Za-z])|\[NH2\]|\[NH\]', s))
        if nco == 0 and amine_h > 0:
            return min(amine_h * 2, 8), "amine"
        if oh > 0:
            return min(oh, 8), "phenol"
        return 0, "unidentified"

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

        # 官能度与当量计算 (严格真实识别，不人为虚构假官能度或当量)
        if is_resin:
            func = self.get_epoxide_count(smi, mol)
            if func > 0:
                props["functionality"] = float(func)
                props["functional_group_status"] = "identified"
            else:
                props["functionality"] = np.nan
                props["functional_group_status"] = "unidentified"
            if fallback_ew is not None and not pd.isna(fallback_ew) and float(fallback_ew) > 0:
                props["ew"] = float(fallback_ew)
            elif not np.isnan(props["mw"]) and func > 0:
                props["ew"] = props["mw"] / func
        else:
            info = inspect_curer_functionality(smi, mol)
            if info.is_valid and info.f_network > 0:
                props["functionality"] = float(info.f_network)
                props["f_network"] = float(info.f_network)
                props["f_stoich"] = float(info.f_stoich)
                props["functional_group_status"] = "identified"
            else:
                props["functionality"] = np.nan
                props["f_network"] = np.nan
                props["f_stoich"] = np.nan
                props["functional_group_status"] = "unidentified"
            props["curer_type"] = info.mechanism
            if fallback_ew is not None and not pd.isna(fallback_ew) and float(fallback_ew) > 0:
                props["ew"] = float(fallback_ew)
            # 固化剂当量重 (AHEW) 必须严格按化学计量口径 f_stoich 计算（酸酐 f_stoich=1.0，EW=MW，绝不能除以 2.0）(P0-5)
            f_st = props.get("f_stoich", np.nan)
            if not (np.isfinite(f_st) and f_st > 0):
                f_st = props.get("functionality", np.nan)
            if fallback_ew is not None and not pd.isna(fallback_ew) and float(fallback_ew) > 0:
                props["ew"] = float(fallback_ew)
            elif not np.isnan(props["mw"]) and np.isfinite(f_st) and f_st > 0:
                props["ew"] = props["mw"] / f_st
        # 外部 fallback 补充
        if fallback_mw is not None and not pd.isna(fallback_mw) and float(fallback_mw) > 0:
            props["mw"] = float(fallback_mw)
        if fallback_ew is not None and not pd.isna(fallback_ew) and float(fallback_ew) > 0:
            props["ew"] = float(fallback_ew)

        # 只有在有确切当量和官能度时才推算分子量，绝不硬编码 180.0 或 50.0
        if np.isnan(props["mw"]) and not np.isnan(props["ew"]) and not np.isnan(props["functionality"]) and props["functionality"] > 0:
            props["mw"] = props["ew"] * props["functionality"]

        # 3D 几何特征（体积、回转半径、各向异性等）
        props["geom_vdw_vol"] = np.nan
        props["geom_rg"] = np.nan
        props["geom_asphericity"] = np.nan
        props["geom_eccentricity"] = np.nan
        props["geom_pbf"] = np.nan
        props["ff_charge_variance"] = np.nan
        props["ff_energy_torsion"] = np.nan
        props["ff_energy_oop"] = np.nan
        props["num_atoms_3d"] = np.nan

        # 骨架刚性碳 (sp, sp2) 统计
        props["rigid_carbon_ratio"] = 0.0
        # 炔基与氰酸酯等特殊反应基团统计
        props["n_terminal_alkyne"] = 0
        props["n_internal_alkyne"] = 0
        props["n_cyanate_ester"] = 0

        if mol is not None and RDKIT_AVAILABLE:
            try:
                # 统计碳杂化
                sp_c = 0
                sp2_c = 0
                tot_c = 0
                for a in mol.GetAtoms():
                    if a.GetAtomicNum() == 6:
                        tot_c += 1
                        hyb = a.GetHybridization()
                        if hyb == Chem.HybridizationType.SP:
                            sp_c += 1
                        elif hyb == Chem.HybridizationType.SP2:
                            sp2_c += 1
                if tot_c > 0:
                    props["rigid_carbon_ratio"] = float((sp_c + sp2_c) / tot_c)

                # 统计炔基与氰酸酯
                patt_term_alkyne = Chem.MolFromSmarts('[C;H1]#[C]')
                patt_int_alkyne = Chem.MolFromSmarts('[C;H0]#[C;H0]')
                patt_cyanate = Chem.MolFromSmarts('[O,S]-[C]#[N]')
                if patt_term_alkyne:
                    props["n_terminal_alkyne"] = len(mol.GetSubstructMatches(patt_term_alkyne))
                if patt_int_alkyne:
                    props["n_internal_alkyne"] = len(mol.GetSubstructMatches(patt_int_alkyne))
                if patt_cyanate:
                    props["n_cyanate_ester"] = len(mol.GetSubstructMatches(patt_cyanate))

                # 尝试轻量嵌入 3D 并获取 MMFF 电荷、分项能量及 3D 描述符
                from rdkit.Chem import Descriptors3D, rdMolDescriptors
                m3d = Chem.AddHs(mol)
                res_embed = -1
                try:
                    res_embed = AllChem.EmbedMolecule(m3d, randomSeed=42, useRandomCoords=True)
                except Exception:
                    res_embed = -1
                if res_embed == 0:
                    props["num_atoms_3d"] = int(m3d.GetNumAtoms())
                    try:
                        AllChem.MMFFOptimizeMolecule(m3d, maxIters=30)
                    except Exception:
                        pass
                    # 计算体积与 3D 描述符
                    try:
                        props["geom_vdw_vol"] = float(AllChem.ComputeMolVolume(m3d))
                    except Exception:
                        pass
                    try:
                        props["geom_rg"] = float(Descriptors3D.RadiusOfGyration(m3d))
                    except Exception:
                        pass
                    try:
                        props["geom_asphericity"] = float(Descriptors3D.Asphericity(m3d))
                    except Exception:
                        pass
                    try:
                        props["geom_eccentricity"] = float(Descriptors3D.Eccentricity(m3d))
                    except Exception:
                        pass
                    try:
                        props["geom_pbf"] = float(rdMolDescriptors.CalcPBF(m3d))
                    except Exception:
                        pass
                    # MMFF 电荷方差与分项能量
                    try:
                        if AllChem.MMFFHasAllMoleculeParams(m3d):
                            ff_p = AllChem.MMFFGetMoleculeProperties(m3d, mmffVariant="MMFF94s")
                            charges = [ff_p.GetMMFFPartialCharge(idx) for idx in range(m3d.GetNumAtoms())]
                            props["ff_charge_variance"] = float(np.var(charges))

                            # 提取扭转能与离面弯曲能 (OOP)
                            p_tor = AllChem.MMFFGetMoleculeProperties(m3d, mmffVariant="MMFF94s")
                            for term in ["Bond", "Angle", "StretchBend", "Oop", "VdW", "Ele"]:
                                getattr(p_tor, f"SetMMFF{term}Term")(False)
                            p_tor.SetMMFFTorsionTerm(True)
                            ff_tor = AllChem.MMFFGetMoleculeForceField(m3d, p_tor)
                            if ff_tor:
                                props["ff_energy_torsion"] = float(ff_tor.CalcEnergy())

                            p_oop = AllChem.MMFFGetMoleculeProperties(m3d, mmffVariant="MMFF94s")
                            for term in ["Bond", "Angle", "StretchBend", "Torsion", "VdW", "Ele"]:
                                getattr(p_oop, f"SetMMFF{term}Term")(False)
                            p_oop.SetMMFFOopTerm(True)
                            ff_oop = AllChem.MMFFGetMoleculeForceField(m3d, p_oop)
                            if ff_oop:
                                props["ff_energy_oop"] = float(ff_oop.CalcEnergy())
                    except Exception:
                        pass
            except Exception:
                pass

        return props

    def compute_formulation_mechanism_features(
        self,
        resin_list: List[Dict[str, Any]],
        curer_list: List[Dict[str, Any]],
        given_r_value: Optional[float] = None,
        given_resin_total_phr: Optional[float] = None,
        given_curer_total_phr: Optional[float] = None,
        assume_stoichiometric_if_missing: bool = True,
        existing_r_source: Optional[str] = None,
        rxn_r_assumed: Optional[float] = None,
        rho_g_cm3: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        计算多组分配方的综合机理特征与高分子网络物理参数
        resin_list: [{'smi': str, 'weight': float, 'mw': float, 'functionality': float, ...}]
        curer_list: [{'smi': str, 'weight': float, 'mw': float, 'functionality': float, ...}]
        """
        # 1. 过滤空组分并归一化组内相对比例
        valid_resins = [r for r in resin_list if r.get("smi")]
        valid_curers = [c for c in curer_list if c.get("smi")]

        if not valid_resins or not valid_curers:
            return {}

        w_r_sum = sum(max(r.get("weight") if r.get("weight") is not None else 1.0, 0.0) for r in valid_resins)
        if w_r_sum <= 0:
            w_r_sum = float(len(valid_resins))
        for r in valid_resins:
            r["norm_weight"] = max(r.get("weight") if r.get("weight") is not None else 1.0, 0.0) / w_r_sum

        w_h_sum = sum(max(c.get("weight") if c.get("weight") is not None else 1.0, 0.0) for c in valid_curers)
        if w_h_sum <= 0:
            w_h_sum = float(len(valid_curers))
        for c in valid_curers:
            c["norm_weight"] = max(c.get("weight") if c.get("weight") is not None else 1.0, 0.0) / w_h_sum

        # 2. 计算树脂端与固化剂端组内加权官能度与当量
        weighted_f_r = sum((r.get("functionality", 0.0) or 0.0) * r["norm_weight"] for r in valid_resins)
        weighted_f_h = sum((c.get("functionality", 0.0) or 0.0) * c["norm_weight"] for c in valid_curers)

        # 加权 EEW 与 AHEW
        inv_eew = sum((r["norm_weight"] / r["ew"]) for r in valid_resins if r.get("ew") and 0 < r["ew"] < 50000)
        weighted_eew = (1.0 / inv_eew) if inv_eew > 0 else 180.0

        inv_ahew = sum((c["norm_weight"] / c["ew"]) for c in valid_curers if c.get("ew") and 0 < c["ew"] < 50000)
        weighted_ahew = (1.0 / inv_ahew) if inv_ahew > 0 else 50.0

        # 3. 确定各组分真实质量与化学计量比 r
        # 树脂总质量基准 (默认为 phr 基准 100.0)
        if given_resin_total_phr is not None and not pd.isna(given_resin_total_phr) and float(given_resin_total_phr) > 0:
            total_mass_r = float(given_resin_total_phr)
        else:
            w_sum = sum(max(r.get("weight") if r.get("weight") is not None else 0.0, 0.0) for r in valid_resins)
            total_mass_r = w_sum if w_sum > 0 else 100.0

        # 固化剂总质量基准推导
        # 严格执行四级优先级：文献 r 绝对优先 -> 完整质量计算 -> 缺失时假设 r = 1.0 (P0-4)
        all_r_have_mass = all(r.get("weight") is not None and np.isfinite(r.get("weight")) and r.get("weight") > 0 for r in valid_resins)
        all_h_have_mass = all(c.get("weight") is not None and np.isfinite(c.get("weight")) and c.get("weight") > 0 for c in valid_curers)
        has_real_mass_both = (all_r_have_mass or (given_resin_total_phr is not None and float(given_resin_total_phr) > 0)) and \
                             (all_h_have_mass or (given_curer_total_phr is not None and float(given_curer_total_phr) > 0))

        if given_r_value is not None and not pd.isna(given_r_value) and float(given_r_value) > 0:
            r_val = float(given_r_value)
            if total_mass_r > 0 and weighted_eew > 0 and weighted_ahew > 0:
                total_mass_h = total_mass_r * r_val * (weighted_ahew / weighted_eew)
            else:
                total_mass_h = 100.0 * r_val * (weighted_ahew / weighted_eew) if (weighted_eew > 0 and weighted_ahew > 0) else 10.0
        elif has_real_mass_both and weighted_eew > 0 and weighted_ahew > 0:
            total_mass_h = float(given_curer_total_phr) if (given_curer_total_phr is not None and float(given_curer_total_phr) > 0) else sum(c["weight"] for c in valid_curers)
            total_mass_r = float(given_resin_total_phr) if (given_resin_total_phr is not None and float(given_resin_total_phr) > 0) else sum(r["weight"] for r in valid_resins)
            r_val = (total_mass_h / weighted_ahew) / (total_mass_r / weighted_eew)
        elif assume_stoichiometric_if_missing:
            r_val = 1.0
            total_mass_r = 100.0 if not (total_mass_r > 0) else total_mass_r
            total_mass_h = total_mass_r * (weighted_ahew / max(weighted_eew, 1e-4))
        else:
            r_val = np.nan
            total_mass_r = 100.0 if not (total_mass_r > 0) else total_mass_r
            total_mass_h = 0.0

        # 4. 严密高分子网络物理计算（Flory-Stockmayer 统计理论，上限对齐 F_MAX=8.0）
        # 计算每个单体实际质量与摩尔数 (n_i = m_i / MW_i)
        for r in valid_resins:
            r["comp_mass"] = total_mass_r * r["norm_weight"]
            mw_r = max(r.get("mw", 0.0) if not pd.isna(r.get("mw", 0.0)) else 0.0, 10.0)
            r["comp_mols"] = r["comp_mass"] / mw_r
            r["comp_func"] = min(max(r.get("functionality", 0.0) if not pd.isna(r.get("functionality", 0.0)) else 0.0, 0.0), 8.0)

        for c in valid_curers:
            c["comp_mass"] = total_mass_h * c["norm_weight"]
            mw_c = max(c.get("mw", 0.0) if not pd.isna(c.get("mw", 0.0)) else 0.0, 10.0)
            c["comp_mols"] = c["comp_mass"] / mw_c
            c["comp_func"] = min(max(c.get("functionality", 0.0) if not pd.isna(c.get("functionality", 0.0)) else 0.0, 0.0), 8.0)

        mols_r_total = sum(r["comp_mols"] for r in valid_resins)
        mols_h_total = sum(c["comp_mols"] for c in valid_curers)
        total_system_mols = mols_r_total + mols_h_total
        total_system_mass = total_mass_r + total_mass_h

        # (1) 配方数均分子量 Mn = Σ m_i / Σ n_i
        formula_mw = (total_system_mass / total_system_mols) if total_system_mols > 0 else (weighted_eew + weighted_ahew)

        # (2) 混合体系摩尔平均官能度 f_avg = Σ (n_i * f_i) / Σ n_i
        total_func_groups = (sum(r["comp_mols"] * r["comp_func"] for r in valid_resins) +
                             sum(c["comp_mols"] * c["comp_func"] for c in valid_curers))
        if total_system_mols > 0 and (has_real_mass_both or np.isfinite(r_val)):
            f_avg = float(total_func_groups / total_system_mols)
        else:
            f_avg = np.nan

        # (3) 当量偏差与最大转化率
        stoich_deviation = abs(r_val - 1.0) if np.isfinite(r_val) else np.nan
        stoich_log_ratio = float(np.log(np.clip(r_val, 1e-4, 1e4))) if np.isfinite(r_val) else np.nan
        theoretical_alpha_max = min(1.0, r_val, 1.0 / r_val) if (np.isfinite(r_val) and r_val > 0) else np.nan

        # (4) 理想网络凝胶理论与交联指标（直接委托 reaction_core 严密求解，保证四模块 100% 一致）
        from .reaction_core import evaluate_reaction_network
        res_dicts = [{"smiles": r.get("smi") or r.get("smiles"), "mass": (r.get("weight") if r.get("weight") is not None else r.get("comp_mass", 0.0)), "mw": r.get("mw", 0.0), "f_stoich": min(float(r.get("f_stoich", r.get("comp_func", 0.0))), 8.0), "f_network": min(float(r.get("f_network", r.get("comp_func", 0.0))), 8.0)} for r in valid_resins]
        cur_dicts = [{"smiles": c.get("smi") or c.get("smiles"), "mass": (c.get("weight") if c.get("weight") is not None else c.get("comp_mass", 0.0)), "mw": c.get("mw", 0.0), "f_stoich": min(float(c.get("f_stoich", c.get("comp_func", 0.0))), 8.0), "f_network": min(float(c.get("f_network", c.get("comp_func", 0.0))), 8.0)} for c in valid_curers]

        # 统一生成密度基准 (P0-3)
        if rho_g_cm3 is not None and np.isfinite(rho_g_cm3) and float(rho_g_cm3) > 0:
            rho_used = float(rho_g_cm3)
            rho_source = "sample_provided"
        else:
            rho_used = 1.2
            rho_source = "reference_default"

        rc_net = evaluate_reaction_network(
            res_dicts,
            cur_dicts,
            explicit_r=given_r_value,
            assume_stoichiometric_if_missing=assume_stoichiometric_if_missing,
            existing_r_source=existing_r_source,
            rxn_r_assumed=rxn_r_assumed,
            rho_g_cm3=rho_used,
        )

        f_avg = rc_net.f_avg if (np.isfinite(rc_net.f_avg) and rc_net.f_avg > 0) else f_avg
        gel_possible_ideal = rc_net.gel_possible_ideal

        # 经验代理 Mc (独立存储，绝不覆盖理论 Mc) (P0-4)
        mc_proxy = float(formula_mw / max(f_avg - 2.0, 0.1)) if (np.isfinite(f_avg) and f_avg > 2.0001) else np.nan

        if gel_possible_ideal is not None and not np.isnan(gel_possible_ideal) and gel_possible_ideal > 0.5 and np.isfinite(rc_net.r_value):
            alpha_gel = rc_net.alpha_gel if np.isfinite(rc_net.alpha_gel) else np.nan
            theoretical_Mc = rc_net.Mc_ideal if np.isfinite(rc_net.Mc_ideal) else np.nan
            crosslink_density_mol_m3 = rc_net.nu_ideal_mol_m3 if np.isfinite(rc_net.nu_ideal_mol_m3) else np.nan
            crosslink_density_mmol_g = (crosslink_density_mol_m3 / (rho_used * 1000.0)) if np.isfinite(crosslink_density_mol_m3) else np.nan
        elif gel_possible_ideal is None or np.isnan(gel_possible_ideal):
            # 严格模式下理论凝胶状态未知 (P1): 理论交联密度必须为 NaN，绝不伪装为 0.0！
            alpha_gel = np.nan
            theoretical_Mc = np.nan
            crosslink_density_mol_m3 = np.nan
            crosslink_density_mmol_g = np.nan
        else:
            # 确定不可凝胶状态 (如单官能或亚临界): 理想交联密度为 0.0
            alpha_gel = np.nan
            theoretical_Mc = np.nan
            crosslink_density_mol_m3 = 0.0
            crosslink_density_mmol_g = 0.0

        # 5. 多组分成对前线轨道能差 (Delta E) 与反应驱动力多列体系
        delta_e_list = []
        delta_e_fwd_list = []
        delta_e_rev_list = []
        delta_mu_list = []
        delta_e_weights = []
        pairwise_cols = {}

        # 主组分能差 (Primary pair: resin 1 vs curer 1)
        r_primary = valid_resins[0] if valid_resins else {}
        c_primary = valid_curers[0] if valid_curers else {}
        r_prim_lumo = r_primary.get("lumo") if r_primary.get("lumo") is not None and not pd.isna(r_primary.get("lumo")) else r_primary.get("lumo_proxy", -1.5)
        r_prim_homo = r_primary.get("homo") if r_primary.get("homo") is not None and not pd.isna(r_primary.get("homo")) else -10.0
        c_prim_homo = c_primary.get("homo") if c_primary.get("homo") is not None and not pd.isna(c_primary.get("homo")) else c_primary.get("homo_proxy", -7.5)
        c_prim_lumo = c_primary.get("lumo") if c_primary.get("lumo") is not None and not pd.isna(c_primary.get("lumo")) else 1.0

        fmo_delta_e_fwd_primary = float(r_prim_lumo) - float(c_prim_homo)
        fmo_delta_e_rev_primary = float(c_prim_lumo) - float(r_prim_homo)

        for r_idx, r in enumerate(valid_resins):
            r_col_idx = r.get("col_idx", r_idx + 1)
            lumo_r = r.get("lumo")
            if lumo_r is None or pd.isna(lumo_r):
                lumo_r = r.get("lumo_proxy", -1.5)
            homo_r = r.get("homo")
            if homo_r is None or pd.isna(homo_r):
                homo_r = -10.0
            mu_r = r.get("mu")
            if mu_r is None or pd.isna(mu_r):
                mu_r = (float(homo_r) + float(lumo_r)) / 2.0

            for c_idx, c in enumerate(valid_curers):
                c_col_idx = c.get("col_idx", c_idx + 1)
                homo_c = c.get("homo")
                if homo_c is None or pd.isna(homo_c):
                    homo_c = c.get("homo_proxy", -7.5)
                lumo_c = c.get("lumo")
                if lumo_c is None or pd.isna(lumo_c):
                    lumo_c = 1.0
                mu_c = c.get("mu")
                if mu_c is None or pd.isna(mu_c):
                    mu_c = (float(homo_c) + float(lumo_c)) / 2.0

                # 正向能差 (固化剂进攻环氧): LUMO(resin) - HOMO(curer)
                gap_fwd = float(lumo_r) - float(homo_c)
                # 逆向能差: LUMO(curer) - HOMO(resin)
                gap_rev = float(lumo_c) - float(homo_r)
                # 绝对能差
                gap_abs = abs(gap_fwd)
                # 化学势驱动力差: mu(curer) - mu(resin)
                d_mu = float(mu_c) - float(mu_r)

                p_ij = r["norm_weight"] * c["norm_weight"]

                delta_e_list.append(gap_abs)
                delta_e_fwd_list.append(gap_fwd)
                delta_e_rev_list.append(gap_rev)
                delta_mu_list.append(d_mu)
                delta_e_weights.append(p_ij)

                # 记录具体配对列（支持多组分如 r1_h1, r1_h2 等）
                pair_key = f"mech_fmo_delta_e_r{r_col_idx}_h{c_col_idx}"
                pairwise_cols[pair_key] = round(gap_fwd, 4)

        if delta_e_list:
            delta_e_min = float(min(delta_e_list))
            delta_e_max = float(max(delta_e_list))
            delta_e_span = delta_e_max - delta_e_min
            delta_e_weighted = float(sum(g * w for g, w in zip(delta_e_list, delta_e_weights)))
            delta_e_fwd_min = float(min(delta_e_fwd_list))
            delta_e_fwd_max = float(max(delta_e_fwd_list))
            delta_e_fwd_weighted = float(sum(g * w for g, w in zip(delta_e_fwd_list, delta_e_weights)))
            delta_e_rev_min = float(min(delta_e_rev_list))
            delta_e_rev_max = float(max(delta_e_rev_list))
            delta_e_rev_weighted = float(sum(g * w for g, w in zip(delta_e_rev_list, delta_e_weights)))
            delta_mu_weighted = float(sum(m * w for m, w in zip(delta_mu_list, delta_e_weights)))
        else:
            delta_e_min = 6.0
            delta_e_max = 6.0
            delta_e_span = 0.0
            delta_e_weighted = 6.0
            delta_e_fwd_min = 6.0
            delta_e_fwd_max = 6.0
            delta_e_fwd_weighted = 6.0
            delta_e_rev_min = 11.0
            delta_e_rev_max = 11.0
            delta_e_rev_weighted = 11.0
            delta_mu_weighted = 0.0

        # 6. 配方加权物理混合物特征与 3D 几何外形
        weighted_tpsa = sum(r["tpsa"] * r["norm_weight"] for r in valid_resins if not np.isnan(r["tpsa"])) + \
                        sum(c["tpsa"] * c["norm_weight"] for c in valid_curers if not np.isnan(c["tpsa"]))
        weighted_aromatic_rings = sum(r["aromatic_rings"] * r["norm_weight"] for r in valid_resins) + \
                                 sum(c["aromatic_rings"] * c["norm_weight"] for c in valid_curers)
        weighted_rotatable_bonds = sum(r["rotatable_bonds"] * r["norm_weight"] for r in valid_resins) + \
                                  sum(c["rotatable_bonds"] * c["norm_weight"] for c in valid_curers)

        # 3D 几何特征加权 (排除体积、回转半径、各向异性等)
        def _get_geom(plist, key, default_val=1.0):
            vals = [p[key] for p in plist if key in p and not np.isnan(p[key])]
            weights = [p["norm_weight"] for p in plist if key in p and not np.isnan(p[key])]
            if vals and sum(weights) > 0:
                return float(sum(v * w for v, w in zip(vals, weights)) / sum(weights))
            return float(default_val)

        geom_vdw_vol_r = _get_geom(valid_resins, "geom_vdw_vol", default_val=200.0)
        geom_vdw_vol_c = _get_geom(valid_curers, "geom_vdw_vol", default_val=150.0)
        geom_vdw_volume_weighted = (geom_vdw_vol_r + geom_vdw_vol_c) / 2.0

        geom_rg_r = _get_geom(valid_resins, "geom_rg", default_val=2.5)
        geom_rg_c = _get_geom(valid_curers, "geom_rg", default_val=2.0)
        geom_rg_weighted = (geom_rg_r + geom_rg_c) / 2.0

        geom_asphericity_weighted = (_get_geom(valid_resins, "geom_asphericity", default_val=0.2) +
                                     _get_geom(valid_curers, "geom_asphericity", default_val=0.2)) / 2.0
        geom_eccentricity_weighted = (_get_geom(valid_resins, "geom_eccentricity", default_val=0.7) +
                                      _get_geom(valid_curers, "geom_eccentricity", default_val=0.7)) / 2.0

        # 分子紧凑度 C_mol = V_vdW / (Rg^3)
        geom_compactness_weighted = geom_vdw_volume_weighted / max(geom_rg_weighted ** 3, 0.1)

        # 自由体积 FFV 代理: 1.0 - 1.3 * (V_vdW * 0.6022 / V_molar)
        # V_molar = formula_mw / rho (以 rho=1.2 g/cm3 计)
        rho_g_cm3 = 1.2
        v_molar = formula_mw / max(rho_g_cm3, 0.5)
        v_occ_molar = geom_vdw_volume_weighted * 0.602214
        mech_proxy_ffv = float(np.clip(1.0 - 1.3 * (v_occ_molar / max(v_molar, 1.0)), 0.10, 0.40))

        # 7. 开环原位产物与凝聚态内聚能（CED）特征
        # 每个环氧基开环生成 1 个仲醇 -CH(OH)- 羟基；在转化率 alpha_max 下
        # 每摩尔配方单体生成的羟基摩尔数 = (100.0 / weighted_eew) * theoretical_alpha_max
        # 仲醇羟基密度 rho_OH = 理论生成羟基数 / M_formula (mmol/g)
        mol_epoxy_per_100g = (100.0 / max(weighted_eew, 10.0))
        mol_oh_generated = mol_epoxy_per_100g * theoretical_alpha_max
        post_hbond_donor_density = float(mol_oh_generated / max(formula_mw / 100.0, 0.1)) * 10.0

        # 固化生成叔胺密度: 反应活性氢转化为叔胺氮原子
        # 每个伯胺(2H)反应生成1个叔胺，每个仲胺(1H)反应生成1个叔胺
        mol_h_per_phr = (100.0 * (weighted_ahew / weighted_eew) * r_val) / max(weighted_ahew, 10.0)
        mol_tert_amine = (mol_h_per_phr / max(weighted_f_h, 1.0)) * theoretical_alpha_max
        post_tertiary_amine_density = float(mol_tert_amine / max(formula_mw / 100.0, 0.1)) * 10.0

        # MMFF 电荷方差（静电各向异性 / CED 代理）
        ff_charge_variance_weighted = (_get_geom(valid_resins, "ff_charge_variance", default_val=0.08) +
                                       _get_geom(valid_curers, "ff_charge_variance", default_val=0.08)) / 2.0

        # 8. 基于量子化学活化能自适应加权的高 Tg 物理特征体系 (指南规范)
        # (1) 动态转化概率 P_rxn 方程: P = 1 / (1 + exp((Delta_E - E0) / tau))
        # 主环氧-胺反应通道概率:
        e0_epoxy = 7.5
        tau_epoxy = 1.0
        prob_primary_epoxy_amine = float(1.0 / (1.0 + math.exp(np.clip((fmo_delta_e_fwd_primary - e0_epoxy) / tau_epoxy, -20.0, 20.0))))

        # 炔基自适应热聚合概率: 端炔能隙小(~7.0eV)转化率高，内炔受位阻能隙大(~9.0eV)转化率受限
        # 统计配方中炔基数量
        n_term_alkyne = sum(r.get("n_terminal_alkyne", 0) * r["norm_weight"] for r in valid_resins) + \
                        sum(c.get("n_terminal_alkyne", 0) * c["norm_weight"] for c in valid_curers)
        n_int_alkyne = sum(r.get("n_internal_alkyne", 0) * r["norm_weight"] for r in valid_resins) + \
                       sum(c.get("n_internal_alkyne", 0) * c["norm_weight"] for c in valid_curers)
        total_alkyne = n_term_alkyne + n_int_alkyne
        if total_alkyne > 0:
            # 端炔 Delta_E 设定为 7.0 eV，内炔设定为 9.0 eV
            eff_alkyne_gap = (n_term_alkyne * 7.0 + n_int_alkyne * 9.0) / total_alkyne
            prob_alkyne_thermal = float(1.0 / (1.0 + math.exp(np.clip((eff_alkyne_gap - 7.5) / 1.0, -20.0, 20.0))))
        else:
            prob_alkyne_thermal = 0.0

        # 氰酸酯原位三聚均三嗪刚性环节点密度 (mmol/g)
        n_cyanate = sum(r.get("n_cyanate_ester", 0) * r["norm_weight"] for r in valid_resins) + \
                    sum(c.get("n_cyanate_ester", 0) * c["norm_weight"] for c in valid_curers)
        # 3 个氰酸酯聚合形成 1 个均三嗪节点
        triazine_node_density = float((n_cyanate / 3.0) / max(formula_mw / 1000.0, 0.1))

        # (2) 广义动态有效官能度 f_eff 与自适应交联网络
        f_eff_r = weighted_f_r * prob_primary_epoxy_amine + n_term_alkyne * 0.95 + n_int_alkyne * 0.20 + n_cyanate * 0.95
        f_eff_h = weighted_f_h * prob_primary_epoxy_amine
        if mols_r_total > 0 and mols_h_total > 0 and total_system_mols > 0:
            gen_f_eff_avg = float((mols_r_total * f_eff_r + mols_h_total * f_eff_h) / total_system_mols)
        else:
            gen_f_eff_avg = float(compute_molar_average_functionality(
                mol_r=mols_r_total, f_r=f_eff_r,
                mol_h=mols_h_total, f_h=f_eff_h,
                r_val=r_val
            ))
        # 广义有效官能度超过2时形成自适应交联网络；否则处于无交联/极低交联边界(5000 g/mol)
        gen_Mc_adaptive = formula_mw / max(gen_f_eff_avg - 2.0, 0.08)
        gen_Mc_adaptive = float(np.clip(gen_Mc_adaptive, 50.0, 5000.0))
        gen_crosslink_density = float(1000.0 / gen_Mc_adaptive) if gen_f_eff_avg > 2.001 else 0.0
        stoich_defect_ratio = float(abs(r_val - 1.0))

        # (3) 量子化学动力学与反应特征
        xtb_reaction_gap_primary = float(fmo_delta_e_fwd_primary)
        # 化学硬度与化学势差
        lumo_r_val = r_primary.get("lumo", -1.5)
        homo_r_val = r_primary.get("homo", -9.5)
        lumo_c_val = c_primary.get("lumo", 1.0)
        homo_c_val = c_primary.get("homo", -7.5)
        eta_r = (float(lumo_r_val) - float(homo_r_val)) / 2.0
        eta_c = (float(lumo_c_val) - float(homo_c_val)) / 2.0
        xtb_chemical_hardness_weighted = float((eta_r + eta_c) / 2.0)
        mu_r_val = r_primary.get("mu", (float(homo_r_val) + float(lumo_r_val)) / 2.0)
        mu_c_val = c_primary.get("mu", (float(homo_c_val) + float(lumo_c_val)) / 2.0)
        xtb_chemical_potential_diff = float(abs(float(mu_r_val) - float(mu_c_val)))

        # (4) 链段构象阻力与杂化刚度维
        rigid_carbon_ratio = float((_get_geom(valid_resins, "rigid_carbon_ratio", default_val=0.8) +
                                    _get_geom(valid_curers, "rigid_carbon_ratio", default_val=0.8)) / 2.0)

        # 二面角扭转能密度 E_tor / (N_rot + 1e-4) 与 离面弯曲能密度 E_oop / N_atoms
        tor_vals = [p["ff_energy_torsion"] for p in (valid_resins + valid_curers) if "ff_energy_torsion" in p and not np.isnan(p["ff_energy_torsion"])]
        oop_vals = [p["ff_energy_oop"] for p in (valid_resins + valid_curers) if "ff_energy_oop" in p and not np.isnan(p["ff_energy_oop"])]
        atom_vals = [p["num_atoms_3d"] for p in (valid_resins + valid_curers) if "num_atoms_3d" in p and not np.isnan(p["num_atoms_3d"])]

        mean_tor_e = float(np.mean(tor_vals)) if tor_vals else (weighted_rotatable_bonds * 0.8)
        mean_oop_e = float(np.mean(oop_vals)) if oop_vals else 0.5
        mean_atoms = float(np.mean(atom_vals)) if atom_vals else 30.0

        ff_torsion_density = float(mean_tor_e / max(weighted_rotatable_bonds, 0.1))
        ff_oop_density = float(mean_oop_e / max(mean_atoms, 1.0))
        flexibility_index = float(weighted_rotatable_bonds / (weighted_aromatic_rings + 1.0))

        # (5) 3D 几何形貌与偏离平面度 PBF
        geom_pbf = float((_get_geom(valid_resins, "geom_pbf", default_val=0.3) +
                          _get_geom(valid_curers, "geom_pbf", default_val=0.3)) / 2.0)
        geom_asphericity = float(geom_asphericity_weighted)
        geom_compactness = float(geom_compactness_weighted)
        geom_pseudo_ffv = float(mech_proxy_ffv)

        # (6) 非线性物理耦合项 (高 Tg 核心决定项)
        cross_rigidity_crosslink = float(gen_crosslink_density * rigid_carbon_ratio)
        cross_torsion_crosslink = float(gen_crosslink_density * ff_torsion_density)

        # 8. 微观刚性与宏观交联密度的非线性交叉项
        # 扭转能代理：若单体有扭转能取其均摊，否则用旋转键比例估算
        torsion_e_per_atom = 0.25
        res_t_vals = [r.get("ff_energy_torsion_per_atom") for r in valid_resins if r.get("ff_energy_torsion_per_atom") is not None]
        if res_t_vals:
            torsion_e_per_atom = float(np.mean(res_t_vals))
        cross_torsion_density_index = float(crosslink_density_mmol_g * torsion_e_per_atom)

        # 柔顺度因子: (N_rot / (N_arom + 1.0)) * Mc
        mc_for_flex = theoretical_Mc if np.isfinite(theoretical_Mc) else (mc_proxy if np.isfinite(mc_proxy) else 0.0)
        cross_flexibility_index = float((weighted_rotatable_bonds / (weighted_aromatic_rings + 1.0)) * (mc_for_flex / 100.0))

        # 反应活性-交联度协同指数: nu / delta_E_fwd_primary
        cross_fmo_rigidity_index = float(crosslink_density_mmol_g / max(fmo_delta_e_fwd_primary, 0.1))

        out_dict = {
            # --- 原有基础机理与物理特征（保持平滑兼容） ---
            "mech_stoichiometry_r": round(r_val, 4) if np.isfinite(r_val) else np.nan,
            "mech_stoich_deviation": round(stoich_deviation, 4) if np.isfinite(stoich_deviation) else np.nan,
            "mech_stoich_log_ratio": round(stoich_log_ratio, 4) if np.isfinite(stoich_log_ratio) else np.nan,
            "mech_theoretical_alpha_max": round(theoretical_alpha_max, 4) if np.isfinite(theoretical_alpha_max) else np.nan,
            "mech_theoretical_alpha_gel": round(alpha_gel, 4) if np.isfinite(alpha_gel) else np.nan,
            "mech_gel_possible_ideal": float(gel_possible_ideal),
            "mech_weighted_epoxy_func": round(weighted_f_r, 3),
            "mech_weighted_curer_func": round(weighted_f_h, 3),
            "mech_average_functionality": round(f_avg, 3) if np.isfinite(f_avg) else np.nan,
            "mech_theoretical_Mc": round(theoretical_Mc, 2) if np.isfinite(theoretical_Mc) else np.nan,
            "mech_Mc_proxy": round(mc_proxy, 2) if np.isfinite(mc_proxy) else np.nan,
            "mech_crosslink_density": round(crosslink_density_mol_m3, 4) if np.isfinite(crosslink_density_mol_m3) else np.nan,
            "mech_crosslink_density_mol_m3": round(crosslink_density_mol_m3, 4) if np.isfinite(crosslink_density_mol_m3) else np.nan,
            "mech_crosslink_density_mmol_g": round(crosslink_density_mmol_g, 6) if np.isfinite(crosslink_density_mmol_g) else np.nan,
            "mech_crosslink_density_proxy": round(crosslink_density_mmol_g, 6) if np.isfinite(crosslink_density_mmol_g) else np.nan,
            "rho_used_for_mech_nu_g_cm3": rho_used,
            "rho_mech_source": rho_source,
            "mech_delta_E_min": round(delta_e_min, 4),
            "mech_delta_E_max": round(delta_e_max, 4),
            "mech_delta_E_span": round(delta_e_span, 4),
            "mech_delta_E_weighted": round(delta_e_weighted, 4),
            "mech_fmo_delta_e_fwd_primary": round(fmo_delta_e_fwd_primary, 4),
            "mech_fmo_delta_e_rev_primary": round(fmo_delta_e_rev_primary, 4),
            "mech_fmo_delta_e_fwd_min": round(delta_e_fwd_min, 4),
            "mech_fmo_delta_e_fwd_max": round(delta_e_fwd_max, 4),
            "mech_fmo_delta_e_fwd_weighted": round(delta_e_fwd_weighted, 4),
            "mech_fmo_delta_e_rev_min": round(delta_e_rev_min, 4),
            "mech_fmo_delta_e_rev_max": round(delta_e_rev_max, 4),
            "mech_fmo_delta_e_rev_weighted": round(delta_e_rev_weighted, 4),
            "mech_fmo_delta_mu_weighted": round(delta_mu_weighted, 4),
            "mech_weighted_formula_mw": round(formula_mw, 2),
            "mech_weighted_tpsa": round(weighted_tpsa, 2),
            "mech_weighted_aromatic_rings": round(weighted_aromatic_rings, 2),
            "mech_weighted_rotatable_bonds": round(weighted_rotatable_bonds, 2),
            "geom_vdw_volume_weighted": round(geom_vdw_volume_weighted, 2),
            "geom_rg_weighted": round(geom_rg_weighted, 3),
            "geom_asphericity_weighted": round(geom_asphericity_weighted, 4),
            "geom_eccentricity_weighted": round(geom_eccentricity_weighted, 4),
            "geom_compactness_weighted": round(geom_compactness_weighted, 3),
            "mech_proxy_ffv": round(mech_proxy_ffv, 4),
            "post_hbond_donor_density": round(post_hbond_donor_density, 4),
            "post_tertiary_amine_density": round(post_tertiary_amine_density, 4),
            "ff_charge_variance_weighted": round(ff_charge_variance_weighted, 5),
            "cross_torsion_density_index": round(cross_torsion_density_index, 4),
            "cross_flexibility_index": round(cross_flexibility_index, 4),
            "cross_fmo_rigidity_index": round(cross_fmo_rigidity_index, 4),

            # --- 《基于量子化学活化能自适应加权的高 Tg 树脂物理特征清单 (22 维精选规范)》 ---
            # 1. 拓扑与广义交联网络维
            "gen_Mc_adaptive": round(gen_Mc_adaptive, 2),
            "gen_crosslink_density": round(gen_crosslink_density, 4),
            "stoich_defect_ratio": round(stoich_defect_ratio, 4),
            "triazine_node_density": round(triazine_node_density, 4),

            # 2. 量子化学动力学与反应概率维
            "prob_primary_epoxy_amine": round(prob_primary_epoxy_amine, 4),
            "prob_alkyne_thermal": round(prob_alkyne_thermal, 4),
            "xtb_reaction_gap_primary": round(xtb_reaction_gap_primary, 4),
            "xtb_chemical_hardness_weighted": round(xtb_chemical_hardness_weighted, 4),
            "xtb_chemical_potential_diff": round(xtb_chemical_potential_diff, 4),

            # 3. 链段构象阻力与杂化刚度维
            "rigid_carbon_ratio": round(rigid_carbon_ratio, 4),
            "ff_torsion_density": round(ff_torsion_density, 4),
            "ff_oop_density": round(ff_oop_density, 5),
            "flexibility_index": round(flexibility_index, 4),

            # 4. 3D 几何形貌与自由体积维
            "geom_pbf": round(geom_pbf, 4),
            "geom_asphericity": round(geom_asphericity, 4),
            "geom_compactness": round(geom_compactness, 3),
            "geom_pseudo_ffv": round(geom_pseudo_ffv, 4),

            # 5. 凝聚态内聚能（CED）与产物生成维
            "ff_charge_variance": round(ff_charge_variance_weighted, 5),

            # 6. 非线性物理耦合项 (高 Tg 核心决定项)
            "cross_rigidity_crosslink": round(cross_rigidity_crosslink, 4),
            "cross_torsion_crosslink": round(cross_torsion_crosslink, 4),
        }
        return out_dict
        return out_dict

    def enrich_dataframe(
        self,
        df: pd.DataFrame,
        wide_df: Optional[pd.DataFrame] = None,
        progress_callback: Optional[Any] = None,
        include_ff_features: bool = True,
        include_xtb_features: bool = True,
        assume_stoichiometric_if_missing: bool = True,
    ) -> pd.DataFrame:
        """
        对输入的 DataFrame (如 ml_qspr_model_tg_c.csv) 进行一键机理特征衍生与增强
        支持一并提取各单体的 MMFF 能量分项与 xTB 电子结构特征
        """
        df_out = df.copy()

        # 检查是否可以与 wide_df 对齐补充 phr/ew
        has_aligned_wide = False
        if wide_df is not None and len(wide_df) == len(df_out):
            has_aligned_wide = True

        results = []
        total = len(df_out)

        # 组分列模式识别：严格自动识别树脂端与固化剂/添加剂端的所有槽位 (1..N)
        # 支持: resin_1_structure, resin_smiles_1, epoxy_1, diluent_structure, etc.
        resin_pat = re.compile(r"^(?:resin|epoxy|diluent|toughener)(?:_smiles|_structure)?(?:_\d+)?$", re.IGNORECASE)
        resin_smi_cols = [c for c in df_out.columns if resin_pat.match(str(c)) and not str(c).endswith(('_phr', '_mw', '_ew'))]
        if not resin_smi_cols:
            resin_smi_cols = [c for c in df_out.columns if ("resin" in str(c).lower() and ("smiles" in str(c).lower() or "structure" in str(c).lower()))]

        # 固化剂端：支持 curing_agent_1, hardener_1, curer_1, accelerator_1 等小分子槽位
        curer_pat = re.compile(r"^(?:curing_agent|hardener|curer|curing|accelerator|catalyst)(?:_smiles|_structure)?(?:_\d+)?$", re.IGNORECASE)
        curer_smi_cols = [c for c in df_out.columns if curer_pat.match(str(c)) and not str(c).endswith(('_phr', '_mw', '_ew'))]
        if not curer_smi_cols:
            curer_smi_cols = [c for c in df_out.columns if ("cur" in str(c).lower() or "hardener" in str(c).lower()) and ("smiles" in str(c).lower() or "structure" in str(c).lower())]

        # 排序保证主组分 (1) 排在最前
        def _sort_slot(c_name):
            m = re.search(r"\d+", str(c_name))
            return int(m.group()) if m else 1

        resin_smi_cols = sorted(resin_smi_cols, key=_sort_slot)
        curer_smi_cols = sorted(curer_smi_cols, key=_sort_slot)

        # [R 修复] 真实 R 列逐行回退链：优先数据集自带 R，缺失时用反应提取链路算出的 R
        r_value_candidates = [c for c in (
            "formulation_r_value",
            "formulation_resin_hardener_equivalent_ratio",
            "crosslink_stoichiometry_r",
            "stoichiometric_ratio_r_cleaned",
            "r_value",
            "cp_r_value",
            "r_actual",
            "rxn_r_actual",
            "rxn_r_effective",
        ) if c in df_out.columns]
        r_value_col = r_value_candidates[0] if r_value_candidates else None
        r_phr_col = "resin_total_phr" if "resin_total_phr" in df_out.columns else None
        h_phr_col = "curing_agent_total_phr" if "curing_agent_total_phr" in df_out.columns else None

        # 缓存单分子计算结果以极大加速计算
        mol_cache: Dict[str, Dict[str, Any]] = {}

        # 尝试初始化 MMFF 与 xTB 提取器
        ff_extractor = None
        if include_ff_features:
            try:
                from core.molecular_features import QuickForceFieldFeatureExtractor
                ff_extractor = QuickForceFieldFeatureExtractor(ff_mode="auto", max_iters=100)
            except Exception:
                ff_extractor = None

        xtb_extractor = None
        if include_xtb_features:
            try:
                from core.molecular_features import XTBFeatureExtractor
                xtb_extractor = XTBFeatureExtractor(xtb_path="xtb")
                if not getattr(xtb_extractor, "AVAILABLE", False):
                    xtb_extractor = None
            except Exception:
                xtb_extractor = None

        # 单分子特征缓存
        single_ff_cache: Dict[str, Dict[str, Any]] = {}
        single_xtb_cache: Dict[str, Dict[str, Any]] = {}

        # 🚀 [高性能预热] 收集全表中所有不重复的单体 SMILES，进行批量并发预提取
        # 避免在后续逐行遍历中重复串行启动外部进程与 3D 优化
        all_unique_smiles = set()
        for c in resin_smi_cols + curer_smi_cols:
            if c in df_out.columns:
                for val in df_out[c].dropna():
                    s_clean = self.clean_structure_string(val)
                    if s_clean:
                        all_unique_smiles.add(s_clean)

        unique_smi_list = list(all_unique_smiles)

        # 1. 批量预计算 xTB (如有可用 xTB，利用其内置的去重与多线程并发)
        if xtb_extractor is not None and unique_smi_list:
            try:
                n_cpu = min(os.cpu_count() or 1, 16)
                df_xtb_pre, valid_xtb_idx = xtb_extractor.featurize(unique_smi_list, n_jobs=n_cpu)
                for idx_pos, row_idx in enumerate(valid_xtb_idx):
                    smi_key = unique_smi_list[row_idx]
                    single_xtb_cache[smi_key] = df_xtb_pre.iloc[idx_pos].to_dict()
            except Exception:
                pass

        # 2. 批量预计算 MMFF 分项能量
        if ff_extractor is not None and unique_smi_list:
            try:
                for smi_key in unique_smi_list:
                    res_ff = ff_extractor._calc_features(smi_key)
                    if res_ff:
                        single_ff_cache[smi_key] = res_ff
            except Exception:
                pass

        for i in range(total):
            row = df_out.iloc[i]
            wide_row = wide_df.iloc[i] if has_aligned_wide else None

            # 收集该行专属的单体特征
            row_monomer_feats = {}

            # 1. 收集树脂组分 (支持从当前宽/窄表中直接读取各槽位用量 phr 与当量)
            resins = []
            for col_idx, col in enumerate(resin_smi_cols, start=1):
                smi = self.clean_structure_string(row.get(col))
                if smi:
                    weight = None
                    fb_ew = None
                    fb_mw = None

                    # 优先从当前行直接取 (如 resin_1_amount_phr, resin_2_phr)
                    for phr_pattern in [f"resin_{col_idx}_amount_phr", f"resin_{col_idx}_phr", f"resin_amount_phr_{col_idx}"]:
                        if phr_pattern in df_out.columns and not pd.isna(row.get(phr_pattern)) and float(row.get(phr_pattern)) > 0:
                            weight = float(row.get(phr_pattern))
                            break

                    # 次选从 wide_df 补充精确 PHR 与 EEW
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

                    # 检查是否有显式 xTB 列或实时提取
                    lumo_col = f"resin_{col_idx}_xtb_lumo"
                    if lumo_col in df_out.columns and not pd.isna(row[lumo_col]):
                        p["lumo"] = float(row[lumo_col])
                    homo_col_r = f"resin_{col_idx}_xtb_homo"
                    if homo_col_r in df_out.columns and not pd.isna(row[homo_col_r]):
                        p["homo"] = float(row[homo_col_r])
                    mu_col_r = f"resin_{col_idx}_xtb_chemical_potential"
                    if mu_col_r in df_out.columns and not pd.isna(row[mu_col_r]):
                        p["mu"] = float(row[mu_col_r])

                    # 若开启联动提取单体 MMFF 特征
                    if ff_extractor is not None:
                        if smi not in single_ff_cache:
                            ff_res = ff_extractor._calc_features(smi)
                            single_ff_cache[smi] = ff_res or {}
                        for k, v in single_ff_cache[smi].items():
                            row_monomer_feats[f"resin_{col_idx}_{k}"] = v

                    # 若开启联动提取单体 xTB 特征
                    if xtb_extractor is not None:
                        if smi not in single_xtb_cache:
                            xtb_res = xtb_extractor._calc_features(smi)
                            single_xtb_cache[smi] = xtb_res or {}
                        xtb_data = single_xtb_cache[smi]
                        for k, v in xtb_data.items():
                            row_monomer_feats[f"resin_{col_idx}_{k}"] = v
                        if "xtb_lumo" in xtb_data and xtb_data["xtb_lumo"] is not None:
                            p["lumo"] = float(xtb_data["xtb_lumo"])
                        if "xtb_homo" in xtb_data and xtb_data["xtb_homo"] is not None:
                            p["homo"] = float(xtb_data["xtb_homo"])
                        if "xtb_chemical_potential" in xtb_data and xtb_data["xtb_chemical_potential"] is not None:
                            p["mu"] = float(xtb_data["xtb_chemical_potential"])

                    p["weight"] = weight
                    p["col_idx"] = col_idx
                    resins.append(p)

            # 2. 收集固化剂组分 (同样优先支持当前行 phr 字段)
            curers = []
            for col_idx, col in enumerate(curer_smi_cols, start=1):
                smi = self.clean_structure_string(row.get(col))
                if smi:
                    weight = None
                    fb_ew = None
                    fb_mw = None

                    for phr_pattern in [f"curing_agent_{col_idx}_amount_phr", f"curing_agent_{col_idx}_phr", f"curer_{col_idx}_phr", f"accelerator_{col_idx}_phr"]:
                        if phr_pattern in df_out.columns and not pd.isna(row.get(phr_pattern)) and float(row.get(phr_pattern)) > 0:
                            weight = float(row.get(phr_pattern))
                            break

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
                    lumo_col_c = f"curing_agent_{col_idx}_xtb_lumo"
                    if lumo_col_c in df_out.columns and not pd.isna(row[lumo_col_c]):
                        p["lumo"] = float(row[lumo_col_c])
                    mu_col_c = f"curing_agent_{col_idx}_xtb_chemical_potential"
                    if mu_col_c in df_out.columns and not pd.isna(row[mu_col_c]):
                        p["mu"] = float(row[mu_col_c])

                    # 若开启联动提取单体 MMFF 特征
                    if ff_extractor is not None:
                        if smi not in single_ff_cache:
                            ff_res = ff_extractor._calc_features(smi)
                            single_ff_cache[smi] = ff_res or {}
                        for k, v in single_ff_cache[smi].items():
                            row_monomer_feats[f"curing_agent_{col_idx}_{k}"] = v

                    # 若开启联动提取单体 xTB 特征
                    if xtb_extractor is not None:
                        if smi not in single_xtb_cache:
                            xtb_res = xtb_extractor._calc_features(smi)
                            single_xtb_cache[smi] = xtb_res or {}
                        xtb_data = single_xtb_cache[smi]
                        for k, v in xtb_data.items():
                            row_monomer_feats[f"curing_agent_{col_idx}_{k}"] = v
                        if "xtb_homo" in xtb_data and xtb_data["xtb_homo"] is not None:
                            p["homo"] = float(xtb_data["xtb_homo"])
                        if "xtb_lumo" in xtb_data and xtb_data["xtb_lumo"] is not None:
                            p["lumo"] = float(xtb_data["xtb_lumo"])
                        if "xtb_chemical_potential" in xtb_data and xtb_data["xtb_chemical_potential"] is not None:
                            p["mu"] = float(xtb_data["xtb_chemical_potential"])

                    p["weight"] = weight
                    p["col_idx"] = col_idx
                    curers.append(p)

            # 3. 计算配方机理特征
            # [R 修复] 检查历史人工假定与来源传递 (P0-2)
            is_row_assumed = False
            if "rxn_r_assumed" in row and pd.to_numeric(row.get("rxn_r_assumed"), errors="coerce") == 1.0:
                is_row_assumed = True
            for sc in ("cp_r_source", "formulation_r_source", "r_source", "xl_r_source"):
                if row.get(sc) == "assumed_stoichiometric":
                    is_row_assumed = True

            row_r_src = None
            for sc in ("cp_r_source", "formulation_r_source", "r_source", "xl_r_source"):
                if sc in row and row.get(sc) is not None:
                    row_r_src = str(row.get(sc))
                    break

            given_r = None
            if not assume_stoichiometric_if_missing and is_row_assumed:
                # 严格模式下，历史人工假设不采信为有效 r
                given_r = None
            else:
                for _rc in r_value_candidates:
                    _rv = row.get(_rc)
                    if _rv is not None and not pd.isna(_rv) and float(_rv) > 0:
                        given_r = float(_rv)
                        break
            given_r_phr = row.get(r_phr_col) if r_phr_col else None
            given_h_phr = row.get(h_phr_col) if h_phr_col else None

            # 提取行级密度 (P0-3: 真实样本密度优先于历史生成密度缓存)
            row_rho = None
            for _rho_c in ("rho_g_cm3", "density_g_cm3", "density", "rho_used_for_mech_nu_g_cm3"):
                _v = row.get(_rho_c)
                if _v is not None and not pd.isna(_v):
                    try:
                        _vf = float(_v)
                        if np.isfinite(_vf) and _vf > 0:
                            row_rho = _vf
                            break
                    except (ValueError, TypeError):
                        pass

            feat = self.compute_formulation_mechanism_features(
                resins,
                curers,
                given_r_value=given_r,
                given_resin_total_phr=given_r_phr,
                given_curer_total_phr=given_h_phr,
                assume_stoichiometric_if_missing=assume_stoichiometric_if_missing,
                existing_r_source=row_r_src,
                rxn_r_assumed=1.0 if is_row_assumed else 0.0,
                rho_g_cm3=row_rho,
            )
            # 合并单体的 MMFF 与 xTB 特征
            feat.update(row_monomer_feats)
            results.append(feat)

            if progress_callback and (i + 1) % 10 == 0 or (i + 1) == total:
                try:
                    progress_callback(i + 1, total)
                except Exception:
                    pass

        feat_df = pd.DataFrame(results, index=df_out.index)

        # 幂等合并逻辑：更新已有同名列，追加新增特征列，严格杜绝重复列与 _x/_y 后缀 (P0-1)
        if not df_out.columns.is_unique:
            df_out = df_out.loc[:, ~df_out.columns.duplicated(keep="last")]

        existing_cols = [c for c in feat_df.columns if c in df_out.columns]
        new_cols = [c for c in feat_df.columns if c not in df_out.columns]

        for c in existing_cols:
            df_out[c] = feat_df[c]

        if new_cols:
            df_out = pd.concat([df_out, feat_df[new_cols]], axis=1)

        return df_out


def extract_epoxy_mechanism_features(
    df: pd.DataFrame,
    include_xtb: bool = False,
    progress_callback: Optional[Any] = None,
    assume_stoichiometric_if_missing: bool = True,
) -> pd.DataFrame:
    """提取环氧机理特征顶层快捷入口，保持与其它模块接口一致"""
    engine = EpoxyMechanismEngine(verbose=False)
    return engine.enrich_dataframe(
        df,
        progress_callback=progress_callback,
        include_ff_features=False,
        include_xtb_features=include_xtb,
        assume_stoichiometric_if_missing=assume_stoichiometric_if_missing,
    )
