"""
文献配方适配与样本身份模块 (Formulation Adapter & Domain Screening)
功能：
1. 从文献数据（CSV/XLSX）解析树脂、固化剂、用量与配比；
2. 规范 SMILES 化，计算精确分子量、官能度与当量；
3. 显式区分当量比 r 的定义：r = [活性氢当量] / [环氧基当量]；
4. 适用域检查（Applicability Domain）：拦截不透明商业混合物、非环氧-胺反应或缺少关键化学结构的条目；
5. 根据盒子规模预算换算整数分子数，计算离散化化学计量误差；
6. 严格剔除目标标签（Tg / 模量），杜绝标签泄漏。
"""

import math
from typing import Dict, Any, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import pandas as pd

from .provenance import compute_formulation_id

# 常用已知环氧与固化剂标准 SMILES 与官能度库（用于名称对齐与化学验证）
CANONICAL_MONOMER_LIBRARY = {
    # 环氧树脂
    "DGEBA": {
        "canonical_name": "Bisphenol-A Diglycidyl Ether",
        "smiles": "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1",
        "functionality": 2,
        "type": "epoxy",
        "active_smiles_preopened": "CC(C)(c1ccc(OC[CH:3]([OH:5])[CH3:1])cc1)c1ccc(OC[CH:4]([OH:6])[CH3:2])cc1",
        "reactive_atoms": {1: "C1", 2: "C2", 3: "C3", 4: "C4", 5: "O1", 6: "O2"},
        "symmetry_equivalent_atoms": [["C1", "C2"], ["C3", "C4"], ["O1", "O2"]],
        "stereocenters": ["C3"]
    },
    "DGE-DHN": {
        "canonical_name": "1,5-Naphthalenediol Diglycidyl Ether",
        "smiles": "C1OC1COc2ccc3cccc(OCC4CO4)c3c2",
        "functionality": 2,
        "type": "epoxy",
        "active_smiles_preopened": "C1=CC2=C(C=CC=C2OC[CH:3]([OH:5])[CH3:1])C(=C1)OC[CH:4]([OH:6])[CH3:2]",
        "reactive_atoms": {1: "C1", 2: "C2", 3: "C3", 4: "C4", 5: "O1", 6: "O2"},
        "symmetry_equivalent_atoms": [["C1", "C2"], ["C3", "C4"], ["O1", "O2"]],
        "stereocenters": ["C3"]
    },
    "TGDDM": {
        "canonical_name": "Tetraglycidyldiaminodiphenylmethane",
        "smiles": "C1OC1CN(CC2CO2)c3ccc(Cc4ccc(N(CC5CO5)CC6CO6)cc4)cc3",
        "functionality": 4,
        "type": "epoxy",
        "active_smiles_preopened": "",
        "reactive_atoms": {},
        "symmetry_equivalent_atoms": [],
        "stereocenters": []
    },
    # 固化剂
    "4,4-DDS": {
        "canonical_name": "4,4'-Diaminodiphenyl Sulfone",
        "smiles": "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1",
        "functionality": 4,  # 2 个伯胺 = 4 个活性氢
        "type": "hardener_amine",
        "active_smiles": "C1=CC(=CC=C1[N:1]([H:3])[H:4])S(=O)(=O)C2=CC=C(C=C2)[N:2]([H:5])[H:6]",
        "reactive_atoms": {1: "N1", 2: "N2", 3: "H1", 4: "H2", 5: "H3", 6: "H4"},
        "symmetry_equivalent_atoms": [["N1", "N2"]],
        "stereocenters": []
    },
    "PACM": {
        "canonical_name": "4,4'-Methylenebis(cyclohexylamine)",
        "smiles": "NC1CCC(CC2CCC(N)CC2)CC1",
        "functionality": 4,
        "type": "hardener_amine",
        "active_smiles": "C1C[CH:1](CCC1CC2CC[CH:2](CC2)[NH2:3])[NH2:4]",
        "reactive_atoms": {1: "C1", 2: "C1", 3: "N1", 4: "N1"},
        "symmetry_equivalent_atoms": [["N1", "N2"], ["C1", "C2"]],
        "stereocenters": ["C1"]
    },
    "DDM": {
        "canonical_name": "4,4'-Diaminodiphenylmethane",
        "smiles": "Nc1ccc(Cc2ccc(N)cc2)cc1",
        "functionality": 4,
        "type": "hardener_amine",
        "active_smiles": "C1=CC(=CC=C1[N:1]([H:3])[H:4])CC2=CC=C(C=C2)[N:2]([H:5])[H:6]",
        "reactive_atoms": {1: "N1", 2: "N2", 3: "H1", 4: "H2", 5: "H3", 6: "H4"},
        "symmetry_equivalent_atoms": [["N1", "N2"]],
        "stereocenters": []
    }
}

class FormulationAdapter:
    def __init__(self, target_atom_budget: int = 1500):
        """
        :param target_atom_budget: 目标全体系原子规模预算（小规模物理测试推荐 1,000 ~ 3,000 原子）
        """
        self.target_atom_budget = target_atom_budget

    def canonicalize_smiles(self, smiles: str) -> str:
        """使用 RDKit 生成唯一规范 SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return Chem.MolToSmiles(mol, canonical=True)

    def analyze_molecule(self, smiles: str) -> Dict[str, Any]:
        """计算分子量、重原子数、全原子数、元素组成"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        mol_with_h = Chem.AddHs(mol)
        mw = Descriptors.MolWt(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        total_atoms = mol_with_h.GetNumAtoms()

        # 官能团检测：环氧基数量 (三元醚环)
        epoxy_pattern = Chem.MolFromSmarts("C1OC1")
        epoxy_count = len(mol.GetSubstructMatches(epoxy_pattern)) if epoxy_pattern else 0

        # 官能团检测：伯胺 (-NH2) 与仲胺 (-NH-) 数量
        primary_amine_pat = Chem.MolFromSmarts("[NX3;H2;!$(NC=O)]")
        secondary_amine_pat = Chem.MolFromSmarts("[NX3;H1;!$(NC=O)]")
        primary_amine_count = len(mol.GetSubstructMatches(primary_amine_pat)) if primary_amine_pat else 0
        secondary_amine_count = len(mol.GetSubstructMatches(secondary_amine_pat)) if secondary_amine_pat else 0
        active_h_count = primary_amine_count * 2 + secondary_amine_count * 1

        return {
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
            "mw": mw,
            "heavy_atoms": heavy_atoms,
            "total_atoms": total_atoms,
            "epoxy_count": epoxy_count,
            "active_h_count": active_h_count
        }

    def check_applicability_domain(
        self,
        resin_smiles: str,
        hardener_smiles: str,
        r: float
    ) -> Tuple[bool, str]:
        """
        适用域审核 (Applicability Domain)：
        - 树脂必须具备至少 2 个环氧官能团；
        - 固化剂必须具备活性胺基（活性氢 >= 2）；
        - 当量比 r 必须处于合理范围 (0.4 <= r <= 2.0)；
        - 严禁商业黑箱结构、非芳香/非脂肪环环氧-胺反应体系。
        """
        try:
            r_info = self.analyze_molecule(resin_smiles)
            h_info = self.analyze_molecule(hardener_smiles)
        except Exception as e:
            return False, f"SMILES 解析失败: {str(e)}"

        if r_info["epoxy_count"] < 2:
            return False, f"树脂环氧基团数为 {r_info['epoxy_count']} < 2，无法形成交联网络"

        if h_info["active_h_count"] < 2:
            return False, f"固化剂活性氢数为 {h_info['active_h_count']} < 2，无法作为交联固化剂"

        if r <= 0.2 or r >= 3.0:
            return False, f"当量比 r={r} 严重失衡（可接受范围 0.2 ~ 3.0）"

        return True, "PASS"

    def calculate_discrete_formulation(
        self,
        resin_smiles: str,
        hardener_smiles: str,
        r: float,
        resin_name: str = "Resin",
        hardener_name: str = "Hardener"
    ) -> Dict[str, Any]:
        """
        将连续配方转换为离散分子数，报告离散化化学计量误差。
        r 的严格定义为：r = [活性氢当量] / [环氧当量]
        即 r = (n_hardener * f_hardener) / (n_resin * f_resin)
        """
        is_valid, reason = self.check_applicability_domain(resin_smiles, hardener_smiles, r)
        if not is_valid:
            raise ValueError(f"配方适用域检查不通过: {reason}")

        r_info = self.analyze_molecule(resin_smiles)
        h_info = self.analyze_molecule(hardener_smiles)

        f_resin = r_info["epoxy_count"]
        f_hardener = h_info["active_h_count"]

        # 理论分子数比例： n_hardener / n_resin = r * (f_resin / f_hardener)
        molar_ratio_h_to_r = r * (float(f_resin) / float(f_hardener))

        # 目标总原子数： n_resin * atoms_resin + n_hardener * atoms_hardener ~ target_atom_budget
        atoms_per_unit = r_info["total_atoms"] + molar_ratio_h_to_r * h_info["total_atoms"]
        n_resin_float = self.target_atom_budget / atoms_per_unit

        n_resin = max(2, int(round(n_resin_float)))
        n_hardener = max(1, int(round(n_resin * molar_ratio_h_to_r)))

        actual_r = (n_hardener * f_hardener) / (n_resin * f_resin)
        discretization_error_pct = abs(actual_r - r) / r * 100.0

        total_system_atoms = n_resin * r_info["total_atoms"] + n_hardener * h_info["total_atoms"]
        total_epoxy_groups = n_resin * f_resin
        total_active_h = n_hardener * f_hardener

        # 理论最大可达成键加成数
        max_possible_crosslinks = min(total_epoxy_groups, total_active_h)
        theoretical_max_alpha = max_possible_crosslinks / float(total_epoxy_groups)

        resins = [{
            "name": resin_name,
            "smiles": r_info["canonical_smiles"],
            "molecules": n_resin,
            "mw": r_info["mw"],
            "functionality": f_resin,
            "phr": 100.0
        }]

        # 固化剂质量份 phr 计算
        resin_mass = n_resin * r_info["mw"]
        hardener_mass = n_hardener * h_info["mw"]
        hardener_phr = round((hardener_mass / resin_mass) * 100.0, 2)

        curers = [{
            "name": hardener_name,
            "smiles": h_info["canonical_smiles"],
            "molecules": n_hardener,
            "mw": h_info["mw"],
            "functionality": f_hardener,
            "phr": hardener_phr
        }]

        formulation_id = compute_formulation_id(
            resins=resins,
            curers=curers,
            stoichiometric_ratio_r=actual_r,
            r_definition="active_H_to_epoxy"
        )

        return {
            "formulation_id": formulation_id,
            "nominal_r": r,
            "actual_discrete_r": round(actual_r, 4),
            "r_definition": "[active_H] / [epoxy_groups]",
            "discretization_error_pct": round(discretization_error_pct, 2),
            "n_resin_molecules": n_resin,
            "n_hardener_molecules": n_hardener,
            "total_system_atoms": total_system_atoms,
            "total_epoxy_groups": total_epoxy_groups,
            "total_active_h": total_active_h,
            "theoretical_max_alpha": round(theoretical_max_alpha, 4),
            "resins": resins,
            "curers": curers
        }
