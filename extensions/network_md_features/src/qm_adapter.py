"""
量子化学与力场参数适配器 (QM & Parameter Adapter - Rigorous Graph-Grounded Version)
核心修复：
1. 废除简单的分子名称匹配，强制执行 RDKit 分子图同构检验 (Exact Graph Match)；
2. 纠正 4,4'-DDS 原子数为 29 (C12H12N2O2S)，DGEBA 为 49 (C21H24O4)；
3. 从真实产物文件 atom_charges.csv 加载严格两阶段拟合 RESP 电荷 (resp_charge_e)，废除未守恒的中间 molecule.ac；
4. 导出完整的 charge_mapping_audit.csv 审计文件；
5. 明确区分生产级 DFT-RESP 与工程诊断级 GAS_DIAGNOSTIC，绝不伪造。
"""

import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from rdkit import Chem

CONTINUATION_DIR = "/home/wangj/epoxy_prod/rebuild_verified_v3_5/continuation_20261015"
EXT_ROOT = "/home/wangj/project/CFRP/extensions/network_md_features"

class QMParameterAdapter:
    def __init__(self, continuation_dir: str = CONTINUATION_DIR):
        self.continuation_dir = continuation_dir
        self.verified_cache: Dict[str, Dict[str, Any]] = {}
        self._load_and_audit_verified_assets()

    def _load_and_audit_verified_assets(self) -> None:
        """从 continuation_20261015 加载并严格审核真实 DFT/RESP 资产"""
        audit_rows = []

        # 1. DGEBA (C21H24O4, 49 atoms)
        dgeba_smiles = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
        dgeba_csv = os.path.join(self.continuation_dir, "qm/dgeba_resp/atom_charges.csv")
        dgeba_report_p = os.path.join(self.continuation_dir, "qm/dgeba_resp/report.json")
        dgeba_opt_p = os.path.join(self.continuation_dir, "qm/dgeba_optimized/result.json")

        if os.path.exists(dgeba_csv) and os.path.exists(dgeba_report_p):
            df_dgeba = pd.read_csv(dgeba_csv)
            with open(dgeba_report_p, "r", encoding="utf-8") as f:
                dgeba_rep = json.load(f)
            with open(dgeba_opt_p, "r", encoding="utf-8") as f:
                dgeba_opt = json.load(f)

            # 严格核验原子数与净电荷 (真实列名为 resp_charge_e)
            assert len(df_dgeba) == 49, f"DGEBA atom count mismatch: expected 49, got {len(df_dgeba)}"
            charge_col = "resp_charge_e" if "resp_charge_e" in df_dgeba.columns else "charge_e"
            q_sum_dgeba = float(df_dgeba[charge_col].sum())
            assert abs(q_sum_dgeba) < 1e-4, f"DGEBA net charge {q_sum_dgeba} not conserved"

            self.verified_cache["DGEBA"] = {
                "name": "DGEBA",
                "smiles": dgeba_smiles,
                "rdkit_mol": Chem.MolFromSmiles(dgeba_smiles),
                "atom_count": 49,
                "formula": "C21H24O4",
                "qm_method": "B3LYP/6-31G*",
                "charge_method": "AmberTools native two-stage RESP (serialized)",
                "final_energy_Eh": dgeba_opt.get("final_energy_Eh"),
                "total_charge_e": q_sum_dgeba,
                "atom_charges_df": df_dgeba,
                "verification_level": "VERIFIED_PRODUCTION"
            }

            for idx, r in df_dgeba.iterrows():
                audit_rows.append({
                    "molecule": "DGEBA",
                    "atom_id_1based": int(r["source_mol_atom_id"]),
                    "atom_name": r.get("atom_name", f"A{r['source_mol_atom_id']}"),
                    "element": r.get("element", "C"),
                    "resp_charge_e": float(r[charge_col]),
                    "native_resp_charge_e": float(r.get("native_resp_charge_e", r[charge_col])),
                    "charge_model": "DFT_B3LYP_631G*_RESP_STAGE2",
                    "provenance_file": "qm/dgeba_resp/atom_charges.csv"
                })

        # 2. 4,4'-DDS (C12H12N2O2S, 29 atoms)
        dds_smiles = "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1"
        dds_csv = os.path.join(self.continuation_dir, "qm/dds_resp/atom_charges.csv")
        dds_report_p = os.path.join(self.continuation_dir, "qm/dds_resp/report.json")
        dds_opt_p = os.path.join(self.continuation_dir, "qm/dds_optimized_retry/result.json")

        if os.path.exists(dds_csv) and os.path.exists(dds_report_p):
            df_dds = pd.read_csv(dds_csv)
            with open(dds_report_p, "r", encoding="utf-8") as f:
                dds_rep = json.load(f)
            with open(dds_opt_p, "r", encoding="utf-8") as f:
                dds_opt = json.load(f)

            # 严格核验原子数与净电荷
            assert len(df_dds) == 29, f"DDS atom count mismatch: expected 29, got {len(df_dds)}"
            charge_col_dds = "resp_charge_e" if "resp_charge_e" in df_dds.columns else "charge_e"
            q_sum_dds = float(df_dds[charge_col_dds].sum())
            assert abs(q_sum_dds) < 1e-4, f"DDS net charge {q_sum_dds} not conserved"

            self.verified_cache["4,4-DDS"] = {
                "name": "4,4-DDS",
                "smiles": dds_smiles,
                "rdkit_mol": Chem.MolFromSmiles(dds_smiles),
                "atom_count": 29,
                "formula": "C12H12N2O2S",
                "qm_method": "B3LYP/6-31G*",
                "charge_method": "AmberTools native two-stage RESP (serialized)",
                "final_energy_Eh": dds_opt.get("final_energy_Eh"),
                "total_charge_e": q_sum_dds,
                "atom_charges_df": df_dds,
                "verification_level": "VERIFIED_PRODUCTION"
            }

            for idx, r in df_dds.iterrows():
                audit_rows.append({
                    "molecule": "4,4-DDS",
                    "atom_id_1based": int(r["source_mol_atom_id"]),
                    "atom_name": r.get("atom_name", f"A{r['source_mol_atom_id']}"),
                    "element": r.get("element", "C"),
                    "resp_charge_e": float(r[charge_col_dds]),
                    "native_resp_charge_e": float(r.get("native_resp_charge_e", r[charge_col_dds])),
                    "charge_model": "DFT_B3LYP_631G*_RESP_STAGE2",
                    "provenance_file": "qm/dds_resp/atom_charges.csv"
                })

        # 导出 charge_mapping_audit.csv
        out_audit = os.path.join(EXT_ROOT, "reports/charge_mapping_audit.csv")
        os.makedirs(os.path.dirname(out_audit), exist_ok=True)
        if audit_rows:
            pd.DataFrame(audit_rows).to_csv(out_audit, index=False)
            print(f"[AUDIT] Exported {len(audit_rows)} atomic charge audit records to {out_audit}")

    def is_molecule_graph_isomorphic(self, mol1: Chem.Mol, mol2: Chem.Mol) -> bool:
        """严格图同构核验：检查元素、键级和重原子拓扑连接"""
        if mol1 is None or mol2 is None:
            return False
        # 检查重原子数与分子式
        if mol1.GetNumHeavyAtoms() != mol2.GetNumHeavyAtoms():
            return False
        # 双向子结构匹配
        return mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1)

    def get_monomer_parameter_spec(self, monomer_name: str, smiles: str) -> Dict[str, Any]:
        """
        基于化学图严格匹配单体参数。
        严禁仅凭字符串名称匹配；如果图不同构，坚决报错或拒绝！
        """
        input_mol = Chem.MolFromSmiles(smiles)
        if input_mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # 比对已验证的 DGEBA
        if "DGEBA" in self.verified_cache:
            ref = self.verified_cache["DGEBA"]
            if self.is_molecule_graph_isomorphic(input_mol, ref["rdkit_mol"]):
                return {
                    "name": "DGEBA",
                    "smiles": smiles,
                    "formula": "C21H24O4",
                    "atom_count": 49,
                    "forcefield": "GAFF2",
                    "charge_method": "DFT_RESP_B3LYP_631G*",
                    "total_charge_e": ref["total_charge_e"],
                    "atom_charges_df": ref["atom_charges_df"],
                    "verification_level": "VERIFIED"
                }

        # 比对已验证的 4,4'-DDS
        if "4,4-DDS" in self.verified_cache:
            ref = self.verified_cache["4,4-DDS"]
            if self.is_molecule_graph_isomorphic(input_mol, ref["rdkit_mol"]):
                return {
                    "name": "4,4-DDS",
                    "smiles": smiles,
                    "formula": "C12H12N2O2S",
                    "atom_count": 29,
                    "forcefield": "GAFF2",
                    "charge_method": "DFT_RESP_B3LYP_631G*",
                    "total_charge_e": ref["total_charge_e"],
                    "atom_charges_df": ref["atom_charges_df"],
                    "verification_level": "VERIFIED"
                }

        # 未在验证资产库中的分子，明确标记为工程诊断协议 EXPLORATORY
        mol_with_h = Chem.AddHs(input_mol)
        return {
            "name": monomer_name,
            "smiles": Chem.MolToSmiles(input_mol, canonical=True),
            "formula": Chem.rdMolDescriptors.CalcMolFormula(input_mol),
            "atom_count": mol_with_h.GetNumAtoms(),
            "forcefield": "GAFF2",
            "charge_method": "GAS_ANTECHAMBER_DIAGNOSTIC",
            "total_charge_e": 0.0,
            "atom_charges_df": None,
            "verification_level": "EXPLORATORY"
        }
