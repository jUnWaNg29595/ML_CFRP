# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from rdkit import Chem
from core.epoxy_mechanism_features import EpoxyMechanismEngine
import core.crosslink_physics as cp

engine = EpoxyMechanismEngine(verbose=False)

print("--- Test 1: 官能团识别与 SMARTS 测试 ---")
mol_acetamide = Chem.MolFromSmiles("CC(N)=O")
h_count, c_type = engine.get_active_hydrogen_count("CC(N)=O", mol_acetamide)
print(f"乙酰胺 CC(N)=O: count={h_count}, type={c_type}")
props_ethanol = engine.calc_single_molecule_properties("CCO", is_resin=True)
print(f"乙醇 CCO (is_resin=True): functionality={props_ethanol['functionality']}")
props_invalid = engine.calc_single_molecule_properties("invalid_smi", is_resin=True)
print(f"无效 SMILES: functionality={props_invalid['functionality']}, ew={props_invalid['ew']}")

print("\n--- Test 2: 多组分平均官能度测试 ---")
r_dict = engine.calc_single_molecule_properties("C1OC1", is_resin=True)
r_dict.update({"mw": 340.0, "functionality": 2.0, "weight": 100.0, "ew": 170.0})
c_dict = engine.calc_single_molecule_properties("NCCN", is_resin=False)
c_dict.update({"mw": 116.0, "functionality": 4.0, "weight": 17.0, "ew": 29.0})
feats = engine.compute_formulation_mechanism_features(
    [r_dict], [c_dict],
    given_resin_total_phr=100.0,
    given_curer_total_phr=17.0
)
print(f"多组分 f_avg: {feats.get('mech_average_functionality')} (期望约为 2.665)")

print("\n--- Test 3: 单官能环氧不凝胶体系测试 ---")
df_mono = pd.DataFrame([{
    "resin_1_molecular_weight_g_mol": 130.0,
    "resin_1_amount_phr": 100.0,
    "resin_1_epoxide_equivalent_weight_g_eq": 130.0,
    "resin_1_functionality": 1.0,
    "curing_agent_1_molecular_weight_g_mol": 116.0,
    "curing_agent_1_amount_phr": 22.3,
    "curing_agent_1_amine_hydrogen_equivalent_weight_g_eq": 29.0,
    "curing_agent_1_active_hydrogen_functionality": 4.0,
    "r_value": 1.0
}])
res = cp.compute_crosslink_features(df_mono)
print(f"单官能环氧 nu_theory: {res['xl_nu_theory_mol_m3'].iloc[0]} mol/m3")

print("\n--- Test 4: DataFrame 索引对齐测试 ---")
df_custom = pd.DataFrame([{
    "resin_1_smiles": "C1OC1",
    "curing_agent_1_smiles": "NCCN"
}], index=["sample-42"])
enriched = engine.enrich_dataframe(df_custom)
print(f"输入 1 行 sample-42，输出行数: {len(enriched)}, index: {list(enriched.index)}")

print("\n--- Test 5: 当量比逐列回退测试 ---")
df_r = pd.DataFrame({
    "formulation_r_value": [np.nan, 1.2],
    "formulation_resin_hardener_equivalent_ratio": [2.0, 1.5]
})
r_series = cp._stoich_r(df_r)
print(f"第 0 行 r_series.iloc[0]: {r_series.iloc[0]} (期望为 2.0)")
