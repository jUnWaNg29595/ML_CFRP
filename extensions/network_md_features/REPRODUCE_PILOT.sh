#!/bin/bash
# ==============================================================================
# REPRODUCE_PILOT.sh: 小规模文献配方试验与物理特征提取一键复现脚本
# ==============================================================================
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "======================================================================"
echo ">>> 1. 运行文献配方适用域筛选与 Dry-Run 计划生成"
echo "======================================================================"
/home/wangj/miniconda3/envs/htpolynet_env/bin/python - << "EOF"
import os, sys
sys.path.insert(0, ".")
from src.formulation_adapter import FormulationAdapter, CANONICAL_MONOMER_LIBRARY
from src.qm_adapter import QMParameterAdapter
import json

adapter = FormulationAdapter(target_atom_budget=1500)
qm_adapter = QMParameterAdapter()

candidates_def = [
    {
        "candidate_id": "PILOT_01_DGEBA_DDS_r1.0",
        "source_record_id": "epoxy_dgeba_dds_prod_dataset.csv:PROD_DGEBA_DDS_r1.0",
        "resin_name": "DGEBA",
        "resin_smiles": CANONICAL_MONOMER_LIBRARY["DGEBA"]["smiles"],
        "hardener_name": "4,4-DDS",
        "hardener_smiles": CANONICAL_MONOMER_LIBRARY["4,4-DDS"]["smiles"],
        "nominal_r": 1.0,
        "description": "双酚A二缩水甘油醚 / 4,4-二氨基二苯砜 (1:1 当量经典基准配方)"
    },
    {
        "candidate_id": "PILOT_02_DGEBA_PACM_r1.0",
        "source_record_id": "htpolynet_depot:3-pacm-dgeba",
        "resin_name": "DGEBA",
        "resin_smiles": CANONICAL_MONOMER_LIBRARY["DGEBA"]["smiles"],
        "hardener_name": "PACM",
        "hardener_smiles": CANONICAL_MONOMER_LIBRARY["PACM"]["smiles"],
        "nominal_r": 1.0,
        "description": "双酚A二缩水甘油醚 / 4,4-亚甲基双环己胺 (脂环固化剂基准)"
    },
    {
        "candidate_id": "PILOT_03_DGEBA_DDM_r1.0",
        "source_record_id": "PINN筛选清单:Row_DDM_baseline",
        "resin_name": "DGEBA",
        "resin_smiles": CANONICAL_MONOMER_LIBRARY["DGEBA"]["smiles"],
        "hardener_name": "DDM",
        "hardener_smiles": CANONICAL_MONOMER_LIBRARY["DDM"]["smiles"],
        "nominal_r": 1.0,
        "description": "双酚A二缩水甘油醚 / 4,4-二氨基二苯甲烷 (芳香二胺)"
    },
    {
        "candidate_id": "PILOT_04_DGEDHN_DDS_r1.0",
        "source_record_id": "epoxy_1000_formulations_design.csv:FORM_0001",
        "resin_name": "DGE-DHN",
        "resin_smiles": CANONICAL_MONOMER_LIBRARY["DGE-DHN"]["smiles"],
        "hardener_name": "4,4-DDS",
        "hardener_smiles": CANONICAL_MONOMER_LIBRARY["4,4-DDS"]["smiles"],
        "nominal_r": 1.0,
        "description": "1,5-萘二酚二缩水甘油醚 / 4,4-二氨基二苯砜 (稠环高耐热体系)"
    },
    {
        "candidate_id": "PILOT_05_DGEDHN_DDM_r1.0",
        "source_record_id": "epoxy_1000_formulations_design.csv:FORM_0002",
        "resin_name": "DGE-DHN",
        "resin_smiles": CANONICAL_MONOMER_LIBRARY["DGE-DHN"]["smiles"],
        "hardener_name": "DDM",
        "hardener_smiles": CANONICAL_MONOMER_LIBRARY["DDM"]["smiles"],
        "nominal_r": 1.0,
        "description": "1,5-萘二酚二缩水甘油醚 / 4,4-二氨基二苯甲烷 (稠环耐温芳香交联)"
    },
    {
        "candidate_id": "PILOT_06_DGEBA_DDS_r0.8",
        "source_record_id": "epoxy_dgeba_dds_prod_dataset.csv:PROD_DGEBA_DDS_r0.80",
        "resin_name": "DGEBA",
        "resin_smiles": CANONICAL_MONOMER_LIBRARY["DGEBA"]["smiles"],
        "hardener_name": "4,4-DDS",
        "hardener_smiles": CANONICAL_MONOMER_LIBRARY["4,4-DDS"]["smiles"],
        "nominal_r": 0.8,
        "description": "双酚A二缩水甘油醚 / 4,4-二氨基二苯砜 (非等当量欠固化对照 r=0.80)"
    }
]

screening_results = []
os.makedirs("reports", exist_ok=True)
os.makedirs("artifacts/run_plans", exist_ok=True)

for cand in candidates_def:
    is_valid, domain_msg = adapter.check_applicability_domain(
        cand["resin_smiles"], cand["hardener_smiles"], cand["nominal_r"]
    )
    discrete_info = adapter.calculate_discrete_formulation(
        resin_smiles=cand["resin_smiles"],
        hardener_smiles=cand["hardener_smiles"],
        r=cand["nominal_r"],
        resin_name=cand["resin_name"],
        hardener_name=cand["hardener_name"]
    )
    cand["screening_status"] = "PASS" if is_valid else "FAIL"
    cand["domain_message"] = domain_msg
    cand["discrete_formulation"] = discrete_info
    screening_results.append(cand)

with open("reports/pilot_formulation_screening.json", "w", encoding="utf-8") as f:
    json.dump(screening_results, f, indent=2, ensure_ascii=False)
with open("artifacts/run_plans/pilot_run_plans.json", "w", encoding="utf-8") as f:
    json.dump(screening_results, f, indent=2, ensure_ascii=False)
print("Screening and run plans generated.")
EOF

echo ""
echo "======================================================================"
echo ">>> 2. 运行真实小试试验与 17 项物理特征提取"
echo "======================================================================"
/home/wangj/miniconda3/envs/htpolynet_env/bin/python /mnt/c/Users/wangj/.pi-desktop/scratch/bdceec79-692a-4cdb-b7a1-bf98573f51ee/run_pilot_pipeline.py

echo ""
echo ">>> 小试全流程执行完毕！产物导出至 artifacts/exports/ 与 reports/。"
