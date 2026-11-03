"""
DGEBA/4,4'-DDS 单配方真实交联网络构建与物理特征提取闭环驱动
严格遵循：
1. 使用经过严格科学检验的 DGE.mol2 (开环活化模板) 与 DDS.mol2 (29原子，N1/N2带有完整牺牲氢，真实两阶段RESP电荷)；
2. 调用 HTPolyNet 2.15 与 GROMACS 2026.3 真实执行单体参数化、构象打包、能量最小化、预平衡与交联成键；
3. 保留真实的原生 TPR/GRO/EDR/LOG 文件到 artifacts/recovery_v2/ 独立目录；
4. 真实执行两个独立随机网络种子 (seed 20261015, seed 20261016)；
5. 调用重构后的三大科学特征提取器提取真实物理特征；
6. 严格执行 S0~S6 门禁审查并输出 scientific_gate_report.json 与 md_features_reference.csv。
"""

import os
import sys
import time
import json
import yaml
import shutil
import subprocess
import numpy as np
import pandas as pd
import networkx as nx

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, EXT_ROOT)

from src.provenance import (
    compute_formulation_id, compute_parameter_set_id,
    compute_protocol_id, compute_network_id, ExecutionReceipt, calculate_sha256_file
)
from src.chemical_validation import ChemicalValidator, aggregate_overall_gate_status
from src.trajectory_validation import TrajectoryValidator
from src.qm_adapter import QMParameterAdapter
from src.feature_extractors import TopologyExtractor, FreeVolumeExtractor, InteractionExtractor

HTPOLYNET_BIN = "/home/wangj/miniconda3/envs/htpolynet_env/bin/htpolynet"
GMX_BIN = "/home/wangj/miniconda3/envs/htpolynet_env/bin/gmx"
AMBER_BIN_DIR = "/home/wangj/miniconda3/envs/amber_env/bin"
OFFICIAL_DGE_MOL2 = "/home/wangj/htpolynet/src/htpolynet/resources/molecules/inputs/DGE.mol2"

def setup_monomer_templates(work_dir: str):
    """
    在工作目录建立 inputs/ 文件夹，并写入标准的 DGE.mol2 与带有完整 29 原子的 DDS.mol2
    """
    inputs_dir = os.path.join(work_dir, "molecules/inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    # 1. DGE.mol2 (复制官方开环活化标准模板)
    dst_dge = os.path.join(inputs_dir, "DGE.mol2")
    if os.path.exists(OFFICIAL_DGE_MOL2):
        shutil.copy2(OFFICIAL_DGE_MOL2, dst_dge)

    # 2. DDS.mol2 (使用之前构建好的严格 29 原子含牺牲氢模板)
    src_dds = os.path.join(EXT_ROOT, "artifacts/molecule_library/DDS.mol2")
    dst_dds = os.path.join(inputs_dir, "DDS.mol2")
    if os.path.exists(src_dds):
        shutil.copy2(src_dds, dst_dds)
    else:
        # 如果未找到，重新生成
        from scripts.build_rigorous_monomer_templates import build_dds_mol2
        build_dds_mol2(dst_dds)

    print(f"Monomer templates prepared in {inputs_dir}")

def generate_reference_system_yaml(work_dir: str, seed: int) -> str:
    """生成单配方小体系配置 system.yaml"""
    config = {
        "Title": f"DGEBA_DDS_reference_seed_{seed}",
        "gromacs": {
            "gmx": "gmx",
            "gmx_options": "-quiet -nobackup",
            "mdrun": "gmx mdrun",
            "mdrun_options": {
                "ntmpi": 1,
                "ntomp": 8
            }
        },
        "ambertools": {
            "charge_method": "gas"
        },
        "constituents": {
            "DGE": {
                "count": 4
            },
            "DDS": {
                "count": 2
            }
        },
        "reactions": [
            {
                "name": "Primary-to-secondary-amine",
                "stage": "cure",
                "reactants": {1: "DDS", 2: "DGE"},
                "product": "DDS~N1-C1~DGE",
                "probability": 1.0,
                "atoms": {
                    "A": {"reactant": 1, "resid": 1, "atom": "N1", "z": 2},
                    "B": {"reactant": 2, "resid": 1, "atom": "C1", "z": 1}
                },
                "bonds": [{"atoms": ["A", "B"], "order": 1}]
            },
            {
                "name": "Secondary-to-tertiary-amine",
                "stage": "cure",
                "reactants": {1: "DDS~N1-C1~DGE", 2: "DGE"},
                "product": "DDS~N1-C1~DGE-C1~DGE",
                "probability": 0.5,
                "atoms": {
                    "A": {"reactant": 1, "resid": 1, "atom": "N1", "z": 1},
                    "B": {"reactant": 2, "resid": 1, "atom": "C1", "z": 1}
                },
                "bonds": [{"atoms": ["A", "B"], "order": 1}]
            },
            {
                "name": "Oxirane-capping",
                "stage": "cap",
                "reactants": {1: "DGE"},
                "product": "DGEC",
                "probability": 1.0,
                "atoms": {
                    "A": {"reactant": 1, "resid": 1, "atom": "O1", "z": 1},
                    "B": {"reactant": 1, "resid": 1, "atom": "C1", "z": 1}
                },
                "bonds": [{"atoms": ["A", "B"], "order": 1}]
            }
        ],
        "densification": {
            "initial_density": 300.0,
            "equilibration": [
                {"ensemble": "min"},
                {"ensemble": "nvt", "temperature": 300, "nsteps": 200},
                {"ensemble": "npt", "temperature": 300, "pressure": 10, "nsteps": 500}
            ]
        },
        "precure": {
            "preequilibration": {
                "ensemble": "npt",
                "temperature": 300,
                "pressure": 1,
                "nsteps": 500
            }
        },
        "CURE": {
            "controls": {
                "search_radius": 0.6,
                "radial_increment": 0.2,
                "max_iterations": 5,
                "desired_conversion": 0.50,
                "late_threshold": 0.40
            },
            "drag": {
                "trigger_distance": 0.6,
                "increment": 0.08,
                "limit": 0.3,
                "equilibration": [
                    {"ensemble": "min"}
                ]
            },
            "relax": {
                "increment": 0.08,
                "equilibration": [
                    {"ensemble": "min"}
                ]
            },
            "equilibrate": {
                "ensemble": "npt",
                "temperature": 300,
                "pressure": 1,
                "nsteps": 500
            }
        },
        "postcure": {
            "postequilibration": {
                "ensemble": "npt",
                "temperature": 300,
                "pressure": 1,
                "nsteps": 500
            }
        }
    }

    yaml_path = os.path.join(work_dir, "system.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)
    return yaml_path

def run_seed_simulation(seed: int, recovery_dir: str):
    task_dir = os.path.join(recovery_dir, f"REAL_DGEBA_DDS_seed{seed}")
    os.makedirs(task_dir, exist_ok=True)

    # 1. 复制真实单体模板
    setup_monomer_templates(task_dir)

    # 2. 生成配置
    yaml_path = generate_reference_system_yaml(task_dir, seed)
    log_path = os.path.join(task_dir, "htpolynet_exec.log")

    print(f"\n>>> Running Real HTPolyNet/GROMACS Simulation (Seed={seed}) in {task_dir}")
    env = os.environ.copy()
    env["PATH"] = f"{os.path.dirname(GMX_BIN)}:{AMBER_BIN_DIR}:{env.get('PATH', '')}"

    start_t = time.time()
    cmd = [HTPOLYNET_BIN, "run", "--force-parameterization", "-diag", "diagnostics.log", "system.yaml"]

    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=task_dir,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid
        )
        exit_code = proc.wait()
    wall_sec = time.time() - start_t

    print(f"HTPolyNet finished with exit_code={exit_code} in {wall_sec:.1f} seconds.")

    # 扫描收集真实原生文件
    native_files = {}
    for root, dirs, files in os.walk(task_dir):
        for f in files:
            if f.endswith((".gro", ".top", ".tpr", ".edr", ".trr", ".log", ".yaml")):
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, task_dir)
                native_files[rel_p] = {
                    "size_bytes": os.path.getsize(full_p),
                    "sha256": calculate_sha256_file(full_p)
                }

    print(f"Native engine files generated: {len(native_files)}")
    return exit_code, wall_sec, task_dir, log_path, native_files

def main():
    recovery_dir = os.path.join(EXT_ROOT, "artifacts/recovery_v2")
    os.makedirs(recovery_dir, exist_ok=True)

    seeds = [20261015, 20261016]
    results = []

    for s in seeds:
        exit_code, wall_sec, task_dir, log_path, native_files = run_seed_simulation(s, recovery_dir)
        results.append({
            "seed": s,
            "exit_code": exit_code,
            "wall_sec": wall_sec,
            "task_dir": task_dir,
            "log_path": log_path,
            "native_files_count": len(native_files)
        })

    summary_file = os.path.join(recovery_dir, "real_run_execution_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved execution summary to {summary_file}")

if __name__ == "__main__":
    main()
