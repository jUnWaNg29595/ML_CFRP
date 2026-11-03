#!/bin/bash
# ==============================================================================
# REPRODUCE_REFERENCE.sh: DGEBA/DDS 参考体系真实后处理特征闭环一键复现脚本
# 完全自包含于 extensions/network_md_features/，产物隔离至 artifacts/postprocess_20261019_01/
# ==============================================================================
set -e

cd /home/wangj/project/CFRP/extensions/network_md_features

echo "======================================================================"
echo ">>> 1. 运行唯一的正式后处理管线 (postprocess_trajectory.py)"
echo "    - 真实读取 10 ps XTC 轨迹 (101 保存帧，无任何伪造随机数)"
echo "    - 原生 EDR 提取非键能量项并核对物理均值 (LJ-SR, Coul-SR, 1-4)"
echo "    - 3D 空间网格占据并集 FFV (严禁原子球体积直接相加)"
echo "    - 真实供受体几何与 PBC 氢键提取"
echo "    - NetworkX 真实成键图拓扑分析"
echo "======================================================================"
/home/wangj/miniconda3/envs/htpolynet_env/bin/python src/postprocess_trajectory.py     --input-attempt artifacts/attempt_20261019_01     --output-dir artifacts/postprocess_20261019_01     --stride 1

echo ""
echo "======================================================================"
echo ">>> 2. 查看生成的真实特征表 (artifacts/postprocess_20261019_01/md_features_diagnostic.csv)"
echo "======================================================================"
/home/wangj/miniconda3/envs/htpolynet_env/bin/python - << 'PYEOF'
import pandas as pd
df = pd.read_csv("artifacts/postprocess_20261019_01/md_features_diagnostic.csv")
cols = [
    "state_name", "validation_status", "training_admission_status",
    "sampling_temperature_K", "md_density_g_cm3",
    "md_vdw_energy_density_MJ_m3", "md_coulomb_energy_density_MJ_m3",
    "md_ffv_probe_0A", "md_ffv_probe_1_2A", "md_hbond_count_average",
    "charge_protocol"
]
print(df[cols].to_string())
PYEOF

echo ""
echo "======================================================================"
echo ">>> 3. 查看旧派生数据撤回清单 (artifacts/postprocess_20261019_01/RETRACTION_MANIFEST.json)"
echo "======================================================================"
head -n 35 artifacts/postprocess_20261019_01/RETRACTION_MANIFEST.json

echo ""
echo "======================================================================"
echo ">>> 4. 运行全量 49 项自动化测试套件 (包含 7 项真实入口故障注入)"
echo "======================================================================"
/home/wangj/miniconda3/envs/htpolynet_env/bin/python -m pytest tests/ -v

echo ""
echo ">>> 真实后处理闭环与核验流程全部完成！"
