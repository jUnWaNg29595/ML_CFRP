#!/bin/bash
# ==============================================================================
# REPRODUCE_BASELINE_AUDIT.sh: 真实文献基线复现与泛化瓶颈诊断一键复现脚本
# 完全自包含于 extensions/ml_generalization_v2/，不修改原系统任何文件
# ==============================================================================
set -e

cd /home/wangj/project/CFRP/extensions/ml_generalization_v2

echo "======================================================================"
echo ">>> 1. 运行数据粒度核验与特征/标签泄漏审查 (彻底撤回伪造 paper_id)"
echo "======================================================================"
/home/wangj/miniconda3/envs/htpolynet_env/bin/python src/audit_data_and_leakage.py

echo ""
echo "======================================================================"
echo ">>> 2. 真实文献基线复现 (Tg 与 25C 储能模量) 与泛化瓶颈诊断"
echo "       - Random 5-Fold (历史随机对照)"
echo "       - Formulation Group 5-Fold (严格配方留出，防同配方跨折泄漏)"
echo "       - 固定计算预算: Dummy, Ridge, RandomForest, GradientBoosting"
echo "======================================================================"
/home/wangj/miniconda3/envs/htpolynet_env/bin/python src/baseline_reproducer.py

echo ""
echo "======================================================================"
echo ">>> 3. 查看诊断核心指标"
echo "======================================================================"
/home/wangj/miniconda3/envs/htpolynet_env/bin/python src/print_metrics.py

echo ""
echo ">>> 文献基线与泛化诊断全流程复现完毕！产物已归档至 reports/ 与 artifacts/oof_predictions/。"
