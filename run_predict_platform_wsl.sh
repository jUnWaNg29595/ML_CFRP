#!/bin/bash
# CFRP 智能预测平台 WSL 启动脚本 (轻量用户门户)
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

CONDA_ENV_PATH="/home/wangj/miniconda3/envs/CFRP_env"

if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "❌ 找不到 Conda 环境: $CONDA_ENV_PATH"
    exit 1
fi

export PATH="$CONDA_ENV_PATH/bin:$PATH"

PORT="${PORTAL_PORT:-8555}"
echo "============================================================"
echo "🚀 启动 CFRP 材料预测平台 (WSL Ubuntu-24.04)"
echo "   平台功能: 轻量预测门户（直接预测性能）"
echo "   Python 环境: $CONDA_ENV_PATH"
echo "   服务地址:    http://localhost:$PORT"
echo "============================================================"

streamlit run UserPrediction.py --server.port "$PORT" --server.address 0.0.0.0
