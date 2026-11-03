#!/bin/bash
# CFRP 材料机器学习平台 WSL 启动脚本 (主平台 19 个页面)
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

CONDA_ENV_PATH="/home/wangj/miniconda3/envs/CFRP_env"

if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "❌ 找不到 Conda 环境: $CONDA_ENV_PATH"
    exit 1
fi

export PATH="$CONDA_ENV_PATH/bin:/home/wangj/software/xtb-6.7.1/bin:$PATH"
export XTBPATH="/home/wangj/miniconda3/envs/CFRP_env/share/xtb"

PORT="${ML_PORT:-8501}"
echo "============================================================"
echo "🚀 启动 CFRP 材料机器学习平台 (WSL Ubuntu-24.04)"
echo "   平台功能: 19 个完整功能页面（数据/特征/建模/筛选/预测等）"
echo "   Python 环境: $CONDA_ENV_PATH"
echo "   服务地址:    http://localhost:$PORT 或 http://127.0.0.1:$PORT"
echo "   提示: 请在 Windows 浏览器中打开上述地址（请保持此终端窗口运行）"
echo "============================================================"

streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0 --browser.serverAddress localhost
