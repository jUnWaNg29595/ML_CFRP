# 碳纤维复合材料智能预测平台 v1.4.0

## 🚀 项目简介

本平台是一个基于机器学习的碳纤维复合材料（CFRP）性能预测系统

## 🛠️ 安装

### 1. 创建环境

```bash
conda create -n CFRP_env python=3.10
conda activate CFRP_env
```

### 2. 安装PyTorch

```bash
# CPU版本
pip install torch torchvision torchaudio

# GPU版本（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 安装PyTorch Geometric（可选）

```bash
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
```
## 🚀 运行

```bash
streamlit run app.py
```
## 📄 许可证

MIT License
